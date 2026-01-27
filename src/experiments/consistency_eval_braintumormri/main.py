from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import atexit
from typing import List, Sequence, cast

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from analize import analize
from dataset import (
    denormalize,
    provider,
    sample_data,
    target_transform,
    test_dataset,
    train_dataset,
)
from grad_cam import is_gradcam
from grad_cam_consistency import calc_consistency
from grad_cam_image import save_result_multiclass
from kappa_calculator import KappaCalculator
from models import models_classes, predict
from torch.utils.data import DataLoader, Subset

from config import config
from utils import ExperimentManager, ModelManager
from utils.metrics import MulticlassAccuracy
from utils.notify import epoch_message, send_discord_notification, timestamp
from utils.random import fix_seed, gen_worker_init_fn


@atexit.register
def on_exit():
    send_discord_notification(f"# Program ended\nTimestamp: {timestamp()}")


# NOTE: 乱数が関わる処理は全てmain.py内に記述すること

# -------------------------------
# 乱数設定
# -------------------------------
seed = fix_seed(seed=None, strict=False)

g = torch.Generator()
g.manual_seed(seed)


# -------------------------------
# データローダー生成
# -------------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    generator=g,
    worker_init_fn=gen_worker_init_fn(seed),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    generator=g,
    worker_init_fn=gen_worker_init_fn(seed),
)

sample_loader = DataLoader(
    dataset=sample_data,
    batch_size=1,
    shuffle=False,
    num_workers=0,
)

# -------------------------------
# モデル生成
# -------------------------------

# モデル管理機構初期化
expt_manager = ExperimentManager()

for i, model_cls in enumerate(models_classes, start=1):
    model = model_cls(num_channels=1, num_classes=provider.get_num_labels())
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=5,
    )

    manager = ModelManager(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        metric=MulticlassAccuracy(),
        scheduler=[(scheduler, "validation")],
        target_transform=target_transform,
        device=config.env.device,
        seed=seed,
        dataset_provider=provider,
    )

    expt_manager.add_model_manager(f"model{i}", manager)


# 追加のメモリ(gradcam一貫性、Cohen’s kappa)初期化
def mm_memory_init(name: str, mm: ModelManager) -> None:
    mm.memory["gradcam_consistency"] = []
    mm.memory["kappa"] = []


expt_manager.for_each_all(mm_memory_init)


# -------------------------------
# 学習・評価
# -------------------------------

# Cohen’s kappa 計算機設定
kappa_calculator = KappaCalculator(
    num_classes=provider.get_num_labels(),
    models=[mm.model for _, mm in expt_manager],
    predict=predict,
    target_transform=target_transform,
    device=config.env.device,
    dataset=test_dataset,
)


# エポック毎フック関数定義
def epoch_hook(
    model_managers: Sequence[ModelManager], epoch: int, total_epochs: int
) -> None:
    # 一貫性評価用データ保存
    print("Calculating Grad-CAM consistency...")
    for mm in model_managers:
        # 型ガード(エディタ補完用)
        if not is_gradcam(mm.model):
            continue

        consistency, total_consistency = calc_consistency(
            dataset=sample_data,
            model=mm.model,
            device=config.env.device,
            predict=predict,
        )

        cast(list[float], mm.memory["gradcam_consistency"]).append(
            (total_consistency)
        )

    # kappa係数計算
    print("Calculating Cohen's Kappa...")
    kappas = kappa_calculator.calc_kappa()
    for i, (_, mm) in enumerate(expt_manager):
        kappa = kappas[i]
        cast(list[float], mm.memory["kappa"]).append(kappa)

    # Discord通知
    message = epoch_message(
        model_managers=model_managers,
        epoch=epoch,
        total_epochs=total_epochs,
    )
    send_discord_notification(message)


# 学習開始
send_discord_notification(f"# Training started\nTimestamp: {timestamp()}")
expt_manager.train_all(
    train_loader=train_loader,
    valid_loader=test_loader,
    epochs=20,
    device=config.env.device,
    epoch_hook=epoch_hook,
)
expt_manager.save_train_logs_all()
expt_manager.save_train_graphs_all()

# テスト開始
send_discord_notification(f"# Test started\nTimestamp: {timestamp()}")
expt_manager.test_all(
    test_loader=test_loader,
    device=config.env.device,
)
expt_manager.save_test_logs_all()

expt_manager.save_model_summary(include_memory=True)

# -------------------------------
# 学習結果保存
# -------------------------------


# CSVで保存
def get_train_results(
    name: str, manager: ModelManager
) -> dict[str, List[str | float | int]]:
    train_history = manager.get_train_data_safety().history
    result = {
        "model": [name for _ in range(len(train_history))],
        "epochs": [i + 1 for i in range(len(train_history))],
        "accuracy": [his.valid_accuracy for his in train_history],
        "consistency": manager.memory["gradcam_consistency"],
        "kappa": manager.memory["kappa"],
    }
    return result


result = expt_manager.map_all(get_train_results)

rows: list[dict[str, str | float | int]] = []

for res in result:
    names = res["model"]
    epochs = res["epochs"]
    accuracies = res["accuracy"]
    consistencies = res["consistency"]
    kappas = res["kappa"]

    for name, epoch, acc, cons, kappa in zip(
        names, epochs, accuracies, consistencies, kappas
    ):
        rows.append(
            {
                "model": name,
                "epoch": epoch,
                "accuracy": acc,
                "consistency": cons,
                "kappa": kappa,
            }
        )

df = pd.DataFrame(
    rows, columns=["model", "epoch", "accuracy", "consistency", "kappa"]
)
csv_path = expt_manager.root_dir / "train_results.csv"
df.to_csv(csv_path, index=False)

# グラフ化 & 相関解析
analize(train_results=df, save_dir=expt_manager.root_dir)

# -------------------------------
# Grad-CAM保存
# -------------------------------

grad_cam_image_sample_loader = DataLoader(
    dataset=Subset(
        test_dataset,
        indices=(0, 1, 2, 3, 4),
    ),
    batch_size=1,
    shuffle=False,
    num_workers=0,
)


def save_grad_cam(name: str, manager: ModelManager):
    if not is_gradcam(manager.model):
        raise ValueError("Model is not Grad-CAM compatible.")

    save_result_multiclass(
        manager=manager,
        model=manager.model,
        data_loader=grad_cam_image_sample_loader,
        classes=provider.get_label_names(),
        denormalize=denormalize,
        predict=predict,
        device=config.env.device,
    )


# CAM画像はあまり意味がないので保存しない(コメントアウト)

# send_discord_notification(
#     f"# Saving Grad-CAM started\nTimestamp: {timestamp()}"
# )
# expt_manager.for_each_all(save_grad_cam)

print("Done.")
