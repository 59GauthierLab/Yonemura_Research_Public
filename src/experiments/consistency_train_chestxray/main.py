from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import atexit
import re
from typing import List, Sequence, cast

import pandas as pd
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from analyze import analyze
from dataset import (
    input_size,
    provider,
    sample_data,
    target_transform,
    test_dataset,
    train_dataset,
)
from kappa_calculator import KappaCalculator
from loss_func import CamConsistencyLossWithCrossEntropy, LambdaScheduler
from models import Model1, predict
from torch.utils.data import DataLoader

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

# λを変えて複数モデル生成
model_name: str = "Model1"
model_cls = Model1
lamda_values: List[float] = [-100.0, 0.0, 50.0, 100.0, 500.0]

for i, lamda_value in enumerate(lamda_values, start=1):
    model = model_cls(
        num_channels=input_size.channels, num_classes=provider.get_num_labels()
    )
    model.cam_enable()
    dummy_model = model_cls(
        num_channels=input_size.channels, num_classes=provider.get_num_labels()
    )  # CAMサイズ取得用ダミーモデル

    # CAMサイズ取得
    cam_h, cam_w = dummy_model.get_cam_size(
        input_size.height, input_size.width, input_size.channels
    )
    del dummy_model  # メモリを使うので明示的に削除
    torch.cuda.empty_cache()  # 念のためキャッシュもクリア

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

    criterion = CamConsistencyLossWithCrossEntropy(
        model=model,
        class_num=provider.get_num_labels(),
        cam_size=(cam_h, cam_w),
        ema_alpha=0.9,
    )

    lambda_scheduler = LambdaScheduler(
        loss_module=criterion,
        start_lambda=0,
        final_lambda=lamda_value,
        t0=2,
        t1=10,
    )

    manager = ModelManager(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        metric=MulticlassAccuracy(),
        scheduler=[(scheduler, "validation"), (lambda_scheduler, "epoch")],
        target_transform=target_transform,
        device=config.env.device,
        seed=seed,
        dataset_provider=provider,
    )

    expt_manager.add_model_manager(
        f"{model_name}_lambda{lamda_value}", manager
    )


# 追加のメモリ(gradcam一貫性、Cohen’s kappa)初期化
def mm_memory_init(name: str, mm: ModelManager) -> None:
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
    epochs=50,
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
        "loss": [his.valid_loss for his in train_history],
        "accuracy": [his.valid_accuracy for his in train_history],
        "kappa": manager.memory["kappa"],
    }
    return result


result = expt_manager.map_all(get_train_results)

rows: list[dict[str, str | float | int]] = []

modelname_pattern = re.compile(r"^(.*)_lambda([-+]?\d*\.?\d+)$")

for res in result:
    modelname = cast(str, res["model"][0])
    match = modelname_pattern.match(modelname)
    if not match:
        raise RuntimeError(
            f"Model name '{modelname}' does not match the pattern."
        )

    name_ = match.group(1)
    lambda_value_ = float(match.group(2))

    names = [name_ for _ in res["model"]]
    lambdas = [lambda_value_ for _ in res["model"]]
    epochs = res["epochs"]
    losses = res["loss"]
    accuracies = res["accuracy"]
    kappas = res["kappa"]

    for name, lambda_, epoch, loss, acc, kappa in zip(
        names, lambdas, epochs, losses, accuracies, kappas
    ):
        rows.append(
            {
                "model": name,
                "lambda": lambda_,
                "epoch": epoch,
                "loss": loss,
                "accuracy": acc,
                "kappa": kappa,
            }
        )

df = pd.DataFrame(
    rows, columns=["model", "lambda", "epoch", "loss", "accuracy", "kappa"]
)
csv_path = expt_manager.root_dir / "train_results.csv"
df.to_csv(csv_path, index=False)

# グラフ化
analyze(train_results=df, save_dir=expt_manager.root_dir)


print("Done.")
