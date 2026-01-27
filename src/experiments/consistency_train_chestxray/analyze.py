import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _get_lambda_color_map(lambdas: list[float]):
    """
    Create a deterministic mapping from lambda value to color.
    """
    unique_lambdas = sorted(lambdas)
    cmap = plt.get_cmap("tab10")  # 10色以上必要なら tab20
    return {lam: cmap(i % cmap.N) for i, lam in enumerate(unique_lambdas)}


def plot_train_loss(df: pd.DataFrame, graph_path: Path) -> None:
    """
    Plot training loss curves.
    """
    required_cols = {"model", "lambda", "epoch", "loss"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_cols}")

    for model_name, df_model in df.groupby("model"):
        plt.figure(figsize=(8, 6))

        lambdas = df_model["lambda"].unique().tolist()
        lambda_color = _get_lambda_color_map(lambdas)

        # lambda 昇順で描画
        for lam in sorted(lambdas):
            df_lam = df_model[df_model["lambda"] == lam].sort_values("epoch")
            plt.plot(
                df_lam["epoch"],
                df_lam["loss"],
                label=f"λ = {lam}",
                color=lambda_color[lam],
            )

        plt.title(f"Training Loss ({model_name})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        out_path = graph_path.with_stem(f"{graph_path.stem}_{model_name}")
        plt.savefig(out_path)
        plt.close()


def plot_train_accuracy(df: pd.DataFrame, graph_path: Path) -> None:
    """
    Plot training accuracy curves.
    """
    required_cols = {"model", "lambda", "epoch", "accuracy"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_cols}")

    for model_name, df_model in df.groupby("model"):
        plt.figure(figsize=(8, 6))

        lambdas = df_model["lambda"].unique().tolist()
        lambda_color = _get_lambda_color_map(lambdas)

        # lambda 昇順で描画
        for lam in sorted(lambdas):
            df_lam = df_model[df_model["lambda"] == lam].sort_values("epoch")
            plt.plot(
                df_lam["epoch"],
                df_lam["accuracy"],
                label=f"λ = {lam}",
                color=lambda_color[lam],
            )

        plt.title(f"Training Accuracy ({model_name})")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        out_path = graph_path.with_stem(f"{graph_path.stem}_{model_name}")
        plt.savefig(out_path)
        plt.close()


def analyze(train_results: pd.DataFrame, save_dir: Path) -> None:
    """学習結果の解析を行い、グラフを保存する。

    Parameters
    ----------
    train_results : pd.DataFrame
        学習結果データフレーム。必須カラム: ["model", "lambda", "epoch", "loss", "accuracy", "kappa"]
    save_dir : Path
        解析結果の保存先ディレクトリ
    """

    save_dir.mkdir(parents=True, exist_ok=True)

    # 学習過程のグラフ
    plot_train_loss(train_results, save_dir / "train_result_loss.png")
    plot_train_accuracy(train_results, save_dir / "train_result_accuracy.png")

    print(f"Analysis results saved to: {save_dir}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze.py <train_results_csv path>")
        sys.exit(1)

    train_results_csv = Path(sys.argv[1])

    train_results = pd.read_csv(train_results_csv)
    save_dir = train_results_csv.parent

    analyze(train_results, save_dir)


if __name__ == "__main__":
    main()
