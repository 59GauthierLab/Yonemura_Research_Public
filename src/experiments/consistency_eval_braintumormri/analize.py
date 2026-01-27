import sys
from pathlib import Path
from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_modelwise_correlation(
    df: pd.DataFrame,
    method: Literal["pearson", "spearman"] = "pearson",
) -> pd.DataFrame:
    """
    model 別に kappa と consistency の相関係数を計算する。

    Parameters
    ----------
    df : pd.DataFrame
        必須カラム: ["model", "kappa", "consistency"]
    method : {"pearson", "spearman"}
        相関係数の種類

    Returns
    -------
    pd.DataFrame
        columns:
            - model
            - correlation
            - n_samples
    """

    required_columns = {"model", "kappa", "consistency"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    rows = []

    for model, sub_df in df.groupby("model"):
        # 欠損値除去
        clean_df = sub_df[["kappa", "consistency"]].dropna()

        n = len(clean_df)
        if n < 2:
            corr = float("nan")
        else:
            corr = clean_df["kappa"].corr(
                clean_df["consistency"], method=method
            )

        rows.append(
            {
                "model": model,
                "correlation": corr,
                "n_samples": n,
            }
        )

    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def plot_train_results(
    df: pd.DataFrame,
    graph_path: Path,
) -> None:
    """
    Kappa vs consistency の散布図を作成する。
    - 横軸: Kappa
    - 縦軸: consistency
    - model ごとに色分け（model名をソートして決定論的に割り当て）
    - model 別に相関（最小二乗）直線を描画
    """

    required_columns = {"model", "kappa", "consistency"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    # model 名をソートして固定順序を作る
    models = sorted(df["model"].unique())

    # colormap から model 数だけ色を取得（順序固定）
    cmap = plt.get_cmap("tab10")
    colors = {model: cmap(i % cmap.N) for i, model in enumerate(models)}

    fig, ax = plt.subplots(figsize=(8, 6))

    for model in models:
        sub_df = df[df["model"] == model]

        x = cast(pd.Series, sub_df["kappa"]).to_numpy()
        y = cast(pd.Series, sub_df["consistency"]).to_numpy()

        # 散布図
        ax.scatter(
            x,
            y,
            label=model,
            color=colors[model],
            alpha=0.7,
        )

        # データ点が2点以上ある場合のみ回帰直線を描画
        if len(x) >= 2:
            # 最小二乗直線 y = a x + b
            a, b = np.polyfit(x, y, deg=1)

            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = a * x_line + b

            ax.plot(
                x_line,
                y_line,
                color=colors[model],
                linestyle="--",
                linewidth=2,
            )

    ax.set_xlabel("Kappa")
    ax.set_ylabel("Consistency")
    ax.set_title("Kappa vs Grad-CAM Consistency (with correlation lines)")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(graph_path)
    plt.close(fig)


def analize(
    train_results: pd.DataFrame,
    save_dir: Path,
) -> None:
    """学習結果の解析を行い、グラフと相関係数を保存する。

    Parameters
    ----------
    train_results : pd.DataFrame
        学習結果データフレーム。必須カラム: ["model", "kappa", "consistency"]
    save_dir : Path
        解析結果の保存先ディレクトリ
    """

    save_dir.mkdir(parents=True, exist_ok=True)

    # 散布図作成
    graph_path = save_dir / "kappa_vs_consistency.png"
    plot_train_results(train_results, graph_path)

    # model1を除いた散布図作成(model1は他モデルと大きく異なるため)
    graph_path_without_model1 = (
        save_dir / "kappa_vs_consistency_without_model1.png"
    )
    plot_train_results(
        train_results[train_results["model"] != "model1"],
        graph_path_without_model1,
    )

    # 相関係数計算
    corr_df = compute_modelwise_correlation(train_results, method="pearson")
    corr_path = save_dir / "modelwise_correlation.csv"
    corr_df.to_csv(corr_path, index=False)

    print(f"Analysis results saved to: {save_dir}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python analize.py <train_results_csv path>")
        sys.exit(1)

    train_results_csv = Path(sys.argv[1])

    train_results = pd.read_csv(train_results_csv)
    save_dir = train_results_csv.parent

    analize(train_results, save_dir)


if __name__ == "__main__":
    main()
