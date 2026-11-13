from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_predictions_v2 import (
    evaluate_booking_curves,
    plot_actual_only_curves_lt_window,
    plot_booking_curves,
    plot_booking_curves_lt_window,
    plot_evaluation_metrics,
    save_cumulative_data,
    save_evaluation_results,
)


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = (
    BASE_DIR
    / "results"
    / "cluster_only_booking_curves_v2_style"
    / "cumulative_bookings_all_clusters_v2_style.csv"
)
OUTPUT_DIR = BASE_DIR / "results" / "december_connected_booking_curves_v2_style"

ANCHOR_LT = 90
LT_START = 90
LT_END = 0


def load_connected_cumulative() -> pd.DataFrame:
    cumulative = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    cumulative["date"] = pd.to_datetime(cumulative["date"])
    cumulative["cluster"] = cumulative["cluster"].astype(int)
    cumulative["leadtime"] = cumulative["leadtime"].astype(int)
    return cumulative.sort_values(["date", "leadtime"], ascending=[True, False]).reset_index(drop=True)


def add_cluster_to_evaluation(evaluation_df: pd.DataFrame, cumulative_df: pd.DataFrame) -> pd.DataFrame:
    date_cluster = (
        cumulative_df[["date", "cluster"]]
        .drop_duplicates()
        .assign(date_key=lambda df: df["date"].dt.strftime("%Y-%m-%d"))
        .set_index("date_key")["cluster"]
        .to_dict()
    )

    result = evaluation_df.copy()
    result.insert(1, "cluster", result["date"].apply(lambda value: date_cluster.get(str(value)[:10], "all")))
    return result


def save_cluster_color_all_dates_plot(cumulative_df: pd.DataFrame, output_dir: Path) -> None:
    window_df = cumulative_df[
        cumulative_df["leadtime"].between(min(LT_START, LT_END), max(LT_START, LT_END))
    ].copy()

    cluster_colors = {
        0: "tab:blue",
        1: "tab:orange",
        3: "tab:green",
    }

    fig, ax = plt.subplots(figsize=(15, 10))
    used_labels = set()

    for date in sorted(window_df["date"].unique()):
        date_df = window_df[window_df["date"] == date].sort_values("leadtime", ascending=False)
        cluster_id = int(date_df["cluster"].iloc[0])
        color = cluster_colors.get(cluster_id, "tab:gray")

        actual_label = f"Cluster {cluster_id} 実測"
        pred_label = f"Cluster {cluster_id} 予測"
        if actual_label in used_labels:
            actual_label = None
        else:
            used_labels.add(actual_label)
        if pred_label in used_labels:
            pred_label = None
        else:
            used_labels.add(pred_label)

        ax.plot(
            date_df["leadtime"],
            date_df["actual_cumsum"],
            "-",
            color=color,
            alpha=0.45,
            linewidth=1.8,
            label=actual_label,
        )
        ax.plot(
            date_df["leadtime"],
            date_df["predicted_cumsum"],
            "--",
            color=color,
            alpha=0.8,
            linewidth=1.8,
            label=pred_label,
        )

    ax.axvline(ANCHOR_LT, color="gray", linestyle=":", label=f"LT{ANCHOR_LT}アンカー")
    ax.set_title(f"12月全体ブッキングカーブ LT{LT_START}〜{LT_END} クラスタ色分け")
    ax.set_xlabel("リードタイム（日）")
    ax.set_ylabel("累積利用客室数")
    ax.invert_xaxis()
    ax.grid(True)
    ax.legend(loc="best")
    fig.tight_layout()

    file_path = output_dir / "booking_curves_december_all_dates_cluster_color_v2.png"
    fig.savefig(file_path, bbox_inches="tight")
    plt.close(fig)
    print(f"12月全体のクラスタ色分けグラフを保存しました: {file_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cumulative_df = load_connected_cumulative()

    save_cumulative_data(cumulative_df, str(OUTPUT_DIR))

    evaluation_df = evaluate_booking_curves(cumulative_df, anchor_lt=ANCHOR_LT)
    evaluation_df = add_cluster_to_evaluation(evaluation_df, cumulative_df)
    save_evaluation_results(evaluation_df, str(OUTPUT_DIR))

    plot_evaluation_metrics(evaluation_df, str(OUTPUT_DIR))
    plot_booking_curves(cumulative_df, str(OUTPUT_DIR), anchor_lt=ANCHOR_LT)
    plot_booking_curves_lt_window(
        cumulative_df,
        output_dir=str(OUTPUT_DIR),
        anchor_lt=ANCHOR_LT,
        lt_start=LT_START,
        lt_end=LT_END,
    )
    plot_actual_only_curves_lt_window(
        cumulative_df,
        output_dir=str(OUTPUT_DIR),
        lt_start=LT_START,
        lt_end=LT_END,
    )
    save_cluster_color_all_dates_plot(cumulative_df, OUTPUT_DIR)

    overall = evaluation_df[evaluation_df["date"] == "Overall"].iloc[0]
    print("\n===== 12月結合版 Overall =====")
    print(
        f"RMSE={overall['RMSE']:.6f}, "
        f"MAE={overall['MAE']:.6f}, "
        f"MAPE={overall['MAPE']:.6f}, "
        f"R2={overall['R2']:.6f}"
    )
    print("=============================")


if __name__ == "__main__":
    main()
