from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_predictions_v2 import (
    calculate_cumulative_bookings,
    evaluate_booking_curves,
    extract_leadtimes,
    plot_actual_only_curves_lt_window,
    plot_booking_curves,
    plot_booking_curves_lt_window,
    plot_evaluation_metrics,
    save_cumulative_data,
    save_evaluation_results,
)


BASE_DIR = Path(__file__).resolve().parent
PREDICTIONS_FILE = (
    BASE_DIR
    / "results"
    / "cluster_only_lt_sarimax_parallel_v1"
    / "cluster_only_2025dec_100trial_cluster_lt_sarimax_predictions_v1.csv"
)
SOURCE_FILE = BASE_DIR / "展開データ_ほぼ処理完了_lt_expanded_v2_with_events_jp_clean_with_cluster.csv"
OUTPUT_DIR = BASE_DIR / "results" / "cluster_only_booking_curves_v2_style"

ANCHOR_LT = 90
LT_START = 90
LT_END = 0


def load_cluster_only_predictions() -> pd.DataFrame:
    predictions = pd.read_csv(PREDICTIONS_FILE, encoding="utf-8-sig")
    predictions["date"] = pd.to_datetime(predictions["date"])
    predictions["cluster"] = predictions["cluster"].astype(int)
    predictions["leadtime"] = predictions["leadtime"].astype(int)
    return predictions


def load_source_actuals() -> pd.DataFrame:
    source = pd.read_csv(
        SOURCE_FILE,
        encoding="utf-8-sig",
        usecols=["チェックイン日", "LT", "t", "クラスタ"],
    )
    source = source.rename(
        columns={
            "チェックイン日": "date",
            "LT": "leadtime",
            "クラスタ": "cluster",
        }
    )
    source["date"] = pd.to_datetime(source["date"])
    source["cluster"] = source["cluster"].astype(int)
    source["leadtime"] = source["leadtime"].astype(int)
    source["t"] = pd.to_numeric(source["t"], errors="coerce").fillna(0.0)
    return source


def build_analyze_v2_wide_input(
    cluster_id: int,
    predictions: pd.DataFrame,
    source_actuals: pd.DataFrame,
) -> pd.DataFrame:
    cluster_predictions = predictions[predictions["cluster"] == cluster_id].copy()
    if cluster_predictions.empty:
        raise ValueError(f"cluster {cluster_id} の予測行がありません。")

    target_dates = sorted(cluster_predictions["date"].unique())
    cluster_actuals = source_actuals[
        (source_actuals["cluster"] == cluster_id)
        & (source_actuals["date"].isin(target_dates))
    ].copy()
    if cluster_actuals.empty:
        raise ValueError(f"cluster {cluster_id} の実測行がありません。")

    actual_leadtimes = sorted(cluster_actuals["leadtime"].unique(), reverse=True)
    predicted_leadtimes = sorted(cluster_predictions["leadtime"].unique(), reverse=True)

    actual_pivot = cluster_actuals.pivot_table(
        index="date",
        columns="leadtime",
        values="t",
        aggfunc="sum",
        fill_value=0.0,
    ).reindex(index=target_dates, columns=actual_leadtimes, fill_value=0.0)

    predicted_pivot = cluster_predictions.pivot_table(
        index="date",
        columns="leadtime",
        values="predicted",
        aggfunc="mean",
    ).reindex(index=target_dates, columns=predicted_leadtimes)

    rows = []
    for date in target_dates:
        row = {
            "date": pd.Timestamp(date),
            "cluster": cluster_id,
            "model_scope": f"cluster_{cluster_id}",
        }
        for lt in actual_leadtimes:
            row[f"actual_LT{lt}"] = actual_pivot.loc[date, lt]
        for lt in predicted_leadtimes:
            row[f"predicted_LT{lt}"] = predicted_pivot.loc[date, lt]
        rows.append(row)

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def save_cluster_summary_plot(summary_df: pd.DataFrame, output_dir: Path) -> None:
    metrics = ["RMSE", "MAE", "MAPE", "R2"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    plot_df = summary_df.sort_values("cluster").copy()
    x_labels = [f"Cluster {int(cluster)}" for cluster in plot_df["cluster"]]

    for i, metric in enumerate(metrics):
        axes[i].bar(x_labels, plot_df[metric], color="tab:blue", alpha=0.85)
        axes[i].set_title(f"クラスタ別{metric}(LT{ANCHOR_LT}〜LT0)")
        axes[i].set_xlabel("クラスタ")
        axes[i].set_ylabel(metric)
        axes[i].grid(axis="y", alpha=0.35)

        for x_index, value in enumerate(plot_df[metric]):
            if pd.notna(value):
                axes[i].text(x_index, value, f"{value:.3f}", ha="center", va="bottom")

    fig.tight_layout()
    file_path = output_dir / "evaluation_metrics_by_cluster_v2_style.png"
    fig.savefig(file_path, bbox_inches="tight")
    plt.close(fig)
    print(f"クラスタ別評価グラフを保存しました: {file_path}")


def main() -> None:
    predictions = load_cluster_only_predictions()
    source_actuals = load_source_actuals()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    combined_evaluations = []
    combined_cumulative = []
    summary_rows = []

    for cluster_id in sorted(predictions["cluster"].unique()):
        cluster_output_dir = OUTPUT_DIR / f"cluster_{cluster_id}"
        cluster_output_dir.mkdir(parents=True, exist_ok=True)

        wide_df = build_analyze_v2_wide_input(cluster_id, predictions, source_actuals)
        wide_file = cluster_output_dir / "wide_predictions_for_analyze_v2.csv"
        wide_df.to_csv(wide_file, index=False, encoding="utf-8-sig")
        print(f"v2解析用の横持ち予測CSVを保存しました: {wide_file}")

        leadtimes = extract_leadtimes(wide_df)
        cumulative_df = calculate_cumulative_bookings(wide_df, leadtimes, anchor_lt=ANCHOR_LT)
        cumulative_df.insert(0, "cluster", cluster_id)

        evaluation_df = evaluate_booking_curves(cumulative_df, anchor_lt=ANCHOR_LT)
        evaluation_df.insert(0, "cluster", cluster_id)

        save_cumulative_data(cumulative_df, str(cluster_output_dir))
        save_evaluation_results(evaluation_df, str(cluster_output_dir))
        plot_evaluation_metrics(evaluation_df, str(cluster_output_dir))
        plot_booking_curves(cumulative_df, str(cluster_output_dir), anchor_lt=ANCHOR_LT)
        plot_booking_curves_lt_window(
            cumulative_df,
            output_dir=str(cluster_output_dir),
            anchor_lt=ANCHOR_LT,
            lt_start=LT_START,
            lt_end=LT_END,
        )
        plot_actual_only_curves_lt_window(
            cumulative_df,
            output_dir=str(cluster_output_dir),
            lt_start=LT_START,
            lt_end=LT_END,
        )

        combined_evaluations.append(evaluation_df)
        combined_cumulative.append(cumulative_df)

        overall = evaluation_df[evaluation_df["date"] == "Overall"].iloc[0].to_dict()
        overall["date_count"] = int(wide_df["date"].nunique())
        summary_rows.append(overall)

    combined_evaluation_df = pd.concat(combined_evaluations, ignore_index=True)
    combined_cumulative_df = pd.concat(combined_cumulative, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    combined_evaluation_file = OUTPUT_DIR / "evaluation_results_all_clusters_v2_style.csv"
    combined_cumulative_file = OUTPUT_DIR / "cumulative_bookings_all_clusters_v2_style.csv"
    summary_file = OUTPUT_DIR / "evaluation_summary_by_cluster_v2_style.csv"

    combined_evaluation_df.to_csv(combined_evaluation_file, index=False, encoding="utf-8-sig")
    combined_cumulative_df.to_csv(combined_cumulative_file, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")
    save_cluster_summary_plot(summary_df, OUTPUT_DIR)

    print(f"全クラスタ評価CSVを保存しました: {combined_evaluation_file}")
    print(f"全クラスタ累積CSVを保存しました: {combined_cumulative_file}")
    print(f"クラスタ別サマリーCSVを保存しました: {summary_file}")
    print("完了しました。")


if __name__ == "__main__":
    main()
