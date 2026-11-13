import argparse
import os
from pathlib import Path

import japanize_matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parent


def load_predictions(file_path):
    """予測結果CSVを読み込む。"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        print(f'ファイルの読み込み中にエラーが発生しました: {str(e)}')
        return None


def extract_leadtimes(df):
    """actual/predicted列からリードタイム一覧を抽出する。"""
    lt_columns = [col for col in df.columns if col.startswith('actual_LT') or col.startswith('predicted_LT')]
    leadtimes = []

    for col in lt_columns:
        if col.startswith('actual_LT'):
            lt = int(col.replace('actual_LT', ''))
        else:
            lt = int(col.replace('predicted_LT', ''))
        if lt not in leadtimes:
            leadtimes.append(lt)

    return sorted(leadtimes, reverse=True)


def calculate_cumulative_bookings(df, leadtimes, anchor_lt: int = 90):
    """LTアンカー固定の累積利用客室数を計算する。"""
    ordered_leadtimes = sorted(leadtimes, reverse=True)
    result_rows = []

    for date in sorted(df['date'].unique()):
        date_df = df[df['date'] == date].copy()
        cumsum_actual = 0.0
        cumsum_blended = 0.0

        for lt in ordered_leadtimes:
            actual_col = f'actual_LT{lt}'
            predicted_col = f'predicted_LT{lt}'

            actual_increment = 0.0
            predicted_increment = np.nan

            if actual_col in date_df.columns and not pd.isna(date_df[actual_col].values[0]):
                actual_increment = float(date_df[actual_col].values[0])

            if predicted_col in date_df.columns and not pd.isna(date_df[predicted_col].values[0]):
                predicted_increment = float(date_df[predicted_col].values[0])

            if lt > anchor_lt or pd.isna(predicted_increment):
                blended_increment = actual_increment
                increment_source = 'actual'
            else:
                blended_increment = predicted_increment
                increment_source = 'predicted'

            cumsum_actual += actual_increment
            cumsum_blended += blended_increment

            result_rows.append({
                'date': date,
                'leadtime': lt,
                'actual_increment': actual_increment,
                'predicted_increment': predicted_increment,
                'blended_increment': blended_increment,
                'increment_source': increment_source,
                'actual_cumsum': cumsum_actual,
                'predicted_cumsum': cumsum_blended,
            })

    return pd.DataFrame(result_rows)


def evaluate_booking_curves(
    cumulative_df,
    anchor_lt: int = 90,
    mape_min_actual: float = 1.0,
):
    """LTアンカー以下の予測区間を評価する。"""
    daily_rows = []
    daily_series = {}

    for date in cumulative_df['date'].unique():
        date_df = cumulative_df[
            (cumulative_df['date'] == date) & (cumulative_df['leadtime'] <= anchor_lt)
        ].sort_values('leadtime', ascending=False)

        if date_df.empty:
            continue

        actual = date_df['actual_cumsum'].values
        predicted = date_df['predicted_cumsum'].values

        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)

        valid_mask = actual >= mape_min_actual
        if np.any(valid_mask):
            actual_valid = actual[valid_mask]
            predicted_valid = predicted[valid_mask]
            mape = np.mean(np.abs((actual_valid - predicted_valid) / actual_valid)) * 100
        else:
            mape = np.nan
            actual_valid = np.array([])
            predicted_valid = np.array([])

        r2 = r2_score(actual, predicted)

        daily_rows.append({
            'date': date,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2,
        })

        daily_series[pd.Timestamp(date)] = {
            'actual': actual,
            'predicted': predicted,
            'mape_actual': actual_valid,
            'mape_predicted': predicted_valid,
        }

    evaluation_df = pd.DataFrame(daily_rows)

    all_actual = []
    all_predicted = []
    mape_actual_values = []
    mape_predicted_values = []
    for date in evaluation_df['date']:
        series = daily_series.get(pd.Timestamp(date))
        if series is None:
            continue
        all_actual.extend(series['actual'])
        all_predicted.extend(series['predicted'])
        if series['mape_actual'].size > 0:
            mape_actual_values.extend(series['mape_actual'])
            mape_predicted_values.extend(series['mape_predicted'])

    if all_actual:
        overall_rmse = np.sqrt(mean_squared_error(all_actual, all_predicted))
        overall_mae = mean_absolute_error(all_actual, all_predicted)
        overall_r2 = r2_score(all_actual, all_predicted)
    else:
        overall_rmse = np.nan
        overall_mae = np.nan
        overall_r2 = np.nan

    if mape_actual_values:
        mape_actual_arr = np.array(mape_actual_values)
        mape_predicted_arr = np.array(mape_predicted_values)
        overall_mape = np.mean(np.abs((mape_actual_arr - mape_predicted_arr) / mape_actual_arr)) * 100
    else:
        overall_mape = np.nan

    overall_row = {
        'date': 'Overall',
        'RMSE': overall_rmse,
        'MAE': overall_mae,
        'MAPE': overall_mape,
        'R2': overall_r2,
    }
    evaluation_df = pd.concat([evaluation_df, pd.DataFrame([overall_row])], ignore_index=True)

    return evaluation_df


def plot_evaluation_metrics(evaluation_df, output_dir='results/booking_curves_v2'):
    os.makedirs(output_dir, exist_ok=True)

    plot_df = evaluation_df[evaluation_df['date'] != 'Overall'].copy()
    plot_df['date'] = pd.to_datetime(plot_df['date'])
    plot_df = plot_df.sort_values('date')

    metrics = ['RMSE', 'MAE', 'MAPE', 'R2']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        axes[i].plot(plot_df['date'], plot_df[metric], 'o-', linewidth=2)
        axes[i].set_title(f'日付ごとの{metric}(LTアンカー以降のみ)')
        axes[i].set_xlabel('日付')
        axes[i].set_ylabel(metric)
        axes[i].grid(True)
        plt.setp(axes[i].get_xticklabels(), rotation=45, ha='right')

        overall_value = evaluation_df[evaluation_df['date'] == 'Overall'][metric].values[0]
        axes[i].axhline(y=overall_value, color='r', linestyle='--', label=f'全体平均: {overall_value:.4f}')
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/evaluation_metrics_v2.png')
    plt.close()


def plot_booking_curves(cumulative_df, output_dir='results/booking_curves_v2', anchor_lt: int = 90):
    os.makedirs(output_dir, exist_ok=True)

    for date in sorted(cumulative_df['date'].unique()):
        date_df = cumulative_df[cumulative_df['date'] == date].sort_values('leadtime', ascending=False)
        predicted_plot = date_df['predicted_cumsum'].where(date_df['leadtime'] <= anchor_lt, np.nan)

        plt.figure(figsize=(12, 6))
        plt.plot(date_df['leadtime'], date_df['actual_cumsum'], 'o-', label='実際の累積利用客室数')
        plt.plot(date_df['leadtime'], predicted_plot, 'o-', label=f'LT{anchor_lt}アンカー予測累積利用客室数')
        plt.axvline(anchor_lt, color='gray', linestyle=':', label=f'LT{anchor_lt}アンカー')

        plt.title(f'{date.strftime("%Y-%m-%d")}のブッキングカーブ(v2)')
        plt.xlabel('リードタイム（日）')
        plt.ylabel('累積利用客室数')
        plt.gca().invert_xaxis()
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        file_name = f'{output_dir}/booking_curve_{date.strftime("%Y%m%d")}_v2.png'
        plt.savefig(file_name)
        plt.close()

        print(f"{date.strftime('%Y-%m-%d')}のブッキングカーブを保存しました: {file_name}")

    plt.figure(figsize=(15, 10))
    dates = cumulative_df['date'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(dates)))

    for i, date in enumerate(dates):
        date_df = cumulative_df[cumulative_df['date'] == date].sort_values('leadtime', ascending=False)
        predicted_plot = date_df['predicted_cumsum'].where(date_df['leadtime'] <= anchor_lt, np.nan)
        plt.plot(date_df['leadtime'], date_df['actual_cumsum'], 'o-', color=colors[i], label=f'{date.strftime("%Y-%m-%d")} 実際')
        plt.plot(date_df['leadtime'], predicted_plot, 's--', color=colors[i], alpha=0.7, label=f'{date.strftime("%Y-%m-%d")} 予測')

    plt.axvline(anchor_lt, color='gray', linestyle=':', label=f'LT{anchor_lt}アンカー')
    plt.title('全日程のブッキングカーブ比較(v2)')
    plt.xlabel('リードタイム（日）')
    plt.ylabel('累積利用客室数')
    plt.gca().invert_xaxis()
    legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    for text in legend.get_texts():
        text.set_rotation(45)
    plt.grid(True)
    plt.gcf().subplots_adjust(right=0.78)

    all_dates_file = f'{output_dir}/booking_curves_all_dates_v2.png'
    plt.savefig(all_dates_file, bbox_inches='tight')
    plt.close()

    print(f'全日程のブッキングカーブ比較を保存しました: {all_dates_file}')

    leadtime_order = sorted(cumulative_df['leadtime'].unique(), reverse=True)
    pivot_actual = cumulative_df.pivot(index='date', columns='leadtime', values='actual_cumsum').sort_index().reindex(columns=leadtime_order)
    pivot_predicted = cumulative_df.pivot(index='date', columns='leadtime', values='predicted_cumsum').sort_index().reindex(columns=leadtime_order)

    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_actual, cmap='viridis', annot=True, fmt='.1f', linewidths=.5)
    plt.title('実際の累積利用客室数ヒートマップ(v2)')
    plt.ylabel('日付')
    plt.xlabel('リードタイム（日）')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/heatmap_actual_cumsum_v2.png')
    plt.close()

    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_predicted, cmap='viridis', annot=True, fmt='.1f', linewidths=.5)
    plt.title(f'LT{anchor_lt}アンカー予測累積利用客室数ヒートマップ(v2)')
    plt.ylabel('日付')
    plt.xlabel('リードタイム（日）')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/heatmap_predicted_cumsum_v2.png')
    plt.close()

    diff = pivot_predicted - pivot_actual
    plt.figure(figsize=(14, 8))
    sns.heatmap(diff, cmap='RdBu_r', annot=True, fmt='.1f', linewidths=.5, center=0)
    plt.title('予測値と実際値の差分ヒートマップ(v2)')
    plt.ylabel('日付')
    plt.xlabel('リードタイム（日）')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/heatmap_difference_v2.png')
    plt.close()


def filter_leadtime_window(cumulative_df, lt_start: int, lt_end: int) -> pd.DataFrame:
    """指定したLT範囲のデータだけを抽出する。"""
    lower = min(lt_start, lt_end)
    upper = max(lt_start, lt_end)
    filtered = cumulative_df[cumulative_df['leadtime'].between(lower, upper)].copy()
    return filtered.sort_values(['date', 'leadtime'], ascending=[True, False]).reset_index(drop=True)


def plot_booking_curves_lt_window(
    cumulative_df,
    output_dir='results/booking_curves_v2_lt90_to_0',
    anchor_lt: int = 90,
    lt_start: int = 90,
    lt_end: int = 0,
):
    """LT範囲を固定した実測値+予測値のブッキングカーブを描画する。"""
    os.makedirs(output_dir, exist_ok=True)
    window_df = filter_leadtime_window(cumulative_df, lt_start, lt_end)

    if window_df.empty:
        print(f'LT{lt_start}〜{lt_end} の範囲に該当するデータがありません。')
        return

    for date in sorted(window_df['date'].unique()):
        date_df = window_df[window_df['date'] == date].sort_values('leadtime', ascending=False)
        predicted_plot = date_df['predicted_cumsum'].where(date_df['leadtime'] <= anchor_lt, np.nan)

        plt.figure(figsize=(12, 6))
        plt.plot(date_df['leadtime'], date_df['actual_cumsum'], 'o-', linewidth=2, label='実測累積値')
        plt.plot(date_df['leadtime'], predicted_plot, 's--', linewidth=2, label='予測累積値')
        plt.axvline(anchor_lt, color='gray', linestyle=':', label=f'LT{anchor_lt}アンカー')

        plt.title(f'{date.strftime("%Y-%m-%d")}のブッキングカーブ LT{lt_start}〜{lt_end}')
        plt.xlabel('リードタイム（日）')
        plt.ylabel('累積利用客室数')
        plt.gca().invert_xaxis()
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        file_name = f'{output_dir}/booking_curve_{date.strftime("%Y%m%d")}_lt{lt_start}_to_{lt_end}_actual_vs_pred_v2.png'
        plt.savefig(file_name)
        plt.close()
        print(f"{date.strftime('%Y-%m-%d')} の実測値+予測値グラフを保存しました: {file_name}")

    plt.figure(figsize=(15, 10))
    dates = sorted(window_df['date'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(dates)))

    for i, date in enumerate(dates):
        date_df = window_df[window_df['date'] == date].sort_values('leadtime', ascending=False)
        predicted_plot = date_df['predicted_cumsum'].where(date_df['leadtime'] <= anchor_lt, np.nan)
        plt.plot(date_df['leadtime'], date_df['actual_cumsum'], 'o-', color=colors[i], label=f'{date.strftime("%Y-%m-%d")} 実測')
        plt.plot(date_df['leadtime'], predicted_plot, 's--', color=colors[i], alpha=0.8, label=f'{date.strftime("%Y-%m-%d")} 予測')

    plt.axvline(anchor_lt, color='gray', linestyle=':', label=f'LT{anchor_lt}アンカー')
    plt.title(f'全日程ブッキングカーブ比較 LT{lt_start}〜{lt_end} 実測値+予測値')
    plt.xlabel('リードタイム（日）')
    plt.ylabel('累積利用客室数')
    plt.gca().invert_xaxis()
    legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    for text in legend.get_texts():
        text.set_rotation(45)
    plt.grid(True)
    plt.gcf().subplots_adjust(right=0.78)

    file_name = f'{output_dir}/booking_curves_all_dates_lt{lt_start}_to_{lt_end}_actual_vs_pred_v2.png'
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()
    print(f'全日程の実測値+予測値グラフを保存しました: {file_name}')


def plot_actual_only_curves_lt_window(
    cumulative_df,
    output_dir='results/booking_curves_v2_lt90_to_0',
    lt_start: int = 90,
    lt_end: int = 0,
):
    """LT範囲を固定した実測値のみのブッキングカーブを描画する。"""
    os.makedirs(output_dir, exist_ok=True)
    window_df = filter_leadtime_window(cumulative_df, lt_start, lt_end)

    if window_df.empty:
        print(f'LT{lt_start}〜{lt_end} の範囲に該当するデータがありません。')
        return

    for date in sorted(window_df['date'].unique()):
        date_df = window_df[window_df['date'] == date].sort_values('leadtime', ascending=False)

        plt.figure(figsize=(12, 6))
        plt.plot(date_df['leadtime'], date_df['actual_cumsum'], 'o-', linewidth=2, color='tab:blue', label='実測累積値')
        plt.title(f'{date.strftime("%Y-%m-%d")}の実測ブッキングカーブ LT{lt_start}〜{lt_end}')
        plt.xlabel('リードタイム（日）')
        plt.ylabel('累積利用客室数')
        plt.gca().invert_xaxis()
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        file_name = f'{output_dir}/booking_curve_{date.strftime("%Y%m%d")}_lt{lt_start}_to_{lt_end}_actual_only_v2.png'
        plt.savefig(file_name)
        plt.close()
        print(f"{date.strftime('%Y-%m-%d')} の実測値のみグラフを保存しました: {file_name}")

    plt.figure(figsize=(15, 10))
    dates = sorted(window_df['date'].unique())
    colors = plt.cm.plasma(np.linspace(0, 1, len(dates)))

    for i, date in enumerate(dates):
        date_df = window_df[window_df['date'] == date].sort_values('leadtime', ascending=False)
        plt.plot(date_df['leadtime'], date_df['actual_cumsum'], 'o-', color=colors[i], label=date.strftime("%Y-%m-%d"))

    plt.title(f'全日程実測ブッキングカーブ比較 LT{lt_start}〜{lt_end}')
    plt.xlabel('リードタイム（日）')
    plt.ylabel('累積利用客室数')
    plt.gca().invert_xaxis()
    legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    for text in legend.get_texts():
        text.set_rotation(45)
    plt.grid(True)
    plt.gcf().subplots_adjust(right=0.78)

    file_name = f'{output_dir}/booking_curves_all_dates_lt{lt_start}_to_{lt_end}_actual_only_v2.png'
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()
    print(f'全日程の実測値のみグラフを保存しました: {file_name}')


def save_cumulative_data(cumulative_df, output_dir='results/booking_curves_v2'):
    os.makedirs(output_dir, exist_ok=True)
    file_path = f'{output_dir}/cumulative_bookings_v2.csv'
    cumulative_df.sort_values(['date', 'leadtime'], ascending=[True, False]).to_csv(file_path, index=False, encoding='utf-8-sig')
    print(f'累積データを保存しました: {file_path}')


def save_evaluation_results(evaluation_df, output_dir='results/booking_curves_v2'):
    os.makedirs(output_dir, exist_ok=True)
    file_path = f'{output_dir}/evaluation_results_v2.csv'
    evaluation_df.to_csv(file_path, index=False, encoding='utf-8-sig')
    print(f'評価結果を保存しました: {file_path}')
    print('\n===== ブッキングカーブの評価結果(v2) =====')
    print(evaluation_df.to_string())
    print('=========================================')


def resolve_path(path_str: str) -> Path:
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate
    return BASE_DIR / candidate


def parse_args():
    parser = argparse.ArgumentParser(description='SARIMAX v2 の予測CSVからLT範囲指定のブッキングカーブを出力します。')
    parser.add_argument(
        '--predictions-file',
        default='results/sarimax_v2/all_predictions_v2.csv',
        help='入力する all_predictions_v2.csv のパス',
    )
    parser.add_argument(
        '--output-dir',
        default='results/booking_curves_v2_lt90_to_0',
        help='出力先ディレクトリ',
    )
    parser.add_argument('--anchor-lt', type=int, default=90, help='予測アンカーに使うLT')
    parser.add_argument('--lt-start', type=int, default=90, help='グラフの開始LT')
    parser.add_argument('--lt-end', type=int, default=0, help='グラフの終了LT')
    return parser.parse_args()


def main():
    args = parse_args()
    predictions_file = resolve_path(args.predictions_file)
    output_dir = resolve_path(args.output_dir)

    if not predictions_file.exists():
        raise FileNotFoundError(f'予測ファイルが見つかりません: {predictions_file}')

    predictions_df = load_predictions(predictions_file)
    if predictions_df is None or predictions_df.empty:
        raise ValueError(f'予測データを読み込めませんでした: {predictions_file}')

    leadtimes = extract_leadtimes(predictions_df)
    cumulative_df = calculate_cumulative_bookings(predictions_df, leadtimes, anchor_lt=args.anchor_lt)
    os.makedirs(output_dir, exist_ok=True)

    save_cumulative_data(cumulative_df, str(output_dir))
    plot_booking_curves_lt_window(
        cumulative_df,
        output_dir=str(output_dir),
        anchor_lt=args.anchor_lt,
        lt_start=args.lt_start,
        lt_end=args.lt_end,
    )
    plot_actual_only_curves_lt_window(
        cumulative_df,
        output_dir=str(output_dir),
        lt_start=args.lt_start,
        lt_end=args.lt_end,
    )


if __name__ == '__main__':
    main()
