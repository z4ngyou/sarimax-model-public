import pandas as pd

from preprocess import load_data, train_test_split_by_date


def add_anchor_feature(
    df: pd.DataFrame,
    anchor_lt: int,
    cumulative_column: str = '累積利用客室合計数',
) -> tuple[pd.DataFrame, str]:
    """指定LT時点の累積利用客室数をアンカー特徴量として追加する。"""
    if cumulative_column not in df.columns:
        raise ValueError(f"'{cumulative_column}' 列が見つかりません。")

    anchor_rows = (
        df[df['leadtime'] == anchor_lt][['date_part', cumulative_column]]
        .sort_values('date_part')
        .drop_duplicates(subset='date_part', keep='last')
    )
    if anchor_rows.empty:
        raise ValueError(f'LT{anchor_lt} の実績累積値が見つかりません。')

    anchor_column = f'anchor_cumsum_LT{anchor_lt}'
    anchor_rows = anchor_rows.rename(columns={cumulative_column: anchor_column})

    merged_df = df.merge(anchor_rows, on='date_part', how='left')
    missing_dates = merged_df.loc[merged_df[anchor_column].isna(), 'date_part'].nunique()
    if missing_dates > 0:
        print(f'警告: LT{anchor_lt} のアンカー実績がない日付が {missing_dates} 件あるため除外します。')
        merged_df = merged_df.dropna(subset=[anchor_column]).copy()

    return merged_df, anchor_column


def build_actual_test_matrix(
    split_test_data: dict[int, pd.DataFrame],
    target_column: str,
) -> pd.DataFrame:
    """各LTの実績テストデータを日付横持ちに変換する。"""
    actual_frames = []

    for lt, test_data in sorted(split_test_data.items(), reverse=True):
        actual_df = (
            test_data[['date_part', target_column]]
            .drop_duplicates(subset='date_part')
            .rename(columns={'date_part': 'date', target_column: f'actual_LT{lt}'})
        )
        actual_frames.append(actual_df)

    if not actual_frames:
        return pd.DataFrame(columns=['date'])

    actual_matrix = actual_frames[0]
    for frame in actual_frames[1:]:
        actual_matrix = pd.merge(actual_matrix, frame, on='date', how='outer')

    return actual_matrix.sort_values('date').reset_index(drop=True)


def preprocess_main_v2(
    file_path,
    anchor_lt: int = 90,
    train_start_date=None,
    train_end_date=None,
    test_start_date=None,
    test_end_date=None,
):
    """LTアンカー特徴量つきでSARIMAX v2用データを作成する。"""
    df, target_column = load_data(file_path)
    df, anchor_column = add_anchor_feature(df, anchor_lt)

    print(f'アンカー列: {anchor_column}')
    print(f'アンカーLT: {anchor_lt}')

    leadtimes = sorted(df['leadtime'].unique(), reverse=True)
    print(f'検出されたリードタイム: {sorted(leadtimes)}')

    leadtime_data = {}
    split_test_data = {}

    for lt in leadtimes:
        df_lt = df[df['leadtime'] == lt].copy()
        if df_lt.empty:
            continue

        train_data, test_data = train_test_split_by_date(
            df_lt,
            target_column,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            test_start_date=test_start_date,
            test_end_date=test_end_date,
        )

        if train_data is None or test_data is None:
            print(f'警告: リードタイム {lt} のデータ分割に失敗しました。')
            continue

        split_test_data[lt] = test_data.copy()

        if lt > anchor_lt:
            print(f'リードタイム {lt} は LT{anchor_lt} より大きいため実績固定として扱います。')
            continue

        if len(df_lt) < 30:
            print(f'警告: リードタイム {lt} のデータが少なすぎます（{len(df_lt)}行）。学習をスキップします。')
            continue

        if len(test_data) < 5:
            print(f'警告: リードタイム {lt} のテストデータが少なすぎます（{len(test_data)}行）。学習をスキップします。')
            continue

        leadtime_data[lt] = {
            'train': train_data,
            'test': test_data,
        }

    actual_test_matrix = build_actual_test_matrix(split_test_data, target_column)
    print(f'実績テストマトリクスの形状: {actual_test_matrix.shape}')
    print(f'学習対象LT数: {len(leadtime_data)}')

    return leadtime_data, actual_test_matrix, target_column, anchor_column


if __name__ == '__main__':
    file_path = 'data/data1.csv'
    train_start_date = '2024-05-16'
    train_end_date = '2024-11-30'
    test_start_date = '2024-12-01'
    test_end_date = '2024-12-31'

    leadtime_data, actual_test_matrix, target_column, anchor_column = preprocess_main_v2(
        file_path,
        anchor_lt=90,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        test_start_date=test_start_date,
        test_end_date=test_end_date,
    )

    print(f'前処理v2が完了しました。学習対象LT数: {len(leadtime_data)}')
    print(f'アンカー列: {anchor_column}')
    print(actual_test_matrix.head())
