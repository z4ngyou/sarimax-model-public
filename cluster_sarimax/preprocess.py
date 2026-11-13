import pandas as pd
import numpy as np
import warnings
import japanize_matplotlib
warnings.filterwarnings('ignore')

def load_data(file_path):
    """データの読み込みと前処理を行う関数"""
    df = pd.read_csv(file_path)
    
    # データの最初の数行を表示して確認
    print("データの最初の数行:")
    print(df.head())
    
    # 列名を確認
    print("カラム名:", df.columns.tolist())
    
    # 「LTまで拡張した日付」列の処理
    lt_date_column = 'LTまで拡張した日付'
    if lt_date_column in df.columns:
        # 日付とリードタイムを分離
        if isinstance(df[lt_date_column].iloc[0], str):
            # 'LT'を含む場合（例: '2024-05-16 LT90'）
            if 'LT' in df[lt_date_column].iloc[0]:
                # スペースで分割して日付部分を取得
                df['date_part'] = df[lt_date_column].str.split(' ').str[0]
                # 'LT'以降の数字部分を取得
                df['leadtime'] = df[lt_date_column].str.extract(r'LT(\d+)').astype(int)
                
                # 日付を日付型に変換
                df['date_part'] = pd.to_datetime(df['date_part'])
            else:
                # 通常の日付列の場合
                df['date_part'] = pd.to_datetime(df[lt_date_column])
                # デフォルトのリードタイム（単一のリードタイムとして0を設定）
                df['leadtime'] = 0
        else:
            # 日付が既に日付型の場合
            df['date_part'] = pd.to_datetime(df[lt_date_column])
            df['leadtime'] = 0
    elif 'date' in df.columns:
        # 'date'列がある場合はそれを使用
        df['date_part'] = pd.to_datetime(df['date'])
        df['leadtime'] = 0
    else:
        # 日付列がない場合のエラーハンドリング
        raise ValueError("データに日付列が見つかりません。")
    
    # 曜日情報の追加
    df['dayofweek'] = df['date_part'].dt.dayofweek  # 0=月曜, 6=日曜
    
    # 曜日列がすでにある場合は使用
    if '曜日' in df.columns:
        print("'曜日'列が見つかりました。この情報を使用します。")
        # 曜日のマッピング
        day_mapping = {
            '月': 0, '火': 1, '水': 2, '木': 3, '金': 4, '土': 5, '日': 6,
            '月曜': 0, '火曜': 1, '水曜': 2, '木曜': 3, '金曜': 4, '土曜': 5, '日曜': 6,
            '月曜日': 0, '火曜日': 1, '水曜日': 2, '木曜日': 3, '金曜日': 4, '土曜日': 5, '日曜日': 6
        }
        
        # 曜日列の値を確認
        unique_days = df['曜日'].unique()
        print(f"ユニークな曜日値: {unique_days}")
        
        # 数値に変換できる場合は変換
        try:
            if all(day in day_mapping for day in df['曜日'].unique()):
                df['dayofweek'] = df['曜日'].map(day_mapping)
            else:
                print("曜日列の値が予期しない形式です。日付から曜日を計算します。")
        except:
            print("曜日列の処理中にエラーが発生しました。日付から曜日を計算します。")
    
    # 曜日を個別の列として追加（日本語名）
    days_jp = ['月', '火', '水', '木', '金', '土', '日']
    for i, day in enumerate(days_jp):
        df[f'曜日_{day}'] = (df['dayofweek'] == i).astype(int)
    
    # 既存v2のブッキングカーブ処理に合わせ、LTごとの増分を目的変数にする。
    target_column = 't'
    
    # ターゲット列が存在するか確認
    if target_column not in df.columns:
        # 利用可能な列名を表示
        print(f"利用可能な列: {df.columns.tolist()}")
        # 最初の数値列をターゲットとして使用
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            target_column = numeric_cols[0]
            print(f"警告: '{target_column}' 列が見つかりません。代わりに '{target_column}' を使用します。")
        else:
            raise ValueError("適切なターゲット列が見つかりません。")
    
    # NaN値を処理する
    if df[target_column].isna().any():
        print(f"警告: データに{df[target_column].isna().sum()}個のNaN値が含まれています。前方補間で処理します。")
        df[target_column] = df[target_column].interpolate(method='ffill')
        
        # 先頭にNaNがある場合は後方補間も行う
        if df[target_column].isna().any():
            df[target_column] = df[target_column].interpolate(method='bfill')
    
    # データの基本情報を表示
    print(f"データの形状: {df.shape}")
    print(f"カラム: {df.columns.tolist()}")
    print(f"日付範囲: {df['date_part'].min()} から {df['date_part'].max()}")
    if 'leadtime' in df.columns:
        print(f"リードタイム範囲: {df['leadtime'].min()} から {df['leadtime'].max()}")
        print(f"ユニークなリードタイム値: {sorted(df['leadtime'].unique())}")
    
    return df, target_column

def train_test_split_by_date(df, target_column, train_start_date=None, train_end_date=None, test_start_date=None, test_end_date=None):
    """日付でデータをトレーニングセットとテストセットに分割する関数"""
    # 日付でソート
    df = df.sort_values('date_part')
    
    if train_start_date is None and train_end_date is None and test_start_date is None and test_end_date is None:
        # 日付指定がない場合は8:2の比率で分割
        train_size = int(len(df) * 0.8)
        train_data = df.iloc[:train_size]
        test_data = df.iloc[train_size:]
    else:
        # 日付指定がある場合
        # トレーニングデータの範囲を設定
        if train_start_date is not None:
            # 文字列の場合は日付型に変換
            if isinstance(train_start_date, str):
                train_start_date = pd.to_datetime(train_start_date)
                
            if train_end_date is not None:
                # トレーニング開始日と終了日が両方指定されている場合
                if isinstance(train_end_date, str):
                    train_end_date = pd.to_datetime(train_end_date)
                train_data = df[(df['date_part'] >= train_start_date) & (df['date_part'] <= train_end_date)]
            else:
                # トレーニング開始日のみ指定されている場合
                train_data = df[df['date_part'] >= train_start_date]
        elif train_end_date is not None:
            # トレーニング終了日のみ指定されている場合
            if isinstance(train_end_date, str):
                train_end_date = pd.to_datetime(train_end_date)
            train_data = df[df['date_part'] <= train_end_date]
        else:
            # トレーニング期間が指定されていない場合
            train_data = df.copy()
        
        # テストデータの範囲を設定
        if test_start_date is not None:
            # 文字列の場合は日付型に変換
            if isinstance(test_start_date, str):
                test_start_date = pd.to_datetime(test_start_date)
            
            if test_end_date is not None:
                # テスト開始日と終了日が両方指定されている場合
                if isinstance(test_end_date, str):
                    test_end_date = pd.to_datetime(test_end_date)
                test_data = df[(df['date_part'] >= test_start_date) & (df['date_part'] <= test_end_date)]
            else:
                # テスト開始日のみ指定されている場合
                test_data = df[df['date_part'] >= test_start_date]
        elif test_end_date is not None:
            # テスト終了日のみ指定されている場合
            if isinstance(test_end_date, str):
                test_end_date = pd.to_datetime(test_end_date)
            # テスト開始日が指定されていない場合、トレーニング終了日の翌日から開始
            if train_end_date is not None:
                test_data = df[(df['date_part'] > train_end_date) & (df['date_part'] <= test_end_date)]
            else:
                test_data = df[df['date_part'] <= test_end_date]
        else:
            # テスト期間が指定されていない場合
            if train_end_date is not None:
                # トレーニング終了日の翌日からテスト
                test_data = df[df['date_part'] > train_end_date]
            else:
                # デフォルトの分割
                train_size = int(len(df) * 0.8)
                train_data = df.iloc[:train_size]
                test_data = df.iloc[train_size:]
    
    # データが空でないか確認
    if len(train_data) == 0:
        print("警告: 指定された条件ではトレーニングデータが存在しません。")
        return None, None
    
    if len(test_data) == 0:
        print("警告: 指定された条件ではテストデータが存在しません。")
        return None, None
    
    print(f"トレーニングデータ期間: {train_data['date_part'].min()} から {train_data['date_part'].max()}")
    print(f"テストデータ期間: {test_data['date_part'].min()} から {test_data['date_part'].max()}")
    print(f"トレーニングデータ: {len(train_data)}行, テストデータ: {len(test_data)}行")
    
    return train_data, test_data

def preprocess_main(file_path, train_start_date=None, train_end_date=None, test_start_date=None, test_end_date=None):
    """前処理のメイン関数"""
    # データの読み込み
    df, target_column = load_data(file_path)
    
    # データの基本情報を表示
    print(f"データの形状: {df.shape}")
    print(f"カラム: {df.columns.tolist()}")
    print(f"ターゲット列の統計情報:")
    print(df[target_column].describe())
    
    # リードタイムの一覧を取得
    leadtimes = sorted(df['leadtime'].unique())
    print(f"検出されたリードタイム: {leadtimes}")
    print(f"リードタイムの数: {len(leadtimes)}")
    
    # 各リードタイムごとのデータを準備
    leadtime_data = {}
    
    for lt in leadtimes:
        # リードタイムでフィルタリング
        df_lt = df[df['leadtime'] == lt]
        
        # データが少なすぎる場合はスキップ
        if len(df_lt) < 30:
            print(f"警告: リードタイム {lt} のデータが少なすぎます（{len(df_lt)}行）。スキップします。")
            continue
        
        # 日付でデータを分割
        train_data, test_data = train_test_split_by_date(
            df_lt, 
            target_column, 
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            test_start_date=test_start_date,
            test_end_date=test_end_date
        )
        
        # 分割に失敗した場合はスキップ
        if train_data is None or test_data is None:
            print(f"警告: リードタイム {lt} のデータ分割に失敗しました。スキップします。")
            continue
        
        # テストデータが少なすぎる場合はスキップ
        if len(test_data) < 5:
            print(f"警告: リードタイム {lt} のテストデータが少なすぎます（{len(test_data)}行）。スキップします。")
            continue
        
        # データを保存
        leadtime_data[lt] = {
            'train': train_data,
            'test': test_data
        }
    
    return leadtime_data, target_column

if __name__ == "__main__":
    # 前処理のテスト実行
    file_path = 'data/data1.csv'
    train_start_date = '2024-05-16'
    train_end_date = '2024-11-30'
    test_start_date = '2024-12-01'
    test_end_date = '2024-12-31'
    
    leadtime_data, target_column = preprocess_main(
        file_path, 
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        test_start_date=test_start_date,
        test_end_date=test_end_date
    )
    
    print(f"前処理が完了しました。{len(leadtime_data)}個のリードタイムのデータが準備されました。") 
