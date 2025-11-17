import os
import json
import time
import random
from typing import Dict, Any, List

# データ処理と機械学習ライブラリ
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from joblib import dump, load

# 外部API (CoinGeckoのSimulated APIとして扱う)
# NOTE: 実際の外部APIコールは、セキュリティと実行環境の制約上、シミュレーションとして記述します。
# 実際のプロジェクトでは、requestsライブラリなどを使用してAPIを叩いてください。
# ここでは、データ取得失敗時の処理を強調します。

# --- 定数設定 ---
REPORT_FILENAME = 'latest_report.json'
MODEL_FILENAME = 'futures_predictor.joblib'
FALLBACK_FILENAME = 'fallback_data.csv'
DAYS_LOOKBACK = 900  # 過去約2.5年分のデータを使用
HORIZON = 5          # 予測する日数 (5日後終値を予測)

# --- カスタム例外 ---
class DataFetchError(Exception):
    """データ取得に失敗した場合のカスタム例外"""
    pass

# --- メインクラス ---
class FuturesMLBot:
    """
    先物市場の価格データ取得、MLモデル学習、予測を担当するクラス。
    データ取得の堅牢性を確保するため、API失敗時にはフォールバックデータを使用する。
    """
    
    def __init__(self):
        """インスタンス初期化時にスケーラーを初期化します。"""
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        print("🤖 FuturesMLBot初期化完了。")

    # --- 1. データ取得 (堅牢性を考慮) ---

    def _simulate_api_fetch(self, days: int) -> pd.DataFrame:
        """
        CoinGecko APIからのデータ取得をシミュレーションします。
        ランダムに失敗する可能性があります。
        """
        print(f"📡 APIから過去 {days} 日間のデータ取得を試行中...")
        
        # 稀にAPIが失敗する状況をシミュレーション (本番環境ではこのランダム失敗は不要)
        if random.random() < 0.05: # 5%の確率で失敗
            raise DataFetchError("CoinGecko APIからのデータ取得に失敗しました。タイムアウトまたはサーバーエラー。")

        # 成功したと仮定して、シミュレーションデータを生成
        # NOTE: 実際のデータはAPIレスポンスから生成されます。
        
        start_date = datetime.now() - timedelta(days=days)
        date_range = pd.date_range(start=start_date, periods=days, freq='D')
        
        # 簡略化のため、フォールバックデータと同じ構造をシミュレーション
        data = {
            'Date': date_range,
            'Close': np.cumsum(np.random.normal(0, 10, days)) + 1000,
            'Volume': np.random.randint(10000, 50000, days)
        }
        df = pd.DataFrame(data).set_index('Date')
        df['Close'] = df['Close'].round(2)
        df['Volume'] = df['Volume'].astype(int)
        
        return df

    def _load_fallback_data(self) -> pd.DataFrame:
        """
        ローカルのフォールバックCSVファイルからデータを読み込みます。
        """
        print(f"📂 フォールバックデータ ({FALLBACK_FILENAME}) を読み込み中...")
        if not os.path.exists(FALLBACK_FILENAME):
            print(f"🚨 フォールバックファイルが見つかりません: {FALLBACK_FILENAME}")
            return pd.DataFrame()
            
        try:
            # Dateをインデックスとしてパースして読み込む
            df = pd.read_csv(FALLBACK_FILENAME, index_col='Date', parse_dates=True)
            # 必要な列 'Close' と 'Volume' があるか確認
            if 'Close' not in df.columns or 'Volume' not in df.columns:
                 print("🚨 フォールバックファイルに必要な列(Close, Volume)がありません。")
                 return pd.DataFrame()
            print(f"✅ フォールバックデータをロードしました。行数: {len(df)}")
            return df
        except Exception as e:
            print(f"🚨 フォールバックデータの読み込み中にエラーが発生しました: {e}")
            return pd.DataFrame()

    def fetch_ohlcv_data(self, days: int) -> pd.DataFrame:
        """
        主要なデータ取得メソッド。APIを試行し、失敗した場合はフォールバックに切り替える。
        """
        try:
            # 1. APIからのデータ取得を試みる
            df = self._simulate_api_fetch(days)
            if df.empty:
                raise DataFetchError("APIから空のデータセットが返されました。")
            print("✅ APIデータ取得成功。")
            return df
            
        except DataFetchError as e:
            print(f"⚠️ データ取得エラー: {e} -> フォールバックに切り替えます。")
            # 2. 失敗した場合、フォールバックデータを読み込む
            df_fallback = self._load_fallback_data()
            
            if df_fallback.empty:
                print("🚨 フォールバックデータも使用できません。")
                return pd.DataFrame() # 最終的に空のDataFrameを返す

            # 過去DAYS_LOOKBACK日数にデータを絞り込む
            if len(df_fallback) > days:
                df_fallback = df_fallback.iloc[-days:]
                
            print("✅ フォールバックデータを使用します。")
            return df_fallback
        except Exception as e:
            print(f"🚨 予期せぬデータ取得エラー: {e}")
            return pd.DataFrame()
    
    # --- 2. 特徴量エンジニアリングと学習 ---

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        移動平均線や出来高のラグなど、MLモデル用の特徴量を生成します。
        """
        df_copy = df.copy()

        # ターゲット変数 (未来の終値)
        # T+HORIZON 日後の終値を予測する
        df_copy['Target'] = df_copy['Close'].shift(-HORIZON) 

        # 特徴量: 短期・長期移動平均線
        df_copy['MA_7'] = df_copy['Close'].rolling(window=7).mean()
        df_copy['MA_30'] = df_copy['Close'].rolling(window=30).mean()
        
        # 特徴量: 出来高のラグ
        df_copy['Volume_Lag_1'] = df_copy['Volume'].shift(1)
        
        # NaN行を削除 (移動平均線計算に必要な過去データがない行)
        df_copy.dropna(inplace=True)
        
        return df_copy
        
    def train_and_save_model(self, df: pd.DataFrame):
        """
        リッジ回帰モデルを学習し、モデルとスケーラーをファイルに保存します。
        モデルが存在しない場合、または定期的な再学習が必要な場合に実行されます。
        """
        df_features = self._create_features(df)
        if df_features.empty:
            print("🚨 特徴量生成後、学習データが空になりました。モデル学習をスキップします。")
            return

        # 特徴量 (X) とターゲット (y) を定義
        X = df_features[['Close', 'MA_7', 'MA_30', 'Volume_Lag_1']].values
        y = df_features['Target'].values

        # データの正規化
        X_scaled = self.scaler.fit_transform(X) # スケーラーを学習データでフィット

        # モデルの学習 (リッジ回帰を使用)
        # NOTE: 実際の予測では、より高度なモデル(LGBM, ARIMAなど)を使用することが推奨されます。
        model = Ridge(alpha=1.0)
        model.fit(X_scaled, y)
        
        # モデルとスケーラーの保存
        try:
            dump(model, MODEL_FILENAME)
            # スケーラーは、モデルの予測時に必要なので、ここではスケーラー自体を保存するのではなく、
            # self.scalerとしてインスタンスに保持し続けます。
            print(f"✅ MLモデルを {MODEL_FILENAME} に保存しました。")
        except Exception as e:
            print(f"🚨 モデル保存中にエラーが発生しました: {e}")

    # --- 3. 予測とレポート生成 ---

    def _load_model(self) -> Any:
        """
        保存されたMLモデルをロードします。存在しない場合はNoneを返します。
        """
        if os.path.exists(MODEL_FILENAME):
            try:
                model = load(MODEL_FILENAME)
                print(f"✅ MLモデルを {MODEL_FILENAME} からロードしました。")
                return model
            except Exception as e:
                print(f"🚨 モデルロード中にエラーが発生しました: {e}")
                return None
        else:
            print(f"⚠️ モデルファイル {MODEL_FILENAME} が見つかりません。学習が必要です。")
            return None

    def fetch_advanced_metrics(self) -> Dict[str, Any]:
        """
        高度な指標 (ファンダメンタルズ、センチメントなど) の取得をシミュレーションします。
        """
        # NOTE: 実際のプロジェクトでは、ニュースAPIやソーシャルメディアAPIなどから取得します。
        metrics = {
            "market_sentiment": random.choice(["Bullish", "Neutral", "Bearish"]),
            "fear_greed_index": random.randint(10, 90),
            "open_interest_change": round(random.uniform(-5.0, 5.0), 2),
            "economic_data_impact": random.choice(["Low", "Medium", "High"])
        }
        return metrics

    def predict_and_report(self, df: pd.DataFrame, advanced_data: Dict[str, Any]):
        """
        最新のデータを使用して予測を行い、結果をJSONレポートとして保存します。
        """
        model = self._load_model()
        if model is None:
            print("🚨 予測を行うためのモデルがロードできません。予測をスキップします。")
            return
            
        df_features = self._create_features(df)
        if df_features.empty:
            print("🚨 予測を行うための特徴量データがありません。予測をスキップします。")
            return

        # 最新日のデータ (df_featuresの最後の行) を予測に使用
        latest_data = df_features.iloc[[-1]] 
        
        # 予測に使用する特徴量を抽出
        X_latest = latest_data[['Close', 'MA_7', 'MA_30', 'Volume_Lag_1']].values
        
        # スケーリング (学習時と同じスケーラーを使用)
        # NOTE: スケーラーは train_and_save_model で fit_transform されている
        try:
            X_latest_scaled = self.scaler.transform(X_latest) 
        except Exception as e:
            print(f"🚨 スケーラー変換中にエラーが発生しました。モデルの再学習が必要です。: {e}")
            return

        # 予測の実行
        predicted_close_price = model.predict(X_latest_scaled)[0]
        
        # 最新の終値と予測値の比較
        current_close = latest_data['Close'].iloc[0]
        prediction_change = ((predicted_close_price - current_close) / current_close) * 100
        
        # レポート生成日
        report_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S JST')
        
        # 予測方向の決定
        if prediction_change > 0.5:
            direction = "上昇トレンド継続 (Bullish)"
            action = "積極的な買い増し"
        elif prediction_change < -0.5:
            direction = "下降トレンド警戒 (Bearish)"
            action = "利確または空売り検討"
        else:
            direction = "レンジ相場または調整局面 (Neutral)"
            action = "様子見または短期トレード"
            
        # 予測結果をJSON形式で構造化
        report = {
            "report_time": report_date,
            "prediction_horizon_days": HORIZON,
            "current_close_price": round(current_close, 2),
            "predicted_close_price": round(predicted_close_price, 2),
            "predicted_change_percent": round(prediction_change, 2),
            "prediction_direction": direction,
            "recommended_action": action,
            "advanced_metrics": advanced_data, # 高度な指標を含める
            "data_source": "API (またはフォールバックデータを使用)", # どちらが使われたかを示唆
            "chart_data": self._prepare_chart_data(df) # チャート用データ
        }

        # JSONレポートをファイルに保存
        try:
            with open(REPORT_FILENAME, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=4)
            print(f"✅ 予測レポートを {REPORT_FILENAME} に保存しました。")
        except Exception as e:
            print(f"🚨 レポート保存中にエラーが発生しました: {e}")

    # --- 4. チャートデータ準備 ---

    def _prepare_chart_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        フロントエンドのチャート表示用に、過去数ヶ月間のデータを準備します。
        """
        # 過去180日分に絞る (チャートが重くなりすぎないように)
        chart_df = df.iloc[-180:].copy() 
        
        # 予測ポイントをチャートに追加するために、予測日を計算
        # 予測は「最新データの日付 + HORIZON日後」とする
        latest_date = chart_df.index[-1]
        prediction_date = latest_date + timedelta(days=HORIZON)

        # 予測データを追加するためのダミー行を作成
        # チャートが予測ポイントまで線でつながるように、最新の終値を予測日の1日前にも追加
        
        # 1. 最後の実績日
        last_real_entry = {
            "date": latest_date.strftime('%Y-%m-%d'),
            "close": round(chart_df['Close'].iloc[-1], 2),
            "type": "Actual"
        }
        
        # 2. 予測日 (値は予測時に書き込まれるため、プレースホルダーとして保持)
        # NOTE: 予測値は予測関数が計算するため、ここでは構造のみ準備
        
        # 過去データから必要な列だけを抽出し、JSON形式のリストに変換
        chart_list = []
        for index, row in chart_df.iterrows():
            chart_list.append({
                "date": index.strftime('%Y-%m-%d'),
                "close": round(row['Close'], 2),
                "type": "Actual"
            })
            
        # 最後の実績データを追加
        # (リストの最後に実際の予測値を追加する処理は predict_and_report の外部で行うか、
        # ここでは過去データのみを返してフロントエンドで処理する方がシンプル)
        
        # シンプルに過去実績データのみを返す
        return chart_list

# --- 実行に必要なフォールバックデータ生成 ---

if __name__ == '__main__':
    # このファイルが単独で実行された場合に、フォールバックデータを作成する
    # これは、アプリ実行前に `fallback_data.csv` が存在しない場合に役立ちます。
    
    if not os.path.exists(FALLBACK_FILENAME):
        print(f"🛠️ {FALLBACK_FILENAME} が見つからないため、シミュレーションデータを作成します。")
        
        days_to_generate = 1000
        start_date = datetime.now() - timedelta(days=days_to_generate)
        date_range = pd.date_range(start=start_date, periods=days_to_generate, freq='D')
        
        # S&P 500または主要先物の動きをシミュレート
        # ランダムウォークにトレンドとノイズを加える
        np.random.seed(42)
        base_price = 4000
        returns = np.random.normal(0.0005, 0.01, days_to_generate)
        prices = base_price * (1 + returns).cumprod()
        volumes = np.random.randint(50000, 150000, days_to_generate)
        
        fallback_df = pd.DataFrame({
            'Date': date_range,
            'Close': prices.round(2),
            'Volume': volumes
        }).set_index('Date')
        
        fallback_df.to_csv(FALLBACK_FILENAME)
        print(f"✅ {FALLBACK_FILENAME} に {len(fallback_df)} 日分のシミュレーションデータを作成しました。")
    
    # テスト実行
    bot = FuturesMLBot()
    
    # データを取得 (API失敗をシミュレートする可能性あり)
    test_df = bot.fetch_ohlcv_data(DAYS_LOOKBACK)

    if not test_df.empty:
        # モデルの学習と保存
        bot.train_and_save_model(test_df)
        
        # 高度な指標の取得
        advanced = bot.fetch_advanced_metrics()
        
        # 予測とレポートの生成
        bot.predict_and_report(test_df, advanced)
