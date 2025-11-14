# futures_ml_bot.py

import os
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import requests
import joblib
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, Dict, Any

# --- 1. 環境変数設定 ---
# .envファイルを使用する場合、app.pyでロードされます
MEXC_API_KEY = os.environ.get('MEXC_API_KEY')
MEXC_SECRET = os.environ.get('MEXC_SECRET')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

FUTURES_SYMBOL = 'BTC_USDT'
TIMEFRAME = '4h'
MODEL_FILENAME = 'btc_futures_ml_model.joblib'


# --- 2. 🚨 実戦ベースのカスタムデータ取得関数 ---
def fetch_futures_metrics(exchange: ccxt.Exchange, symbol: str) -> Dict[str, float]:
    """
    実戦: MEXCのAPIを使い、最新のFR, OI, L/S Ratioのデータを取得・計算する。
    この関数は、MEXCのAPI仕様に合わせて正確に実装が必要です。
    """
    try:
        ticker = exchange.fetch_ticker(symbol)
        
        # 資金調達率 (FR)
        funding_rate = float(ticker.get('fundingRate', 0) or 0)
        
        # L/S Ratio (LSR) - ⚠️ 要実装: MEXCの専用APIから取得
        ls_ratio = 1.05  # 実装するまでのプレースホルダー
        
        # OI Change (OIの変化率) - ⚠️ 要実装: 過去4hのOI時系列データと比較
        oi_change_4h = 0.01  # 実装するまでのプレースホルダー

        return {
            'funding_rate': funding_rate,
            'ls_ratio': ls_ratio,
            'oi_change_4h': oi_change_4h
        }
    except Exception as e:
        print(f"先物指標データ取得失敗: {e}")
        return {'funding_rate': 0.0, 'ls_ratio': 1.0, 'oi_change_4h': 0.0}


# --- 3. メイン BOT クラス ---
class FuturesMLBot:
    def __init__(self):
        # 🚨 環境変数がNoneの場合、ccxtの初期化が失敗する可能性があるため、チェック
        if not all([MEXC_API_KEY, MEXC_SECRET]):
             raise ValueError("APIキーが設定されていません。環境変数を確認してください。")
             
        self.exchange = ccxt.mexc({
            'apiKey': MEXC_API_KEY,
            'secret': MEXC_SECRET,
            'options': {'defaultType': 'future'},
            'enableRateLimit': True,
        })
        self.target_threshold = 0.0005
        self.prediction_period = 1
        self.feature_cols = [] 

    # --- (A) データ取得 ---
    def fetch_ohlcv_data(self, limit: int = 100, timeframe: str = TIMEFRAME) -> pd.DataFrame:
        """OHLCVデータを取得する (学習時にはlimitを大きくする)"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(FUTURES_SYMBOL, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            raise Exception(f"OHLCVデータ取得エラー: {e}")

    # --- (B) 特徴量エンジニアリング ---
    def create_ml_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """実戦ベースの特徴量を作成する"""
        
        df['SMA'] = ta.sma(df['Close'], length=20)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['MACD_H'] = ta.macd(df['Close'])['MACDh_12_26_9']
        df['Vol_Diff'] = df['Volume'] / ta.sma(df['Volume'], length=20)

        for lag in [1, 2, 3]:
            df[f'RSI_L{lag}'] = df['RSI'].shift(lag)
            df[f'Price_L{lag}'] = df['Close'].pct_change(lag).shift(lag)
            
        future_change = df['Close'].pct_change(periods=-self.prediction_period).shift(self.prediction_period)
        df['Target'] = np.select(
            [future_change > self.target_threshold, future_change < -self.target_threshold],
            [1, -1], default=0
        )
        
        df.dropna(inplace=True)
        
        if not self.feature_cols:
            cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'timestamp']]
            self.feature_cols = [col for col in cols if df[col].dtype in [np.float64, np.int64]]
        
        return df[self.feature_cols], df['Target']

    # --- (C) モデルの学習と保存（再構築） ---
    def train_and_save_model(self, df_long_term: pd.DataFrame) -> bool:
        """長期データからモデルを再学習し、ファイルに保存する"""
        print("🧠 モデル再学習タスク開始...")
        X_train, Y_train = self.create_ml_features(df_long_term.copy())
        
        model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', max_depth=10)
        model.fit(X_train, Y_train)
        
        joblib.dump(model, MODEL_FILENAME)
        print("✅ モデル再学習完了し、ファイルに保存しました。")
        return True

    # --- (D) リアルタイム予測と通知 (コア実行部) ---
    def predict_and_report(self, df_latest: pd.DataFrame, futures_data: Dict[str, float]) -> bool:
        """最新データで予測を実行し、2つの報告書を生成・通知する"""
        
        try:
            model = joblib.load(MODEL_FILENAME)
        except FileNotFoundError:
            report = "🚨 エラー: モデルファイルが見つかりません。最初に学習とコミットを行ってください。"
            self.send_telegram_notification(report)
            return False

        X_latest, _ = self.create_ml_features(df_latest.copy())
        latest_X = X_latest.iloc[[-1]] 
        
        prediction_val = model.predict(latest_X)[0]
        prediction_proba = model.predict_proba(latest_X)[0]
        
        # 報告書生成ロジックを呼び出し、2つのレポートを受け取る
        report_structure, report_conclusion = self._generate_two_part_reports(
            df_latest.iloc[-1], 
            futures_data, 
            prediction_val, 
            prediction_proba
        )
        
        # 報告書を順番に送信
        self.send_telegram_notification(report_structure)
        self.send_telegram_notification(report_conclusion)
        
        return True

    # --- (E) 報告書生成の補助関数 (2つのレポートを生成) ---
    def _generate_two_part_reports(self, latest_price_data: pd.Series, futures_data: Dict[str, float], ml_prediction: int, proba: np.ndarray) -> Tuple[str, str]:
        """
        レポートを「市場構造分析」と「最終結論と戦略」の2つに分けて生成する
        """
        price = latest_price_data['Close']
        sma = latest_price_data['SMA']
        rsi = latest_price_data['RSI']
        
        pred_map = {-1: "📉 下落", 0: "↔️ レンジ", 1: "📈 上昇"}
        ml_result = pred_map.get(ml_prediction, "不明")
        
        fr = futures_data.get('funding_rate', 0)
        lsr = futures_data.get('ls_ratio', 1.0)
        oi_chg = futures_data.get('oi_change_4h', 0.0)
        
        current_time = datetime.now(timezone.utc).astimezone(None).strftime('%Y-%m-%d %H:%M JST')

        # ---------------------------------------------------
        # A. レポート 1: 市場構造分析レポート (データとテクニカル)
        # ---------------------------------------------------
        report_structure = f"""
📈 **BTC/USDT 市場構造分析 (4H)**
📅 {current_time}

---
### 📊 複合指標詳細

| 指標 | 現在値 | 評価 | 示唆するリスク/機会 |
| :--- | :--- | :--- | :--- |
| **現在価格** | **${price:.2f}** | - | - |
| **20-SMA** | ${sma:.2f} | {'🟢 上回る' if price > sma else '🔴 下回る'} | 短期トレンドの方向性。 |
| **RSI (14)** | {rsi:.2f} | {'🟢' if rsi > 60 else '🔴' if rsi < 40 else '🟡'} | 買われすぎ/売られすぎの判断。 |
| **FR** | {fr*100:.5f}% | {'🚨 強いプラス' if fr > 0.00015 else '✅ 強いマイナス' if fr < -0.00015 else '🟡 中立'} | スクイーズリスクの判断。 |
| **L/S 比率** | {lsr:.2f} | {'🔴 ロング優勢' if lsr > 1.2 else '✅ ショート優勢' if lsr < 0.9 else '🟡 均衡'} | ポジションの偏り。 |
| **OI 変化率 (4H)** | {oi_chg*100:.1f}% | {'🔴 増加' if oi_chg > 0.03 else '🟢 減少' if oi_chg < -0.03 else '🟡 安定'} | トレンドの勢いと継続性。 |

### 🛠️ テクニカル環境

* **現在のトレンド:** {'強気' if price > sma else '弱気'}。20-SMAは現在、{'サポート' if price > sma else 'レジスタンス'}として機能しています。
* **市場の過熱度:** RSIが{rsi:.2f}であるため、{'過熱感があり反落リスクに注意。' if rsi > 70 else '売られすぎで反発の可能性。' if rsi < 30 else '次の動きのエネルギーを蓄積中。'}
* **結論：市場構造は** {'強気バイアス' if price > sma and lsr < 1.0 else '弱気バイアス' if price < sma and lsr > 1.0 else '中立/レンジ'}です。
"""
        
        # ---------------------------------------------------
        # B. レポート 2: 最終結論と戦略レポート (ML予測とアクション)
        # ---------------------------------------------------
        
        main_reasons = []
        if price > sma and lsr < 1.0:
            main_reasons.append("ポジティブなテクニカル構造と、ショート優勢のセンチメントが重なり、上昇への圧力が強い。")
        elif price < sma and oi_chg > 0.03:
            main_reasons.append("価格下落中にOIが大幅増加。新規ショート参入が下落トレンドの継続を強く示唆。")
        elif fr > 0.00015:
            main_reasons.append("FRが大幅なプラスであり、ロング過熱感が高い。モデルはロングスクイーズ（下落）を予測。")
        else:
             main_reasons.append("テクニカルとセンチメントが均衡しており、モデルの予測に基づいたレンジ戦略を推奨。")

        # 最終結論の調整 (スクイーズ警戒)
        final_conclusion = ml_result
        if (ml_result == "📈 上昇" and fr > 0.00015) or (ml_result == "📉 下落" and fr < -0.00015):
             final_conclusion = f"⚠️ {ml_result} (スクイーズ警戒)"
        
        report_conclusion = f"""
🚨 **BTC/USDT 最終結論とアクションプラン**
📅 {current_time}

---
### 🤖 最終予測と根拠

| 項目 | 分析結果 | 確率 |
| :--- | :--- | :--- |
| **ML 予測結論** | **{final_conclusion}** | **{proba[np.argmax(proba)]*100:.1f}%** |
| **予測される位置** | **${price:.2f}** を起点に**{final_conclusion}**方向に動く可能性が高い。 | - |

#### 🧠 なぜこの結論なのか？ (主要な根拠)

* **主要な根拠:** {main_reasons[0]}
* **モデルの判断:** 継続学習により、過去の類似パターンと比較した結果、**{final_conclusion}**への確率が最も高いと判断されました。

### 🎯 推奨戦略

| 戦略 | 詳細 |
| :--- | :--- |
| **推奨方向** | **{final_conclusion}**の方向に沿った取引を検討。リスクが極めて高いため、明確なトレンド転換シグナルを待つ選択肢も考慮してください。 |
| **アクション** | **エントリー**は20-SMA (${sma:.2f}) のブレイク/反発を確認後。**損切り**は直近の強いサポート/レジスタンスの外側に設定し、リスクを限定してください。 |
"""
        return report_structure, report_conclusion
        
    # --- (F) Telegram 通知関数 ---
    def send_telegram_notification(self, message: str):
        """通知の実装"""
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
        try:
            requests.post(url, data=payload)
            print("✅ Telegramへの通知が完了しました。")
        except Exception as e:
            print(f"Telegram通知エラー: {e}")
