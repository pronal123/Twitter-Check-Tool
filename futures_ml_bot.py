import os
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import requests
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# --- 環境変数設定 ---
MEXC_API_KEY = os.environ.get('MEXC_API_KEY')
MEXC_SECRET = os.environ.get('MEXC_SECRET')
FUTURES_SYMBOL = 'BTC_USDT'
MODEL_FILENAME = 'btc_futures_ml_model.joblib'
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

# --- 🚨 実戦ベースのカスタムデータ取得関数 ---
def fetch_futures_metrics(exchange, symbol):
    """
    実戦: MEXCのAPIを使い、最新のFR, OI, L/S Ratioのデータを取得・計算する。
    この関数は、MEXCのFutures APIの仕様に合わせてユーザー自身が実装する必要があります。
    """
    try:
        # ccxtのfetch_ticker, fetch_funding_rate, fetch_open_interestなどを利用
        ticker = exchange.fetch_ticker(symbol)
        
        # 資金調達率 (FR)
        funding_rate = float(ticker['info'].get('fundingRate', 0))
        
        # L/S Ratio (LSR) - 取引所APIに依存
        # ユーザーはここでLSRの最新値を取得するカスタムロジックを実装
        ls_ratio = 1.0 # ⚠️ 要実装
        
        # OI Change (OIの変化率) - OIの時系列データ取得と比較が必要
        # ユーザーはここでOIの過去データと比較し、4hの変化率を計算するロジックを実装
        oi_change_4h = 0.0 # ⚠️ 要実装

        return {
            'funding_rate': funding_rate,
            'ls_ratio': ls_ratio,
            'oi_change_4h': oi_change_4h
        }
    except Exception as e:
        print(f"先物指標データ取得失敗: {e}")
        return {'funding_rate': 0.0, 'ls_ratio': 1.0, 'oi_change_4h': 0.0}


class FuturesMLBot:
    def __init__(self):
        self.exchange = ccxt.mexc({
            'apiKey': MEXC_API_KEY,
            'secret': MEXC_SECRET,
            'options': {'defaultType': 'future'},
            'enableRateLimit': True,
        })
        self.target_threshold = 0.0005
        self.prediction_period = 1
        self.feature_cols = [] 

    # --- (1) データ取得 (OHLCV) ---
    def fetch_ohlcv_data(self, limit=100, timeframe='4h'):
        """OHLCVデータを取得する (学習時にはlimitを大きくする)"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(FUTURES_SYMBOL, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            raise Exception(f"OHLCVデータ取得エラー: {e}")

    # --- (2) 特徴量エンジニアリング（学習と予測で共通） ---
    def create_ml_features(self, df):
        """実戦ベースの特徴量を作成する"""
        
        # a) テクニカル指標
        df['SMA'] = ta.sma(df['Close'], length=20)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['MACD_H'] = ta.macd(df['Close'])['MACDh_12_26_9']
        df['Vol_Diff'] = df['Volume'] / ta.sma(df['Volume'], length=20)

        # b) ラグ特徴量（過去のパターン学習）
        for lag in [1, 2, 3]:
            df[f'RSI_L{lag}'] = df['RSI'].shift(lag)
            df[f'Price_L{lag}'] = df['Close'].pct_change(lag).shift(lag)
            
        # c) ターゲットの定義 (学習時のみ使用)
        future_change = df['Close'].pct_change(periods=-self.prediction_period).shift(self.prediction_period)
        df['Target'] = np.select(
            [future_change > self.target_threshold, future_change < -self.target_threshold],
            [1, -1], default=0
        )
        
        df.dropna(inplace=True)
        
        # 特徴量カラムリストの確定
        if not self.feature_cols:
            cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'timestamp']]
            self.feature_cols = [col for col in cols if df[col].dtype in [np.float64, np.int64]] # 数値型のみを特徴量とする
        
        # 学習に使用する特徴量とターゲットを返す
        return df[self.feature_cols], df['Target']

    # --- (3) モデルの学習と保存（再構築） ---
    def train_and_save_model(self, df_long_term):
        """長期データからモデルを再学習し、ファイルに保存する"""
        X_train, Y_train = self.create_ml_features(df_long_term.copy())
        
        # 全データを訓練データとして使用 (継続学習のため)
        model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', max_depth=10)
        model.fit(X_train, Y_train)
        
        joblib.dump(model, MODEL_FILENAME)
        return True

    # --- (4) リアルタイム予測 ---
    def predict_and_report(self, df_latest, futures_data):
        """最新データとリアルタイム指標で予測を実行し、報告書を生成する"""
        
        # モデルと特徴量の準備
        model = joblib.load(MODEL_FILENAME)
        X_latest, _ = self.create_ml_features(df_latest.copy())
        latest_X = X_latest.iloc[[-1]] 
        
        # 予測実行
        prediction_val = model.predict(latest_X)[0]
        prediction_proba = model.predict_proba(latest_X)[0]
        
        # 報告書生成ロジック (前回の実戦モデル報告書ロジックを使用)
        report = self._generate_final_report(df_latest.iloc[-1], futures_data, prediction_val, prediction_proba)
        return report

    # --- (5) 報告書生成の補助関数 ---
    def _generate_final_report(self, latest_price_data, futures_data, ml_prediction, proba):
        """実戦で使える詳細なレポートを生成する"""
        price = latest_price_data['Close']
        sma = latest_price_data['SMA']
        
        pred_map = {-1: "📉 下落", 0: "↔️ レンジ", 1: "📈 上昇"}
        ml_result = pred_map.get(ml_prediction, "不明")
        
        fr = futures_data.get('funding_rate', 0)
        lsr = futures_data.get('ls_ratio', 1.0)
        oi_chg = futures_data.get('oi_change_4h', 0.0)
        
        reasons = []
        
        # a) 機械学習の根拠
        reasons.append(f"🤖 **機械学習予測:** **{ml_result}** (UP: {proba[2]*100:.1f}%, DOWN: {proba[0]*100:.1f}%)")
        
        # b) テクニカルの根拠
        if price > sma:
            reasons.append(f"🟢 **価格トレンド:** 4h足は20-SMA (${sma:.2f}) の上にあり、短期は強気。")
        else:
            reasons.append(f"🔴 **価格トレンド:** 4h足は20-SMA (${sma:.2f}) の下にあり、短期は弱気。")

        # c) 先物センチメントの根拠
        if fr > 0.00015 or lsr > 1.3:
            reasons.append(f"🚨 **ロング過熱:** FR({fr*100:.3f}%) と L/S比率({lsr:.2f}) からロング過熱と判断。下落リスクが高い。")
        elif fr < -0.00015 or lsr < 0.8:
            reasons.append(f"✅ **ショート過熱:** FR({fr*100:.3f}%) が大幅マイナス。ショートスクイーズ（上昇）リスクが高い。")
        
        if oi_chg > 0.03 and price < sma:
             reasons.append(f"⚠️ **OI増加:** 下落中にOI増加({oi_chg*100:.1f}%)。新規ショート参入による下落トレンドの継続リスク。")
        
        # 最終結論の調整 (スクイーズ警戒)
        final_conclusion = ml_result
        if (ml_result == "📈 上昇" and fr > 0.00015) or (ml_result == "📉 下落" and fr < -0.00015):
             final_conclusion = f"⚠️ {ml_result} (スクイーズ警戒)"


        report = f"""
📈 **MEXC BTC/USDT 先物市場 複合分析レポート (継続学習型)**
📅 **分析日時:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S JST')}
---
### 🎯 4時間後の最終予測動向
**結論:** **{final_conclusion}**

### 📍 4時間後のBTCの位置予測
モデルと複合指標の分析に基づき、現在の価格 **${price:.2f}** を起点に**{final_conclusion}**方向に動く可能性が最も高いです。

### 🧠 根拠となる詳細分析
---
""" + "\n".join(reasons) + """
---
* **現在の価格:** ${price:.2f}
* **資金調達率 (FR):** {fr*100:.4f}%
* **L/S比率 (LSR):** {lsr:.2f}
"""
        return report
        
    # --- (6) Telegram 通知関数 ---
    def send_telegram_notification(self, message):
        """通知の実装 (省略) """
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
        try:
            requests.post(url, data=payload)
            print("✅ Telegramへの通知が完了しました。")
        except Exception as e:
            print(f"Telegram通知エラー: {e}")
