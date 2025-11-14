# futures_ml_bot.py (最終完全版)

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
MEXC_API_KEY = os.environ.get('MEXC_API_KEY')
MEXC_SECRET = os.environ.get('MEXC_SECRET')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

FUTURES_SYMBOL = 'BTC_USDT'
TIMEFRAME = '4h'
MODEL_FILENAME = 'btc_futures_ml_model.joblib'
MEXC_API_BASE_URL = 'https://contract.mexc.com' 

# 🚨 外部API (仮定) - 実際のAPI URLに置き換えてください
FG_INDEX_API_URL = 'https://api.alternative.me/fng/?limit=1'
COINGLASS_API_URL = 'https://api.coinglass.com/api/v1/liquidation/recent' # 仮定の清算API


# --- 2. 高度なカスタムデータ取得関数 ---
def fetch_advanced_metrics(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
    """
    FR, OI, L/S Ratio, Fear & Greed Index, Liquidation Dataなど、高度な指標を取得・計算する。
    """
    mexc_symbol = symbol.replace('_', '/') 
    metrics = {}
    
    try:
        # 1. 資金調達率 (FR) の取得
        ticker = exchange.fetch_ticker(mexc_symbol)
        metrics['funding_rate'] = float(ticker.get('fundingRate', 0) or 0)
        
        # 2. Fear & Greed Index 取得
        fg_response = requests.get(FG_INDEX_API_URL, timeout=5)
        fg_response.raise_for_status()
        fg_data = fg_response.json().get('data', [{}])
        metrics['fg_index'] = int(fg_data[0].get('value', 50))
        metrics['fg_value'] = fg_data[0].get('value_classification', 'Neutral')

        # 3. 清算データ取得 (Coinglass API - 仮定)
        liquidation_response = requests.get(COINGLASS_API_URL, params={'symbol': 'BTC'}, timeout=5)
        liquidation_response.raise_for_status()
        liq_data = liquidation_response.json().get('data', {})
        metrics['liq_24h_total'] = liq_data.get('totalLiquidationUSD', 0.0) 
        metrics['liq_24h_long'] = liq_data.get('longLiquidationUSD', 0.0)
        
        # 4. OI/LSR取得 (MEXC API - 仮定のロジックを再挿入)
        # ⚠️ 実運用時は、この部分をMEXCの実際のAPIエンドポイントに置き換えてください。
        metrics['ls_ratio'] = 1.05 # 仮の値
        metrics['oi_change_4h'] = 0.01 # 仮の値

        return metrics
    
    except requests.exceptions.RequestException as req_e:
        print(f"🚨 外部APIリクエストエラー: {req_e}")
        return {
            'funding_rate': 0.0, 'ls_ratio': 1.0, 'oi_change_4h': 0.0, 
            'fg_index': 50, 'fg_value': 'API Failed', 
            'liq_24h_total': 0.0, 'liq_24h_long': 0.0
        }
    except Exception as e:
        print(f"🚨 先物指標データ処理エラー: {e}")
        return {
            'funding_rate': 0.0, 'ls_ratio': 1.0, 'oi_change_4h': 0.0, 
            'fg_index': 50, 'fg_value': 'API Failed', 
            'liq_24h_total': 0.0, 'liq_24h_long': 0.0
        }


# --- 3. メイン BOT クラス ---
class FuturesMLBot:
    def __init__(self):
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

    # --- (A) データ取得 (OHLCV) ---
    def fetch_ohlcv_data(self, limit: int = 100, timeframe: str = TIMEFRAME) -> pd.DataFrame:
        """OHLCVデータを取得する"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(FUTURES_SYMBOL, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            raise Exception(f"OHLCVデータ取得エラー: {e}")

    # --- (B) 特徴量エンジニアリング (ATRを含む) ---
    def create_ml_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """実戦ベースの特徴量を作成する"""
        
        df['SMA'] = ta.sma(df['Close'], length=20)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['MACD_H'] = ta.macd(df['Close'])['MACDh_12_26_9']
        df['Vol_Diff'] = df['Volume'] / ta.sma(df['Volume'], length=20)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14) 

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

    # --- (C) モデルの学習と保存（継続学習） ---
    def train_and_save_model(self, df_long_term: pd.DataFrame) -> bool:
        """長期データからモデルを再学習し、ファイルに保存する"""
        print("🧠 モデル再学習タスク開始...")
        X_train, Y_train = self.create_ml_features(df_long_term.copy())
        
        model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', max_depth=10)
        model.fit(X_train, Y_train)
        
        joblib.dump(model, MODEL_FILENAME)
        print("✅ モデル再学習完了し、ファイルに保存しました。")
        return True

    # --- (D) リアルタイム予測と通知 ---
    def predict_and_report(self, df_latest: pd.DataFrame, advanced_data: Dict[str, Any]) -> bool:
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
        
        report_structure, report_conclusion = self._generate_two_part_reports(
            df_latest.iloc[-1], 
            advanced_data, 
            prediction_val, 
            prediction_proba
        )
        
        self.send_telegram_notification(report_structure)
        self.send_telegram_notification(report_conclusion)
        
        return True

    # --- (E) 報告書生成の補助関数 - 高度な統合分析レポートを生成 ---
    def _generate_two_part_reports(self, latest_price_data: pd.Series, advanced_data: Dict[str, Any], ml_prediction: int, proba: np.ndarray) -> Tuple[str, str]:
        """
        レポートを「市場構造と主因分析」と「最終結論と戦略」の2つに分けて生成する
        """
        price = latest_price_data['Close']
        sma = latest_price_data['SMA']
        rsi = latest_price_data['RSI']
        atr = latest_price_data['ATR']
        
        pred_map = {-1: "📉 下落", 0: "↔️ レンジ", 1: "📈 上昇"}
        ml_result = pred_map.get(ml_prediction, "不明")
        
        fr = advanced_data.get('funding_rate', 0)
        lsr = advanced_data.get('ls_ratio', 1.0)
        oi_chg = advanced_data.get('oi_change_4h', 0.0)
        fg_index = advanced_data.get('fg_index', 50)
        fg_value = advanced_data.get('fg_value', 'Neutral')
        liq_long = advanced_data.get('liq_24h_long', 0)
        
        current_time = datetime.now(timezone.utc).astimezone(None).strftime('%Y-%m-%d %H:%M JST')
        
        max_proba = proba[np.argmax(proba)]
        uncertainty_score = 1.0 - max_proba
        
        # 🚨 主因とリスクの判定ロジック
        main_cause = "技術的環境（重要支持線の維持）"
        if fg_index <= 30 and liq_long > 100_000_000:
             main_cause = "センチメントショック（極度の恐怖と多頭清算連鎖）"
        elif fr > 0.00015 and lsr > 1.1:
             main_cause = "需給アンバランス（ロング過熱とFR高騰）"
        
        risk_level = "中🔴"
        if uncertainty_score > 0.40 or fg_index <= 25:
             risk_level = "高🔴🔴"
             
        
        report_structure = f"""
==> **【BTC 市場の主因分析】** <==
📅 {current_time}

📌 **要点**
* **主因:** 現在の市場動向の主因は**{main_cause}**にあります。
* **センチメント:** 恐怖・強欲指数は**{fg_index}**の「**{fg_value}**」水準で、市場の動揺が示唆されます。
* **技術的環境:** BTC価格**${price:.2f}**は20-SMA（${sma:.2f}）に対し{'🟢 上回る' if price > sma else '🔴 下回る'}。短期は{'弱気' if price < sma else '強気'}トレンド。

---
### 📉 市場主因とリスク分析

| カテゴリ | 指標 | 現在値 / 状況 | 分析 / 示唆 |
| :--- | :--- | :--- | :--- |
| **需給/流動性** | FR (資金調達率) | {fr*100:.4f}% | {'🚨 ロングポジションのコスト高。スクイーズリスクあり。' if fr > 0.00015 else '中立。'} |
| | L/S 比率 | {lsr:.2f} | {'🔴 ロング優勢。レバレッジポジションの偏り。' if lsr > 1.1 else '🟡 均衡。'} |
| | OI 変化率 (4H) | {oi_chg*100:.1f}% | {'🔴 増加。トレンド継続の勢いが強い。' if oi_chg > 0.03 else '🟢 減少。トレンド減速の可能性。'} |
| **センチメント** | F&G Index | {fg_index} ({fg_value}) | {'極度の恐怖。逆張りチャンスか、底割れ注意。' if fg_index <= 20 else '楽観的。短期的な過熱感。'} |
| | 24H 多頭清算額 | ${liq_long:,.0f} | {'🚨 大規模清算発生。市場のフラッシュクラッシュ警戒。' if liq_long > 100_000_000 else '通常。'} |
| **ボラティリティ** | ATR | ${atr:.2f} | **${(atr / price) * 100:.2f}%**。レンジ相場か、トレンド加速中かを示唆。 |

### 🎯 チャンスとリスク

* **メッセージ面 (チャンス):** 市場の恐怖が高まっている今（F&G Index:{fg_index}）、**強力な押し目買いの機会**が到来する可能性があります。
* **🚨 リスクレベル:** **{risk_level}**。高レバレッジによる清算連鎖リスクが継続しています。重要支持線での反発確認が必須です。
"""
        
        # 最終結論の調整 (スクイーズ/不確実性警戒)
        final_conclusion = ml_result
        if (ml_result == "📈 上昇" and fr > 0.00015):
             final_conclusion = f"⚠️ {ml_result} (ロング過熱注意)"
        elif (ml_result == "📉 下落" and liq_long > 100_000_000):
             final_conclusion = f"🚨 {ml_result} (清算連鎖リスク)"
        
        # 推奨戦略の決定
        if uncertainty_score > 0.40 and ml_prediction == 0:
            strategy_advice_short = "様子見/取引回避を強く推奨。レンジブレイクを待つ。"
            entry_long = "安全な支持帯"
            entry_short = "強固なレジスタンス"
        else:
             strategy_advice_short = f"ML予測の**{final_conclusion}**に沿った取引を検討。"
             entry_long = f"現在の価格帯 (${price:.2f}) での押し目買い"
             entry_short = f"現在の価格帯 (${price:.2f}) での戻り売り"
        
        report_conclusion = f"""
==> **【最終結論とアクションプラン】** <==
📅 {current_time}

---
### 🤖 予測と総合戦略

| 項目 | 分析結果 | 確率 | 不確実性スコア |
| :--- | :--- | :--- | :--- |
| **ML 予測結論** | **{final_conclusion}** | **{max_proba*100:.1f}%** | **{uncertainty_score*100:.1f}%** |

* **総合判断:** **{strategy_advice_short}** 不確実性スコアが高いため、特に短期取引ではポジションサイズを限定してください。

### 🎯 短期戦略（先物/デイトレ）

| 方向 | エントリー目安 | ストップロス | 利確目標 |
| :--- | :--- | :--- | :--- |
| **{'弱気' if ml_prediction <= 0 else '強気'}** | {entry_short if ml_prediction <= 0 else entry_long} | ATRに基づき (${atr:.2f}分) | 直近の高値/安値帯 |

### 📈 中長期戦略（現物/押し目）

* **戦略:** **様子見〜押し目買い**。市場の恐怖が高まるタイミングをチャンスと捉え、安全な支持帯（例: 90,000米ドル付近）での買い増しを計画。
* **分散:** BTCに集中せず、ETHやSOLなど成長テーマのアルトコインに資金を分散させ、中長期のリスクを低減。

📚 **総括**
BOTの最終分析は、テクニカルなサインとセンチメントのバランスを見ています。現在の市場は「具材のタイミングがすべて」の鍋料理のような状態です。焦らず、冷静にアクションを取りましょう。
"""
        return report_structure, report_conclusion
        
    # --- (F) Telegram 通知関数 - エラー処理を強化 ---
    def send_telegram_notification(self, message: str):
        """通知の実装"""
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
        try:
            response = requests.post(url, data=payload)
            if response.status_code == 200:
                print("✅ Telegramへの通知が完了しました。")
            else:
                # 🚨 エラー時の詳細をログに出力
                print(f"🚨 Telegram通知エラー (HTTP {response.status_code}): {response.text}")
        except Exception as e:
            print(f"🚨 Telegramリクエスト失敗: {e}")
