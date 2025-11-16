# futures_ml_bot.py (1時間足に最適化された最高峰の市場分析レポート生成バージョン - 堅牢性向上)

import os
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import requests
import joblib
import json
import time
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, Dict, Any, List

# --- 1. 環境変数設定 ---
MEXC_API_KEY = os.environ.get('MEXC_API_KEY')
MEXC_SECRET = os.environ.get('MEXC_SECRET')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

FUTURES_SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h' 
MODEL_FILENAME = 'btc_futures_ml_model.joblib'

# 外部APIエンドポイント (Fear & Greed Index)
FG_INDEX_API_URL = 'https://api.alternative.me/fng/?limit=1'

# --- 2. Advanced Custom Data Fetching Function (変更なし) ---
def fetch_advanced_metrics(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
    """FR, Fear & Greed Indexなど、確実に取得できる公開実践データのみを取得します。"""
    metrics = {}
    default_fallbacks = {
        'funding_rate': 0.0, 
        'fg_index': 50, 
        'fg_value': 'Neutral (API失敗)'
    }
    metrics.update(default_fallbacks)

    try:
        # この関数は認証済みインスタンスを使用し、FRなどを取得します
        ticker = exchange.fetch_ticker(symbol)
        metrics['funding_rate'] = float(ticker.get('fundingRate', 0) or 0)
        
        try:
            fg_response = requests.get(FG_INDEX_API_URL, timeout=5)
            fg_response.raise_for_status()
            fg_data = fg_response.json().get('data', [{}])
            metrics['fg_index'] = int(fg_data[0].get('value', 50))
            metrics['fg_value'] = fg_data[0].get('value_classification', 'Neutral')
        except Exception as e:
            print(f"⚠️ F&G Index APIエラー: {e}")
            
        return metrics
    
    except Exception as e:
        # APIキー認証失敗時でも、公開情報（F&G Index）は取得を試みる
        print(f"🚨 主要データ取得エラー (CCXT/その他): {e}")
        try:
            fg_response = requests.get(FG_INDEX_API_URL, timeout=5)
            fg_response.raise_for_status()
            fg_data = fg_response.json().get('data', [{}])
            default_fallbacks['fg_index'] = int(fg_data[0].get('value', 50))
            default_fallbacks['fg_value'] = fg_data[0].get('value_classification', 'Neutral')
        except:
             pass # F&G Indexも失敗した場合はそのままフォールバック
        return default_fallbacks


# --- 3. メインBOTクラス ---
class FuturesMLBot:
    def __init__(self):
        # 認証済みインスタンス (トレード操作用 - APIキーが正しく設定されていない場合、認証が必要なAPIコールは失敗します)
        self.exchange = ccxt.mexc({
            'apiKey': MEXC_API_KEY if MEXC_API_KEY else 'dummy',
            'secret': MEXC_SECRET if MEXC_SECRET else 'dummy',
            'options': {'defaultType': 'future'},
            'enableRateLimit': True,
        })
        
        # 🆕 公開データ取得用インスタンス (OHLCVデータは公開されているため、APIキーなしで初期化)
        self.public_exchange = ccxt.mexc({
            'options': {'defaultType': 'future'},
            'enableRateLimit': True,
        })
        
        self.target_threshold = 0.0005 
        self.prediction_period = 1 
        self.feature_cols = [] 

    # --- (A) データ取得 (OHLCV) ---
    def fetch_ohlcv_data(self, limit: int = 100, timeframe: str = TIMEFRAME) -> pd.DataFrame:
        """OHLCVデータを公開用インスタンスから取得します。"""
        try:
            # 🆕 公開用インスタンス (self.public_exchange) を使用し、403エラーを回避
            ohlcv = self.public_exchange.fetch_ohlcv(FUTURES_SYMBOL, timeframe, limit=limit)
            if not ohlcv:
                print("🚨 OHLCVデータが空です。")
                return pd.DataFrame()
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"🚨 OHLCVデータ取得エラー (公開APIを使用中): {e}")
            return pd.DataFrame()

    # --- (B), (C), (D) 特徴量作成、学習、予測 (変更なし) ---
    def create_ml_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """実践的なテクニカル特徴量を作成"""
        if df.empty:
            return pd.DataFrame(), pd.Series(dtype=int)

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
        
        if not self.feature_cols and not df.empty:
            cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'timestamp']]
            self.feature_cols = [col for col in cols if df[col].dtype in [np.float64, np.int64]]
        
        if not self.feature_cols:
            return pd.DataFrame(), df['Target']
            
        return df[self.feature_cols], df['Target']

    def train_and_save_model(self, df_long_term: pd.DataFrame) -> bool:
        print("🧠 モデルの再学習タスクを開始...")
        X_train, Y_train = self.create_ml_features(df_long_term.copy())
        
        if X_train.empty:
            print("🚨 致命的なエラー: 学習データが不足しているため、モデルを構築できませんでした。")
            return False
        
        model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', max_depth=10)
        model.fit(X_train, Y_train)
        
        joblib.dump(model, MODEL_FILENAME)
        print("✅ モデルの再学習が完了し、ファイルに保存されました。")
        return True

    def predict_and_report(self, df_latest: pd.DataFrame, advanced_data: Dict[str, Any]) -> bool:
        try:
            model = joblib.load(MODEL_FILENAME)
        except FileNotFoundError:
            report = "🚨 <b>エラー:</b> モデルファイルが見つかりません。まず学習とコミットを行ってください。"
            self.send_telegram_notification(report) 
            return False

        X_latest, _ = self.create_ml_features(df_latest.copy())
        
        if X_latest.empty:
            report = ("🚨 <b>予測スキップ通知:</b> OHLCVデータが不足しています。")
            self.send_telegram_notification(report)
            return False
            
        latest_X = X_latest.iloc[[-1]] 
        
        prediction_val = model.predict(latest_X)[0]
        prediction_proba = model.predict_proba(latest_X)[0]
        
        # プレミアムレポートを生成
        full_report = self._generate_premium_report(
            latest_price_data=df_latest.iloc[-1],
            latest_features=latest_X.iloc[-1],
            advanced_data=advanced_data, 
            ml_prediction=prediction_val, 
            proba=prediction_proba
        )
        
        self.send_telegram_notification(full_report)
        
        return True
    # --- プレミアムレポートのためのヘルパー関数 (変更なし) ---
    def _determine_market_regime(self, price: float, sma: float, atr: float, high: float, low: float) -> Tuple[str, str]:
        """SMAとATRを用いて市場構造（レジーム）を判断する"""
        
        sma_deviation = abs(price - sma)
        is_trending = sma_deviation > (atr * 0.5)
        
        price_range = high - low
        is_tight_range = price_range < (atr * 0.5)

        if is_trending:
            if price > sma:
                regime_status = "短期上昇トレンド継続"
                regime_emoji = "🚀"
            else:
                regime_status = "短期下降トレンド継続"
                regime_emoji = "🌊"
        else:
            if is_tight_range:
                regime_status = "ブレイクアウト前夜 (低ボラティリティ収束)"
                regime_emoji = "⏳"
            else:
                regime_status = "横ばいレンジ (方向性欠如)"
                regime_emoji = "⚖️"
        
        return regime_status, regime_emoji

    def _analyze_macro_sentiment(self, fg_index: int, fr: float) -> Tuple[str, List[str], str]:
        """F&G IndexとFRからマクロなセンチメントと核心リスクを判断する"""
        
        if fg_index >= 70:
            sentiment_summary = "極度の楽観（Greed）。ロングポジション過多による調整リスク高。"
            risk_color = "🔴"
        elif fg_index <= 30:
            sentiment_summary = "極度の恐怖（Fear）。パニック売りからの短期的な強力反発期待（逆張り妙味）。"
            risk_color = "🟢"
        else:
            sentiment_summary = "中立。特定の要因（FRなど）でリスクが増加する可能性。"
            risk_color = "🟡"

        core_risks = []
        if fr > 0.00015:
            core_risks.append(f"<b>資金調達率 (FR):</b> {fr*100:.4f}%と極めて高水準。強制的な<b>ロングスクイーズリスク</b>が主要因。")
        else:
             core_risks.append("マクロ的リスクは、主に外部要因（金利、ETF動向）に依存。ポジションの傾きは現在中立。")
        
        if fg_index >= 70 and fr > 0.0001:
            core_risks.append("<b>過熱警告:</b> 楽観（FGI）とポジションの傾き（FR）が一致。調整は急激になる可能性あり。")

        return sentiment_summary, core_risks, risk_color
        
    def _generate_premium_report(self, latest_price_data: pd.Series, latest_features: pd.Series, advanced_data: Dict[str, Any], ml_prediction: int, proba: np.ndarray) -> str:
        """ML予測と実データを統合し、最高峰の分析レポートを生成する。"""
        
        price = latest_price_data['Close']
        high = latest_price_data['High']
        low = latest_price_data['Low']
        sma = latest_features.get('SMA', price)
        atr = latest_features.get('ATR', price * 0.01)
        
        pred_map = {-1: "📉 下落", 0: "↔️ レンジ", 1: "📈 上昇"}
        ml_result = pred_map.get(ml_prediction, "不明")
        max_proba = proba[np.argmax(proba)]
        
        fg_index = advanced_data.get('fg_index', 50)
        fr = advanced_data.get('funding_rate', 0)
        
        current_time = datetime.now(timezone.utc).astimezone(None).strftime('%Y-%m-%d %H:%M JST')
        
        regime_status, regime_emoji = self._determine_market_regime(price, sma, atr, high, low)
        sentiment_summary, core_risks, risk_color = self._analyze_macro_sentiment(fg_index, fr)
        
        R1 = price + atr
        S1 = price - atr
        R2 = price + (atr * 2)
        S2 = price - (atr * 2)
        
        ml_interpretation = f"MLモデルは次の1時間で<b>{ml_result}</b>を予測しています (信頼度: {max_proba*100:.1f}%)。"
        if ml_prediction == 0 and max_proba < 0.4:
            ml_interpretation += "MLの判断が分かれており、不確実性が高いため、レンジ内での取引を推奨します。"

        core_reason_list = [f"<b>ML予測:</b> {ml_interpretation}"]
        core_reason_list.extend(core_risks)
        
        if regime_status.startswith("短期上昇トレンド"):
            core_reason_list.append(f"<b>テクニカル要因:</b> 価格は20-SMA (${sma:.2f}) を上回り、短期モメンタムは継続中。")
        elif regime_status.startswith("短期下降トレンド"):
            core_reason_list.append(f"<b>テクニカル要因:</b> 20-SMA (${sma:.2f}) をレジスタンスとして機能させており、短期的な下落圧力が支配的。")
        else:
             core_reason_list.append(f"<b>テクニカル要因:</b> {regime_status}。ボラティリティ (${atr:.2f}) が収束/拡散の兆候。")

        chance_list = [
            f"<b>ML予測との一致:</b> {ml_result}の方向にエントリーする場合、信頼度 ({max_proba*100:.1f}%) を裏付けとして活用可能。",
            f"<b>市場心理の逆張り:</b> F&G指数が<b>{fg_index}</b> ({advanced_data['fg_value']}) の場合、過去の統計では強力な逆張りの買い場を提供する傾向がある。",
        ]
        
        risk_list = [
            f"<b>{risk_color} 総合リスク警告:</b> 市場構造は現在 <b>{regime_status}</b> であり、FRやFGIに基づくセンチメントは {sentiment_summary} です。",
            f"<b>ボラティリティリスク (ATR):</b> 過去14時間の平均変動幅は <b>${atr:.2f}</b> です。ストップロスは最低この値幅を考慮に入れる必要があります。",
            f"<b>重要レベル割れ:</b> 2-ATRサポートS2 (${S2:.2f}) を割り込んだ場合、次の主要な節目まで急落するリスクが高い。"
        ]
        
        if ml_prediction == 1 or fg_index <= 30:
            strategy_title = "📈 <b>推奨戦略: 短期ロング/押し目買い</b>"
            entry_zone = f"<b>S1: ${S1:.2f}〜現在価格</b>（市場の弱さを利用したエントリー）"
            sl_level = f"<b>S2: ${S2:.2f}</b>（ここを割ると短期トレンド転換の可能性）"
            tp_targets = f"R1: <b>${R1:.2f}</b> (50%)、R2: <b>${R2:.2f}</b> (30%)、R2+ATR: <b>${R2+atr:.2f}</b> (20%)"
        elif ml_prediction == -1 or fr > 0.00015:
            strategy_title = "📉 <b>推奨戦略: 短期ショート/戻り売り</b>"
            entry_zone = f"<b>現在価格〜R1: ${R1:.2f}</b>（一時的な戻りを狙った売り）"
            sl_level = f"<b>R2: ${R2:.2f}</b>（ここを突破するとショートスクイーズの可能性）"
            tp_targets = f"S1: <b>${S1:.2f}</b> (50%)、S2: <b>${S2:.2f}</b> (30%)、S2-ATR: <b>${S2-atr:.2f}</b> (20%)"
        else:
            strategy_title = "⚖️ <b>推奨戦略: レンジ内取引/ブレイクアウト待機</b>"
            entry_zone = f"<b>R1/S1 ({R1:.2f} / {S1:.2f})</b> の極値"
            sl_level = f"エントリーポイントから <b>ATRの0.5倍</b> の外側"
            tp_targets = f"<b>R1/S1</b>の反対側の極値"
        
        report = f"""
<b>【👑 BTC 1時間足 最新状況レポート 👑】</b>
📅 <b>{current_time}</b> | <b>{TIMEFRAME}足分析</b> (次期予測: 1時間後)
<p>
    <b>現在の市場構造:</b> <b>{regime_emoji} {regime_status}</b> | <b>現在価格: ${price:.2f} USDT</b>
</p>

---------------------------------------
<h3><b>🔍 1. 核心理由と構造的リスク</b></h3>
<ul>
    {''.join([f'<li>{reason}</li>' for reason in core_reason_list])}
</ul>
<p>
    <b>市場心理:</b> 恐怖と欲望指数: <b>{fg_index}</b> ({advanced_data['fg_value']})。
</p>

<h3><b>💡 2. チャンスと期待される反発点</b></h3>
<ul>
    {''.join([f'<li>{chance}</li>' for chance in chance_list])}
</ul>

<h3><b>🚨 3. リスク（定量評価）と警戒レベル</b></h3>
<ul>
    {''.join([f'<li>{risk}</li>' for risk in risk_list])}
    <li><b>主要サポートS1/S2:</b> ${S1:.2f} / ${S2:.2f}</li>
    <li><b>主要レジスタンスR1/R2:</b> ${R1:.2f} / ${R2:.2f}</li>
</ul>

---------------------------------------
<h3><b>⚡️ 4. 行動計画と最適化された戦略</b></h3>

<h4>{strategy_title}</h4>
<pre>
<b>入場区間:</b> {entry_zone}
<b>損切り（SL）:</b> {sl_level}
<b>利確（TP）:</b> {tp_targets}
</pre>
<p>
<b>💡 戦略のヒント:</b> 1時間足はノイズが多いため、推奨レベルでの<b>部分利確・部分損切り</b>の徹底が不可欠です。
</p>
---------------------------------------
<b>📚 まとめ：トレーダーへのメッセージ</b>
現在の市場は <b>{regime_status}</b> の段階にあり、短期的な動向を予測するにはMLモデルの信頼度 ({max_proba*100:.1f}%) とATRによるレベルの厳守が鍵です。
緻密な価格変動 (${atr:.2f}) に対応するため、高い集中力を持って取引に臨んでください。
"""
        return report
        
    # --- (F) Telegram通知機能 (変更なし) ---
    def send_telegram_notification(self, message: str):
        """通知の実装"""
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'HTML'}
        try:
            response = requests.post(url, data=payload)
            if response.status_code != 200:
                print(f"🚨 Telegram通知エラー (HTTP {response.status_code}): {response.text}")
        except Exception as e:
            print(f"🚨 Telegramリクエストに失敗しました: {e}")
