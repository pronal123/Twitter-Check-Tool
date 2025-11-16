# futures_ml_bot.py (1時間足に最適化された最高峰の市場分析レポート生成バージョン - 分析専用)

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
# 注: このBOTは分析専用であり、トレード操作は行いません。
# Telegram通知に必要な変数のみを保持します。
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

FUTURES_SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h' 
MODEL_FILENAME = 'btc_futures_ml_model.joblib'

# 外部APIエンドポイント (Fear & Greed Index)
FG_INDEX_API_URL = 'https://api.alternative.me/fng/?limit=1'

# --- 2. Advanced Custom Data Fetching Function ---
def fetch_advanced_metrics() -> Dict[str, Any]:
    """Fear & Greed Indexなど、確実に取得できる公開実践データのみを取得します。"""
    metrics = {
        'funding_rate': 0.0, # 分析専用のため、Funding Rateは公開APIから取得できる場合に限り使用 (Binanceの公開TickerはFRを含まないため0で初期化)
        'fg_index': 50, 
        'fg_value': 'Neutral (API失敗)'
    }

    try:
        # F&G Indexの取得
        fg_response = requests.get(FG_INDEX_API_URL, timeout=5)
        fg_response.raise_for_status()
        fg_data = fg_response.json().get('data', [{}])
        metrics['fg_index'] = int(fg_data[0].get('value', 50))
        metrics['fg_value'] = fg_data[0].get('value_classification', 'Neutral')
    except Exception as e:
        print(f"⚠️ F&G Index APIエラー: {e}")
        
    # Funding Rateを補完する公開APIがあれば追加しても良いが、ここではシンプルに0とする
    
    return metrics


# --- 3. メインBOTクラス ---
class FuturesMLBot:
    def __init__(self):
        # 認証情報なしの公開APIインスタンスのみを使用
        self.exchange = ccxt.binance({
            'options': {'defaultType': 'future'},
            'enableRateLimit': True,
        })
        
        self.target_threshold = 0.0005 
        self.prediction_period = 1 
        self.feature_cols: List[str] = [] 

    # --- (A) データ取得 (OHLCV) ---
    def fetch_ohlcv_data(self, limit: int = 1000, timeframe: str = TIMEFRAME) -> pd.DataFrame:
        """OHLCVデータをBinance公開APIから取得します。"""
        try:
            # 公開インスタンスを使用
            ohlcv = self.exchange.fetch_ohlcv(FUTURES_SYMBOL, timeframe, limit=limit)
            if not ohlcv:
                print("🚨 OHLCVデータが空です。")
                return pd.DataFrame()
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            print(f"✅ Binanceから{len(df)}件のOHLCVデータを取得しました。")
            return df
        except Exception as e:
            # 致命的なAPI障害
            print(f"🚨 OHLCVデータ取得エラー (Binance公開APIを使用中): {e}")
            return pd.DataFrame()

    # --- (B), (C), (D) 特徴量作成、学習、予測 (変更なし) ---
    def create_ml_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """実践的なテクニカル特徴量を作成"""
        if df.empty:
            return pd.DataFrame(), pd.Series(dtype=int)

        # 基礎的なテクニカル指標
        df['SMA'] = ta.sma(df['Close'], length=20)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['MACD_H'] = ta.macd(df['Close'])['MACDh_12_26_9']
        df['Vol_Diff'] = df['Volume'] / ta.sma(df['Volume'], length=20)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14) 

        # ラグ特徴量
        for lag in [1, 2, 3]:
            df[f'RSI_L{lag}'] = df['RSI'].shift(lag)
            df[f'Price_L{lag}'] = df['Close'].pct_change(lag).shift(lag)
            
        # 予測対象（Target）: 次の1時間で設定した閾値以上動くか (+1: 上昇, -1: 下落, 0: レンジ)
        future_change = df['Close'].pct_change(periods=-self.prediction_period).shift(self.prediction_period)
        
        df['Target'] = np.select(
            [future_change > self.target_threshold, future_change < -self.target_threshold],
            [1, -1], default=0
        )
        
        df.dropna(inplace=True)
        
        # 特徴量カラムリストの更新
        if not self.feature_cols and not df.empty:
            cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'timestamp']]
            self.feature_cols = [col for col in cols if df[col].dtype in [np.float64, np.int64]]
        
        if not self.feature_cols:
            return pd.DataFrame(), df['Target']
            
        return df[self.feature_cols], df['Target']

    def train_and_save_model(self, df_long_term: pd.DataFrame) -> bool:
        print("🧠 モデルの再学習タスクを開始...")
        # 過去の長期データから特徴量とターゲットを作成
        X_train, Y_train = self.create_ml_features(df_long_term.copy())
        
        if X_train.empty:
            print("🚨 致命的なエラー: 学習データが不足しているため、モデルを構築できませんでした。")
            return False
        
        # ランダムフォレスト分類器を使用
        model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', max_depth=10)
        model.fit(X_train, Y_train)
        
        # モデルをファイルに保存
        joblib.dump(model, MODEL_FILENAME)
        print("✅ モデルの再学習が完了し、ファイルに保存されました。")
        return True

    def predict_and_report(self, df_latest: pd.DataFrame, advanced_data: Dict[str, Any]) -> bool:
        """最新データに基づいて予測を行い、レポートを生成し、Telegramに送信する。"""
        try:
            model = joblib.load(MODEL_FILENAME)
        except FileNotFoundError:
            report = "🚨 <b>エラー:</b> モデルファイルが見つかりません。まず学習とコミットを行ってください。"
            self.send_telegram_notification(report) 
            return False

        # 最新データから特徴量を作成
        X_latest, _ = self.create_ml_features(df_latest.copy())
        
        if X_latest.empty:
            report = ("🚨 <b>予測スキップ通知:</b> OHLCVデータが不足しています。")
            self.send_telegram_notification(report)
            return False
            
        latest_X = X_latest.iloc[[-1]] 
        
        # 予測の実行
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
        """F&G Indexからマクロなセンチメントと核心リスクを判断する (FRは参考情報)"""
        
        if fg_index >= 70:
            sentiment_summary = "極度の楽観（Greed）。過熱感による調整リスク高。"
            risk_color = "🔴"
        elif fg_index <= 30:
            sentiment_summary = "極度の恐怖（Fear）。パニック売りからの短期的な強力反発期待（逆張り妙味）。"
            risk_color = "🟢"
        else:
            sentiment_summary = "中立。特定の要因でリスクが増加する可能性。"
            risk_color = "🟡"

        core_risks = []
        # FRデータが公開APIで取得できないため、F&G Indexを主要なセンチメントリスクとする
        core_risks.append(f"<b>資金調達率 (FR):</b> データ非取得のため中立 (0.00%)。")
        if fg_index >= 70:
            core_risks.append("<b>過熱警告:</b> 楽観（FGI）が非常に高い。調整は急激になる可能性あり。")
        elif fg_index <= 30:
            core_risks.append("<b>反発期待:</b> 恐怖（FGI）が非常に高い。短期的な反発の可能性を探る。")

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
            f"<b>{risk_color} 総合リスク警告:</b> 市場構造は現在 <b>{regime_status}</b> であり、FGIに基づくセンチメントは {sentiment_summary} です。",
            f"<b>ボラティリティリスク (ATR):</b> 過去14時間の平均変動幅は <b>${atr:.2f}</b> です。リスク許容度を決定する際の基準値としてください。",
            f"<b>重要レベル割れ:</b> 2-ATRサポートS2 (${S2:.2f}) を割り込んだ場合、次の主要な節目まで急落するリスクが高い。"
        ]
        
        if ml_prediction == 1 or fg_index <= 30:
            strategy_title = "📈 <b>分析結果に基づいた推奨アクション: 短期ロング戦略の検討</b>"
            action_guide = f"""
<b>検討ゾーン:</b> S1: ${S1:.2f}〜現在価格（押し目を待つ）
<b>リスク管理基準 (SL):</b> S2: ${S2:.2f}（分析上のサポートライン）
<b>利確目標 (TP):</b> R1: ${R1:.2f}, R2: ${R2:.2f}
"""
        elif ml_prediction == -1 or fg_index >= 70:
            strategy_title = "📉 <b>分析結果に基づいた推奨アクション: 短期ショート戦略の検討</b>"
            action_guide = f"""
<b>検討ゾーン:</b> 現在価格〜R1: ${R1:.2f}（戻りを待つ）
<b>リスク管理基準 (SL):</b> R2: ${R2:.2f}（分析上のレジスタンスライン）
<b>利確目標 (TP):</b> S1: ${S1:.2f}, S2: ${S2:.2f}
"""
        else:
            strategy_title = "⚖️ <b>分析結果に基づいた推奨アクション: レンジ内取引戦略の検討</b>"
            action_guide = f"""
<b>検討ゾーン:</b> R1/S1 ({R1:.2f} / {S1:.2f}) の極値付近での反転
<b>リスク管理基準 (SL):</b> 各極値からATRの0.5倍の外側
<b>利確目標 (TP):</b> R1/S1の反対側の極値
"""
        
        report = f"""
<b>【👑 BTC 1時間足 最新状況レポート 👑】</b>
📅 <b>{current_time}</b> | <b>{TIMEFRAME}足分析</b>
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
<h3><b>⚡️ 4. 分析に基づく行動ガイド</b></h3>

<h4>{strategy_title}</h4>
<pre>
{action_guide}
</pre>
<p>
<b>💡 注意点:</b> これは機械学習とテクニカル指標に基づく<b>分析情報</b>であり、トレードの推奨ではありません。最終的な意思決定は自己責任で行ってください。
</p>
---------------------------------------
<b>📚 まとめ：分析の焦点</b>
現在の市場は <b>{regime_status}</b> の段階にあり、短期的な動向を予測するにはMLモデルの信頼度 ({max_proba*100:.1f}%) とATRによるレベルの厳守が鍵です。
緻密な価格変動 (${atr:.2f}) に対応するため、高い集中力を持って分析を深めてください。
"""
        return report
        
    # --- (F) Telegram通知機能 (変更なし) ---
    def send_telegram_notification(self, message: str):
        """通知の実装"""
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            print("⚠️ Telegram BOT TOKENまたはCHAT IDが設定されていません。レポートの送信をスキップします。")
            print("--- レポート内容（コンソール出力） ---")
            print(message)
            print("---------------------------------")
            return

        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'HTML'}
        try:
            response = requests.post(url, data=payload)
            if response.status_code != 200:
                print(f"🚨 Telegram通知エラー (HTTP {response.status_code}): {response.text}")
        except Exception as e:
            print(f"🚨 Telegramリクエストに失敗しました: {e}")
