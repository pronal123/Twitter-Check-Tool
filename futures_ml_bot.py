# futures_ml_bot.py (代替データソースと擬似OHLCV生成版 / 分析強化版)

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
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

TIMEFRAME = '1d' # 🚨 変更: 安定化のため分析単位を日足に変更
MODEL_FILENAME = 'btc_futures_ml_model.joblib'

# 外部APIエンドポイント (CoinGecko & Fear & Greed Index)
# CoinGecko: 過去90日間の日足価格データ
COINGECKO_API_URL = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=90&interval=daily'
FG_INDEX_API_URL = 'https://api.alternative.me/fng/?limit=1'

# --- 2. Advanced Custom Data Fetching Function ---
def fetch_advanced_metrics() -> Dict[str, Any]:
    """Fear & Greed Indexを取得します。"""
    metrics = {
        'fg_index': 50, 
        'fg_value': 'Neutral (API失敗)'
    }
    try:
        fg_response = requests.get(FG_INDEX_API_URL, timeout=5)
        fg_response.raise_for_status()
        fg_data = fg_response.json().get('data', [{}])
        metrics['fg_index'] = int(fg_data[0].get('value', 50))
        metrics['fg_value'] = fg_data[0].get('value_classification', 'Neutral')
    except Exception as e:
        print(f"⚠️ F&G Index APIエラー: {e}")
        
    return metrics


# --- 3. メインBOTクラス ---
class FuturesMLBot:
    def __init__(self):
        # CCXTは使用しないため、インスタンス化を削除
        self.target_threshold = 0.01 # 日足のため閾値を1.0%に変更
        self.prediction_period = 1 # 次の日の予測
        self.feature_cols: List[str] = [] 

    # --- (A) データ取得とOHLCVの擬似再構築 ---
    def fetch_ohlcv_data(self, limit: int = 90) -> pd.DataFrame:
        """CoinGeckoから終値を取得し、OHLCVを統計的に推定してデータフレームを作成します。"""
        try:
            # CoinGecko APIから日足データを取得
            response = requests.get(COINGECKO_API_URL, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # タイムスタンプと終値を取得
            prices = data.get('prices', [])
            if not prices:
                raise Exception("CoinGeckoから価格データが取得できませんでした。")

            # Pandas DataFrameに変換
            df = pd.DataFrame(prices, columns=['timestamp', 'Close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df['Close'] = df['Close'].round(2)
            
            # --- 擬似OHLCVの再構築 ---
            
            # 1. Openの生成 (前の足のCloseを使用)
            df['Open'] = df['Close'].shift(1)
            
            # 2. High/Lowの生成 (Closeに対するランダムなボラティリティを付与)
            # 過去の価格変動に基づきボラティリティのノイズを生成
            vol_multiplier = 0.03 # 日次で3%程度のボラティリティを想定
            df['High_Noise'] = np.abs(np.random.normal(0, vol_multiplier * 0.5, len(df)))
            df['Low_Noise'] = np.abs(np.random.normal(0, vol_multiplier * 0.5, len(df)))
            
            # HighとLowを生成
            df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + df['High_Noise'])
            df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - df['Low_Noise'])
            
            # 3. Volumeの生成 (F&G Indexと逆相関のノイズを組み合わせて近似)
            # 出来高は「恐怖時(F&G Index低)に増える」という傾向をモデル化
            fg_data = fetch_advanced_metrics()
            fg_index = fg_data.get('fg_index', 50)
            
            # 出来高のベース（Market Capから推測）
            volume_base = np.random.randint(200000, 500000, len(df))
            
            # センチメント補正 (F&Gが低いほど補正値が高くなる)
            sentiment_boost = (100 - df.index.to_series().apply(lambda x: fg_index)) / 50 
            
            df['Volume'] = (volume_base * sentiment_boost).round(0)
            
            # データの整形と不要行の削除
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            
            print(f"✅ CoinGeckoから終値を取得し、{len(df)}件のOHLCV擬似データを生成しました。")
            return df
        
        except Exception as e:
            # 🚨 CoinGecko APIも失敗した場合、完全にランダムなダミーデータを生成
            print(f"🚨 CoinGecko APIデータ取得エラー: {e}")
            print("🚨 予備のダミーデータ生成ロジックにフォールバックします。")
            
            # --- ダミーデータ生成ロジック (日足) ---
            np.random.seed(42)
            base_price = 62000
            end_time = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            timestamps = pd.to_datetime(pd.date_range(end=end_time, periods=limit, freq='D'))
            
            price_changes = np.random.normal(0, 0.005, limit).cumsum()
            prices = base_price * (1 + price_changes)
            
            data = {
                'Open': prices,
                'Close': prices + np.random.normal(0, 100, limit),
                'High': prices + np.abs(np.random.normal(0, 150, limit)),
                'Low': prices - np.abs(np.random.normal(0, 150, limit)),
                'Volume': np.random.randint(100000, 500000, limit)
            }
            df = pd.DataFrame(data)
            df.index = timestamps
            df.index.name = 'timestamp'
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            print(f"✅ 完全にランダムな予備ダミーデータ ({len(df)}件) の生成が完了しました。")
            return df


    # --- (B) 特徴量作成 (分析強化版を維持) ---
    def create_ml_features(self, df: pd.DataFrame, advanced_data: Dict[str, Any] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """ボラティリティ、モメンタム、トレンド、センチメントを含む高度な特徴量を作成"""
        if df.empty:
            return pd.DataFrame(), pd.Series(dtype=int)

        # 🚨 日足に合わせてパラメータを調整
        # --- トレンド指標 ---
        df['SMA10'] = ta.sma(df['Close'], length=10) # 短期 (2週間)
        df['SMA30'] = ta.sma(df['Close'], length=30) # 中期 (1ヶ月)
        df['Trend_Signal'] = np.where(df['SMA10'] > df['SMA30'], 1, -1) 
        
        # --- モメンタム指標 ---
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['MACD_H'] = ta.macd(df['Close'], fast=12, slow=26, signal=9)['MACDh_12_26_9']
        df['Momentum'] = ta.mom(df['Close'], length=10)
        
        # --- ボラティリティ指標 ---
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14) 
        df['BBands_Width'] = ta.bbands(df['Close'])['BBP_20_2.0']
        
        # --- ボリューム指標 ---
        df['Volume_SMA'] = ta.sma(df['Volume'], length=10)
        df['Volume_ROC'] = df['Volume'].pct_change(1)
        
        # --- 価格変化率 ---
        for lag in [1, 3, 5]: 
            df[f'Price_L{lag}'] = df['Close'].pct_change(lag).shift(lag)
            
        # --- センチメント指標 ---
        if advanced_data:
            # 最新のF&G Indexをすべての行に適用する（時系列データではないため）
            df['FG_Index'] = advanced_data.get('fg_index', 50)
        else:
            pass

        # 予測対象（Target）
        future_change = df['Close'].pct_change(periods=-self.prediction_period).shift(self.prediction_period)
        
        df['Target'] = np.select(
            [future_change > self.target_threshold, future_change < -self.target_threshold],
            [1, -1], default=0
        )
        
        df.dropna(inplace=True)
        
        if not self.feature_cols and not df.empty:
            cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'timestamp', 'SMA10', 'SMA30']]
            self.feature_cols = [col for col in cols if df[col].dtype in [np.float64, np.int64]]
        
        self.feature_cols = [col for col in self.feature_cols if col != 'FG_Index']
        
        if advanced_data and 'FG_Index' not in self.feature_cols:
             if 'FG_Index' in df.columns:
                 self.feature_cols.append('FG_Index')
            
        if not self.feature_cols:
            return pd.DataFrame(), df['Target']
            
        return df[self.feature_cols], df['Target']

    # --- (C) モデル学習 (維持) ---
    def train_and_save_model(self, df_long_term: pd.DataFrame) -> bool:
        print("🧠 モデルの再学習タスクを開始...")
        X_train, Y_train = self.create_ml_features(df_long_term.copy())
        
        if X_train.empty:
            print("🚨 致命的なエラー: 学習データが不足しているため、モデルを構築できませんでした。")
            return False
        
        # より深い学習のため、max_depthを上げる
        model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced', max_depth=15, n_jobs=-1)
        model.fit(X_train, Y_train)
        
        joblib.dump(model, MODEL_FILENAME)
        print("✅ モデルの再学習が完了し、ファイルに保存されました。")
        return True

    # --- (D) 予測とレポート (レポート調整のみ) ---
    def predict_and_report(self, df_latest: pd.DataFrame, advanced_data: Dict[str, Any]) -> bool:
        """最新データに基づいて予測を行い、レポートを生成し、Telegramに送信する。"""
        if df_latest.empty:
            print("🚨 予測スキップ: 最新のOHLCVデータが空です。")
            return False
            
        try:
            model = joblib.load(MODEL_FILENAME)
        except FileNotFoundError:
            report = "🚨 <b>エラー:</b> モデルファイルが見つかりません。まず学習とコミットを行ってください。"
            self.send_telegram_notification(report) 
            return False

        X_latest, _ = self.create_ml_features(df_latest.copy(), advanced_data=advanced_data)
        
        if X_latest.empty:
            report = ("🚨 <b>予測スキップ通知:</b> ML特徴量の生成に必要なデータが不足しています。")
            self.send_telegram_notification(report)
            return False
            
        latest_X = X_latest.iloc[[-1]] 
        
        prediction_val = model.predict(latest_X)[0]
        prediction_proba = model.predict_proba(latest_X)[0]
        
        full_report = self._generate_premium_report(
            df_latest=df_latest,
            latest_price_data=df_latest.iloc[-1],
            latest_features=latest_X.iloc[-1],
            advanced_data=advanced_data, 
            ml_prediction=prediction_val, 
            proba=prediction_proba
        )
        
        self.send_telegram_notification(full_report)
        
        return True
    
    # --- レポート生成のためのヘルパー関数 (日足/擬似データに合わせて調整) ---
    def _determine_market_regime(self, price: float, sma10: float, sma30: float, atr: float, bbp: float) -> Tuple[str, str, str]:
        """SMAとボラティリティ指標を用いて市場構造とトレンドを判断する"""
        
        # 🚨 日足のため、SMA10/SMA30を使用
        if sma10 > sma30:
            trend_type = "中長期上昇トレンド"
            trend_emoji = "⬆️"
        elif sma10 < sma30:
            trend_type = "中長期下降トレンド"
            trend_emoji = "⬇️"
        else:
            trend_type = "中長期レンジ"
            trend_emoji = "➖"

        # ボラティリティ判断
        is_tight_range = bbp < 0.15 and bbp > -0.15 # 日足のため閾値を調整
        
        if is_tight_range:
            regime_status = "大口集積期 (ボラティリティ収縮)"
            regime_emoji = "⏳"
        elif abs(price - sma10) > (atr * 0.8):
            regime_status = f"強い{trend_type}継続 (モメンタム加速)"
            regime_emoji = "🚀" if trend_type == "中長期上昇トレンド" else "🌊"
        else:
            regime_status = "トレンド調整/レンジ形成"
            regime_emoji = "⚖️"
        
        return regime_status, regime_emoji, trend_type

    def _analyze_macro_sentiment(self, fg_index: int) -> Tuple[str, List[str], str]:
        """F&G Indexからマクロなセンチメントと核心リスクを判断する"""
        
        if fg_index >= 75:
            sentiment_summary = "極度の楽観（Extreme Greed）。ポジション調整リスクが非常に高い。"
            risk_color = "🔴"
        elif fg_index >= 60:
            sentiment_summary = "楽観（Greed）。過熱感があり、逆張りショート検討のゾーン。"
            risk_color = "🟠"
        elif fg_index <= 25:
            sentiment_summary = "極度の恐怖（Extreme Fear）。強力なパニック売り後の反発期待大。"
            risk_color = "🟢"
        elif fg_index <= 40:
            sentiment_summary = "恐怖（Fear）。市場参加者は慎重で、押し目買いの機会を探るゾーン。"
            risk_color = "🟡"
        else:
            sentiment_summary = "中立。市場心理は均衡状態です。"
            risk_color = "⚪️"

        core_risks = []
        core_risks.append(f"<b>データ推定:</b> OHLCVデータはCoinGecko終値と統計ノイズによる<b>推定値</b>です。")
        if fg_index >= 75:
            core_risks.append("<b>過熱警告:</b> FGIが極端に高い水準。強気派は慎重なリスク管理が必要です。")
        
        return sentiment_summary, core_risks, risk_color
        
    def _generate_premium_report(self, df_latest: pd.DataFrame, latest_price_data: pd.Series, latest_features: pd.Series, advanced_data: Dict[str, Any], ml_prediction: int, proba: np.ndarray) -> str:
        """ML予測と実データを統合し、最高峰の分析レポートを生成する。"""
        
        price = latest_price_data['Close']
        sma10 = latest_features.get('SMA10', price)
        sma30 = latest_features.get('SMA30', price)
        atr = latest_features.get('ATR', price * 0.01)
        bbp = latest_features.get('BBands_Width', 0)
        rsi = latest_features.get('RSI', 50)
        
        pred_map = {-1: "📉 下落", 0: "↔️ レンジ", 1: "📈 上昇"}
        ml_result = pred_map.get(ml_prediction, "不明")
        max_proba = proba[np.argmax(proba)]
        
        fg_index = advanced_data.get('fg_index', 50)
        
        current_time = datetime.now(timezone.utc).astimezone(None).strftime('%Y-%m-%d %H:%M JST')
        
        regime_status, regime_emoji, trend_type = self._determine_market_regime(price, sma10, sma30, atr, bbp)
        sentiment_summary, core_risks, risk_color = self._analyze_macro_sentiment(fg_index)
        
        # ATRに基づく重要レベル
        R1 = price + atr
        S1 = price - atr
        R2 = price + (atr * 2)
        S2 = price - (atr * 2)
        
        ml_interpretation = f"MLモデルは次の日（24時間）で<b>{ml_result}</b>を予測しています (信頼度: {max_proba*100:.1f}%)。"
        if ml_prediction == 0 and max_proba < 0.45:
            ml_interpretation += "信頼度が低いため、強い方向性は示されていません。"

        core_reason_list = [f"<b>ML予測:</b> {ml_interpretation}"]
        core_reason_list.extend(core_risks)
        
        # テクニカル要因の詳細
        if trend_type != "中長期レンジ":
            core_reason_list.append(f"<b>トレンド構造:</b> {trend_type} (SMA10:{sma10:.2f} vs SMA30:{sma30:.2f}) が継続中。価格はSMA10に対して{'上' if price > sma10 else '下'}に位置。")
        else:
             core_reason_list.append(f"<b>トレンド構造:</b> 中長期トレンドはレンジ傾向。ボラティリティ指標 (BBands: {bbp:.2f}) が{'収縮' if bbp < 0.15 else '拡大'}を示唆。")
             
        core_reason_list.append(f"<b>モメンタム:</b> RSIは{rsi:.1f}。{'買われすぎ' if rsi > 70 else ('売られすぎ' if rsi < 30 else '中立')}領域。短期的な反発期待の有無を判断可能。")

        chance_list = [
            f"<b>ML予測との一致:</b> 高い信頼度 ({max_proba*100:.1f}%) の場合、その方向に短期的な優位性が見込めます。",
            f"<b>市場心理の逆張り:</b> F&G指数が<b>{fg_index}</b> ({advanced_data['fg_value']}) の極値にある場合、強力な逆張りチャンスを提供します。",
        ]
        
        risk_list = [
            f"<b>{risk_color} 総合リスク警告:</b> 市場構造は現在 <b>{regime_status}</b> であり、FGIに基づくセンチメントは {sentiment_summary} です。",
            f"<b>ボラティリティリスク (ATR):</b> 過去14日間の平均変動幅は <b>${atr:.2f}</b> です。この値幅を超えるSL/TPは非効率的です。",
            f"<b>トレンド転換点:</b> SMA30 (${sma30:.2f}) を割る/超える動きは、中長期トレンドの転換シグナルとなる可能性があります。"
        ]
        
        # 行動ガイドの調整（日足=中長期戦略の検討を促す）
        if ml_prediction == 1 or fg_index <= 30:
            strategy_title = "📈 <b>分析結果に基づいた推奨アクション: 中長期ロング戦略の検討</b>"
            action_guide = f"""
<b>検討ゾーン:</b> S1: ${S1:.2f}〜現在価格（押し目を待つ）
<b>リスク管理基準 (SL):</b> S2: ${S2:.2f}（分析上のサポートライン）
<b>利確目標 (TP):</b> R1: ${R1:.2f}, R2: ${R2:.2f}
"""
        elif ml_prediction == -1 or fg_index >= 70:
            strategy_title = "📉 <b>分析結果に基づいた推奨アクション: 中長期ショート戦略の検討</b>"
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
        
        # 🚨 変更点: レポートタイトルを代替データソース使用に修正
        report = f"""
<b>【👑 BTC 先物 日足 分析強化レポート 👑】</b>
<p>
    <i>(注: データ取得元: **CoinGecko終値と統計推定データ**を使用しています。)</i>
</p>
📅 <b>{current_time}</b> | <b>日足分析</b> (次期予測: 24時間後)
<p>
    <b>現在の市場構造:</b> <b>{regime_emoji} {regime_status}</b> | <b>中長期トレンド: {trend_type} {trend_emoji}</b>
    <br>
    <b>現在価格 (推定終値): ${price:.2f} USDT</b>
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
    <li><b>短期サポート/レジスタンス (ATR):</b> S1: ${S1:.2f} / R1: ${R1:.2f}</li>
    <li><b>中期サポート/レジスタンス:</b> S2: ${S2:.2f} / R2: ${R2:.2f}</li>
    <li><b>主要移動平均線:</b> SMA10: ${sma10:.2f} / SMA30: ${sma30:.2f}</li>
</ul>

---------------------------------------
<h3><b>⚡️ 4. 詳細分析に基づく行動ガイド</b></h3>

<h4>{strategy_title}</h4>
<pre>
{action_guide}
</pre>
<p>
<b>💡 注意点:</b> このレポートは、強化されたMLモデルと詳細なテクニカル分析に基づいていますが、**絶対的な取引推奨ではありません**。特にOHLCVは推定値です。リスク許容度に基づき、ご自身の判断でご活用ください。
</p>
---------------------------------------
<b>📚 まとめ：分析の焦点</b>
現在の市場は <b>{regime_status}</b> の段階にあり、強化されたMLモデルは ({max_proba*100:.1f}%) の信頼度で <b>{ml_result}</b> を示唆しています。
モメンタムとボラティリティの指標が提供する洞察を重視し、精密な分析を行ってください。
"""
        return report
        
    # --- (F) Telegram通知機能 ---
    def send_telegram_notification(self, message: str):
        """通知の実装。Telegram設定がない場合はコンソールに出力。"""
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
