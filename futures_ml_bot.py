# futures_ml_bot.py (MEXC分析強化版 / 即時通知対応 / 特徴量大幅追加)

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

FUTURES_SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h' 
MODEL_FILENAME = 'btc_futures_ml_model.joblib'

# 外部APIエンドポイント (Fear & Greed Index)
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
        # MEXC公開APIインスタンス
        self.exchange = ccxt.mexc({
            'options': {'defaultType': 'future'},
            'enableRateLimit': True,
        })
        
        self.target_threshold = 0.0005 
        self.prediction_period = 1 
        self.feature_cols: List[str] = [] 

    # --- (A) データ取得 (OHLCV) ---
    def fetch_ohlcv_data(self, limit: int = 2000, timeframe: str = TIMEFRAME) -> pd.DataFrame:
        """OHLCVデータをMEXC公開APIから取得します。"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(FUTURES_SYMBOL, timeframe, limit=limit)
            if not ohlcv:
                print("🚨 OHLCVデータが空です。")
                return pd.DataFrame()
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            print(f"✅ MEXCから{len(df)}件のOHLCVデータを取得しました。")
            return df
        except Exception as e:
            print(f"🚨 OHLCVデータ取得エラー (MEXC公開APIを使用中): {e}")
            return pd.DataFrame()

    # --- (B) 特徴量作成 (大幅に強化) ---
    def create_ml_features(self, df: pd.DataFrame, advanced_data: Dict[str, Any] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """ボラティリティ、モメンタム、トレンド、センチメントを含む高度な特徴量を作成"""
        if df.empty:
            return pd.DataFrame(), pd.Series(dtype=int)

        # --- トレンド指標 ---
        df['SMA20'] = ta.sma(df['Close'], length=20)
        df['SMA50'] = ta.sma(df['Close'], length=50)
        df['Trend_Signal'] = np.where(df['SMA20'] > df['SMA50'], 1, -1) # 短期 > 長期 = 1 (上昇トレンド)
        
        # --- モメンタム指標 ---
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['MACD_H'] = ta.macd(df['Close'])['MACDh_12_26_9']
        df['StochRSI_K'] = ta.stochrsi(df['Close'])['STOCHRSId_14_14_3_3'] # Stochastic RSI
        df['Momentum'] = ta.mom(df['Close'], length=10) # 10期間モメンタム
        
        # --- ボラティリティ指標 ---
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14) 
        df['BBands_Width'] = ta.bbands(df['Close'])['BBP_20_2.0'] # Bollinger Band Percent B (Percent B)
        df['Keltner_Width'] = (ta.kc(df['High'], df['Low'], df['Close'])['KCBu_20_2.0'] - ta.kc(df['High'], df['Low'], df['Close'])['KCLl_20_2.0']) / df['Close'] # Keltner Channel Width Normalized
        
        # --- ボリューム指標 ---
        df['Volume_SMA'] = ta.sma(df['Volume'], length=20)
        df['Volume_ROC'] = df['Volume'].pct_change(1) # Volume Rate of Change
        
        # --- 価格変化率 ---
        for lag in [1, 2, 3, 5]: # ラグの数を増やす
            df[f'Price_L{lag}'] = df['Close'].pct_change(lag).shift(lag)
            
        # --- センチメント指標 (予測時のみ使用) ---
        if advanced_data:
            # F&G Indexを特徴量として追加
            df['FG_Index'] = advanced_data.get('fg_index', 50)
        else:
            # 学習データ生成時は、最新のF&G Indexは未来情報となるため、50で埋めるか、より堅牢な方法を使う
            # 今回は学習データとして使うには危険なため、予測時のみ使うようにリストから除外
            pass

        # 予測対象（Target）: 次の1時間で設定した閾値以上動くか (+1: 上昇, -1: 下落, 0: レンジ)
        future_change = df['Close'].pct_change(periods=-self.prediction_period).shift(self.prediction_period)
        
        df['Target'] = np.select(
            [future_change > self.target_threshold, future_change < -self.target_threshold],
            [1, -1], default=0
        )
        
        df.dropna(inplace=True)
        
        # 特徴量カラムリストの更新
        if not self.feature_cols and not df.empty:
            cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'timestamp', 'SMA20', 'SMA50']]
            self.feature_cols = [col for col in cols if df[col].dtype in [np.float64, np.int64]]
        
        # 学習時にはFG_Indexは含めない (未来情報混入防止)
        self.feature_cols = [col for col in self.feature_cols if col != 'FG_Index']
        
        # 予測時、FG_Indexが追加された場合は特徴量リストに追加する
        if advanced_data and 'FG_Index' not in self.feature_cols:
             if 'FG_Index' in df.columns:
                 self.feature_cols.append('FG_Index')
            
        if not self.feature_cols:
            return pd.DataFrame(), df['Target']
            
        return df[self.feature_cols], df['Target']

    # --- (C) モデル学習 ---
    def train_and_save_model(self, df_long_term: pd.DataFrame) -> bool:
        print("🧠 モデルの再学習タスクを開始...")
        # 学習時にはadvanced_data (FG_Index) を渡さない
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

    # --- (D) 予測とレポート ---
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

        # 予測時には advanced_data (FG_Index) を渡す
        X_latest, _ = self.create_ml_features(df_latest.copy(), advanced_data=advanced_data)
        
        if X_latest.empty:
            report = ("🚨 <b>予測スキップ通知:</b> ML特徴量の生成に必要なデータが不足しています。")
            self.send_telegram_notification(report)
            return False
            
        latest_X = X_latest.iloc[[-1]] 
        
        # 予測の実行
        prediction_val = model.predict(latest_X)[0]
        prediction_proba = model.predict_proba(latest_X)[0]
        
        # プレミアムレポートを生成
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
    
    # --- レポート生成のためのヘルパー関数 (洞察力強化) ---
    def _determine_market_regime(self, price: float, sma20: float, sma50: float, atr: float, bbp: float) -> Tuple[str, str, str]:
        """SMAとボラティリティ指標を用いて市場構造とトレンドを判断する"""
        
        # トレンド判断
        if sma20 > sma50:
            trend_type = "中期上昇トレンド"
            trend_emoji = "⬆️"
        elif sma20 < sma50:
            trend_type = "中期下降トレンド"
            trend_emoji = "⬇️"
        else:
            trend_type = "中期レンジ"
            trend_emoji = "➖"

        # ボラティリティ判断
        is_high_vol = atr > (atr * 1.5) # 過去平均ATRとの比較など、より詳細なロジックを組むことも可能だが、今回はシンプルに
        is_tight_range = bbp < 0.2 and bbp > -0.2 # ボリンジャーバンドの収縮を示す
        
        if is_tight_range:
            regime_status = "ブレイクアウト前夜 (ボラティリティ収縮)"
            regime_emoji = "⏳"
        elif abs(price - sma20) > (atr * 1.0):
            regime_status = f"強い{trend_type}継続 (モメンタム加速)"
            regime_emoji = "🚀" if trend_type == "中期上昇トレンド" else "🌊"
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
        core_risks.append(f"<b>資金調達率 (FR):</b> データ非取得のため中立 (0.00%)。")
        if fg_index >= 75:
            core_risks.append("<b>過熱警告:</b> FGIが極端に高い水準。強気派は慎重なリスク管理が必要です。")
        
        return sentiment_summary, core_risks, risk_color
        
    def _generate_premium_report(self, df_latest: pd.DataFrame, latest_price_data: pd.Series, latest_features: pd.Series, advanced_data: Dict[str, Any], ml_prediction: int, proba: np.ndarray) -> str:
        """ML予測と実データを統合し、最高峰の分析レポートを生成する。"""
        
        price = latest_price_data['Close']
        high = latest_price_data['High']
        low = latest_price_data['Low']
        
        sma20 = latest_features.get('SMA20', price)
        sma50 = latest_features.get('SMA50', price)
        atr = latest_features.get('ATR', price * 0.01)
        bbp = latest_features.get('BBands_Width', 0) # Percent Bを代用
        rsi = latest_features.get('RSI', 50)
        
        pred_map = {-1: "📉 下落", 0: "↔️ レンジ", 1: "📈 上昇"}
        ml_result = pred_map.get(ml_prediction, "不明")
        max_proba = proba[np.argmax(proba)]
        
        fg_index = advanced_data.get('fg_index', 50)
        
        current_time = datetime.now(timezone.utc).astimezone(None).strftime('%Y-%m-%d %H:%M JST')
        
        regime_status, regime_emoji, trend_type = self._determine_market_regime(price, sma20, sma50, atr, bbp)
        sentiment_summary, core_risks, risk_color = self._analyze_macro_sentiment(fg_index)
        
        # ATRに基づく重要レベル
        R1 = price + atr
        S1 = price - atr
        R2 = price + (atr * 2)
        S2 = price - (atr * 2)
        
        ml_interpretation = f"MLモデルは次の1時間で<b>{ml_result}</b>を予測しています (信頼度: {max_proba*100:.1f}%)。"
        if ml_prediction == 0 and max_proba < 0.45:
            ml_interpretation += "信頼度が低いため、強い方向性は示されていません。"

        core_reason_list = [f"<b>ML予測:</b> {ml_interpretation}"]
        core_reason_list.extend(core_risks)
        
        # テクニカル要因の詳細
        if trend_type != "中期レンジ":
            core_reason_list.append(f"<b>トレンド構造:</b> {trend_type} ({sma20:.2f} vs {sma50:.2f}) が継続中。価格はSMA20に対して{'上' if price > sma20 else '下'}に位置。")
        else:
             core_reason_list.append(f"<b>トレンド構造:</b> 中期トレンドはレンジ傾向。ボラティリティ指標 (BBands: {bbp:.2f}) が{'収縮' if bbp < 0.3 else '拡大'}を示唆。")
             
        core_reason_list.append(f"<b>モメンタム:</b> RSIは{rsi:.1f}。{'買われすぎ' if rsi > 70 else ('売られすぎ' if rsi < 30 else '中立')}領域。短期的な反発期待の有無を判断可能。")

        chance_list = [
            f"<b>ML予測との一致:</b> 高い信頼度 ({max_proba*100:.1f}%) の場合、その方向に短期的な優位性が見込めます。",
            f"<b>市場心理の逆張り:</b> F&G指数が<b>{fg_index}</b> ({advanced_data['fg_value']}) の極値にある場合、強力な逆張りチャンスを提供します。",
        ]
        
        risk_list = [
            f"<b>{risk_color} 総合リスク警告:</b> 市場構造は現在 <b>{regime_status}</b> であり、FGIに基づくセンチメントは {sentiment_summary} です。",
            f"<b>ボラティリティリスク (ATR):</b> 過去14時間の平均変動幅は <b>${atr:.2f}</b> です。この値幅を超えるSL/TPは非効率的です。",
            f"<b>トレンド転換点:</b> SMA50 (${sma50:.2f}) を割る/超える動きは、中期トレンドの転換シグナルとなる可能性があります。"
        ]
        
        # 行動ガイドの調整（分析専門のため、より一般的な「検討」を促す）
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
<b>【👑 BTC MEXC 1時間足 分析強化レポート 👑】</b>
📅 <b>{current_time}</b> | <b>{TIMEFRAME}足分析</b> (次期予測: 1時間後)
<p>
    <b>現在の市場構造:</b> <b>{regime_emoji} {regime_status}</b> | <b>中期トレンド: {trend_type} {trend_emoji}</b>
    <br>
    <b>現在価格: ${price:.2f} USDT</b>
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
    <li><b>主要移動平均線:</b> SMA20: ${sma20:.2f} / SMA50: ${sma50:.2f}</li>
</ul>

---------------------------------------
<h3><b>⚡️ 4. 詳細分析に基づく行動ガイド</b></h3>

<h4>{strategy_title}</h4>
<pre>
{action_guide}
</pre>
<p>
<b>💡 注意点:</b> このレポートは、強化されたMLモデルと詳細なテクニカル分析に基づいていますが、**絶対的な取引推奨ではありません**。リスク許容度に基づき、ご自身の判断でご活用ください。
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
