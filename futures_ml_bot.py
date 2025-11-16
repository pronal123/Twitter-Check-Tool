# futures_ml_bot.py (無料データに基づく過熱シグナル分析版 - 日本語版)

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
from typing import Tuple, Dict, Any

# --- 1. 環境変数設定 ---
MEXC_API_KEY = os.environ.get('MEXC_API_KEY')
MEXC_SECRET = os.environ.get('MEXC_SECRET')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

FUTURES_SYMBOL = 'BTC/USDT'
TIMEFRAME = '4h'
MODEL_FILENAME = 'btc_futures_ml_model.joblib'

# 外部APIエンドポイント (現在動作確認済みで無料のAPIのみを使用)
FG_INDEX_API_URL = 'https://api.alternative.me/fng/?limit=1'

# Coinglassへの直接アクセスはエラーとなるため、このバージョンでは削除します。

# --- 2. Advanced Custom Data Fetching Function ---
def fetch_advanced_metrics(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
    """
    FR, Fear & Greed Indexなど、確実に取得できる公開実践データのみを取得します。
    高度な集計データは有料APIが必要なため、フォールバック値を設定します。
    """
    metrics = {}
    
    # 取得失敗時、および有料APIが必要な場合のフォールバック値
    default_fallbacks = {
        'funding_rate': 0.0, 
        'fg_index': 50, 
        'fg_value': 'Neutral (API失敗)',
        # 🚨 Coinglassの集計データは無料で取得できないため、ステータスを報告します。
        'oi_current_usd': '取得不可 (有料API)',          
        'oi_change_4h': '取得不可 (有料API)',            
        'ls_ratio': "取得不可 (有料API)",
        'liq_24h_long': "取得不可 (有料API)",
        'aggregated_oi_trend': '取得不可 (有料API)',
        'liquidation_cluster': '取得不可 (有料API)'
    }
    metrics.update(default_fallbacks)

    try:
        # 1. ファンディングレート (FR) の取得 (実践データ: CCXT/MEXC)
        ticker = exchange.fetch_ticker(symbol)
        metrics['funding_rate'] = float(ticker.get('fundingRate', 0) or 0)
        
        # 2. Fear & Greed Index の取得 (実践データ: 外部API)
        try:
            fg_response = requests.get(FG_INDEX_API_URL, timeout=5)
            fg_response.raise_for_status()
            fg_data = fg_response.json().get('data', [{}])
            metrics['fg_index'] = int(fg_data[0].get('value', 50))
            metrics['fg_value'] = fg_data[0].get('value_classification', 'Neutral')
        except Exception as e:
            print(f"⚠️ F&G Index APIエラー: {e}")
            
        print("ℹ️ 実践データ取得プロセスを完了しました。高度な集計データはスキップされています。")
        return metrics
    
    except Exception as e:
        print(f"🚨 主要データ取得エラー (CCXT/その他): {e}")
        return default_fallbacks


# --- 3. メインBOTクラス ---
class FuturesMLBot:
    def __init__(self):
        if not all([MEXC_API_KEY, MEXC_SECRET]):
             raise ValueError("APIキーが設定されていません。環境変数を確認してください。")
             
        # CCXT MEXC 先物クライアントの初期化
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
        """OHLCVデータを取得 (実践データ)"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(FUTURES_SYMBOL, timeframe, limit=limit)
            if not ohlcv:
                print("🚨 OHLCVデータが空です。")
                return pd.DataFrame()
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"🚨 OHLCVデータ取得エラー: {e}")
            return pd.DataFrame()

    # --- (B) 特徴量エンジニアリング (ATRを含む) ---
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

    # --- (C) モデルの学習と保存 (継続的学習) ---
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

    # --- (D) リアルタイム予測と通知 ---
    def predict_and_report(self, df_latest: pd.DataFrame, advanced_data: Dict[str, Any]) -> bool:
        try:
            model = joblib.load(MODEL_FILENAME)
        except FileNotFoundError:
            report = "🚨 <b>エラー:</b> モデルファイルが見つかりません。まず学習とコミットを行ってください。"
            self.send_telegram_notification(report) 
            return False

        X_latest, _ = self.create_ml_features(df_latest.copy())
        
        if X_latest.empty:
            report = (
                "🚨 <b>予測スキップ通知:</b> OHLCVデータが不足しています。"
            )
            self.send_telegram_notification(report)
            return False
            
        latest_X = X_latest.iloc[[-1]] 
        
        prediction_val = model.predict(latest_X)[0]
        prediction_proba = model.predict_proba(latest_X)[0]
        
        report_structure, report_conclusion = self._generate_two_part_reports(
            latest_price_data=df_latest.iloc[-1],
            latest_features=latest_X.iloc[-1],
            advanced_data=advanced_data, 
            ml_prediction=prediction_val, 
            proba=prediction_proba
        )
        
        self.send_telegram_notification(report_structure)
        self.send_telegram_notification(report_conclusion)
        
        return True

    # --- (E) レポート生成ヘルパー関数 - 日本語版 (実践数値に基づく) ---
    def _generate_two_part_reports(self, latest_price_data: pd.Series, latest_features: pd.Series, advanced_data: Dict[str, Any], ml_prediction: int, proba: np.ndarray) -> Tuple[str, str]:
        """
        レポートを「市場構造と主要ドライバー分析」と「最終結論と行動計画」の2部構成で生成（日本語版、HTML形式）
        -> <p>タグを削除し、\n\n に置換
        """
        # 価格データとテクニカル指標 (実践データ)
        price = latest_price_data['Close']
        sma = latest_features.get('SMA', price)
        atr = latest_features.get('ATR', price * 0.01)
        rsi = latest_features.get('RSI', 50)
        macd_h = latest_features.get('MACD_H', 0)

        # 予測結果マップ
        pred_map = {-1: "📉 下落", 0: "↔️ レンジ", 1: "📈 上昇"}
        ml_result = pred_map.get(ml_prediction, "不明")
        
        # --- 実践データと集計データ ---
        fr = advanced_data.get('funding_rate', 0)
        fg_index = advanced_data.get('fg_index', 50)
        fg_value = advanced_data.get('fg_value', 'Neutral')
        
        # 🚨 有料APIが必要な集計データ
        ls_ratio = advanced_data.get('ls_ratio', '取得不可 (有料API)')
        oi_trend = advanced_data.get('aggregated_oi_trend', '取得不可 (有料API)')
        liq_long = advanced_data.get('liq_24h_long', '取得不可 (有料API)')
        liq_cluster_info = advanced_data.get('liquidation_cluster', '取得不可 (有料API)')

        # ----------------------------------------------------------------
        # 🆕 無料データに基づく「市場過熱シグナル」のロジック (OI/LSRの代わり)
        # ----------------------------------------------------------------
        overheat_score = 0
        risk_level = "中🔴"
        market_signal = "中立/テクニカル主導"

        # 1. 貪欲シグナル (ロング過熱の可能性)
        if fg_index >= 75:
            overheat_score += 1
        # 2. FRシグナル (ロングのコスト高・調整リスク)
        if fr >= 0.00015: 
            overheat_score += 1
            
        # テクニカル分析の簡潔な洞察
        rsi_comment = "中立（上昇余地あり）"
        if rsi >= 70:
            rsi_comment = "買われすぎ水準 (短期調整注意)"
        elif rsi <= 30:
            rsi_comment = "売られすぎ水準 (反発期待)"
            
        macd_comment = "勢いなし"
        # MACDがマイナスからプラスに転換した場合 (ゴールデンクロスに近い状態)
        if macd_h > 0 and latest_features.get('MACD_H_L1', 0) < 0:
            macd_comment = "📈 ゴールデンクロス発生の兆候（強い強気シグナル）"
        elif macd_h > 0:
            macd_comment = "強気モメンタム継続"
        elif macd_h < 0:
            macd_comment = "弱気モメンタム継続"


        if overheat_score >= 2:
            market_signal = "🚨 極端なロング過熱リスク（調整警戒）"
            risk_level = "高🔴🔴"
            main_cause = "過度な楽観と調整リスク"
        elif overheat_score == 1:
             market_signal = "⚠️ 過熱の兆候（FRまたはFGI）"
             risk_level = "中高🔴"
             main_cause = "センチメントの傾き"
        elif fg_index <= 25:
             market_signal = "🟢 極度の恐怖（底打ちの可能性）"
             risk_level = "中高🔴"
             main_cause = "極端な恐怖によるポジション調整"
        else:
             main_cause = "テクニカル環境と市場センチメント"
        # ----------------------------------------------------------------

        current_time = datetime.now(timezone.utc).astimezone(None).strftime('%Y-%m-%d %H:%M JST')
        
        max_proba = proba[np.argmax(proba)]
        uncertainty_score = 1.0 - max_proba
        
        
        # --- レポートA: BTC市場 現状と短期見通し 分析 (HTML形式) ---
        report_structure = f"""
<b>【BTC市場 現状と短期見通し 分析レポート】</b>
📅 {current_time} | 4h足分析

<b>🔍 1. 市場構造の現状把握</b>

📌 <b>現在のドライバー:</b> <b>{main_cause}</b> が市場の方向性を主導しています。
📌 <b>現在価格:</b> <b>${price:.2f}</b>

-------------------------------------
<b>📊 テクニカル分析（短期トレンド）</b>
<pre>
指標           | 現在値/ステータス     | 洞察
--------------------------------------------------------------------------------
20-SMA         | ${sma:.2f}            | 価格はSMAを{'🟢 上回る' if price > sma else '🔴 下回る'}。短期トレンドは{'強気' if price > sma else '弱気'}。
RSI (14)       | {rsi:.2f}              | <b>{rsi_comment}</b>。
MACD Hist.     | {macd_h:.2f}           | <b>{macd_comment}</b>。
ATR (ボラティリティ) | ${atr:.2f}            | 過去14期間の平均変動幅。現在価格の<b>{atr/price*100:.2f}%</b>。
</pre>
-------------------------------------
<b>📈 2. センチメントと過熱シグナル（OI/LSR代替分析）</b>
<pre>
カテゴリ        | 指標         | 現在値/ステータス     | 分析/示唆
--------------------------------------------------------------------------------
需給・流動性    | FR           | {fr*100:.4f}%             | {'🚨 ロングのコストが高騰。調整圧力に注意。' if fr > 0.00015 else '中立水準。'}
                | F&G指数      | {fg_index} ({fg_value}) | {'極度の恐怖。底打ちの可能性。' if fg_index <= 25 else '楽観的。過熱度を警戒。'}
                | <b>過熱シグナル</b> | <b>{market_signal}</b>  | 無料データに基づく市場の傾き。
</pre>

<b>⚠️ リスクサマリー:</b> <b>{risk_level}</b>。無料データ（FR/FGI）分析に基づく総合評価。
"""
        
        # --- 予測結果の調整 ---
        final_conclusion = ml_result
        
        # 予測が上昇だが、過熱シグナルが強い場合、調整リスクを警告
        if ml_prediction == 1 and overheat_score >= 1:
             final_conclusion = f"⚠️ {ml_result} (注意: 過熱シグナルに基づく短期調整リスク)"
        
        # 推奨戦略の決定
        # 予測が不明確 (レンジ or 不確実性が高い) の場合は待機を推奨
        if uncertainty_score > 0.40 or ml_prediction == 0:
            overall_advice = "🚨 <b>リスク回避/待機推奨</b>。市場がレンジまたは方向性が不明確です。重要なブレイクアウトを待機してください。"
            entry_long = f"ATRサポート付近 (${price - atr:.2f})"
            entry_short = f"ATRレジスタンス付近 (${price + atr:.2f})"
        else:
             overall_advice = f"✅ <b>ML予測に合わせた取引を検討</b>: <b>{final_conclusion}</b>。リスク管理を徹底してください。"
             # 予測が上昇/下落の場合、現在価格の0.2%程度の押し/戻しをエントリー目標とする（例）
             entry_long = f"現在のトレンドに沿ったエントリー ({price * 0.998:.2f}付近)"
             entry_short = f"現在のトレンドに沿ったエントリー ({price * 1.002:.2f}付近)"
        
        # HTMLテーブルを、Telegramで確実に表示できる構造化テキストに置き換える
        strategy_block = f"""
<b>🟢 強気シナリオ (ML予測が強気の場合の指針)</b>
  - <b>エントリー目標:</b> {entry_long if ml_prediction >= 0 else '---'}
  - <b>利益確定目標:</b> 直近の高値/主要レジスタンスゾーン

<b>🔴 弱気シナリオ (リスク管理の指針)</b>
  - <b>エントリー目標:</b> {entry_short if ml_prediction <= 0 else '---'}
  - <b>損切りライン:</b> ATRに基づく ${atr:.2f} (厳守)
"""
        
        # --- レポートB: BOTの最終見解と具体的な行動計画 (HTML形式) ---
        report_conclusion = f"""
<b>【BOTの最終見解と行動計画】</b>
📅 {current_time}

<b>🎯 3. BOTの最終見解（これからのBTC見通し）</b>
<b>ML予測結論:</b> <b>{final_conclusion}</b> (信頼度: {max_proba*100:.1f}%)

{overall_advice}

{strategy_block}

-------------------------------------
<b>📚 まとめと今後の監視ポイント</b>
市場は<b>「{main_cause}」</b>を背景に、<b>{market_signal}</b>の状況にあります。
MLモデルは短期的には<b>{final_conclusion}</b>を示唆していますが、不確実性スコア<b>{uncertainty_score*100:.1f}%</b>を考慮し、ポジションサイズを調整することを推奨します。

<b><span style="color:#007bff;">✅ 次の4時間監視ポイント:</span></b>
1. RSIが{'70を突破/下回る' if rsi < 70 else '70以下に冷却する'}か。
2. FRが{'0.00015%を下回る' if fr > 0.00015 else 'さらに上昇する'}か。
3. 価格が20-SMA (${sma:.2f}) を{'維持できる' if price > sma else '回復できる'}か。
"""
        return report_structure, report_conclusion
        
    # --- (F) Telegram通知機能 - 強化されたエラー処理 ---
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
