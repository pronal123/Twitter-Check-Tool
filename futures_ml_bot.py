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
        """
        # 価格データとテクニカル指標 (実践データ)
        price = latest_price_data['Close']
        sma = latest_features.get('SMA', price)
        atr = latest_features.get('ATR', price * 0.01)

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
        overheat_detail = "中立"
        risk_level = "中🔴"
        market_signal = "中立/テクニカル主導"

        # 1. 貪欲シグナル (ロング過熱の可能性)
        if fg_index >= 75:
            overheat_score += 1
            overheat_detail = "極端な貪欲"
        # 2. FRシグナル (ロングのコスト高・調整リスク)
        if fr >= 0.00015: # 0.015% 以上で高水準と見なす
            overheat_score += 1
            if overheat_detail == "中立":
                overheat_detail = "高いFR"
            else:
                overheat_detail += " & 高いFR"

        if overheat_score >= 2:
            market_signal = "🚨 極端なロング過熱リスク（調整警戒）"
            risk_level = "高🔴🔴"
            main_cause = "無料データ分析に基づく過度な楽観と調整リスク"
        elif overheat_score == 1:
             market_signal = "⚠️ 過熱の兆候（FRまたはFGI）"
             risk_level = "中高🔴"
             main_cause = "無料データ分析に基づくセンチメントの傾き"
        elif fg_index <= 25:
             market_signal = "🟢 極度の恐怖（ショート解消または底打ちの可能性）"
             risk_level = "中高🔴"
             main_cause = "無料データ分析に基づく極端な恐怖によるポジション調整"
        else:
             main_cause = "テクニカル環境と市場センチメント"
        # ----------------------------------------------------------------

        current_time = datetime.now(timezone.utc).astimezone(None).strftime('%Y-%m-%d %H:%M JST')
        
        max_proba = proba[np.argmax(proba)]
        uncertainty_score = 1.0 - max_proba
        
        
        # --- レポートA: 市場構造と主要ドライバー分析 (HTML形式) ---
        report_structure = f"""
<b>【BTC 市場ドライバー分析 - 無料データに基づく】</b>
📅 {current_time}

📌 <b>主要ポイント</b>
<b>主要ドライバー:</b> 現在の市場トレンドの主要なドライバーは <b>{main_cause}</b> です。
<b>センチメント:</b> Fear & Greed Indexは <b>{fg_index}</b>（「{fg_value}」レベル）であり、市場のボラティリティを示唆しています。
<b>テクニカル環境:</b> BTC価格 <b>${price:.2f}</b> は、20日SMA（${sma:.2f}）を {'🟢 上回っています' if price > sma else '🔴 下回っています'}。短期トレンドは {'強気' if price > sma else '弱気'} です。

-------------------------------------
<b>📉 無料データに基づくセンチメント分析 (OI/LSRの代わり)</b>
<pre>
カテゴリ        | 指標         | 現在値/ステータス     | 分析/示唆
--------------------------------------------------------------------------------
需給・流動性    | FR           | {fr*100:.4f}%             | {'🚨 ロングのコスト高。スクイーズリスクあり。' if fr > 0.00015 else '中立。'}
                | F&G指数      | {fg_index} ({fg_value}) | {'極度の恐怖。逆張り機会か、底値割れの警告。' if fg_index <= 20 else '楽観的。短期的な過熱の可能性。'}
                | <b>過熱シグナル</b> | <b>{market_signal}</b>  | FRとF&G指数を組み合わせた無料分析。
--------------------------------------------------------------------------------
<b>⚠️ 有料APIが必要な集計データ</b>
                | L/S比率(集計)| {ls_ratio}           | 取得不可。
                | OI総量(集計)| 取得不可             | {oi_trend}
</pre>
-------------------------------------

<b>🎯 機会とリスク</b>
- <b>🚨 リスクレベル:</b> <b>{risk_level}</b>。無料データ（FR/FGI）分析に基づく評価。
"""
        
        # --- 予測結果の調整 ---
        final_conclusion = ml_result
        
        # 予測が上昇だが、過熱シグナルが強い場合、調整リスクを警告
        if ml_prediction == 1 and overheat_score >= 1:
             final_conclusion = f"⚠️ {ml_result} (注意: 過熱シグナルに基づく短期調整リスク)"
        
        # 推奨戦略の決定
        if uncertainty_score > 0.40 or ml_prediction == 0:
            strategy_advice_short = "トレードを待ち/避けることを強く推奨。レンジブレイクを待機。"
            entry_long = f"ATRサポート付近 (${price - atr:.2f})"
            entry_short = f"ATRレジスタンス付近 (${price + atr:.2f})"
        else:
             strategy_advice_short = f"ML予測に合わせた取引を検討してください: <b>{final_conclusion}</b>。"
             entry_long = f"ATRサポート付近 (${price - atr:.2f})"
             entry_short = f"ATRレジスタンス付近 (${price + atr:.2f})"
        
        # --- レポートB: 最終結論と行動計画 (HTML形式) ---
        report_conclusion = f"""
<b>【最終結論と行動計画】</b>
📅 {current_time}

-------------------------------------
<b>🤖 予測と全体戦略</b>
<pre>
項目         | 分析結果                         | 確率           | 不確実性スコア
------------------------------------------------------------------------------------
ML予測結論   | <b>{final_conclusion}</b>             | {max_proba*100:.1f}%          | {uncertainty_score*100:.1f}%
</pre>

<b>全体判断:</b> <b>{strategy_advice_short}</b>。無料で取得できるデータに基づき、過熱シグナルも考慮しています。

-------------------------------------
<b>🎯 短期戦略（先物/デイトレード）</b>
<pre>
方向性           | エントリー目標                  | 損切り(SL)           | 利益確定目標
------------------------------------------------------------------------------------
{'弱気' if ml_prediction <= 0 else '強気'} | {entry_short if ml_prediction <= 0 else entry_long} | ATRに基づく ${atr:.2f} (リスク許容度) | 直近の高値/安値ゾーン
</pre>

-------------------------------------
<b>📚 まとめ</b>
CoinglassのOI/LSRの代わりに、無料公開データ（FRとF&G指数）から「市場の過熱感」を独自に分析するロジックを導入しました。これにより、外部APIエラーを回避しつつ、無料でより深いセンチメント分析を提供できます。
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
