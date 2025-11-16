# futures_ml_bot.py (Coinglass無料アクセス試行版 - 日本語版)

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
# APIキーはMEXC/Telegram用のみ必要です。Coinglassのキーは不要です。
MEXC_API_KEY = os.environ.get('MEXC_API_KEY')
MEXC_SECRET = os.environ.get('MEXC_SECRET')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

FUTURES_SYMBOL = 'BTC/USDT'
TIMEFRAME = '4h'
MODEL_FILENAME = 'btc_futures_ml_model.joblib'

# 外部APIエンドポイント (公開されている可能性のある非認証URLを試行)
FG_INDEX_API_URL = 'https://api.alternative.me/fng/?limit=1'

# 🚨 Coinglass公開エンドポイント (無料利用を試みるURL - 変更リスクあり)
# 市場全体のOIデータ取得を試行
COINGLASS_OI_API_URL = 'https://coinglass.com/api/futures/openInterest?symbol=BTC&interval=4h&exchange=binance,bybit,okx' 
# 市場全体のLSRデータ取得を試行
COINGLASS_LSR_API_URL = 'https://coinglass.com/api/longShort?symbol=BTC&interval=4h&exchange=binance,bybit,okx' 

# --- 2. Advanced Custom Data Fetching Function ---
def fetch_advanced_metrics(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
    """
    FR, Fear & Greed Index, および市場全体の建玉 (OI) などの公開された実践データを取得します。
    LSR/OIはCoinglassの公開エンドポイントから無料で取得を試みます。
    """
    metrics = {}
    
    # 取得失敗時のフォールバック値
    default_fallbacks = {
        'funding_rate': 0.0, 
        'fg_index': 50, 
        'fg_value': 'Neutral (API失敗)',
        'oi_current_usd': 0.0,          # 市場全体の建玉 (USD)
        'oi_change_4h': 0.0,            # 市場全体の建玉変化率 (4h)
        # 複雑な認証が必要なデータには「無料アクセス試行失敗」を報告
        'ls_ratio': "無料アクセス試行失敗",
        'liq_24h_long': "清算額取得未実装",
        'aggregated_oi_trend': 'データ取得試行失敗',
        'liquidation_cluster': 'データ取得試行失敗'
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
            
        # 3. Aggregated Market Open Interest (OI) の取得 (Coinglass公開アクセス試行)
        try:
            oi_response = requests.get(COINGLASS_OI_API_URL, timeout=10)
            oi_response.raise_for_status()
            oi_data = oi_response.json()
            
            # --- API応答処理ロジックの例 ---
            if oi_data and 'data' in oi_data and len(oi_data['data']) >= 2:
                latest_oi = oi_data['data'][-1].get('openInterest', 0.0)
                previous_oi = oi_data['data'][-2].get('openInterest', 0.0)
                
                metrics['oi_current_usd'] = latest_oi
                if previous_oi > 0:
                    metrics['oi_change_4h'] = (latest_oi - previous_oi) / previous_oi
                
                metrics['aggregated_oi_trend'] = 'OI Increased' if metrics['oi_change_4h'] > 0.005 else 'OI Decreased'
            
        except Exception as e:
            print(f"🚨 Coinglass OI APIエラー (無料アクセス失敗): {e} (フォールバック値を使用)")
            
        # 4. Long/Short Ratio (LSR) の取得 (Coinglass公開アクセス試行)
        try:
            lsr_response = requests.get(COINGLASS_LSR_API_URL, timeout=10)
            lsr_response.raise_for_status()
            lsr_data = lsr_response.json()
            
            # 🚨 LSRデータ解析ロジックの例 (応答構造に合わせてユーザーが調整が必要)
            if lsr_data and 'data' in lsr_data and len(lsr_data['data']) > 0:
                # 最新のLSR値を取得すると仮定
                latest_lsr = lsr_data['data'][-1].get('longShortRatio', 'データ解析失敗')
                # 値が数値であることを確認し、文字列で表示するために丸めます
                if isinstance(latest_lsr, (int, float)):
                    metrics['ls_ratio'] = f"{latest_lsr:.2f}" 
                else:
                    metrics['ls_ratio'] = latest_lsr
            
        except Exception as e:
            print(f"🚨 Coinglass LSR APIエラー (無料アクセス失敗): {e}")
        
        # 5. 🚨 Liquidation Cluster (清算ヒートマップ) - これは通常、無料では提供されていません
        
        print("ℹ️ 実践データ取得プロセスを完了しました。LSR/OIは無料公開エンドポイントを試行しました。")
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
        
        # 🚨 Coinglass 公開アクセス試行データ
        oi_usd = advanced_data.get('oi_current_usd', 0.0)
        oi_chg = advanced_data.get('oi_change_4h', 0.0)
        oi_trend = advanced_data.get('aggregated_oi_trend', 'データ取得試行失敗')
        ls_ratio = advanced_data.get('ls_ratio', '無料アクセス試行失敗') # 無料アクセス試行の結果
        
        # 🚨 未実装の複雑な実践データ
        liq_long = advanced_data.get('liq_24h_long', '清算額取得未実装')
        liq_cluster_info = advanced_data.get('liquidation_cluster', 'データ取得試行失敗')
        
        current_time = datetime.now(timezone.utc).astimezone(None).strftime('%Y-%m-%d %H:%M JST')
        
        max_proba = proba[np.argmax(proba)]
        uncertainty_score = 1.0 - max_proba
        
        # 主要な原因とリスクレベルの決定ロジック (実践データに依存)
        main_cause = "テクニカル環境と市場センチメント"
        risk_level = "中🔴"
        
        # LSRが取得できている場合、分析に組み込む
        if isinstance(ls_ratio, str) and ls_ratio.replace('.', '', 1).isdigit():
             ls_ratio_val = float(ls_ratio)
             if ls_ratio_val > 1.10 and fr > 0.0001:
                 main_cause = "市場全体でのロング過熱と需給の不均衡"
                 risk_level = "高🔴🔴"
        elif oi_chg < -0.01 and fg_index < 30: 
             main_cause = "恐怖によるポジション解消とボラティリティの低下"
             risk_level = "中高🔴"
        
        if uncertainty_score > 0.40:
             risk_level = "高🔴🔴"
             
        
        # --- レポートA: 市場構造と主要ドライバー分析 (HTML形式) ---
        report_structure = f"""
<b>【BTC 市場ドライバー分析 - 実践数値に基づく】</b>
📅 {current_time}

📌 <b>主要ポイント</b>
<b>主要ドライバー:</b> 現在の市場トレンドの主要なドライバーは <b>{main_cause}</b> です。
<b>センチメント:</b> Fear & Greed Indexは <b>{fg_index}</b>（「{fg_value}」レベル）であり、市場のボラティリティを示唆しています。
<b>テクニカル環境:</b> BTC価格 <b>${price:.2f}</b> は、20日SMA（${sma:.2f}）を {'🟢 上回っています' if price > sma else '🔴 下回っています'}。短期トレンドは {'強気' if price > sma else '弱気'} です。

-------------------------------------
<b>📉 市場ドライバーとリスク分析 (実践データ)</b>
<pre>
カテゴリ        | 指標         | 現在値/ステータス     | 分析/示唆
--------------------------------------------------------------------------------
需給・流動性    | FR           | {fr*100:.4f}%             | {'🚨 ロングのコスト高。スクイーズリスクあり。' if fr > 0.00015 else '中立。'}
                | OI総量(集計)| ${oi_usd:,.0f}       | {oi_trend}の傾向。市場への資金流入/流出を示す。
                | OI変化率(4H集計)| {oi_chg*100:.2f}%            | {'🟢 増加。トレンド継続の勢い。' if oi_chg > 0.01 else '中立/減少。'}
センチメント    | F&G指数      | {fg_index} ({fg_value}) | {'極度の恐怖。逆張り機会か、底値割れの警告。' if fg_index <= 20 else '楽観的。短期的な過熱の可能性。'}
                | L/S比率(集計)| {ls_ratio}           | Coinglass無料アクセス試行結果。
                | 24H清算額(集計)| {liq_long}           | ⚠️ <b>高度なデータは通常有料/未実装。</b>
ボラティリティ  | ATR          | ${atr:.2f}             | {(atr / price) * 100:.2f}% (市場ボラティリティの目安)。
</pre>
-------------------------------------

<b>📊 市場集計データ洞察（Coinglass等から取得）</b>
- <b>総建玉トレンド:</b> {oi_trend}
- <b>清算ヒートマップ:</b> {liq_cluster_info}

<b>🎯 機会とリスク</b>
- <b>🚨 リスクレベル:</b> <b>{risk_level}</b>。公開データに基づき、市場ボラティリティに警戒してください。
"""
        
        # --- 予測結果の調整 ---
        final_conclusion = ml_result
        if (ml_result == "📈 上昇" and (isinstance(ls_ratio, str) and ls_ratio.replace('.', '', 1).isdigit() and float(ls_ratio) > 1.15)):
             final_conclusion = f"⚠️ {ml_result} (注意: LSRに基づく短期調整リスク)"
        
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

<b>全体判断:</b> <b>{strategy_advice_short}</b>。公開された実践データに基づいています。

-------------------------------------
<b>🎯 短期戦略（先物/デイトレード）</b>
<pre>
方向性           | エントリー目標                  | 損切り(SL)           | 利益確定目標
------------------------------------------------------------------------------------
{'弱気' if ml_prediction <= 0 else '強気'} | {entry_short if ml_prediction <= 0 else entry_long} | ATRに基づく ${atr:.2f} (リスク許容度) | 直近の高値/安値ゾーン
</pre>

-------------------------------------
<b>🚨 重要な注意事項</b>
本BOTは、**Coinglassの無料公開エンドポイント**からOIとLSRデータを取得しようと試みています。この方法は、APIキーが不要ですが、**APIの安定性は保証されません**。データ取得に失敗した場合は、フォールバック値（例: '無料アクセス試行失敗'）が表示されます。

📚 <b>まとめ</b>
無料の公開データに切り替えることで、外部APIキーなしで市場センチメントを分析できるようになりました。
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
