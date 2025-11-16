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
# Telegramや取引所APIの認証情報が環境変数から取得されます
MEXC_API_KEY = os.environ.get('MEXC_API_KEY')
MEXC_SECRET = os.environ.get('MEXC_SECRET')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

FUTURES_SYMBOL = 'BTC/USDT'
TIMEFRAME = '4h' # 予測の期間（4時間足）
MODEL_FILENAME = 'btc_futures_ml_model.joblib'

# 外部APIエンドポイント (現在動作確認済みで無料のAPIのみを使用)
FG_INDEX_API_URL = 'https://api.alternative.me/fng/?limit=1'

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
        # APIキーの存在チェック
        if not all([MEXC_API_KEY, MEXC_SECRET]):
             raise ValueError("APIキーが設定されていません。環境変数を確認してください。")
             
        # CCXT MEXC 先物クライアントの初期化
        self.exchange = ccxt.mexc({
            'apiKey': MEXC_API_KEY,
            'secret': MEXC_SECRET,
            'options': {'defaultType': 'future'},
            'enableRateLimit': True,
        })
        self.target_threshold = 0.0005 # 予測対象の変動閾値 (0.05%以上の変動をターゲット)
        self.prediction_period = 1 # 1期間先の変動を予測
        self.feature_cols = [] # 特徴量カラム名リスト

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

        # テクニカル指標の計算
        df['SMA'] = ta.sma(df['Close'], length=20)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['MACD_H'] = ta.macd(df['Close'])['MACDh_12_26_9']
        df['Vol_Diff'] = df['Volume'] / ta.sma(df['Volume'], length=20)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14) 

        # ラグ特徴量（過去の値）の作成
        for lag in [1, 2, 3]:
            df[f'RSI_L{lag}'] = df['RSI'].shift(lag)
            # 過去の価格変化率をラグ特徴量として使用
            df[f'Price_L{lag}'] = df['Close'].pct_change(lag).shift(lag)
            
        # ターゲット変数（未来の価格変動）の作成
        future_change = df['Close'].pct_change(periods=-self.prediction_period).shift(self.prediction_period)
        
        # ターゲットに1(上昇), -1(下落), 0(レンジ)を割り当てる
        df['Target'] = np.select(
            [future_change > self.target_threshold, future_change < -self.target_threshold],
            [1, -1], default=0
        )
        
        # 欠損値を含む行を削除
        df.dropna(inplace=True)
        
        # 特徴量カラムのリストを一度だけ作成
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
        
        # ランダムフォレストモデルを使用
        model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', max_depth=10)
        model.fit(X_train, Y_train)
        
        # モデルをファイルに保存
        joblib.dump(model, MODEL_FILENAME)
        print("✅ モデルの再学習が完了し、ファイルに保存されました。")
        return True

    # --- (D) リアルタイム予測と通知 ---
    def predict_and_report(self, df_latest: pd.DataFrame, advanced_data: Dict[str, Any]) -> bool:
        # モデルのロード
        try:
            model = joblib.load(MODEL_FILENAME)
        except FileNotFoundError:
            report = "🚨 <b>エラー:</b> モデルファイルが見つかりません。まず学習とコミットを行ってください。"
            self.send_telegram_notification(report) 
            return False

        # 最新データの特徴量作成
        X_latest, _ = self.create_ml_features(df_latest.copy())
        
        if X_latest.empty:
            report = (
                "🚨 <b>予測スキップ通知:</b> OHLCVデータが不足しています。"
            )
            self.send_telegram_notification(report)
            return False
            
        latest_X = X_latest.iloc[[-1]] 
        
        # 予測の実行
        prediction_val = model.predict(latest_X)[0]
        prediction_proba = model.predict_proba(latest_X)[0]
        
        # レポートの生成と送信
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
        レポートを「市場構造と詳細分析」と「ML予測、根拠と行動計画」の2部構成で生成（日本語版、HTML形式）
        分析深度を最大化し、定性的な市場評価を組み込む。
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
        
        # ----------------------------------------------------------------
        # 🆕 詳細分析と定性評価ロジック
        # ----------------------------------------------------------------
        
        # 1. テクニカル分析の定性評価
        is_above_sma = price > sma
        trend_direction = "上昇トレンド" if is_above_sma else "下降トレンド"
        
        # 2. 過熱シグナルとリスクレベル
        overheat_score = 0
        
        # 貪欲シグナル (ロング過熱の可能性)
        if fg_index >= 75: overheat_score += 1
        # FRシグナル (ロングのコスト高・調整リスク)
        if fr >= 0.00015: overheat_score += 1
        
        if overheat_score >= 2:
            market_state = "🚨 極端な過熱・調整リスクが高い局面"
            risk_level = "高🔴🔴"
            market_health = "不安定（ロング調整待ち）"
            main_driver = "過度な楽観（FGI）とポジションの傾き（FR）"
        elif overheat_score == 1:
             market_state = "⚠️ 過熱の兆候が見られる局面"
             risk_level = "中高🔴"
             market_health = "警戒が必要"
             main_driver = "センチメントの傾き、またはテクニカル主導"
        elif fg_index <= 25:
             market_state = "🟢 極度の恐怖に支配された局面"
             risk_level = "中高🔴" # V字反発リスクでリスクは高い
             market_health = "反発期待（逆張り候補）"
             main_driver = "極端な恐怖によるポジション調整"
        else:
             market_state = "⚖️ 中立的なレンジまたはトレンド形成中"
             risk_level = "中🔴"
             market_health = "比較的健全"
             main_driver = "テクニカル環境と価格変動"
             
        # 3. MACD/RSIのモメンタム判断
        momentum_strength = "中立"
        if rsi > 60 and macd_h > 0:
            momentum_strength = "強い強気モメンタム"
        elif rsi < 40 and macd_h < 0:
            momentum_strength = "強い弱気モメンタム"

        # 4. ATRに基づく主要サポート/レジスタンス
        # 1 ATR レジスタンス/サポート (短期取引の目安)
        res_1atr = price + atr
        sup_1atr = price - atr
        
        # 2 ATR 重要なブレイクポイント (トレンド継続/転換の目安)
        res_2atr = price + (atr * 2)
        sup_2atr = price - (atr * 2)

        # ----------------------------------------------------------------
        
        current_time = datetime.now(timezone.utc).astimezone(None).strftime('%Y-%m-%d %H:%M JST')
        max_proba = proba[np.argmax(proba)]
        uncertainty_score = 1.0 - max_proba
        
        
        # --- レポートA: BTC市場 現状と詳細分析 (HTML形式) ---
        report_structure = f"""
<b>【BTC市場 徹底分析レポート】</b>
📅 {current_time} | <b>{TIMEFRAME}足分析</b>

<b>🌐 総合市場評価:</b> <b>{market_state}</b>
📌 <b>市場の健全性:</b> <b>{market_health}</b> | <b>主要ドライバー:</b> {main_driver}
📌 <b>現在価格:</b> <b>${price:.2f}</b>

-------------------------------------
<b>🔍 1. テクニカル分析の詳細な根拠</b>
<pre>
指標         | 現在値/レベル     | 洞察と根拠
--------------------------------------------------------------------------------
トレンド     | {trend_direction}   | 価格は20-SMA(${sma:.2f})を{'<b>上回っており</b>' if is_above_sma else '<b>下回っており</b>'}、短期トレンドを示唆。
RSI (14)     | {rsi:.2f}           | {'<b>買われすぎ</b>(>70)' if rsi >= 70 else ('<b>売られすぎ</b>(<30)' if rsi <= 30 else '中立')}: モメンタムは現在 <b>{momentum_strength}</b>。
MACD Hist.   | {macd_h:.2f}         | {'上昇勢い継続' if macd_h > 0 else '下降勢い継続'}: 勢い転換の有無を監視。
ATR (ボラティリティ) | ${atr:.2f}        | 過去14期間の平均変動幅。現在価格の<b>{atr/price*100:.2f}%</b>の変動幅を予想。
</pre>

<b>🔑 主要なサポート/レジスタンスレベル (ATRに基づく):</b>
- **短期レジスタンス (R1):** <b>${res_1atr:.2f}</b> (ここを突破すれば次のトレンドへ)
- **短期サポート (S1):** <b>${sup_1atr:.2f}</b> (ここを割ると調整局面入り)
- **重要ブレイクアウトレベル (R2/S2):** <b>${res_2atr:.2f}</b> / <b>${sup_2atr:.2f}</b>

-------------------------------------
<b>📈 2. センチメントと過熱リスク</b>
<pre>
カテゴリ    | 指標      | 現在値/ステータス     | 示唆
--------------------------------------------------------------------------------
ポジション  | FR        | {fr*100:.4f}%             | {'🚨 <b>極端に高い</b>: ロング側への資金調達コストが増加。短期的に調整リスクが極めて高い。' if fr > 0.00015 else '中立。'}
感情        | F&G指数   | {fg_index} ({fg_value}) | {'極度の恐怖。逆張りロングの候補。' if fg_index <= 25 else '楽観的。'}
<b>総合評価</b> | <b>リスクレベル</b> | <b>{risk_level}</b>        | FR/FGIの傾きと市場の過熱度に基づきます。
</pre>
"""
        
        # --- ML予測と根拠の統合 ---
        final_conclusion = ml_result
        integrated_reasoning = []
        
        # 予測が上昇の場合
        if ml_prediction == 1:
            integrated_reasoning.append(f"<b>テクニカル要因:</b> {trend_direction}継続、モメンタム ({momentum_strength}) が{'強い' if rsi > 60 else '中立'}ため。")
            
            if overheat_score >= 1:
                # 警告を追加
                integrated_reasoning.append(f"<b>センチメント要因:</b> FGIとFRによる過熱シグナル（{market_state}）が見られます。短期的な急落（ロングスクイーズ）に警戒し、利益確定を推奨します。")
                final_conclusion = f"⚠️ {ml_result} (調整リスク高)"
            elif fg_index <= 25:
                integrated_reasoning.append(f"<b>センチメント要因:</b> 極度の恐怖（FGI={fg_index}）からの反発シグナルが、MLの上昇予測を補強しています。底打ちの可能性があります。")
                
        # 予測が下落の場合
        elif ml_prediction == -1:
            integrated_reasoning.append(f"<b>テクニカル要因:</b> 20-SMA割れ（下降トレンド）またはMACD/RSIの勢い ({momentum_strength}) 減速による下落圧力が優勢なため。")
            if overheat_score >= 1:
                integrated_reasoning.append(f"<b>センチメント要因:</b> 市場の過熱感（FR/FGI）が、調整/下落圧力を強く後押ししています。ショートポジションの優位性が高いと判断されます。")
            elif fg_index >= 75:
                integrated_reasoning.append(f"<b>センチメント要因:</b> 極端な貪欲（FGI={fg_index}）の反動とFRの高さが、強制的な調整（ロング清算）を誘発する可能性が高いため。")

        # 予測がレンジの場合
        elif ml_prediction == 0:
            integrated_reasoning.append(f"<b>テクニカル要因:</b> 主要な指標（SMA/MACD）が方向性を示しておらず、モメンタムが弱く、価格がS1とR1の間に収束しやすいため。")
            integrated_reasoning.append(f"<b>センチメント要因:</b> 市場センチメントが中立（{fg_value}）であり、ポジションの傾きも控えめなため、レンジ相場が継続すると予測されます。")
            
        
        # 推奨戦略の決定
        # 予測が不明確 (レンジ or 不確実性が高い) の場合は待機を推奨
        if uncertainty_score > 0.40 or ml_prediction == 0:
            overall_advice = "🚨 <b>リスク回避/待機推奨</b>\n市場がレンジまたはML予測の信頼度が低いです。重要なブレイクアウト（S1/R1レベル）を待機するか、小ロットでの短期取引に限定してください。"
            action_plan_long = f"<b>守備的エントリー:</b> 1-ATRサポート付近 (${sup_1atr:.2f}付近) での反発狙い。"
            action_plan_short = f"<b>守備的エントリー:</b> 1-ATRレジスタンス付近 (${res_1atr:.2f}付近) での押し目売り狙い。"
            stop_loss = f"ATRの変動を考慮し、最大変動許容幅は <b>${atr:.2f}</b> です。"

        else:
             overall_advice = f"✅ <b>ML予測に合わせた取引を推奨</b>\n結論: <b>{final_conclusion}</b>。リスク管理を徹底し、ATRレベルを活用してください。"
             
             if ml_prediction == 1: # 上昇予測
                 action_plan_long = f"<b>推奨エントリー:</b> S1 (${sup_1atr:.2f}) への押し目買い、またはR1突破後のエントリー。"
                 action_plan_short = f"<b>短期ショート:</b> R2 (${res_2atr:.2f}) への到達後の利食いショートのみ検討。"
                 stop_loss = f"損切りは S1 (${sup_1atr:.2f}) の少し下（約 {atr*0.5:.2f}幅）に設定。"
                 
             else: # 下落予測
                 action_plan_long = f"<b>短期ロング:</b> S2 (${sup_2atr:.2f}) への到達後の反発狙い（逆張り）のみ検討。"
                 action_plan_short = f"<b>推奨エントリー:</b> R1 (${res_1atr:.2f}) への戻り売り、またはS1突破後のエントリー。"
                 stop_loss = f"損切りは R1 (${res_1atr:.2f}) の少し上（約 {atr*0.5:.2f}幅）に設定。"
        
        
        # --- レポートB: ML予測、根拠と具体的な行動計画 (HTML形式) ---
        report_conclusion = f"""
<b>【ML予測、根拠と具体的な行動計画】</b>
📅 {current_time}

<b>🎯 3. ML予測の結論と総合根拠</b>
ML予測結論: <b>{final_conclusion}</b> (信頼度: {max_proba*100:.1f}%)

<p><b>🤔 なぜそう考えるのか (総合根拠):</b></p>
<ul>
    {''.join([f'<li>{reason}</li>' for reason in integrated_reasoning])}
</ul>

-------------------------------------
<b>📝 行動計画と推奨ストラテジー</b>

{overall_advice}

🟢 <b>ロング戦略（買い）の指針:</b>
  - エントリー目標: {action_plan_long}
  - 利益確定: R1 (${res_1atr:.2f}) やR2 (${res_2atr:.2f}) などの主要レジスタンスゾーン。
  
🔴 <b>ショート戦略（売り）の指針:</b>
  - エントリー目標: {action_plan_short}
  - 損切りライン: <b>{stop_loss}</b>

-------------------------------------
<b>📚 今後の監視ポイント (次の4時間):</b>
1. 価格が<b>短期レジスタンス R1 (${res_1atr:.2f})</b> を突破できるか。
2. <b>FR/FGI</b>が過熱度を解消するか、または悪化するか。
3. 価格が<b>20-SMA</b> (${sma:.2f}) を維持できるか。
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
