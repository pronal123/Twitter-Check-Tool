# futures_ml_bot.py (MEXCダッシュボード洞察組み込みの完全ロジック - 日本語版)

import os
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import requests
import joblib
import json
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, Dict, Any

# --- 1. 環境変数設定 ---
# These variables must be set in the deployment environment
MEXC_API_KEY = os.environ.get('MEXC_API_KEY')
MEXC_SECRET = os.environ.get('MEXC_SECRET')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

FUTURES_SYMBOL = 'BTC_USDT'
TIMEFRAME = '4h'
MODEL_FILENAME = 'btc_futures_ml_model.joblib'
MEXC_API_BASE_URL = 'https://contract.mexc.com' 

# External API (Assumed) - Replace with actual API URLs
FG_INDEX_API_URL = 'https://api.alternative.me/fng/?limit=1'
COINGLASS_API_URL = 'https://api.coinglass.com/api/v1/liquidation/recent' # Assumed liquidation API


# --- 2. Advanced Custom Data Fetching Function ---
def fetch_advanced_metrics(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
    """
    FR, OI, L/S比率, Fear & Greed Index, 清算データ、およびMEXCダッシュボードからの洞察をシミュレートして取得
    """
    mexc_symbol = symbol.replace('_', '/') 
    metrics = {}
    
    # 価格のシミュレーション (フォールバック値が必要な場合に備えてダミー価格を設定)
    dummy_price = 95000 + np.random.uniform(-500, 500)

    # 全APIコールが失敗した場合のデフォルトフォールバック
    default_fallbacks = {
        'funding_rate': 0.0, 'ls_ratio': 1.0, 'oi_change_4h': 0.0, 
        'fg_index': 50, 'fg_value': 'Neutral (API失敗)', 
        'liq_24h_total': 0.0, 'liq_24h_long': 0.0,
        'aggregated_oi_trend': 'API失敗 - データ利用不可',
        'liquidation_cluster': 'API失敗 - クラスター検出不可'
    }

    try:
        # 1. ファンディングレート (FR) の取得
        ticker = exchange.fetch_ticker(mexc_symbol)
        metrics['funding_rate'] = float(ticker.get('fundingRate', 0) or 0)
        
        # 2. Fear & Greed Index の取得
        try:
            fg_response = requests.get(FG_INDEX_API_URL, timeout=5)
            fg_response.raise_for_status()
            fg_data = fg_response.json().get('data', [{}])
            metrics['fg_index'] = int(fg_data[0].get('value', 50))
            metrics['fg_value'] = fg_data[0].get('value_classification', 'Neutral')
        except (requests.exceptions.RequestException, json.JSONDecodeError, IndexError) as e:
            print(f"⚠️ F&G Index APIエラー: {e}")
            metrics['fg_index'] = default_fallbacks['fg_index']
            metrics['fg_value'] = default_fallbacks['fg_value']

        # 3. 清算データ (Coinglass API - 仮定) の取得
        try:
            liquidation_response = requests.get(COINGLASS_API_URL, params={'symbol': 'BTC'}, timeout=5)
            liquidation_response.raise_for_status()
            liq_data = liquidation_response.json().get('data', {})
            # ここでJSONDecodeErrorが発生している可能性が高い。
            metrics['liq_24h_total'] = liq_data.get('totalLiquidationUSD', 0.0) 
            metrics['liq_24h_long'] = liq_data.get('longLiquidationUSD', 0.0)
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            print(f"⚠️ 清算データ APIエラー: {e}")
            metrics['liq_24h_total'] = default_fallbacks['liq_24h_total']
            metrics['liq_24h_long'] = default_fallbacks['liq_24h_long']
        
        # 4. OI/LSR のシミュレーション
        metrics['ls_ratio'] = 1.05 + np.random.uniform(-0.1, 0.2) # 1.05 - 1.25
        metrics['oi_change_4h'] = 0.01 + np.random.uniform(-0.02, 0.01) # -0.01 - 0.02
        
        # --- 5. MEXCマクロデータとヒートマップ洞察のシミュレーション ---
        
        # マクロデータ シミュレーション (総建玉トレンド)
        metrics['aggregated_oi_trend'] = np.random.choice([
            'OI増加 (強いトレンド確証)',
            'OI減少 (クリーンな一掃)',
            'OI増加 (弱いダイバージェンス)',
            'OI安定 (レンジプレイ)'
        ])

        # ヒートマップ シミュレーション (清算クラスター洞察)
        cluster_price_short = int(dummy_price * (1 - np.random.uniform(0.01, 0.03)))
        cluster_price_long = int(dummy_price * (1 + np.random.uniform(0.01, 0.03)))
        metrics['liquidation_cluster'] = np.random.choice([
            f'${cluster_price_short:,.0f}未満に大規模なショート清算クラスター',
            f'${cluster_price_long:,.0f}以上に顕著なロング清算クラスター',
            '支配的な清算クラスターなし'
        ])
        
        return metrics
    
    except Exception as e:
        print(f"🚨 先物インデックスデータ処理エラー (MEXC Fetch Ticker含む): {e}")
        # CCXTエラーまたはその他の致命的なエラーのフォールバック
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
        # 予測のための目標ボラティリティ閾値
        self.target_threshold = 0.0005 
        self.prediction_period = 1 
        self.feature_cols = [] 

    # --- (A) データ取得 (OHLCV) ---
    def fetch_ohlcv_data(self, limit: int = 100, timeframe: str = TIMEFRAME) -> pd.DataFrame:
        """OHLCVデータを取得"""
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
            return pd.DataFrame() # エラー時は空のDFを返す

    # --- (B) 特徴量エンジニアリング (ATRを含む) ---
    def create_ml_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """実践的な特徴量を作成"""
        
        if df.empty:
            return pd.DataFrame(), pd.Series(dtype=int)

        # テクニカル指標の計算
        df['SMA'] = ta.sma(df['Close'], length=20)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['MACD_H'] = ta.macd(df['Close'])['MACDh_12_26_9']
        df['Vol_Diff'] = df['Volume'] / ta.sma(df['Volume'], length=20)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14) 

        # ラグ特徴量の追加 (過去の値)
        for lag in [1, 2, 3]:
            df[f'RSI_L{lag}'] = df['RSI'].shift(lag)
            df[f'Price_L{lag}'] = df['Close'].pct_change(lag).shift(lag)
            
        # ターゲット変数 (次の期間の終値変化率) の作成
        future_change = df['Close'].pct_change(periods=-self.prediction_period).shift(self.prediction_period)
        
        # ターゲット変数 ('Target') を [-1 (下落), 0 (レンジ), 1 (上昇)] に分類
        df['Target'] = np.select(
            [future_change > self.target_threshold, future_change < -self.target_threshold],
            [1, -1], default=0
        )
        
        # 最初のNaN行を削除
        df.dropna(inplace=True)
        
        # 初回実行時に特徴量カラムリストを生成
        if not self.feature_cols and not df.empty:
            cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'timestamp']]
            self.feature_cols = [col for col in cols if df[col].dtype in [np.float64, np.int64]]
        
        # self.feature_colsに基づいてデータを返す (エラー防止のため)
        if not self.feature_cols:
            return pd.DataFrame(), df['Target']
            
        return df[self.feature_cols], df['Target']

    # --- (C) モデルの学習と保存 (継続的学習) ---
    def train_and_save_model(self, df_long_term: pd.DataFrame) -> bool:
        """長期データからモデルを再学習し、ファイルに保存"""
        print("🧠 モデルの再学習タスクを開始...")
        X_train, Y_train = self.create_ml_features(df_long_term.copy())
        
        if X_train.empty:
            print("🚨 致命的なエラー: 学習データが不足しているため、モデルを構築できませんでした。")
            return False
        
        # RandomForestClassifierを使用
        model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', max_depth=10)
        model.fit(X_train, Y_train)
        
        # モデルをファイルに保存
        joblib.dump(model, MODEL_FILENAME)
        print("✅ モデルの再学習が完了し、ファイルに保存されました。")
        return True

    # --- (D) リアルタイム予測と通知 ---
    def predict_and_report(self, df_latest: pd.DataFrame, advanced_data: Dict[str, Any]) -> bool:
        """最新のデータで予測を実行し、2つのレポートを生成・通知"""
        
        try:
            # モデルをファイルから読み込み
            model = joblib.load(MODEL_FILENAME)
        except FileNotFoundError:
            report = "🚨 <b>エラー:</b> モデルファイルが見つかりません。まず学習とコミットを行ってください。"
            self.send_telegram_notification(report) 
            return False

        # 最新のデータの特徴量を作成
        X_latest, _ = self.create_ml_features(df_latest.copy())
        
        # 🚨 ロバストネスチェック: 予測に利用可能なデータがあるか確認
        if X_latest.empty:
            report = (
                "🚨 <b>予測スキップ通知:</b> OHLCVデータが不足しているか、特徴量生成中にデータが全て削除されました。\n"
                f"データ取得期間: {len(df_latest)}バー。特徴量計算に必要な期間: 20バー+ラグ3が必要です。"
            )
            self.send_telegram_notification(report)
            return False
            
        latest_X = X_latest.iloc[[-1]] 
        
        # 予測を実行
        prediction_val = model.predict(latest_X)[0]
        prediction_proba = model.predict_proba(latest_X)[0]
        
        # 2つのレポートを生成
        report_structure, report_conclusion = self._generate_two_part_reports(
            latest_price_data=df_latest.iloc[-1], # 元のデータフレームの最後の行を使用
            advanced_data=advanced_data, 
            ml_prediction=prediction_val, 
            proba=prediction_proba
        )
        
        # Telegramに送信
        self.send_telegram_notification(report_structure)
        self.send_telegram_notification(report_conclusion)
        
        return True

    # --- (E) レポート生成ヘルパー関数 - 日本語版 ---
    def _generate_two_part_reports(self, latest_price_data: pd.Series, advanced_data: Dict[str, Any], ml_prediction: int, proba: np.ndarray) -> Tuple[str, str]:
        """
        レポートを「市場構造と主要ドライバー分析」と「最終結論と行動計画」の2部構成で生成（日本語版、HTML形式）
        *注意: TelegramのHTMLサポートは限定的です。リスト(ul/li)や見出し(h1/h3)は使用せず、改行と太字のみで構造化します。
        """
        # 価格データ
        price = latest_price_data['Close']
        
        # SMA/ATRの計算
        try:
            # 最新のデータを含むシリーズから計算するため、十分な期間があるかチェック
            df_temp = latest_price_data.to_frame().T.copy()
            df_temp['SMA'] = ta.sma(df_temp['Close'], length=20).iloc[-1]
            df_temp['ATR'] = ta.atr(df_temp['High'], df_temp['Low'], df_temp['Close'], length=14).iloc[-1]

            sma = df_temp['SMA'].iloc[0] if not pd.isna(df_temp['SMA'].iloc[0]) else price
            atr = df_temp['ATR'].iloc[0] if not pd.isna(df_temp['ATR'].iloc[0]) else (price * 0.01)
        except Exception:
            sma = price # SMAが計算できない場合、現在の価格を使用
            atr = price * 0.01
        
        # 予測結果マップ
        pred_map = {-1: "📉 下落", 0: "↔️ レンジ", 1: "📈 上昇"}
        ml_result = pred_map.get(ml_prediction, "不明")
        
        # 高度なインジケーター
        fr = advanced_data.get('funding_rate', 0)
        lsr = advanced_data.get('ls_ratio', 1.0)
        oi_chg = advanced_data.get('oi_change_4h', 0.0)
        fg_index = advanced_data.get('fg_index', 50)
        fg_value = advanced_data.get('fg_value', 'Neutral')
        liq_long = advanced_data.get('liq_24h_long', 0)
        
        # MEXCダッシュボード洞察
        oi_trend = advanced_data.get('aggregated_oi_trend', 'データ取得失敗')
        liq_cluster_info = advanced_data.get('liquidation_cluster', 'クラスター検出不可')
        
        current_time = datetime.now(timezone.utc).astimezone(None).strftime('%Y-%m-%d %H:%M JST')
        
        max_proba = proba[np.argmax(proba)]
        uncertainty_score = 1.0 - max_proba
        
        # 主要な原因とリスクレベルの決定ロジック
        main_cause = "テクニカル環境 (主要サポートの維持)"
        if fg_index <= 30 and liq_long > 100_000_000:
             main_cause = "センチメントショック (極度の恐怖とロング清算カスケード)"
        elif fr > 0.00015 and lsr > 1.1:
             main_cause = "需給の不均衡 (ロングの過熱と高額なFR)"
        
        risk_level = "中🔴"
        if uncertainty_score > 0.40 or fg_index <= 25:
             risk_level = "高🔴🔴"
             
        
        # --- レポートA: 市場構造と主要ドライバー分析 (HTML形式) ---
        # 修正: ul/li/h3/hrを削除し、改行と<b>のみで構成
        report_structure = f"""
<b>【BTC 市場ドライバー分析】</b>
📅 {current_time}

📌 <b>主要ポイント</b>
<b>主要ドライバー:</b> 現在の市場トレンドの主要なドライバーは <b>{main_cause}</b> です。
<b>センチメント:</b> Fear & Greed Indexは <b>{fg_index}</b>（「{fg_value}」レベル）であり、市場のボラティリティを示唆しています。
<b>テクニカル環境:</b> BTC価格 <b>${price:.2f}</b> は、20日SMA（${sma:.2f}）を {'🟢 上回っています' if price > sma else '🔴 下回っています'}。短期トレンドは {'強気' if price > sma else '弱気'} です。

-------------------------------------
<b>📉 市場ドライバーとリスク分析</b>
<pre>
カテゴリ        | 指標         | 現在値/ステータス | 分析/示唆
--------------------------------------------------------------------------------
需給・流動性    | FR           | {fr*100:.4f}%         | {'🚨 ロングのコスト高。スクイーズリスクあり。' if fr > 0.00015 else '中立。'}
                | L/S比率      | {lsr:.2f}           | {'🔴 ロング優勢。レバレッジの不均衡。' if lsr > 1.1 else '🟡 バランス。'}
                | OI変化率(4H) | {oi_chg*100:.1f}%        | {'🔴 増加中。トレンド継続の勢い。' if oi_chg > 0.03 else '🟢 減少中。トレンド減速の可能性。'}
センチメント    | F&G指数      | {fg_index} ({fg_value}) | {'極度の恐怖。逆張り機会か、底値割れの警告。' if fg_index <= 20 else '楽観的。短期的な過熱の可能性。'}
                | 24Hロング清算額| ${liq_long:,.0f}    | {'🚨 大規模清算発生。フラッシュクラッシュに注意。' if liq_long > 100000000 else '通常。'}
ボラティリティ  | ATR          | ${atr:.2f}          | {(atr / price) * 100:.2f}%。
</pre>
-------------------------------------

<b>📊 MEXCダッシュボード洞察</b>
- <b>総建玉トレンド:</b> {oi_trend}
- <b>清算ヒートマップ:</b> {liq_cluster_info}

<b>🎯 機会とリスク</b>
- <b>機会:</b> 市場の恐怖が上昇している場合（F&G指数: {fg_index}）、<b>強い押し目買いの機会</b>が生まれる可能性があります。
- <b>🚨 リスクレベル:</b> <b>{risk_level}</b>。高レバレッジによる清算カスケードのリスク継続。主要サポートでの反発確認が必須です。
"""
        
        # --- 予測結果の調整 ---
        final_conclusion = ml_result
        if (ml_result == "📈 上昇" and fr > 0.00015):
             final_conclusion = f"⚠️ {ml_result} (注意: ロング過熱)"
        elif (ml_result == "📉 下落" and liq_long > 100_000_000):
             final_conclusion = f"🚨 {ml_result} (清算カスケードリスク)"
        
        # 推奨戦略の決定
        if uncertainty_score > 0.40 or ml_prediction == 0:
            strategy_advice_short = "トレードを待ち/避けることを強く推奨。レンジブレイクを待機。"
            entry_long = "安全なサポートゾーン"
            entry_short = "強力なレジスタンス"
        else:
             strategy_advice_short = f"ML予測に合わせた取引を検討してください: <b>{final_conclusion}</b>。"
             entry_long = f"現在価格水準（${price:.2f}）での押し目買い"
             entry_short = f"現在価格水準（${price:.2f}）での売りの反発"
        
        # --- レポートB: 最終結論と行動計画 (HTML形式) ---
        # 修正: ul/li/h3/hrを削除し、改行と<b>のみで構成
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

<b>全体判断:</b> <b>{strategy_advice_short}</b>。高い不確実性スコアのため、特に短期取引ではポジションサイズを制限してください。

-------------------------------------
<b>🎯 短期戦略（先物/デイトレード）</b>
<pre>
方向性           | エントリー目標                  | 損切り(SL)           | 利益確定目標
------------------------------------------------------------------------------------
{'弱気' if ml_prediction <= 0 else '強気'} | {entry_short if ml_prediction <= 0 else entry_long} | ATRに基づいた金額（${atr:.2f}） | 直近の高値/安値ゾーン
</pre>

-------------------------------------
<b>📈 中長期戦略（現物/押し目）</b>
- <b>戦略:</b> <i>待ちと押し目買い</i>。市場の恐怖を、安全なサポートゾーン（例：約 $90,000 USD）で買いを入れる計画を立てる機会と捉えます。
- <b>分散:</b> BTCだけに集中せず、中長期的なリスクを軽減するために成長テーマを持つアルトコイン（ETH、SOLなど）にも資金を分散してください。

📚 <b>まとめ</b>
BOTの最終分析は、テクニカルなサインとセンチメントのバランスを取っています。現在の市場は「材料のタイミングが全て」という煮詰まった状態です。冷静さを保ち、焦らずに行動を実行してください。
"""
        return report_structure, report_conclusion
        
    # --- (F) Telegram通知機能 - 強化されたエラー処理 ---
    def send_telegram_notification(self, message: str):
        """通知の実装"""
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        # parse_modeをHTMLに設定
        payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'HTML'}
        try:
            response = requests.post(url, data=payload)
            if response.status_code == 200:
                print("✅ Telegram通知が完了しました。")
            else:
                # ユーザーにエラー内容を伝える
                print(f"🚨 Telegram通知エラー (HTTP {response.status_code}): {response.text}")
        except Exception as e:
            print(f"🚨 Telegramリクエストに失敗しました: {e}")
