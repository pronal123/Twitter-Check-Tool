import os
import json
import time
from datetime import datetime
import threading
import logging

import yfinance as yf
import pandas as pd
import pandas_ta as ta

from flask import Flask, render_template, jsonify
from apscheduler.schedulers.background import BackgroundScheduler

# ロギング設定
logging.basicConfig(level=logging.INFO, 
                    format='[%(asctime)s] %(levelname)s: %(message)s')

# --- グローバル変数と初期設定 ---

# アプリケーションのグローバル設定（Canvas環境変数から取得）
app_id = os.environ.get('__app_id', 'default-bot-app')
firebase_config_str = os.environ.get('__firebase_config', '{}')
try:
    FIREBASE_CONFIG = json.loads(firebase_config_str)
except json.JSONDecodeError:
    logging.error("❌ Firebase configのパースに失敗しました。")
    FIREBASE_CONFIG = {}

# スケジュール設定
REPORT_UPDATE_INTERVAL_HOURS = 6

# BOTの現在の分析結果を保持するグローバルオブジェクト
# ロックを使用してスレッドセーフにする
report_data_lock = threading.Lock()
report_data = {
    "scheduler_status": "初期化中",
    "current_price": 0.0,
    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S (JST)"),
    "data_count": 0,
    "data_range": "N/A",
    "bias": "N/A",
    "strategy": "データ処理中...",
    "stop_loss_level": 0.0,  # 新規追加: 推奨損切りレベル
    "stop_loss_reason": "N/A", # 新規追加: 損切り理由
    "predictions": {},
}

# --- Flask アプリケーション設定 ---

app = Flask(__name__)

# --- データ取得と前処理 ---

def fetch_historical_data(ticker="BTC-USD", period="60d", interval="1d", max_retries=3):
    """
    yfinanceから過去データを取得し、エラー発生時はリトライを行う。
    """
    for attempt in range(1, max_retries + 1):
        try:
            logging.info(f"yfinanceから{ticker}の過去データ（{period}）を取得中... (試行 {attempt}/{max_retries})")
            
            # auto_adjust=Trueはデフォルトになっているため、明示的に指定しない
            df = yf.download(ticker, period=period, interval=interval, progress=False)

            if df.empty:
                raise ValueError("取得したデータが空です。レート制限の可能性があります。")

            # MultiIndexをチェックし、フラット化する (MultiIndexの警告対策)
            if isinstance(df.columns, pd.MultiIndex):
                logging.warning("⚠️ yfinanceデータがMultiIndexを返しました。カラム名をフラット化し、再設定します。")
                df.columns = ['_'.join(col).strip() for col in df.columns.values]
                # 必要な列名をシンプルな名前に戻す
                df.rename(columns={
                    'Open_': 'Open',
                    'High_': 'High',
                    'Low_': 'Low',
                    'Close_': 'Close',
                    'Volume_': 'Volume'
                }, inplace=True)
                
            # 'Close'列が存在するか確認（フラット化後の確認）
            if 'Close' not in df.columns:
                 raise KeyError("Close列が見つかりません。データ構造を確認してください。")

            logging.info(f"✅ 過去データ取得成功。件数: {len(df)}")
            return df

        except Exception as e:
            logging.error(f"❌ yfinanceからデータ取得中にエラーが発生しました: {e}")
            if attempt < max_retries:
                wait_time = 6 * attempt
                logging.warning(f"⚠️ レート制限の可能性があります。{wait_time}秒待ってリトライします (試行 {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                logging.error("全リトライ試行が失敗しました。")
                return pd.DataFrame() # 空のDataFrameを返す

def calculate_technical_indicators(df):
    """
    DataFrameにテクニカル指標（MA, BBANDS, RSI, MACD）を追加する。
    """
    if df.empty:
        return df

    # シンプルな移動平均 (MA)
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()

    # ボリンジャーバンド (BBANDS)
    # デフォルトの期間(20)と標準偏差(2)を使用
    # columns: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0
    df.ta.bbands(close=df['Close'], length=20, std=2, append=True)

    # 相対力指数 (RSI)
    df.ta.rsi(close=df['Close'], length=14, append=True) # column: RSI_14

    # MACD
    df.ta.macd(close=df['Close'], fast=12, slow=26, signal=9, append=True) # columns: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9

    logging.info("✅ テクニカル指標の計算完了。")
    return df

# --- 分析と戦略生成 ---

def generate_strategy(df_analyzed):
    """
    分析されたデータフレームに基づいて、戦略、バイアス、推奨損切りレベルを生成する。
    """
    if df_analyzed.empty or len(df_analyzed) < 200:
        return {
            "bias": "分析データ不足",
            "strategy": "過去データが200日分（MA200の計算に必要）に満たないため、中期バイアスの判断を保留します。",
            "stop_loss_level": 0.0,
            "stop_loss_reason": "データ不足"
        }

    latest = df_analyzed.iloc[-1]
    
    # 最後の計算結果（列名に注意してアクセス）
    close = latest['Close']
    ma50 = latest['MA50']
    ma200 = latest['MA200']
    rsi = latest['RSI_14']
    
    # 【重要修正】ボリンジャーバンドの列名アクセスを修正
    # pandas_taの列名は BBL_20_2.0, BBM_20_2.0, BBU_20_2.0 であることを確認
    # 以前のKeyErrorを回避するため、列名の存在をチェックするか、確実な名前を使用します。
    # ここでは、MultiIndexフラット化後の列名整合性を信頼し、標準的な名前を使用します。
    bbl = latest.get('BBL_20_2.0', 0) # Lower Band
    bbu = latest.get('BBU_20_2.0', 0) # Upper Band

    # --- 1. 中期バイアス判定（MA50とMA200基準） ---
    
    bias = "中立"
    
    if ma50 > ma200:
        bias = "強気 (Bullish)"
        bias_color = "green"
    elif ma50 < ma200:
        bias = "弱気 (Bearish)"
        bias_color = "red"
        
    # --- 2. 戦略とアクション ---

    strategy_text = f"現在の市場バイアスは【{bias}】です。"
    stop_loss_level = 0.0
    stop_loss_reason = "現状維持"

    if bias == "強気 (Bullish)":
        if close > bbu:
            strategy_text += "価格はボリンジャーバンド上限を大きく超えており、短期的な過熱感が確認されます（RSI: {rsi:.2f}）。新規のロング（買い）エントリーは避け、利確・様子見推奨。"
            stop_loss_level = ma50  # 損切りラインをMA50に設定
            stop_loss_reason = "MA50ブレイク"
        elif rsi < 50 and close < ma50:
            strategy_text += "強気バイアス下で、短期的な調整（MA50付近への戻り）が確認されます。RSIが50以下でMA50付近での反発を確認できれば、押し目買いの機会となります。"
            stop_loss_level = ma200 # 損切りラインをMA200に設定
            stop_loss_reason = "MA200ブレイク"
        else:
            strategy_text += "MA50がMA200の上にあり、全体的に上昇傾向です。MA50を割るまではロング（買い）継続推奨。"
            stop_loss_level = bbl if bbl else ma50 
            stop_loss_reason = "BB下限/MA50ブレイク"
            
    elif bias == "弱気 (Bearish)":
        if close < bbl:
            strategy_text += "価格はボリンジャーバンド下限を下回っており、短期的な売られすぎ感があります（RSI: {rsi:.2f}）。新規のショート（売り）エントリーは避け、ショートの利確・様子見推奨。"
            stop_loss_level = ma50 # 損切りラインをMA50に設定
            stop_loss_reason = "MA50ブレイク"
        elif rsi > 50 and close > ma50:
            strategy_text += "弱気バイアス下で、短期的な反発（MA50付近への戻り）が確認されます。RSIが50以上でMA50付近での押し戻しを確認できれば、戻り売りの機会となります。"
            stop_loss_level = bbu if bbu else ma50
            stop_loss_reason = "BB上限/MA50ブレイク"
        else:
            strategy_text += "MA50がMA200の下にあり、全体的に下降傾向です。MA50を上回るまではショート（売り）継続推奨。"
            stop_loss_level = ma200 # 損切りラインをMA200に設定
            stop_loss_reason = "MA200ブレイク"

    elif bias == "中立":
        strategy_text += "MA50とMA200が接近しており、明確なトレンドが確認できません。レンジブレイクを待つか、ボリンジャーバンドの上下限を利用した短期逆張り戦略を検討してください。"
        stop_loss_level = 0.0
        stop_loss_reason = "レンジ相場"
        
    # 予測は簡易的なものとしてダミーデータを生成 (必要に応じて高度な予測モデルに置き換え可能)
    predictions = {
        "6時間": "レンジ継続",
        "12時間": "様子見",
        "24時間": "バイアス方向へ進展期待",
    }
    
    if bias == "強気 (Bullish)":
        predictions["6時間"] = "短期上昇"
    elif bias == "弱気 (Bearish)":
        predictions["6時間"] = "短期下降"

    return {
        "bias": bias,
        "strategy": strategy_text.format(rsi=rsi),
        "stop_loss_level": round(stop_loss_level, 2) if stop_loss_level > 0 else 0.0,
        "stop_loss_reason": stop_loss_reason,
        "predictions": predictions
    }

# --- バックグラウンドジョブ ---

def update_report_data():
    """
    データ取得、分析、戦略生成、グローバルデータの更新を行うバックグラウンドタスク。
    """
    global report_data
    
    try:
        # 1. データ取得
        # MA200に必要な日数を確保するため、period="1y"または"200d"などを使うのがより安全だが、
        # ここでは以前の設定を尊重しつつ、エラー処理で対応
        df = fetch_historical_data(period="300d", interval="1d") 
        if df.empty:
            raise Exception("データ取得失敗。処理を中断します。")

        # 2. テクニカル指標計算
        df_analyzed = calculate_technical_indicators(df)

        # 3. 分析と戦略生成
        analysis_result = generate_strategy(df_analyzed)

        # 4. 現在価格の取得（最新のClose価格）
        current_price = df_analyzed['Close'].iloc[-1]
        
        # 5. グローバルデータの更新 (スレッドセーフな更新)
        with report_data_lock:
            report_data.update({
                "scheduler_status": "稼働中",
                "current_price": round(current_price, 2),
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S (JST)"),
                "data_count": len(df_analyzed),
                "data_range": f"{df_analyzed.index[0].strftime('%Y-%m-%d')} - {df_analyzed.index[-1].strftime('%Y-%m-%d')}",
                "bias": analysis_result["bias"],
                "strategy": analysis_result["strategy"],
                "stop_loss_level": analysis_result["stop_loss_level"],
                "stop_loss_reason": analysis_result["stop_loss_reason"],
                "predictions": analysis_result["predictions"]
            })
            
        logging.info("✅ レポートデータ更新成功。Telegram通知もスケジュールされています。")

    except Exception as e:
        logging.error(f"❌ レポート更新タスクで重大なエラーが発生しました: {e}", exc_info=True)
        with report_data_lock:
            report_data.update({
                "scheduler_status": "エラー",
                "strategy": f"【エラー】データ処理中に問題が発生しました: {e}",
                "stop_loss_level": 0.0,
                "stop_loss_reason": "システムエラー",
            })
    
# --- スケジューラーの初期設定 ---

def init_scheduler():
    """
    APSchedulerを設定し、バックグラウンドでレポート更新タスクを実行する。
    """
    scheduler = BackgroundScheduler()
    
    # 起動直後に一度実行し、その後は設定された間隔で実行
    scheduler.add_job(
        update_report_data, 
        'interval', 
        hours=REPORT_UPDATE_INTERVAL_HOURS, 
        id='report_update_job', 
        next_run_time=datetime.now() # 起動直後に実行
    )
    
    try:
        scheduler.start()
        logging.info("✅ スケジューラーを開始しました。")
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logging.info("❌ スケジューラーを停止しました。")

# アプリケーション起動時にスケジューラーを起動
init_scheduler()

# --- Flask ルーティング ---

@app.route('/')
def index():
    """
    ダッシュボードを表示する。
    """
    return render_template('index.html')

@app.route('/status')
def get_status():
    """
    BOTの現在の分析ステータスをJSONで返す（フロントエンドのポーリング用）。
    """
    with report_data_lock:
        # グローバルデータをコピーして返す
        return jsonify(report_data.copy())


if __name__ == '__main__':
    # Gunicorn環境下では実行されないが、ローカルデバッグ用に残しておく
    app.run(host='0.0.0.0', port=10000, debug=False)
