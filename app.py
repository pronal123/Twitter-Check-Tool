# app.py (修正版)

import os
from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from dotenv import load_dotenv 
from futures_ml_bot import FuturesMLBot, fetch_futures_metrics, FUTURES_SYMBOL

# ローカルテスト時に .env ファイルを読み込む
load_dotenv() 

# --- 環境変数設定 ---
WEB_SERVICE_PORT = int(os.environ.get('PORT', 8080))
RETRAIN_INTERVAL_HOURS = int(os.environ.get('RETRAIN_INTERVAL_HOURS', 24))
PREDICTION_INTERVAL_HOURS = int(os.environ.get('PREDICTION_INTERVAL_HOURS', 1))

app = Flask(__name__)
scheduler = BackgroundScheduler()

# 🚨 BOTの初期化 (APIキーチェックとccxt接続がここで行われる)
bot = None
try:
    bot = FuturesMLBot() 
except ValueError as e:
    print(f"致命的な初期化エラー: {e}")
    
# --- 予測実行タスク (1時間ごと) ---
def run_prediction_and_notify():
    """予測を実行し、Telegramに通知する関数"""
    if bot is None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚨 BOTインスタンスがありません。タスクスキップ。")
        return

    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙️ 予測タスク開始...")
        
        # 🚨 実装された fetch_futures_metrics を呼び出す
        futures_data = fetch_futures_metrics(bot.exchange, FUTURES_SYMBOL)
        df_latest = bot.fetch_ohlcv_data(limit=100) 
        bot.predict_and_report(df_latest, futures_data)
        
        print("✅ 予測・通知タスク完了。")
             
    except Exception as e:
        print(f"🚨 予測タスクエラー: {e}")

# --- モデル再学習タスク (24時間ごと) ---
def run_retrain_and_improve():
    """モデルの再学習と構築を行う関数"""
    if bot is None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚨 BOTインスタンスがありません。再学習スキップ。")
        return
        
    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🧠 再学習タスク開始...")
        
        df_long_term = bot.fetch_ohlcv_data(limit=2000) 
        bot.train_and_save_model(df_long_term)
        
    except Exception as e:
        print(f"🚨 致命的な再学習タスクエラーが発生しました: {e}")

# --- スケジューラの初期化と起動 ---
def start_scheduler():
    """APSchedulerを設定し、バックグラウンドで開始する"""
    if bot is None:
        print("⚠️ BOT初期化失敗のため、スケジューラは起動しません。")
        return

    print("--- スケジューラ設定開始 ---")

    scheduler.add_job(func=run_prediction_and_notify, trigger='interval', hours=PREDICTION_INTERVAL_HOURS, id='prediction_job')
    scheduler.add_job(func=run_retrain_and_improve, trigger='interval', hours=RETRAIN_INTERVAL_HOURS, id='retrain_job')

    scheduler.start()
    print(f"✅ スケジューラ起動済み。予測:{PREDICTION_INTERVAL_HOURS}時間ごと, 再学習:{RETRAIN_INTERVAL_HOURS}時間ごと")
    
@app.route('/')
def health_check():
    """Renderのヘルスチェック用エンドポイント"""
    return "ML Bot Scheduler is running!" if bot else "ML Bot Initialization Failed.", 200

if __name__ == '__main__':
    # スケジューラを同期的に起動してから、Flaskアプリをメインスレッドで実行
    start_scheduler()
    
    app.run(host='0.0.0.0', port=WEB_SERVICE_PORT)
