# app.py

import os
import threading
from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
# 🚨 作成したBOTクラスと指標取得関数をインポート
from futures_ml_bot import FuturesMLBot, fetch_futures_metrics, FUTURES_SYMBOL

# --- 設定情報 (Render環境変数から読み込み) ---
WEB_SERVICE_PORT = int(os.environ.get('PORT', 8080))
RETRAIN_INTERVAL_HOURS = int(os.environ.get('RETRAIN_INTERVAL_HOURS', 24))
PREDICTION_INTERVAL_HOURS = int(os.environ.get('PREDICTION_INTERVAL_HOURS', 1))

app = Flask(__name__)
scheduler = BackgroundScheduler()
bot = FuturesMLBot() 

# --- 予測実行タスク (1時間ごと) ---
def run_prediction_and_notify():
    """予測を実行し、Telegramに通知する関数"""
    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙️ 予測タスク開始...")
        
        # 1. リアルタイム先物指標の取得
        futures_data = fetch_futures_metrics(bot.exchange, FUTURES_SYMBOL)
        
        # 2. 最新OHLCVデータの取得と予測
        df_latest = bot.fetch_ohlcv_data(limit=100) 
        report_success = bot.predict_and_report(df_latest, futures_data)
        
        if report_success:
             print("✅ 予測・通知タスク完了。")
        else:
             print("⚠️ 予測・通知タスクはエラーのためスキップされました。")
             
    except Exception as e:
        print(f"🚨 致命的な予測タスクエラーが発生しました: {e}")

# --- モデル再学習タスク (24時間ごと) ---
def run_retrain_and_improve():
    """モデルの再学習と構築を行う関数"""
    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🧠 再学習タスク開始...")
        
        # 1. 長期データを取得 (例: 過去2000本の4h足データ)
        # ⚠️ Renderの無料ティアの制限に注意。データ取得が長時間にわたる場合は、外部データベースを利用すべきです。
        df_long_term = bot.fetch_ohlcv_data(limit=2000) 
        
        # 2. モデルの学習と保存
        bot.train_and_save_model(df_long_term)
        
    except Exception as e:
        print(f"🚨 致命的な再学習タスクエラーが発生しました: {e}")

# --- スケジューラの初期化と起動 ---
def start_scheduler():
    """APSchedulerを設定し、バックグラウンドで開始する"""
    print("--- スケジューラ設定開始 ---")

    # 🚨 予測ジョブの追加 (PREDICTION_INTERVAL_HOURS ごと)
    scheduler.add_job(func=run_prediction_and_notify, trigger='interval', hours=PREDICTION_INTERVAL_HOURS, id='prediction_job')
    
    # 🚨 再学習ジョブの追加 (RETRAIN_INTERVAL_HOURS ごと)
    scheduler.add_job(func=run_retrain_and_improve, trigger='interval', hours=RETRAIN_INTERVAL_HOURS, id='retrain_job')

    scheduler.start()
    print(f"✅ スケジューラ起動済み。予測:{PREDICTION_INTERVAL_HOURS}時間ごと, 再学習:{RETRAIN_INTERVAL_HOURS}時間ごと")
    
@app.route('/')
def health_check():
    """Renderのヘルスチェック用エンドポイント"""
    return "ML Bot Scheduler is running!", 200

if __name__ == '__main__':
    # Webサービスが起動した後にスケジューラを起動し、タスクを継続実行
    threading.Thread(target=start_scheduler).start()
    
    app.run(host='0.0.0.0', port=WEB_SERVICE_PORT)
