import os
import threading
from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
# 🚨 作成したBOTクラスと指標取得関数をインポート
from futures_ml_bot import FuturesMLBot, fetch_futures_metrics 

# --- 設定情報 ---
WEB_SERVICE_PORT = int(os.environ.get('PORT', 8080))
RETRAIN_INTERVAL_HOURS = 24 
PREDICTION_INTERVAL_HOURS = 1

app = Flask(__name__)
scheduler = BackgroundScheduler()
bot = FuturesMLBot() 

def run_prediction_and_notify():
    """予測を実行し、Telegramに通知する関数 (1時間ごと)"""
    try:
        # 1. リアルタイム先物指標の取得
        futures_data = fetch_futures_metrics(bot.exchange, bot.FUTURES_SYMBOL)
        
        # 2. 最新OHLCVデータの取得と予測
        df_latest = bot.fetch_ohlcv_data(limit=100) 
        report = bot.predict_and_report(df_latest, futures_data)
        
        # 3. 通知
        bot.send_telegram_notification(report)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙️ 予測タスク完了。")
    except Exception as e:
        print(f"🚨 予測タスクエラー: {e}")

def run_retrain_and_improve():
    """モデルの再学習と構築を行う関数 (24時間ごと)"""
    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🧠 再学習タスク開始...")
        
        # 1. 長期データを取得 (例: 過去2000本の4h足データ)
        df_long_term = bot.fetch_ohlcv_data(limit=2000) 
        
        # 2. モデルの学習と保存
        bot.train_and_save_model(df_long_term)
        
        print("✅ モデル再学習完了し、ファイルに保存しました。")
    except Exception as e:
        print(f"🚨 再学習タスクエラー: {e}")

def start_scheduler():
    """APSchedulerを設定し、バックグラウンドで開始する"""
    scheduler.add_job(func=run_prediction_and_notify, trigger='interval', hours=PREDICTION_INTERVAL_HOURS, id='prediction_job')
    scheduler.add_job(func=run_retrain_and_improve, trigger='interval', hours=RETRAIN_INTERVAL_HOURS, id='retrain_job')

    scheduler.start()
    print("✅ スケジューラ起動済み。")
    
@app.route('/')
def health_check():
    """Renderのヘルスチェック用エンドポイント"""
    return "ML Bot Scheduler is running!", 200

if __name__ == '__main__':
    # Webサービスが起動した後にスケジューラを起動し、タスクを継続実行
    threading.Thread(target=start_scheduler).start()
    
    app.run(host='0.0.0.0', port=WEB_SERVICE_PORT)
