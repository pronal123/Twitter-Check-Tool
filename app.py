# app.py (再学習間隔を4時間に修正)

import os
from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from dotenv import load_dotenv 
from futures_ml_bot import FuturesMLBot, fetch_advanced_metrics, FUTURES_SYMBOL

# ローカルテスト時に .env ファイルを読み込む
load_dotenv() 

# --- 環境変数設定 ---
WEB_SERVICE_PORT = int(os.environ.get('PORT', 8080))
# 🚨 修正: モデル再学習の間隔を、分から時間に変更し、初期値を4時間に設定
# 大量のデータ取得とモデル学習は時間がかかるため、安全のため4時間ごとに設定します。
RETRAIN_INTERVAL_HOURS = int(os.environ.get('RETRAIN_INTERVAL_HOURS', 4))
PREDICTION_INTERVAL_HOURS = int(os.environ.get('PREDICTION_INTERVAL_HOURS', 1))

app = Flask(__name__)
scheduler = BackgroundScheduler()

# BOTの初期化 (BOTインスタンスはグローバルに保持)
bot = None
try:
    # 認証情報が不足しているとここでエラーが発生
    bot = FuturesMLBot() 
except ValueError as e:
    print(f"致命的な初期化エラー: {e}")
    
# --- 予測実行タスク (定時) ---
def run_prediction_and_notify():
    """予測を実行し、Telegramに通知する関数"""
    if bot is None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚨 BOTインスタンスがありません。タスクスキップ。")
        return

    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙️ 予測タスク開始...")
        
        # 高度な分析指標を取得
        advanced_data = fetch_advanced_metrics(bot.exchange, FUTURES_SYMBOL)
        df_latest = bot.fetch_ohlcv_data(limit=100) 
        bot.predict_and_report(df_latest, advanced_data)
        
        print("✅ 予測・通知タスク完了。")
             
    except Exception as e:
        print(f"🚨 予測タスクエラー: {e}")

# --- モデル再学習タスク (定時) ---
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

    # 初回起動通知
    boot_message = (
        "✅ **BOT起動成功とスケジューラ設定完了**\n\n"
        f"サービス名: MEXC分析BOT (高度分析バージョン)\n"
        f"予測間隔: {PREDICTION_INTERVAL_HOURS}時間ごと\n"
        f"再学習間隔: {RETRAIN_INTERVAL_HOURS}時間ごと (モデル学習の安全性を確保)\n\n"
        "間もなく初回または定時予測タスクが実行されます。"
    )
    bot.send_telegram_notification(boot_message)

    # ジョブの追加
    scheduler.add_job(func=run_prediction_and_notify, trigger='interval', hours=PREDICTION_INTERVAL_HOURS, id='prediction_job')
    # 修正された間隔を使用
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
