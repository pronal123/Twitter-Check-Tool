# app.py (起動時にモデルを強制学習・初回予測を行い、次回時刻を通知する完全版)

import os
from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
# 🚨 修正: datetime, timedelta, timezoneをインポート
from datetime import datetime, timedelta, timezone 
from dotenv import load_dotenv

# futures_ml_botモジュールから必要なコンポーネントをインポート
from futures_ml_bot import FuturesMLBot, fetch_advanced_metrics, FUTURES_SYMBOL, MODEL_FILENAME

# ローカルテスト時に .env ファイルを読み込む
# デプロイ環境では効果はありません
load_dotenv() 

# --- 環境変数設定 ---
# 環境変数 'PORT' が設定されていない場合、8080を使用
WEB_SERVICE_PORT = int(os.environ.get('PORT', 8080))
# 環境変数から再学習間隔を取得。デフォルトは24時間
RETRAIN_INTERVAL_HOURS = int(os.environ.get('RETRAIN_INTERVAL_HOURS', 24))
# 環境変数から予測間隔を取得。デフォルトは1時間
PREDICTION_INTERVAL_HOURS = int(os.environ.get('PREDICTION_INTERVAL_HOURS', 1))

app = Flask(__name__)
scheduler = BackgroundScheduler()

# BOTの初期化 (BOTインスタンスはグローバルに保持)
bot = None
try:
    # 認証情報が不足しているとValueErrorが発生
    bot = FuturesMLBot() 
except ValueError as e:
    # APIキー不足などの致命的なエラーをコンソールに出力
    print(f"致命的な初期化エラー: {e}")
    
# --- ヘルパー関数: 次回実行時間の計算と通知 (新規追加) ---
def calculate_and_notify_next_run(bot_instance, interval_hours):
    """次の実行時間を計算し、Telegramで通知する"""
    if bot_instance is None: return

    try:
        # 現在時刻 (タスクが完了した時刻)
        now_utc = datetime.now(timezone.utc)
        
        # 次の実行時刻は、現在時刻 + インターバル
        next_run_utc = now_utc + timedelta(hours=interval_hours)
        
        # JSTに変換してフォーマット (PythonのdatetimeはローカルタイムゾーンをNoneとして扱う)
        # タイムゾーン情報を付加
        JST = timezone(timedelta(hours=9))
        next_run_jst = next_run_utc.astimezone(JST).strftime('%Y-%m-%d %H:%M JST')
        
        # 通知メッセージ
        message = (
            f"⏰ <b>次のレポート通知予定時刻:</b>\n"
            f"次の予測タスクは <b>{interval_hours}時間後</b> に実行されます。\n"
            f"予定時刻 (JST): <b>{next_run_jst}</b>"
        )
        bot_instance.send_telegram_notification(message)
        print(f"✅ 次回実行通知を送信しました。予定時刻: {next_run_jst}")
        
    except Exception as e:
        print(f"🚨 次回実行通知の計算/送信エラー: {e}")


# --- 予測実行タスク (定時) ---
def run_prediction_and_notify():
    """予測を実行し、Telegramに通知する関数"""
    if bot is None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚨 BOTインスタンスがありません。タスクスキップ。")
        return

    try:
        # モデルファイルが存在しない場合は、予測をスキップする（初回強制実行でカバーされているはず）
        if not os.path.exists(MODEL_FILENAME):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ モデルファイル '{MODEL_FILENAME}' が存在しません。予測をスキップし、再学習待ち。")
            return

        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙️ 予測タスク開始...")
        
        # 高度な分析指標を取得
        advanced_data = fetch_advanced_metrics(bot.exchange, FUTURES_SYMBOL)
        # 最新のOHLCVデータを取得 (100期間)
        df_latest = bot.fetch_ohlcv_data(limit=100) 
        # 予測を実行し、Telegramにレポートを送信
        bot.predict_and_report(df_latest, advanced_data)
        
        print("✅ 予測・通知タスク完了。")
        
        # 🚨 修正: レポート完了後、次の実行時間を通知
        calculate_and_notify_next_run(bot, PREDICTION_INTERVAL_HOURS)
             
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚨 予測タスクエラー: {e}")

# --- モデル再学習タスク (定時) ---
def run_retrain_and_improve():
    """モデルの再学習と構築を行う関数"""
    if bot is None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚨 BOTインスタンスがありません。再学習スキップ。")
        return
        
    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🧠 再学習タスク開始...")
        
        # モデル学習のための長期データ (2000期間) を取得
        df_long_term = bot.fetch_ohlcv_data(limit=2000) 
        # モデルを学習し、ファイルに保存 (MODEL_FILENAMEで定義されたパス)
        bot.train_and_save_model(df_long_term)
        
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚨 致命的な再学習タスクエラーが発生しました: {e}")

# --- スケジューラの初期化と起動 ---
def start_scheduler():
    """APSchedulerを設定し、バックグラウンドで開始する"""
    if bot is None:
        print("⚠️ BOT初期化失敗のため、スケジューラは起動しません。")
        return

    print("--- スケジューラ設定を開始します ---")

    # 🚨 【重要修正】BOT起動時に、最初の予測の前に必ずモデルを初回学習する
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 初回モデル構築を強制実行中...")
    run_retrain_and_improve()
    print("✅ 初回モデル構築が完了しました。")
    
    # 🚨 初回起動時に初回レポートを強制実行
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 初回レポートを強制実行中...")
    # run_prediction_and_notify()内で次回時刻通知も実行される
    run_prediction_and_notify()
    print("✅ 初回レポート送信が完了しました。")

    # 初回起動通知（ステータスのみ）
    boot_message = (
        "✅ **BOT起動成功とスケジューラ設定完了**\n\n"
        f"サービス名: MEXC分析BOT (高度分析バージョン)\n"
        f"予測間隔: {PREDICTION_INTERVAL_HOURS}時間ごと\n"
        f"再学習間隔: {RETRAIN_INTERVAL_HOURS}時間ごと\n\n"
        "スケジューラが起動しました。"
    )
    bot.send_telegram_notification(boot_message)
    

    # ジョブの追加
    scheduler.add_job(func=run_prediction_and_notify, trigger='interval', hours=PREDICTION_INTERVAL_HOURS, id='prediction_job')
    scheduler.add_job(func=run_retrain_and_improve, trigger='interval', hours=RETRAIN_INTERVAL_HOURS, id='retrain_job')

    scheduler.start()
    print(f"✅ スケジューラが起動しました。予測:{PREDICTION_INTERVAL_HOURS}時間ごと, 再学習:{RETRAIN_INTERVAL_HOURS}時間ごと")

@app.route('/')
def health_check():
    """Renderなどのデプロイサービスでのヘルスチェック用エンドポイント"""
    # スケジューラが起動しているか、BOTが初期化されているかを返す
    return "ML Bot Scheduler is running!" if bot else "ML Bot Initialization Failed.", 200

if __name__ == '__main__':
    # スケジューラを同期的に起動してから、Flaskアプリをメインスレッドで実行
    start_scheduler()
    
    app.run(host='0.0.0.0', port=WEB_SERVICE_PORT)
