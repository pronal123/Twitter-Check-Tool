import os
import json
from datetime import datetime
from flask import Flask, render_template, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
# dotenvは環境変数をローカルで読み込むために使用されます
from dotenv import load_dotenv

# --- 実践的なBOTロジックと定数をインポート ---
# futures_ml_bot.py から実際のクラスと定数をインポート
from futures_ml_bot import (
    FuturesMLBot, 
    fetch_advanced_metrics, 
    REPORT_FILENAME,
    MODEL_FILENAME
)

# ローカルテスト時に .env ファイルを読み込む (デプロイ環境では通常不要)
load_dotenv() 

# --- 環境変数と設定 ---
WEB_SERVICE_PORT = int(os.environ.get('PORT', 8080))
RETRAIN_INTERVAL_HOURS = int(os.environ.get('RETRAIN_INTERVAL_HOURS', 24)) # 24時間ごとに再学習
PREDICTION_INTERVAL_HOURS = int(os.environ.get('PREDICTION_INTERVAL_HOURS', 1)) # 1時間ごとに予測

app = Flask(__name__)
scheduler = BackgroundScheduler()

# 🚨 BOTの初期化 (グローバルインスタンス)
bot = None
try:
    # FuturesMLBotが初期化時にモデルのロードを試みます
    bot = FuturesMLBot() 
except Exception as e:
    # 致命的なエラーログ
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚨 致命的なBOT初期化エラー: {e}")
    
# --- 予測実行タスク (定時) ---
def run_prediction_and_report_generation():
    """予測を実行し、REPORT_FILENAMEにJSONレポートを保存する関数。"""
    if bot is None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚨 BOTインスタンスがありません。タスクスキップ。")
        return

    try:
        # モデルファイルが存在しない場合は、予測をスキップ
        if not os.path.exists(MODEL_FILENAME):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ モデルファイルが存在しません。予測をスキップし、再学習待ち。")
            return

        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙️ 予測タスク開始...")
        
        # NOTE: fetch_advanced_metricsは futures_ml_bot.py で引数なしのダミー実装のため、そのまま呼び出し
        advanced_data = fetch_advanced_metrics() 
        
        # 最新のOHLCVデータを取得 (100期間)
        # NOTE: FuturesMLBotがfetch_ohlcv_dataを持つようになりました
        df_latest = bot.fetch_ohlcv_data(days=100) 
        
        # 予測を実行し、レポートJSONを生成・保存
        bot.predict_and_report(df_latest, advanced_data)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 予測・レポート生成タスク完了。")
             
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚨 予測タスクエラー: {e}")

# --- モデル再学習タスク (定時) ---
def run_retrain_and_save():
    """モデルの再学習と構築を行う関数。"""
    if bot is None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚨 BOTインスタンスがありません。再学習スキップ。")
        return
        
    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🧠 再学習タスク開始...")
        
        # モデル学習のための長期データ (900期間) を取得
        df_long_term = bot.fetch_ohlcv_data(days=900) 
        # モデルを学習し、ファイルに保存
        bot.train_and_save_model(df_long_term)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 再学習タスク完了。")
        
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚨 致命的な再学習タスクエラーが発生しました: {e}")


# --- スケジューラの初期化と起動 ---
def start_scheduler():
    """APSchedulerを設定し、バックグラウンドで開始する"""
    if bot is None:
        print("⚠️ BOT初期化失敗のため、スケジューラは起動しません。")
        return

    print("--- スケジューラ設定開始 ---")

    # 🚨 【重要】BOT起動時に、最初の予測の前に必ずモデルを初回学習する
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 初回モデル構築を強制実行中...")
    run_retrain_and_save()
    print("✅ 初回モデル構築完了。")
    
    # 初回学習完了後、予測を実行
    run_prediction_and_report_generation()
    print("✅ 初回予測完了。")


    # ジョブの追加
    scheduler.add_job(func=run_prediction_and_report_generation, trigger='interval', hours=PREDICTION_INTERVAL_HOURS, id='prediction_job')
    scheduler.add_job(func=run_retrain_and_save, trigger='interval', hours=RETRAIN_INTERVAL_HOURS, id='retrain_job')

    scheduler.start()
    print(f"✅ スケジューラ起動済み。予測:{PREDICTION_INTERVAL_HOURS}時間ごと, 再学習:{RETRAIN_INTERVAL_HOURS}時間ごと")
    
# --- Flask Webサーバーのルーティング ---

@app.route('/')
def index():
    """メインダッシュボード (index.html) をレンダリングする"""
    return render_template('index.html', title="ML活用先物BOT分析レポート")

@app.route('/get_report')
def get_report():
    """最新の予測レポートJSONデータを返すエンドポイント。"""
    if not os.path.exists(REPORT_FILENAME):
         # レポートファイルが存在しない場合
         return jsonify({
             "status": "error", 
             "message": "レポートはまだ生成されていません。初期の学習と予測が完了するまでお待ちください。"
         }), 503
    
    try:
        with open(REPORT_FILENAME, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        return jsonify(report_data)
    except Exception as e:
        # JSONパースエラーなど、ファイルの読み込み中にエラーが発生した場合
        return jsonify({
            "status": "error", 
            "message": f"レポートデータの読み込み中にエラーが発生しました: {str(e)}"
        }), 500

@app.route('/report_status')
def report_status():
    """スケジューラーのステータス情報と次回の実行時間を返すエンドポイント。"""
    jobs = scheduler.get_jobs()
    
    # 次の予測実行時間を検索
    next_prediction_run = "N/A"
    next_training_run = "N/A"
    
    for job in jobs:
        if job.id == 'prediction_job' and job.next_run_time:
            next_prediction_run = job.next_run_time.strftime('%Y-%m-%d %H:%M:%S JST')
        if job.id == 'retrain_job' and job.next_run_time:
            next_training_run = job.next_run_time.strftime('%Y-%m-%d %H:%M:%S JST')

    status = {
        'status': '稼働中 (Scheduler running)',
        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S JST'),
        'next_prediction': next_prediction_run,
        'next_training': next_training_run
    }
    return jsonify(status)

# --- メイン実行ブロック ---
if __name__ == '__main__':
    # スケジューラを同期的に起動してから、Flaskアプリをメインスレッドで実行
    start_scheduler()
    
    print("🌐 Flask Webサーバーを起動中...")
    # use_reloader=False は、APSchedulerが二重起動するのを防ぐために推奨されます。
    app.run(host='0.0.0.0', port=WEB_SERVICE_PORT, use_reloader=False)
