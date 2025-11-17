import os
import json
import threading
import time
from datetime import datetime, timedelta

# 追加: pandasがフォールバッククラス内で必要
import pandas as pd

# Flask for Web Interface and API
# 修正: template_folderを明示的に指定
from flask import Flask, render_template, jsonify, send_file
from flask_apscheduler import APScheduler 

# Custom ML Bot Logic
try:
    from futures_ml_bot import FuturesMLBot, REPORT_FILENAME, DAYS_LOOKBACK
except ImportError:
    # ロジックファイルが見つからない場合のフォールバッククラス
    class FuturesMLBot:
        def __init__(self):
            print("🚨 futures_ml_bot.pyが見つからないため、ML機能を無効にします。")
        def fetch_ohlcv_data(self, days): 
            # 空のDataFrameを返すように修正
            return pd.DataFrame() 
        def train_and_save_model(self, df): pass
        def predict_and_report(self, df, advanced_data): return {}
        def fetch_advanced_metrics(self): return {'status': 'Unavailable'}
    REPORT_FILENAME = 'latest_report.json'
    DAYS_LOOKBACK = 900


# --- アプリケーション設定 ---
class Config:
    """APSchedulerの設定"""
    SCHEDULER_API_ENABLED = True
    # UTCではなく、ローカルタイムゾーン (JST)でスケジュールを設定
    SCHEDULER_TIMEZONE = "Asia/Tokyo" 
    # MLレポートの更新頻度 (例: 毎日午前9時)
    REPORT_UPDATE_HOUR = 9
    REPORT_UPDATE_MINUTE = 0

# 修正: template_folder='templates' を指定
app = Flask(__name__, template_folder='templates') 
app.config.from_object(Config())

# グローバルインスタンス
scheduler = APScheduler()
ml_bot = FuturesMLBot()

# ロック機構: スケジューラがレポート処理中に競合を避けるため
report_lock = threading.Lock() 

# --- スケジュールされたタスク ---

def update_ml_report():
    """
    定期的に実行されるタスク：データ取得、モデル学習、予測、レポート保存を行います。
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🤖 スケジュールされたレポート更新タスク開始...")
    
    # データの取得 (学習用と予測用を兼ねる)
    df = ml_bot.fetch_ohlcv_data(days=DAYS_LOOKBACK) 
    
    if df.empty:
        print("🚨 データが取得できないため、レポート更新をスキップします。")
        return

    # Advanced Dataの取得
    try:
        advanced_metrics = ml_bot.fetch_advanced_metrics() 
    except AttributeError:
        # fetch_advanced_metrics が bot にない場合のフォールバック (テスト環境用)
        advanced_metrics = {'status': 'Unavailable'} 

    with report_lock:
        # モデルの学習 (ファイルが存在しない場合のみ、または定期的に再学習)
        ml_bot.train_and_save_model(df)
        
        # 最新の予測の実行とレポートのファイル保存
        ml_bot.predict_and_report(df, advanced_metrics)
        
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ レポート更新タスク完了。")


# --- Flask ルーティング ---

@app.route('/')
def index():
    """メインダッシュボードページをレンダリングします。"""
    # NOTE: テンプレートファイルは、このコードと同時に生成された `index.html` または
    # 以前に生成された `index.html` に依存します。
    return render_template('index.html', title='ML活用先物BOT分析レポート')

@app.route('/get_report')
def get_report():
    """最新の予測レポートJSONファイルを返します。"""
    with report_lock:
        if os.path.exists(REPORT_FILENAME):
            with open(REPORT_FILENAME, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            # JSONを返す
            return jsonify(report_data)
        else:
            # レポートがまだ生成されていない場合は、503 Service Unavailableを返す
            return jsonify({"error": "Report not yet generated", "message": "MLレポートがまだ生成されていません。数分後にリロードしてください。"}), 503

@app.route('/report_status')
def report_status():
    """スケジューラの状態と次回の実行時間を返します。"""
    
    status = "稼働中"
    next_run_time_str = "未定" # デフォルト値

    job = scheduler.get_job('ml_report_job')

    if job:
        status = "スケジューラ稼働中"
        next_run_time = job.next_run_time
        
        if next_run_time:
            # タイムゾーンをJSTに変換してフォーマット
            next_run_time_jst = next_run_time.astimezone(app.config['SCHEDULER_TIMEZONE'])
            next_run_time_str = next_run_time_jst.strftime('%Y/%m/%d %H:%M:%S JST')
        else:
            status = "スケジューラ停止中またはジョブ実行待ち"
            
    else:
        status = "スケジューラ未起動"

    return jsonify({
        "status": status,
        "next_prediction": next_run_time_str
    })


# --- アプリケーションの初期化と実行 ---

if __name__ == '__main__':
    # スケジューラの初期化
    scheduler.init_app(app)
    
    # MLレポート更新ジョブを定義 (毎日Config.REPORT_UPDATE_HOURに実行)
    scheduler.add_job(
        id='ml_report_job',
        func=update_ml_report,
        trigger='cron',
        hour=app.config['REPORT_UPDATE_HOUR'],
        minute=app.config['REPORT_UPDATE_MINUTE'],
        timezone=app.config['SCHEDULER_TIMEZONE'],
        replace_existing=True
    )
    
    # サーバー起動時にもすぐに実行する (レポートがない場合)
    def run_initial_job():
        # 初回実行時、レポートファイルがない場合にのみ実行
        if not os.path.exists(REPORT_FILENAME):
            print("⏳ 初回起動時のレポート生成を実行します...")
            update_ml_report()
            
    # メインスレッドとは別に初期ジョブを実行
    initial_thread = threading.Thread(target=run_initial_job)
    initial_thread.start()
    
    # スケジューラ起動
    scheduler.start()
    
    print("🚀 Flaskアプリケーションを起動中...")
    # NOTE: ポートは環境によって変わる可能性があるため、適切なポートを使用
    # use_reloader=Falseを設定することで、初期スレッドが二重に実行されるのを防ぎます
    app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)
