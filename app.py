import datetime
import logging
import time
from threading import Thread

# Flask関連のインポート
from flask import Flask, render_template, jsonify

# スケジューラーのインポート
from flask_apscheduler import APScheduler

# ロギング設定
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# グローバル変数 (データとスケジューラーの状態を保持)
global_data = {
    "last_updated": "未実行",
    "data_range": "N/A",
    "data_count": 0,
    "scheduler_status": "初期化中"
}

# -----------------
# データ取得/処理関数 (トレードデータ取得のダミー関数)
# -----------------
def fetch_data(days_ago=900):
    """
    APIから過去のトレードデータを取得するダミー関数。
    """
    try:
        logging.info(f"APIから過去 {days_ago} 日間のデータ取得を試行中...")
        
        # 取得期間の計算
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days_ago)
        
        # 実際のAPI呼び出しの遅延をシミュレート
        time.sleep(2) 
        
        # 成功時のダミーデータ
        data_count = 1000 # ダミーのデータ件数
        
        # グローバルデータの更新
        global global_data
        global_data.update({
            "last_updated": end_date.strftime('%Y-%m-%d %H:%M:%S'),
            "data_range": f"{start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}",
            "data_count": data_count,
        })
        
        logging.info(f"データ取得が成功しました。期間: {global_data['data_range']}, 件数: {data_count}")
        return {"status": "success"}

    except Exception as e:
        logging.error(f"予期せぬデータ取得エラー: {e}")
        return {"status": "error", "message": str(e)}

def update_report_task():
    """定期的に実行されるレポート更新タスク。"""
    logging.info("スケジュールされたレポート更新タスク開始...")
    fetch_data()
    logging.info("レポート更新タスク完了。")

# -----------------
# Flaskアプリケーション本体
# -----------------
# 階層修正: template_folder='./' を設定し、テンプレートファイルをapp.pyと同じ階層からロードします。
app = Flask(__name__, template_folder='./')
app.config.update({
    'SCHEDULER_API_ENABLED': True
})
logging.info("🤖 FuturesMLBot初期化完了。")

# スケジューラーの初期化
scheduler = APScheduler()

@app.before_first_request
def initial_setup():
    """アプリケーションの初回リクエスト時にスケジューラーを設定し、実行を開始します。"""
    
    # 既存のジョブを削除（リロード対策）
    for job in scheduler.get_jobs():
        job.remove()
        
    # 定期実行ジョブの追加 (例: 1分ごとに実行)
    scheduler.add_job(
        id='scheduled_report_update',
        func=update_report_task,
        trigger='interval',
        minutes=1,
        max_instances=1,
        name='レポート定期更新'
    )
    
    # スケジューラーの起動
    if not scheduler.running:
        scheduler.init_app(app)
        scheduler.start()
        global_data["scheduler_status"] = "稼働中 (1分ごと)"
        logging.info("⏳ スケジューラーを起動し、初回レポート生成を実行します...")
        # 初回起動時のデータ取得
        update_report_task() 
    else:
        global_data["scheduler_status"] = "既に稼働中"
        logging.info("⏳ スケジューラーは既に稼働中です。")


@app.route('/')
def index():
    # index.htmlをapp.pyと同じ階層からロードします。
    # グローバルデータをテンプレートに渡します
    return render_template('index.html', title='ML活用先物BOT分析レポート', data=global_data)

@app.route('/status')
def status():
    # AJAXでデータを取得するためのエンドポイント
    return jsonify(global_data)

# -----------------
# サーバー起動 (開発用)
# -----------------
if __name__ == '__main__':
    # 開発環境で直接実行される場合
    # Note: 本番環境(Gunicorn)では 'app.run' は実行されません
    logging.info("🚀 Flaskアプリケーションを起動中...")
    app.run(host='0.0.0.0', port=8080)
