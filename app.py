import datetime
import logging
import time
import os
from threading import Thread

# Flask関連のインポート
from flask import Flask, render_template, jsonify
from flask_apscheduler import APScheduler # <--- スケジューラーをインポート

# -----------------
# ロギング設定
# -----------------
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# -----------------
# アプリケーション初期化
# -----------------
# Flaskアプリのインスタンスを作成
# 修正点: template_folderの指定を削除し、Flaskがルートディレクトリの
# index.htmlをテンプレートとして使用できるようにします。
app = Flask(__name__) 
# スケジューラーのインスタンスを作成
scheduler = APScheduler()

# ダミーデータとグローバル状態
global_data = {
    'last_updated': 'N/A',
    'data_range': '2023-01-01 - 2025-11-18', # 初期ダミー期間
    'data_count': 0,
    'scheduler_status': '初期化中'
}
data_item_count = 0

# -----------------
# スケジューリングタスク
# -----------------
def update_report_data():
    """定期的に実行されるタスク：データ取得とレポート更新のシミュレーション"""
    global global_data
    global data_item_count

    logging.info("スケジュールされたレポート更新タスク開始...")
    
    # 1. データ取得のシミュレーション (ここではダミーで900日間としています)
    days_to_fetch = 900
    logging.info(f"APIから過去 {days_to_fetch} 日間のデータ取得を試行中...")
    
    # ダミー処理時間（2秒）
    time.sleep(2) 
    
    # 2. ダミーデータの更新
    data_item_count += 1000 # 毎回1000件ずつデータが増加したと仮定
    now = datetime.datetime.now()
    
    # 3. グローバル状態の更新
    global_data['last_updated'] = now.strftime('%Y-%m-%d %H:%M:%S')
    global_data['data_count'] = data_item_count
    global_data['scheduler_status'] = '稼働中'
    
    logging.info(f"データ取得が成功しました。期間: {global_data['data_range']}, 件数: {global_data['data_count']}")
    logging.info("レポート更新タスク完了。")


# -----------------
# ルート（エンドポイント）
# -----------------
@app.route('/')
def index():
    """ダッシュボードの表示"""
    # テンプレート名を参照 (ルートにある index.html を使用)
    return render_template('index.html', title='ML BOT分析レポート ダッシュボード', data=global_data)

# -----------------
# スケジューラーの初期設定と開始
# -----------------
if not scheduler.running:
    # スケジューラー設定
    app.config.update({
        'SCHEDULER_JOBSTORES': {
            'default': {'type': 'memory'}
        },
        'SCHEDULER_EXECUTORS': {
            'default': {'type': 'threadpool', 'max_workers': 20}
        },
        'SCHEDULER_API_ENABLED': False # API経由での制御を無効化
    })
    
    # アプリケーションにスケジューラーを登録
    scheduler.init_app(app)
    
    # 1分間隔でジョブを追加
    scheduler.add_job(id='report_update_job', func=update_report_data, 
                      trigger='interval', minutes=1, replace_existing=True)
    
    # スケジューラーを開始
    scheduler.start()
    logging.info("✅ スケジューラーを開始しました。")

# 最初のデータロードを強制的に実行し、初期表示に備える
update_report_data()


# -----------------
# サーバー起動 (ローカル開発用)
# -----------------
if __name__ == '__main__':
    # 環境変数 'PORT' が設定されていればそれを使用し、なければデフォルトの8080を使用
    port = int(os.environ.get('PORT', 8080))
    
    logging.info(f"🚀 Flaskアプリケーションを起動中... (ポート: {port})")
    
    # ホストを '0.0.0.0' にバインドし、指定されたポートでサーバーを起動
    # **注意**: 本番環境では Gunicorn (requirements.txtに含まれている) などのWSGIサーバーを使用してください。
    # 例: gunicorn app:app
    app.run(host='0.0.0.0', port=port, debug=False)
