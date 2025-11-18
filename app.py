import datetime
import logging
import time
import os
import requests 
from threading import Thread
# import random # <-- 削除: ランダムな予測生成を停止

# Flask関連のインポート
from flask import Flask, render_template, jsonify
from flask_apscheduler import APScheduler # スケジューラーをインポート

# -----------------
# Telegram Bot設定
# -----------------
# 🚨 実際のBotトークンとチャットIDに置き換えてください
# 環境変数から取得。設定がない場合はダミー値を使用します。
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE') 
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '-1234567890') 
TELEGRAM_API_URL = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'

# -----------------
# ロギング設定
# -----------------
# ログ形式を設定
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# -----------------
# アプリケーション初期化
# -----------------
# Flaskアプリのインスタンスを作成
# template_folderをカレントディレクトリ('.')に指定し、
# app.pyと同じ階層の index.html をテンプレートとして読み込むように設定
app = Flask(__name__, template_folder='.') 
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
# Telegram通知関数
# -----------------
def send_telegram_message(message: str):
    """
    指定されたメッセージをTelegramチャットに送信します。
    """
    if TELEGRAM_BOT_TOKEN == 'YOUR_BOT_TOKEN_HERE' or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram BotトークンまたはチャットIDが設定されていません。通知をスキップします。")
        return

    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown'
    }
    
    try:
        # HTTP POSTリクエストを送信
        response = requests.post(TELEGRAM_API_URL, data=payload, timeout=10) # 10秒のタイムアウトを設定
        response.raise_for_status() # 4xx, 5xxエラーを発生させる
        logging.info("Telegram通知を送信しました。")
    except requests.exceptions.RequestException as e:
        # リクエスト失敗時のエラー処理
        logging.error(f"Telegram通知の送信に失敗しました: {e}")
        # response変数が定義されているかチェックし、レスポンス内容をログに出力
        if 'response' in locals() and response.text:
            logging.error(f"Telegram APIレスポンス: {response.text}")


# -----------------
# スケジューリングタスク
# -----------------
def update_report_data():
    """定期的に実行されるタスク：データ取得とレポート更新のシミュレーション"""
    global global_data
    global data_item_count

    logging.info("スケジュールされたレポート更新タスク開始...")
    
    # 1. データ取得のシミュレーション 
    days_to_fetch = 900
    logging.info(f"APIから過去 {days_to_fetch} 日間のデータ取得を試行中...")
    
    # 起動を高速化するため、シミュレーションの待機時間（time.sleep(2)）は削除済み
    
    # 2. ダミーデータの更新
    data_item_count += 1000 # 毎回1000件ずつデータが増加したと仮定
    now = datetime.datetime.now()
    
    # 3. グローバル状態の更新
    last_updated_str = now.strftime('%Y-%m-%d %H:%M:%S')
    
    global_data['last_updated'] = last_updated_str
    global_data['data_count'] = data_item_count
    global_data['scheduler_status'] = '稼働中'
    
    logging.info(f"データ取得が成功しました。期間: {global_data['data_range']}, 件数: {global_data['data_count']}")
    logging.info("レポート更新タスク完了。")
    
    # 4. Telegram通知の実行
    
    # BTC予測のシミュレーション (実践的なシミュレーションロジックを導入)
    data_count = global_data['data_count']
    timeframes = ["1h", "4h", "12h", "24h"]
    outcomes = ["上昇 📈", "下降 📉", "レンジ ↔️"]
    predictions = {}
    
    # --- 実践的なシミュレーションロジック ---
    # 1h: 1000件の倍数に基づき、短期的なモメンタムをシミュレート
    if (data_count // 1000) % 3 == 0:
        predictions["1h"] = outcomes[0] # 上昇
        short_term_bias = "上昇"
    elif (data_count // 1000) % 3 == 1:
        predictions["1h"] = outcomes[1] # 下降
        short_term_bias = "下降"
    else:
        predictions["1h"] = outcomes[2] # レンジ
        short_term_bias = "レンジ"
        
    # 4h: 奇数/偶数でRSIの過熱感をシミュレート
    if data_count % 2 != 0:
        predictions["4h"] = outcomes[1] # 下降 (RSIが買われすぎ水準をシミュレート)
    else:
        predictions["4h"] = outcomes[0] # 上昇
        
    # 12h: データカウントの末尾でMACDのクロスをシミュレート
    if data_count % 10 < 5:
        predictions["12h"] = outcomes[0] # 上昇
    else:
        predictions["12h"] = outcomes[2] # レンジ
        
    # 24h: 長期の移動平均線の傾きをシミュレート
    if data_count > 5000:
        predictions["24h"] = outcomes[0] # 上昇
    else:
        predictions["24h"] = outcomes[2] # レンジ
    # --- ロジック終了 ---
    
    # メッセージ本文の組み立て
    analysis_lines = []
    for tf in timeframes:
        # Markdownで太字に装飾
        analysis_lines.append(f"• {tf}後予測: *{predictions[tf]}*") 
        
    analysis_text = "\n".join(analysis_lines)
    
    # 総合サマリーの抽出
    # 短期（1h）の傾向を抽出（例: 「上昇 📈」から「上昇」を取得）
    short_term_trend = predictions['1h'].split(' ')[0]
    long_term_trend = predictions['24h'].split(' ')[0]
    
    formatted_data_count = f"{data_item_count:,}"
    
    report_message = (
        f"🚨 *BTC詳細分析レポート (ML BOT)* 🚨\n\n"
        f"📅 最終データ更新: `{last_updated_str}`\n"
        f"📊 処理データ件数: *{formatted_data_count}* 件\n\n"
        f"--- *BTC 価格動向予測* ---\n"
        f"{analysis_text}\n\n"
        f"💡 *総合サマリー*:\n"
        f"短期（1h）は*{short_term_trend}* 傾向、長期（24h）は*{long_term_trend}* 傾向です。\n"
        f"現在のデータセットに基づき、{short_term_bias}バイアスが確認されています。\n"
        f"_※ 予測は分析シミュレーションに基づき、定期的に更新されます。_"
    )
    
    # スレッド化して通知関数を呼び出し
    Thread(target=send_telegram_message, args=(report_message,)).start()


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
# Gunicorn環境でscheduler.runningのチェックは非常に重要です
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
# サーバー起動 (ローカル開発用ブロックはGunicornの仕様により削除済み)
# -----------------
# Gunicornが直接 app:app を読み込むため、ここに app.run() は含めません。
