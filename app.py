import os
import requests
import time
import psycopg2
from flask import Flask, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from urllib.parse import urlparse
from requests.exceptions import HTTPError, RequestException 
from datetime import datetime, timedelta # datetimeモジュールを追加

# --- 環境変数から設定を取得 ---
# これらの値はRenderの環境変数として設定する必要があります
TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
DATABASE_URL = os.environ.get("DATABASE_URL")

# 環境変数の必須チェック
if not all([TWITTER_BEARER_TOKEN, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DATABASE_URL]):
    print("エラー: 必要な環境変数が不足しています。デプロイ設定を確認してください。")
    
# --- データベース接続関数 ---
def get_db_connection():
    """環境変数からDBに接続する"""
    # Render PostgreSQLの接続情報解析
    result = urlparse(DATABASE_URL)
    
    conn = psycopg2.connect(
        database=result.path[1:],
        user=result.username,
        password=result.password,
        host=result.hostname,
        port=result.port
    )
    return conn

# --- DB初期化（テーブル作成） ---
def setup_database():
    """BOTの実行に必要なテーブルを初期化（存在しない場合のみ作成）"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 'checked_tweets' テーブルを作成し、ツイートIDをプライマリキーとして重複を防止
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS checked_tweets (
                tweet_id BIGINT PRIMARY KEY,
                created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        print("DBセットアップ完了: checked_tweetsテーブルを確認/作成しました。")
    except Exception as e:
        print(f"DBセットアップ中にエラー: {e}")
    finally:
        if conn:
            conn.close()

# --- DBヘルパー関数 ---
def is_tweet_checked(tweet_id):
    """ツイートIDがDBに存在するかチェックする"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM checked_tweets WHERE tweet_id = %s", (str(tweet_id),))
        return cursor.fetchone() is not None
    except Exception as e:
        print(f"DB読み取りエラー (is_tweet_checked): {e}")
        return True # エラー時は安全のため True を返し、処理をスキップ
    finally:
        if conn:
            conn.close()

def mark_tweet_as_checked(tweet_id):
    """ツイートIDをDBに記録する"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # ON CONFLICT DO NOTHING: 競合が発生した場合（既に存在する場合）は何もしない
        cursor.execute("INSERT INTO checked_tweets (tweet_id) VALUES (%s) ON CONFLICT (tweet_id) DO NOTHING", (str(tweet_id),))
        conn.commit()
    except Exception as e:
        print(f"DB書き込みエラー (mark_tweet_as_checked): {e}")
    finally:
        if conn:
            conn.close()


# --- X（Twitter）アカウントのステータス確認関数 ---
def check_twitter_account(user_id, old_username):
    """アカウントの存在を確認し、削除/凍結をチェックする"""
    url = f"https://api.twitter.com/2/users/{user_id}"
    headers = {"Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"}
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            return None # アカウントは存在する
        elif response.status_code == 404:
            # 404 Not Found はアカウント削除または凍結
            return f"アカウント削除または凍結❌: (@{old_username})"
        else:
            # その他APIエラー
            data = response.json()
            error_msg = data.get('detail', f"Unknown Error (Status: {response.status_code})")
            return f"APIエラー⚠️ (Status: {response.status_code}): (@{old_username}) - {error_msg}"
            
    except RequestException as e:
        return f"ネットワークエラー: {e.__class__.__name__}"


# --- Telegramにメッセージを送信する関数 ---
def send_telegram_message(message):
    """Telegram Bot APIで指定のチャットIDにメッセージを送信する"""
    telegram_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown',
        'disable_web_page_preview': True
    }
    
    try:
        response = requests.post(telegram_url, json=payload)
        response.raise_for_status()
        print(f"Telegram通知成功: {message[:20]}...")
    except RequestException as e:
        print(f"Telegram通知失敗 (ネットワーク/API): {e}")

# --- メインの定期実行ロジック ---
def scheduled_check():
    """仮想通貨関連の当選ツイートを検索し、アカウントが存在しない当選者をチェックする"""
    print(f"--- 仮想通貨特化検知実行開始: {time.ctime()} ---")
    
    # 検索クエリの定義: 仮想通貨特化
    crypto_keywords = 'BTC OR ETH OR NFT OR エアドロ OR GiveAway OR Airdrop OR 仮想通貨 OR 暗号資産'
    query = f'("{crypto_keywords}") ("当選" OR "DM" OR "おめでとう" OR "配布") has:mentions lang:ja -filter:retweets -filter:replies'
    
    # APIの制約と効率化のため、最新の100件をチェック
    search_url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&max_results=100&tweet.fields=entities&expansions=author_id"
    headers = {"Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"}
    
    try:
        # X APIへのリクエストを試行し、4xx/5xxの場合はHTTPError例外を発生させる
        response = requests.get(search_url, headers=headers)
        response.raise_for_status() 

        data = response.json()
        
        if 'data' not in data:
            print("検索結果なし。")
            return

        for tweet in data['data']:
            tweet_id = int(tweet['id'])
            
            # DBでチェック済みか確認し、重複ならスキップ
            if is_tweet_checked(tweet_id):
                continue
            
            tweet_author_id = tweet.get('author_id', '不明')
            tweet_url = f"https://twitter.com/i/web/status/{tweet_id}"
            
            # ツイートからメンションされているユーザーを抽出
            if 'entities' in tweet and 'mentions' in tweet['entities']:
                for mention in tweet['entities']['mentions']:
                    mentioned_id = mention['id']
                    mentioned_username = mention['username']
                    
                    # メンションされたユーザーのアカウント状態をチェック
                    detection_message = check_twitter_account(mentioned_id, mentioned_username)
                    
                    if detection_message:
                        # 異常が検知された場合、Telegramに通知
                        notification_text = (
                            f"【仮想通貨当選者 アカウント無し検知】\n"
                            f"異常内容: {detection_message}\n"
                            f"該当ツイートの主催者ID: {tweet_author_id}\n"
                            f"該当ツイート: [ツイートはこちら]({tweet_url})"
                        )
                        send_telegram_message(notification_text)
                    
                    time.sleep(1) # APIのレート制限回避のため待機
            
            # 処理後、このツイートIDをDBに「チェック済み」として記録
            mark_tweet_as_checked(tweet_id)

    except HTTPError as e:
        # 4xx (Client Error) や 5xx (Server Error) の特定のエラー処理
        status_code = e.response.status_code
        error_msg = f"🚨X APIエラーが発生 (Status: {status_code}): TWITTER_BEARER_TOKENまたは権限を確認してください。"
        send_telegram_message(error_msg)
        print(error_msg)
        
    except RequestException as e:
        # ネットワークレベルのエラー処理
        error_msg = f"🚨ネットワーク接続エラーが発生: {e.__class__.__name__}"
        send_telegram_message(error_msg)
        print(error_msg)
        
    except Exception as e:
        # その他の予期せぬエラー処理
        error_msg = f"🚨予期せぬ実行時エラー: {e.__class__.__name__}: {str(e)}"
        send_telegram_message(error_msg)
        print(error_msg)


# ----------------------------------------------------
# --- FLASKとSCHEDULERの設定 ---
# ----------------------------------------------------

app = Flask(__name__)

# アプリ起動時にDBセットアップを実行
try:
    setup_database()
    scheduler = BackgroundScheduler()
    
    # 実行開始時刻を「現在から1日前の過去の時刻」に設定し、起動直後にジョブが実行されるように強制します。
    start_time = datetime.now() - timedelta(days=1)
    
    scheduler.add_job(
        scheduled_check, 
        'interval', 
        seconds=900, 
        start_date=start_time.strftime('%Y-%m-%d %H:%M:%S') # 過去の時刻を強制的に設定
    ) 
    
    scheduler.start()
except Exception as e:
    print(f"BOT初期化失敗: {e}")

# Web Serviceとしてプロセスを維持するためのシンプルなルート
@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "service": "X Crypto Winner Compliance BOT",
        "check_interval": "15 minutes"
    })

if __name__ == '__main__':
    # Renderデプロイ時にはGunicornがこのプロセスを起動します
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))
