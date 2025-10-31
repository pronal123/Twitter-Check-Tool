import os
import requests
import time
import psycopg2
from flask import Flask, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from urllib.parse import urlparse
from requests.exceptions import HTTPError, RequestException 
from datetime import datetime, timedelta
import threading 

# X API Base URL
X_API_URL = "https://api.twitter.com/2"

# --- 環境変数から設定を取得 ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
DATABASE_URL = os.environ.get("DATABASE_URL")
BEARER_TOKEN = os.environ.get("BEARER_TOKEN")

if not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DATABASE_URL]):
    print("エラー: 必要な環境変数が不足しています。デプロイ設定を確認してください。")

# --- データベース接続関数 ---
def get_db_connection():
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
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS checked_tweets (
                tweet_id TEXT PRIMARY KEY,
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
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM checked_tweets WHERE tweet_id = %s", (str(tweet_id),))
        return cursor.fetchone() is not None
    except Exception as e:
        print(f"DB読み取りエラー (is_tweet_checked): {e}")
        return True
    finally:
        if conn:
            conn.close()

def mark_tweet_as_checked(tweet_id):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO checked_tweets (tweet_id) VALUES (%s) ON CONFLICT (tweet_id) DO NOTHING", (str(tweet_id),))
        conn.commit()
    except Exception as e:
        print(f"DB書き込みエラー (mark_tweet_as_checked): {e}")
    finally:
        if conn:
            conn.close()

# --- Telegramにメッセージを送信する関数 ---
def send_telegram_message(message):
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

# --- X APIヘルパー関数 ---
def check_x_account_status(session, username):
    """X APIを使用してユーザーのステータスを確認し、凍結・削除を判断する"""
    url = f"{X_API_URL}/users/by/username/{username}"
    
    headers = {
        'Authorization': f'Bearer {BEARER_TOKEN}',
        'User-Agent': 'v2CheckFrozenAccountBot'
    }
    
    try:
        response = session.get(url, headers=headers, timeout=10)
        
        # 404 Not Found はアカウント削除の可能性が高い
        if response.status_code == 404:
            return f"アカウント削除の可能性❌: (@{username})"

        response.raise_for_status()
        
        data = response.json()
        
        # APIがエラーを返した場合（凍結などの場合はエラーが返る）
        if 'errors' in data:
            for error in data['errors']:
                # 凍結アカウントの典型的なエラーをチェック
                if 'suspended' in error.get('title', '').lower() or 'not found' in error.get('title', '').lower():
                     return f"アカウント凍結/削除の可能性❌: (@{username})"
                if error.get('title') == 'Forbidden':
                     # API制限超過のエラーコードが返る場合
                     return "X API権限エラー (制限超過またはアクセス拒否) 🚫"
                
            return f"X API処理エラー⚠️: (@{username})"
        
        # 正常にユーザーデータが取得できた場合
        if 'data' in data and 'id' in data['data']:
            return None # 正常なアカウント
            
        return f"X API不明な応答⚠️: (@{username})"

    except HTTPError as e:
        if e.response.status_code == 401:
             return "X API認証エラー (Bearer Tokenが無効/期限切れ) 🔑"
        if e.response.status_code == 429: # Rate Limit Exceeded
             return "X API制限超過 (リクエストが多すぎます) 🚫"
        return f"X API HTTPエラー ({e.response.status_code}) 🚨"
    except RequestException as e:
        return f"ネットワークエラー: {e.__class__.__name__}"
    except Exception as e:
        return f"予期せぬエラー: {e.__class__.__name__}"


# --- メインの定期実行ロジック (X API版) ---
def scheduled_check():
    """X APIを使用して当選ツイートを検索し、アカウントの状態をチェックする (無料枠対応)"""
    
    if not BEARER_TOKEN:
        print("X API Bearer Tokenが設定されていません。スキップします。")
        send_telegram_message("🚨BOTエラー: `BEARER_TOKEN`が設定されていないため、X APIによる検知をスキップしました。")
        return

    # 実行開始ログを強力に表示
    print(f"--------------------------------------------------")
    print(f"--- 仮想通貨特化検知実行開始 (X API V2 Free Tier): {time.ctime()} ---")
    print(f"--------------------------------------------------")
    
    # 検索クエリ: 日本語の仮想通貨関連の「当選/配布」ツイートからRTを除外
    query = '("BTC" OR "ETH" OR "NFT" OR "エアドロ" OR "GiveAway" OR "Airdrop" OR "仮想通貨" OR "暗号資産") ("当選" OR "DM" OR "おめでとう" OR "配布") lang:ja -is:retweet'
    
    search_url = f"{X_API_URL}/tweets/search/recent"
    
    # 無料枠の制限に配慮し、max_results=10（最小限）に設定
    params = {
        'query': query,
        'max_results': 10, 
        'expansions': 'entities.mentions.username',
        'tweet.fields': 'created_at,entities'
    }

    headers = {
        'Authorization': f'Bearer {BEARER_TOKEN}',
        'User-Agent': 'v2CheckFrozenAccountBot'
    }
    
    session = requests.Session()
    
    try:
        # 1. ツイート検索 (1リクエスト消費)
        response = session.get(search_url, headers=headers, params=params, timeout=20)
        response.raise_for_status() 
        search_data = response.json()
        
        if 'data' not in search_data or not search_data['data']:
            print("検索結果のツイートは見つかりませんでした。")
            return

        for tweet in search_data['data']:
            tweet_id = tweet['id']
            tweet_url = f"https://twitter.com/i/web/status/{tweet_id}"
            
            if is_tweet_checked(tweet_id):
                continue
            
            mentioned_usernames = []
            
            # entities.mentions からメンションされたユーザー名を取得
            if 'entities' in tweet and 'mentions' in tweet['entities']:
                mentioned_usernames = [m['username'] for m in tweet['entities']['mentions']]

            if not mentioned_usernames:
                mark_tweet_as_checked(tweet_id)
                continue

            # 2. アカウント状態のチェック (ユーザー数に応じてリクエスト消費)
            
            # Free Tierの制限のため、各チェックの間に間隔を設ける
            for mentioned_username in mentioned_usernames:
                
                # リクエストを消費するため、連続実行を防ぐ
                time.sleep(5) 
                
                detection_message = check_x_account_status(session, mentioned_username)
                
                if detection_message and "X API認証エラー" in detection_message:
                    send_telegram_message(detection_message)
                    return
                
                if detection_message and "X API制限超過" in detection_message:
                    send_telegram_message("🚫X API制限超過: 無料枠のリクエスト制限を超過したため、次のスケジュールまでスキップします。")
                    return 

                if detection_message:
                    notification_text = (
                        f"【仮想通貨当選者 アカウント無し検知 (X API)】\n"
                        f"異常内容: {detection_message}\n"
                        f"当選者候補: @{mentioned_username}\n"
                        f"該当ツイート: [ツイートはこちら]({tweet_url})"
                    )
                    send_telegram_message(notification_text)
            
            mark_tweet_as_checked(tweet_id)
        
        print("検知処理が正常に終了しました。")
        
    except HTTPError as e:
        status_code = e.response.status_code
        error_msg = f"🚨X API検索エラー (Status: {status_code}): APIキーや権限を確認してください。"
        send_telegram_message(error_msg)
        print(error_msg)
        
    except RequestException as e:
        error_msg = f"🚨ネットワーク接続エラーが発生: {e.__class__.__name__}"
        send_telegram_message(error_msg)
        print(error_msg)
        
    except Exception as e:
        error_msg = f"🚨予期せぬ実行時エラー: {e.__class__.__name__}: {str(e)}"
        send_telegram_message(error_msg)
        print(error_msg)


# ----------------------------------------------------
# --- FLASKとSCHEDULERの設定 ---
# ----------------------------------------------------

app = Flask(__name__)

try:
    setup_database()
    scheduler = BackgroundScheduler()
    
    # 実行間隔を1時間1回に変更
    start_time = datetime.now() - timedelta(hours=1)
    
    job = scheduler.add_job(
        scheduled_check, 
        'interval', 
        hours=1, # 1時間ごと
        start_date=start_time.strftime('%Y-%m-%d %H:%M:%S')
    ) 
    
    print(f"✅ APScheduler: ジョブ '{job.id}' が1時間ごとの実行にスケジュールされました。")
    
    scheduler.start()
    
    # 起動直後の即時実行をスレッドで強制
    def force_run_on_startup():
        # gunicornのワーカー起動を待つための短い遅延
        time.sleep(1) 
        print("💡 起動直後の即時実行をスレッドで強制します...")
        scheduled_check()
    
    # 起動を妨げないよう、新しいスレッドで実行
    threading.Thread(target=force_run_on_startup).start()

except Exception as e:
    print(f"BOT初期化失敗: {e}")

@app.route('/')
def home():
    return jsonify({
        "status": "running (X API V2 Free Tier Mode)",
        "service": "X Crypto Winner Compliance BOT (API)",
        "check_interval": "1 hour", # レスポンスも更新
        "notice": "This mode is heavily constrained by X Free API limits (1500 req/month)."
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))
