import os
import requests
import time
import psycopg2
from flask import Flask, jsonify
from urllib.parse import urlparse
from requests.exceptions import HTTPError, RequestException 
from datetime import datetime
import threading # /run_checkエンドポイントの実行を非同期にするために使用

# X API Base URL
X_API_URL = "https://api.twitter.com/2"

# --- 環境変数から設定を取得 ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
DATABASE_URL = os.environ.get("DATABASE_URL")
BEARER_TOKEN = os.environ.get("BEARER_TOKEN")

if not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DATABASE_URL, BEARER_TOKEN]):
    # 環境変数が不足している場合、BOTは機能しませんが、Webサービス自体は起動します
    print("エラー: 必要な環境変数が不足しています。BOTは機能しません。")

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
        # DB接続エラー時は、二重通知を防ぐため「チェック済み」とみなす
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
        
        if response.status_code == 404:
            return f"アカウント削除の可能性❌: (@{username})"

        response.raise_for_status()
        
        data = response.json()
        
        if 'errors' in data:
            for error in data['errors']:
                if 'suspended' in error.get('title', '').lower() or 'not found' in error.get('title', '').lower():
                     return f"アカウント凍結/削除の可能性❌: (@{username})"
                if error.get('title') == 'Forbidden':
                     return "X API権限エラー (制限超過またはアクセス拒否) 🚫"
                
            return f"X API処理エラー⚠️: (@{username})"
        
        if 'data' in data and 'id' in data['data']:
            return None
            
        return f"X API不明な応答⚠️: (@{username})"

    except HTTPError as e:
        if e.response.status_code == 401:
             return "X API認証エラー (Bearer Tokenが無効/期限切れ) 🔑"
        if e.response.status_code == 429:
             return "X API制限超過 (リクエストが多すぎます) 🚫"
        return f"X API HTTPエラー ({e.response.status_code}) 🚨"
    except RequestException as e:
        return f"ネットワークエラー: {e.__class__.__name__}"
    except Exception as e:
        return f"予期せぬエラー: {e.__class__.__name__}"


# --- 検知ロジック関数 ---
def run_detection_check():
    """X APIを使用して当選ツイートを検索し、アカウントの状態をチェックする (無料枠対応)"""
    
    if not BEARER_TOKEN:
        print("X API Bearer Tokenが設定されていません。スキップします。")
        return "ERROR: BEARER_TOKEN not set."

    # 実行開始ログ
    print(f"--------------------------------------------------")
    print(f"--- 仮想通貨特化検知実行開始 (外部トリガー): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    print(f"--------------------------------------------------")
    
    query = '("BTC" OR "ETH" OR "NFT" OR "エアドロ" OR "GiveAway" OR "Airdrop" OR "仮想通貨" OR "暗号資産") ("当選" OR "DM" OR "おめでとう" OR "配布") lang:ja -is:retweet'
    
    search_url = f"{X_API_URL}/tweets/search/recent"
    
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
            return "SUCCESS: No new relevant tweets found."

        for tweet in search_data['data']:
            tweet_id = tweet['id']
            tweet_url = f"https://twitter.com/i/web/status/{tweet_id}"
            
            if is_tweet_checked(tweet_id):
                continue
            
            mentioned_usernames = []
            
            if 'entities' in tweet and 'mentions' in tweet['entities']:
                mentioned_usernames = [m['username'] for m in tweet['entities']['mentions']]

            if not mentioned_usernames:
                mark_tweet_as_checked(tweet_id)
                continue

            # 2. アカウント状態のチェック (ユーザー数に応じてリクエスト消費)
            
            for mentioned_username in mentioned_usernames:
                
                # Free Tierの制限のため、各チェックの間に間隔を設ける
                time.sleep(5) 
                
                detection_message = check_x_account_status(session, mentioned_username)
                
                if detection_message and "X API認証エラー" in detection_message:
                    send_telegram_message(detection_message)
                    return f"ERROR: Authentication issue detected. {detection_message}"
                
                if detection_message and "X API制限超過" in detection_message:
                    # 制限超過時は通知して、処理を中断
                    send_telegram_message("🚫X API制限超過: 無料枠のリクエスト制限を超過しました。次の外部トリガーまでスキップします。")
                    return "ERROR: X API Rate Limit exceeded." 

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
        return "SUCCESS: Detection completed."
        
    except HTTPError as e:
        status_code = e.response.status_code
        error_msg = f"🚨X API検索エラー (Status: {status_code}): APIキーや権限を確認してください。"
        send_telegram_message(error_msg)
        print(error_msg)
        return f"ERROR: HTTP Error {status_code} during search."
        
    except RequestException as e:
        error_msg = f"🚨ネットワーク接続エラーが発生: {e.__class__.__name__}"
        send_telegram_message(error_msg)
        print(error_msg)
        return "ERROR: Network Request failed."
        
    except Exception as e:
        error_msg = f"🚨予期せぬ実行時エラー: {e.__class__.__name__}: {str(e)}"
        send_telegram_message(error_msg)
        print(error_msg)
        return f"ERROR: Unexpected runtime exception: {e.__class__.__name__}"


# ----------------------------------------------------
# --- FLASKのエンドポイント設定 ---
# ----------------------------------------------------

app = Flask(__name__)

try:
    setup_database()
except Exception as e:
    print(f"BOT初期化失敗: {e}")

@app.route('/run_check', methods=['GET'])
def trigger_check():
    """外部のCronサービスなどからのアクセスを受けて検知を実行するエンドポイント"""
    
    # 実行完了を待つとタイムアウトするため、スレッドで非同期実行
    # Webサービスの応答自体はすぐに返す
    thread = threading.Thread(target=run_detection_check)
    thread.start()
    
    return jsonify({
        "status": "Detection triggered",
        "message": "Detection job started in background thread. Check logs for results.",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/')
def home():
    return jsonify({
        "status": "Ready for external trigger",
        "service": "X Crypto Winner Compliance BOT (External Trigger Mode)",
        "trigger_endpoint": "/run_check",
        "instructions": "Set up an external monitoring service (e.g., Render Cron Job or external uptime monitoring) to hit the /run_check endpoint hourly for detection.",
        "notice": "Internal scheduling disabled due to environment limitations. Requires external trigger."
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))
