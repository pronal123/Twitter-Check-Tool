import os
import requests
import time
import psycopg2
from flask import Flask, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from urllib.parse import urlparse
from requests.exceptions import HTTPError, RequestException 
from datetime import datetime, timedelta
from bs4 import BeautifulSoup 

# --- 環境変数から設定を取得 ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
DATABASE_URL = os.environ.get("DATABASE_URL")

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

# --- アカウントのステータス確認（スクレイピングによる代替） ---
def check_twitter_account_by_scrape(session, username):
    """ユーザーのプロフィールページをチェックし、凍結/削除を判断する（Requests Sessionを使用）"""
    url = f"https://twitter.com/{username}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # セッションを使用してリクエスト
        response = session.get(url, headers=headers, timeout=10)
        
        if response.status_code == 404:
            return f"アカウント削除の可能性❌: (@{username})"
            
        # ログイン画面へリダイレクト
        if 'login' in response.url.lower() or 'captcha' in response.text.lower():
            return f"スクレイピング制限/認証要求⚠️: (@{username})"
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 凍結/削除されたアカウントの典型的な表示をチェック
        if soup.find('span', string=lambda t: t and ('このアカウントは凍結されています' in t or 'アカウントは存在しません' in t)):
             return f"アカウント凍結/削除の可能性❌: (@{username})"

        # ユーザー名がタイトルに含まれているかチェックすることで、ページ自体は存在すると推測
        if soup.title and username.lower() in soup.title.string.lower():
            return None # アカウントは存在する可能性が高い
        
        if soup.title and "Error" in soup.title.string:
            return f"ページ解析エラー⚠️: (@{username})"
            
        return None # デフォルトではアカウントは存在すると見なす
            
    except RequestException as e:
        return f"ネットワークエラー: {e.__class__.__name__}"


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

# --- メインの定期実行ロジック (スクレイピング版) ---
def scheduled_check():
    """仮想通貨関連の当選ツイートを検索し、アカウントが存在しない当選者をチェックする (スクレイピング版)"""
    print(f"--- 仮想通貨特化検知実行開始 (スクレイピング): {time.ctime()} ---")
    
    # 検索クエリの定義: URLエンコード済み
    encoded_query = 'BTC+OR+ETH+OR+NFT+OR+%E3%82%A8%E3%82%A2%E3%83%89%E3%83%AD+OR+GiveAway+OR+Airdrop+OR+%E4%BB%AE%E6%83%B3%E9%80%9A%E8%B2%A8+OR+%E6%9A%97%E5%8F%B7%E8%B3%87%E7%94%A3+%28%22%E5%BD%93%E9%81%B8%22+OR+%22DM%22+OR+%22%E3%81%8A%E3%82%81%E3%81%A7%E3%81%A8%E3%81%86%22+OR+%22%E9%85%8D%E5%B8%83%22%29'
    search_url = f"https://twitter.com/search?q={encoded_query}&f=live"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Requests Sessionを作成し、Cookieを保持させる
    session = requests.Session()

    try:
        response = session.get(search_url, headers=headers, timeout=20)
        response.raise_for_status() 

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # ページがログインを要求しているかチェック
        if soup.find('input', {'name': 'session[username_or_email]'}):
             print("スクレイピング失敗: Twitterがログインを要求しています。")
             send_telegram_message("🚨スクレイピングエラー: TwitterがログインまたはCAPTCHAを要求しています。BOTを停止/チェックしてください。")
             return

        # ツイートの要素を特定 (セレクタの推測と緩和)
        tweets = soup.find_all('article', {'data-testid': 'tweet'})
        
        if not tweets:
             # data-testid="tweet" が見つからない場合、親要素を広く探す（最後の頼みの綱）
             print("data-testid='tweet' 要素が見つかりませんでした。別の構造を試します。")
             
             # このセレクタは非常に不安定
             tweets = soup.find_all('div', {'data-testid': 'cellInnerDiv'})
             
             if not tweets:
                 print("検索結果からツイート要素を抽出できませんでした。セレクタを確認してください。")
                 return

        for tweet_element in tweets[:20]: # 取得したうちの最初の20件だけをチェック
            
            try:
                # ユーザー名とツイートIDの抽出ロジック（不安定）
                tweet_link = tweet_element.find('a', href=lambda href: href and '/status/' in href)
                if not tweet_link: continue

                href = tweet_link.get('href')
                parts = href.split('/')
                
                if len(parts) < 4: continue
                
                tweet_author_username = parts[1]
                tweet_id = parts[3] 
                
                tweet_url = f"https://twitter.com/{tweet_author_username}/status/{tweet_id}"
                
                # メンションされているユーザー名のリストを生成
                mentioned_usernames = []
                # ツイート本文内の a タグで href が /@から始まるものをメンションとみなす
                for link in tweet_element.find_all('a'):
                    link_href = link.get('href')
                    if link_href and link_href.startswith('/@'):
                        # リンクのテキストがユーザー名と一致するか確認
                        link_text = link.get_text().strip()
                        if link_text and link_text.startswith('@'):
                            mentioned_usernames.append(link_text.lstrip('@'))
                        # それ以外の場合はhrefから取得（より確実だが、構造に依存）
                        elif link_href.count('/') == 1:
                            mentioned_usernames.append(link_href.lstrip('/'))

                # ユニークなユーザー名だけを保持
                mentioned_usernames = list(set(mentioned_usernames))

                if not mentioned_usernames:
                    continue

                if is_tweet_checked(tweet_id):
                    continue

                # メンションされたユーザーをチェック
                for mentioned_username in mentioned_usernames:
                    
                    # メンションされたユーザーのアカウント状態をチェック（Requests Sessionを使用）
                    detection_message = check_twitter_account_by_scrape(session, mentioned_username)
                    
                    if detection_message:
                        notification_text = (
                            f"【仮想通貨当選者 アカウント無し検知 (スクレイピング)】\n"
                            f"異常内容: {detection_message}\n"
                            f"該当ツイートの主催者ユーザー: @{tweet_author_username}\n"
                            f"当選者候補: @{mentioned_username}\n"
                            f"該当ツイート: [ツイートはこちら]({tweet_url})"
                        )
                        send_telegram_message(notification_text)
                    
                    time.sleep(2) # レート制限回避のため待機
            
                mark_tweet_as_checked(tweet_id)
                
            except Exception as e:
                print(f"個別のツイート処理中にエラー: {e}")
                
    except HTTPError as e:
        status_code = e.response.status_code
        error_msg = f"🚨検索URLアクセスエラー (Status: {status_code}): TwitterがBOTをブロックしている可能性があります。"
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

# アプリ起動時にDBセットアップを実行
try:
    setup_database()
    scheduler = BackgroundScheduler()
    
    # 【即時実行のトリガー】開始時刻を過去に設定することで、BOT起動時にジョブが即時実行されるように強制する
    # デプロイ直後と、その後の15分間隔での実行を実現します。
    start_time = datetime.now() - timedelta(days=1)
    
    scheduler.add_job(
        scheduled_check, 
        'interval', 
        seconds=900, # 15分ごと
        start_date=start_time.strftime('%Y-%m-%d %H:%M:%S')
    ) 
    
    scheduler.start()
except Exception as e:
    print(f"BOT初期化失敗: {e}")

# Web Serviceとしてプロセスを維持するためのシンプルなルート
@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "service": "X Crypto Winner Compliance BOT (Scraping)",
        "check_interval": "15 minutes"
    })

if __name__ == '__main__':
    # Renderデプロイ時にはGunicornがこのプロセスを起動します
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))
