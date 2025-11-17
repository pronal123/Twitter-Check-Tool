import os
import schedule
import time
from threading import Thread
from flask import Flask, render_template, jsonify
import requests

# 🚨 修正: FUTURES_SYMBOL のインポートを削除
from futures_ml_bot import FuturesMLBot, fetch_advanced_metrics, MODEL_FILENAME

# --- 初期設定 ---
# Flaskアプリ設定
app = Flask(__name__)
app.config['ENV'] = 'development'
app.config['DEBUG'] = True

# BOT初期化
bot = FuturesMLBot()

# --- スケジューリング関数 ---

def run_model_training():
    """MLモデルの再学習を実行します。"""
    print(f"[{time.strftime('%H:%M:%S')}] 🧠 再学習タスク開始...")
    try:
        df = bot.fetch_ohlcv_data()
        bot.train_and_save_model(df)
    except Exception as e:
        print(f"🚨 再学習エラー: {e}")
    finally:
        print(f"[{time.strftime('%H:%M:%S')}] ✅ 再学習タスクが完了しました。")

def run_prediction_and_report():
    """最新データに基づき予測を行い、レポートを生成します。"""
    print(f"[{time.strftime('%H:%M:%S')}] 🚀 予測タスク開始...")
    try:
        # 予測には最新のデータのみが必要
        df = bot.fetch_ohlcv_data()
        advanced_data = fetch_advanced_metrics()
        bot.predict_and_report(df, advanced_data)
    except FileNotFoundError:
        print(f"[{time.strftime('%H:%M:%S')}] ⚠️ モデルファイル '{MODEL_FILENAME}' が存在しません。予測をスキップし、再学習待ち。")
    except Exception as e:
        print(f"🚨 予測/レポート生成エラー: {e}")
    finally:
        print(f"[{time.strftime('%H:%M:%S')}] ✅ 予測タスクが完了しました。")

# --- スケジューラ起動ロジック ---

def run_scheduler():
    """スケジュールに従ってタスクを実行するスレッド関数。"""
    # 初回起動時に強制実行 (データがないと予測ができないため)
    print(f"[{time.strftime('%H:%M:%S')}] 🚀 初回モデル構築を強制実行中...")
    run_model_training()
    
    print(f"[{time.strftime('%H:%M:%S')}] 🚀 初回レポートを強制実行中...")
    run_prediction_and_report()

    # スケジュール設定
    # 日足分析なので、モデル再学習は毎日、予測レポートも毎日1回で十分ですが、
    # 動作確認のため、予測を1時間ごと、再学習を24時間ごとにしておきます。
    schedule.every(24).hours.do(run_model_training)
    schedule.every(1).hour.do(run_prediction_and_report)

    print(f"[{time.strftime('%H:%M:%S')}] ✅ スケジューラが起動しました。予測:1時間ごと, 再学習:24時間ごと")
    
    while True:
        schedule.run_pending()
        time.sleep(1)

# --- Flask Webサーバー ---

@app.route('/')
def index():
    return render_template('index.html', title="ML-Powered Futures BOT Analysis Report")

@app.route('/report_status')
def report_status():
    # 簡易ステータスチェック
    status = {
        'status': 'Running',
        'last_update': time.strftime('%Y-%m-%d %H:%M:%S JST'),
        'next_report': schedule.next_run().strftime('%Y-%m-%d %H:%M:%S JST')
    }
    return jsonify(status)

# メイン実行ブロック
if __name__ == '__main__':
    # スケジューラをバックグラウンドスレッドで起動
    scheduler_thread = Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    # Flaskサーバーを起動
    # developmentサーバーはシングルスレッドなので、schedulerを別スレッドで動かすのが安全
    app.run(host='0.0.0.0', port=8080, use_reloader=False)
