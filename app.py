import datetime
import logging
import time
import os
import requests
from threading import Thread
import io
import random
import math
import json # JSONのインポートを追加

# グラフ描画とデータ処理のためのインポート
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, DayLocator

# 実践的な分析のための新しいライブラリ
# これらのライブラリは、実行環境にインストールされている必要があります (pip install pandas_ta yfinance)
import yfinance as yf
import pandas_ta as ta
import numpy as np 

# Flask関連のインポート
from flask import Flask, render_template, jsonify
from flask_apscheduler import APScheduler

# -----------------
# ロギング設定
# -----------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------
# Matplotlib 日本語フォント設定
# -----------------
# 環境に応じて 'Noto Sans CJK JP' などをインストールしてください
try:
    plt.rcParams['font.family'] = 'sans-serif'
    # Noto Sans CJK JP や IPAexGothic は一般的な日本語フォントです
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'IPAexGothic', 'Hiragino Sans GB', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False # マイナス記号の表示を正しくする
except Exception as e:
    logging.warning(f"日本語フォント設定に失敗しました: {e}. 英語フォントで続行します。")

# -----------------
# Telegram Bot設定
# -----------------
# 環境変数からトークンとチャットIDを取得。未設定の場合はプレースホルダーを使用。
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE')
# chat_id はユーザーまたはグループのID
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '5890119671') 

TELEGRAM_API_BASE_URL = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}'
TELEGRAM_API_URL_MESSAGE = f'{TELEGRAM_API_BASE_URL}/sendMessage'
TELEGRAM_API_URL_PHOTO = f'{TELEGRAM_API_BASE_URL}/sendPhoto'


# -----------------
# グローバル状態
# -----------------
# BOTの現在のステータスや分析結果を格納するグローバル辞書
global_data = {
    'last_updated': '未実行',
    'scheduler_status': '初期化中',
    'current_price': 0.0,
    'strategy': '待機',
    'bias': 'ニュートラル',
    'dominance': '未分析',
    'predictions': '特になし',
    'data_count': 0,
    'backtest': 'バックテスト結果なし'
}

# -----------------
# Flask & Scheduler
# -----------------
app = Flask(__name__)
scheduler = APScheduler()

# -----------------
# ヘルパー関数 (Telegram)
# -----------------
def send_telegram_message(message):
    """Telegramにテキストメッセージを送信する"""
    try:
        response = requests.post(TELEGRAM_API_URL_MESSAGE, data={
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'Markdown'
        })
        response.raise_for_status()
        logging.info(f"Telegramメッセージ送信完了: {message[:50]}...")
    except requests.exceptions.RequestException as e:
        logging.error(f"Telegramメッセージ送信エラー: {e}")

def send_telegram_photo(photo_buffer, caption):
    """Telegramに画像を送信する"""
    try:
        photo_buffer.seek(0)
        files = {'photo': ('chart.png', photo_buffer, 'image/png')}
        data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': caption, 'parse_mode': 'Markdown'}
        
        response = requests.post(TELEGRAM_API_URL_PHOTO, data=data, files=files)
        response.raise_for_status()
        logging.info(f"Telegram画像送信完了: {caption[:50]}...")
    except requests.exceptions.RequestException as e:
        logging.error(f"Telegram画像送信エラー: {e}")

# -----------------
# データ取得
# -----------------
def fetch_btc_ohlcv_data(ticker='BTC-USD', interval='1d', period='6mo'):
    """指定されたティッカー、インターバル、期間のOHLCVデータを取得する"""
    try:
        # yfinanceを使用してデータを取得
        df = yf.download(ticker, interval=interval, period=period)
        if df.empty:
            raise ValueError("データが空です。ティッカーまたは期間を確認してください。")
        # インデックス名を'Date'に統一（Firestoreに保存する場合はタイムスタンプに変換が必要）
        df.index.name = 'Date' 
        return df
    except Exception as e:
        logging.error(f"データ取得エラー ({ticker}): {e}")
        return pd.DataFrame()

# -----------------
# テクニカル分析と戦略生成
# -----------------
def generate_strategy(df_analyzed):
    """
    テクニカル分析に基づいた戦略とバイアスを生成する
    
    Args:
        df_analyzed (pd.DataFrame): 必要なテクニカル指標が計算されたデータフレーム

    Returns:
        dict: 分析結果 (価格、戦略、バイアスなど)
    """
    analysis_result = {
        'price': df_analyzed['Close'].iloc[-1],
        'strategy': '待機',
        'bias': 'ニュートラル',
        'predictions': '特になし',
        'dominance': '未分析'
    }

    # 1. RSIによる過熱感
    rsi = df_analyzed['RSI_14'].iloc[-1]
    
    # 2. MACDによるトレンドの方向
    macd_hist = df_analyzed['MACDh_12_26_9'].iloc[-1]
    
    # 3. SMA (長期トレンド)
    sma_50 = df_analyzed['SMA_50'].iloc[-1]
    sma_200 = df_analyzed['SMA_200'].iloc[-1]
    
    last_close = analysis_result['price']
    
    # トレンドバイアスの決定
    if sma_50 > sma_200 and last_close > sma_50:
        bias = '強気 (上昇トレンド)'
    elif sma_50 < sma_200 and last_close < sma_50:
        bias = '弱気 (下降トレンド)'
    else:
        bias = 'レンジ/ニュートラル'

    # 戦略の決定
    if rsi < 30 and macd_hist > 0:
        strategy = '買いを検討 (押し目)'
        predictions = f'RSIが30以下({rsi:.2f})で売られすぎを示唆。MACDヒストグラムがプラス({macd_hist:.2f})を維持しており、短期的な反発の可能性が高いです。'
    elif rsi > 70 and macd_hist < 0:
        strategy = '売りを検討 (戻り売り)'
        predictions = f'RSIが70以上({rsi:.2f})で買われすぎを示唆。MACDヒストグラムがマイナス({macd_hist:.2f})に転換しており、短期的な下落の可能性が高いです。'
    elif last_close > sma_50 and last_close > sma_200:
        strategy = 'ホールド (強気相場)'
        predictions = '長期・短期の移動平均線が上向きで、価格がその上を推移しています。強い上昇トレンド継続中です。'
    else:
        strategy = '待機 (様子見)'
        predictions = '相場に明確な方向性が見られません。主要な抵抗線/支持線での動きを待ちます。'

    analysis_result.update({
        'strategy': strategy,
        'bias': bias,
        'predictions': predictions
    })
    
    return analysis_result

# -----------------
# チャート生成
# -----------------
def generate_chart_image(df_analyzed):
    """分析データに基づいてチャート画像を生成する"""
    # Matplotlibの図を初期化
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # --- 1. 価格と移動平均線のプロット (ax1) ---
    ax1.plot(df_analyzed.index, df_analyzed['Close'], label='BTC/USD 終値', color='blue', linewidth=1.5)
    ax1.plot(df_analyzed.index, df_analyzed['SMA_50'], label='SMA 50', color='red', linewidth=1.5)
    ax1.plot(df_analyzed.index, df_analyzed['SMA_200'], label='SMA 200', color='purple', linewidth=1.5)

    ax1.set_title(f'BTC/USD 日足テクニカル分析: 最新価格 ${df_analyzed["Close"].iloc[-1]:,.2f}', fontsize=18, fontweight='bold')
    ax1.set_ylabel('価格 (USD)', fontsize=14)
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # --- 2. RSIのプロット (ax2) ---
    ax2.plot(df_analyzed.index, df_analyzed['RSI_14'], label='RSI (14)', color='green', linewidth=1.5)
    ax2.axhline(70, linestyle='--', color='red', alpha=0.7, label='買われすぎ (70)')
    ax2.axhline(30, linestyle='--', color='green', alpha=0.7, label='売られすぎ (30)')

    ax2.set_xlabel('日付', fontsize=14)
    ax2.set_ylabel('RSI', fontsize=14)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # 日付フォーマット
    ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 画像をメモリバッファに保存
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close(fig) # メモリ解放
    
    return buffer

# -----------------
# バックテスト（簡易版）
# -----------------
def simple_backtest(df_analyzed):
    """
    RSIとMACDに基づくシンプルな売買戦略のバックテスト
    
    戦略: 
    - 買い (Buy): RSI < 30 かつ MACDh > 0 
    - 売り (Sell): RSI > 70 かつ MACDh < 0
    """
    initial_balance = 10000 # 開始残高
    balance = initial_balance
    btc_held = 0
    in_position = False
    
    trades = []

    # バックテスト期間を直近100日間に限定
    df_test = df_analyzed.iloc[-100:] 

    for i in range(1, len(df_test)):
        # 前日と当日のデータ
        yesterday = df_test.iloc[i-1]
        today = df_test.iloc[i]
        
        # 買いシグナル: RSIが30以下から回復、かつMACDがプラス
        buy_signal = (yesterday['RSI_14'] <= 30) and (today['RSI_14'] > 30) and (today['MACDh_12_26_9'] > 0)
        
        # 売りシグナル: RSIが70以上から下落、かつMACDがマイナス
        sell_signal = (yesterday['RSI_14'] >= 70) and (today['RSI_14'] < 70) and (today['MACDh_12_26_9'] < 0)

        # 買い執行
        if buy_signal and not in_position:
            buy_price = today['Open']
            btc_held = balance / buy_price
            balance = 0
            in_position = True
            trades.append({'date': today.name.strftime('%Y-%m-%d'), 'action': 'BUY', 'price': buy_price})
            
        # 売り執行
        elif sell_signal and in_position:
            sell_price = today['Open']
            balance = btc_held * sell_price
            btc_held = 0
            in_position = False
            trades.append({'date': today.name.strftime('%Y-%m-%d'), 'action': 'SELL', 'price': sell_price})

    # 最終的なパフォーマンス計算
    if in_position:
        # ポジションを保有している場合、最終日の終値で決済
        final_value = btc_held * df_test['Close'].iloc[-1]
    else:
        final_value = balance

    profit_loss_pct = ((final_value - initial_balance) / initial_balance) * 100
    
    # 結果の整形
    backtest_results = {
        'initial_balance': initial_balance,
        'final_value': round(final_value, 2),
        'profit_loss_pct': round(profit_loss_pct, 2),
        'trades_count': len(trades) // 2,
        'last_3_trades': trades[-3:] # 直近3回の取引
    }
    
    return backtest_results

# -----------------
# メインのレポート更新ロジック (スケジューラーから呼び出される)
# -----------------
def update_report_data():
    """
    BTCの最新データを取得し、分析、チャート生成、Telegram通知、グローバルデータ更新を一括で行う
    
    **【価格不一致対策の核心】**
    通知メッセージの作成と、`global_data` の更新を、分析結果 (analysis_result) を確定させた直後の
    単一のブロックで実行することで、データの一貫性を確保します。
    """
    logging.info("レポート更新タスクを開始します...")
    
    try:
        # 1. データ取得
        # 長期 (1日足) のデータ取得
        df_long = fetch_btc_ohlcv_data(interval='1d', period='1y') 
        
        if df_long.empty:
            raise Exception("データ取得に失敗しました。")

        # 2. テクニカル指標の計算
        # 長期トレンド分析用の指標
        df_long.ta.sma(length=50, append=True)
        df_long.ta.sma(length=200, append=True)
        df_long.ta.rsi(length=14, append=True)
        df_long.ta.macd(append=True)
        df_long_analyzed = df_long.dropna()

        if df_long_analyzed.empty:
            raise Exception("テクニカル指標の計算に必要なデータが不足しています。")

        # 3. 戦略の生成
        analysis_result = generate_strategy(df_long_analyzed)

        # 4. 簡易バックテストの実行
        backtest_results = simple_backtest(df_long_analyzed)

        # 5. チャート画像の生成
        chart_buffer = generate_chart_image(df_long_analyzed.iloc[-90:]) # 直近90日分をプロット

        # 6. レポートメッセージの整形
        current_price = analysis_result['price']
        
        # バックテストの要約を整形
        backtest_summary = (
            f"**💰バックテスト結果 (100日間)💰**\n"
            f"  - 初期資産: ${backtest_results['initial_balance']:,.2f}\n"
            f"  - 最終資産: ${backtest_results['final_value']:,.2f}\n"
            f"  - 損益率: **{backtest_results['profit_loss_pct']:.2f}%**\n"
            f"  - 取引回数: {backtest_results['trades_count']}回\n"
        )
        
        # Telegramに送信するレポートのメッセージを作成
        report_message = (
            f"🔔 *BTCテクニカル分析レポート* 🔔\n\n"
            f"📅 **更新日時**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"📈 **最新価格**: **${current_price:,.2f}**\n"
            f"📊 **トレンドバイアス**: {analysis_result['bias']}\n\n"
            f"💡 **推奨戦略**: **{analysis_result['strategy']}**\n"
            f"🔎 **分析サマリー**: {analysis_result['predictions']}\n\n"
            f"---\n"
            f"{backtest_summary}"
        )

        photo_caption = (
            f"**BTC/USD テクニカル分析チャート**\n"
            f"最新価格: ${current_price:,.2f} | 推奨戦略: {analysis_result['strategy']}"
        )
        
        # 7. グローバル状態の最終更新 **【価格不一致対策】**
        # 通知メッセージの作成に使用した最新データで、`global_data` を一括更新する
        global_data.update({
            'last_updated': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'scheduler_status': '稼働中', 
            'current_price': current_price,
            'strategy': analysis_result['strategy'],
            'bias': analysis_result['bias'],
            'dominance': analysis_result['dominance'],
            'predictions': analysis_result['predictions'],
            'data_count': len(df_long_analyzed),
            # バックテスト結果はJSON文字列として保存
            'backtest': json.dumps(backtest_results) 
        })
        logging.info("✅ グローバルデータ (`global_data`) を最新の分析結果で更新しました。")

        # 8. 通知の実行 (グローバルデータ更新後)
        # 画像送信は重いので非同期スレッドで実行
        Thread(target=send_telegram_photo, args=(chart_buffer, photo_caption)).start()
        # テキストメッセージは必ず最後に送信
        Thread(target=send_telegram_message, args=(report_message,)).start()

        logging.info("レポート更新タスク完了。通知キューに追加されました。")

    except Exception as e:
        error_caption = f"⚠️ BTCレポート更新エラーが発生しました: {e}"
        logging.error(error_caption)
        
        # エラー時もグローバルデータを更新
        global_data.update({
            'scheduler_status': f'エラー: {e}',
            'last_updated': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        
        # エラー通知も非同期で実行
        Thread(target=send_telegram_message, args=(error_caption,)).start()


# -----------------
# ルート（エンドポイント）
# -----------------
@app.route('/')
def index():
    """ダッシュボードの表示 (簡易HTMLテンプレートを使用)"""
    # グローバルデータの 'backtest' はJSON文字列なので、表示用にパースして渡す
    data_for_template = global_data.copy()
    try:
        data_for_template['backtest'] = json.loads(global_data.get('backtest', '{}'))
    except json.JSONDecodeError:
        data_for_template['backtest'] = {'error': 'パースエラー'}
        
    return render_template('index.html', title='BTC実践テクニカル分析 BOT ダッシュボード', data=data_for_template)

@app.route('/status')
def status():
    """現在のステータスをJSONで返すAPIエンドポイント"""
    # `/status` APIは、更新されたばかりの `global_data` を返します
    return jsonify(global_data)

# -----------------
# スケジューラーの初期設定と開始
# -----------------
if not scheduler.running:
    app.config.update({
        'SCHEDULER_JOBSTORES': {'default': {'type': 'memory'}},
        'SCHEDULER_EXECUTORS': {'default': {'type': 'threadpool', 'max_workers': 20}},
        # スケジューラーのAPIは無効にする (セキュリティ上の理由)
        'SCHEDULER_API_ENABLED': False 
    })

    scheduler.init_app(app)

    # 6時間ごとにupdate_report_dataを実行
    scheduler.add_job(id='report_update_job', func=update_report_data,
                      trigger='interval', hours=6, replace_existing=True, 
                      # 初回起動時にすぐに実行することで初期データをセット
                      next_run_time=datetime.datetime.now()) 

    scheduler.start()
    logging.info("✅ スケジューラーを開始しました...")
    global_data['scheduler_status'] = '待機中 (初回実行待ち)'


# -----------------
# HTMLテンプレート (index.html)
# -----------------
# Flaskはデフォルトで 'templates' フォルダの 'index.html' を探します。
# 動作確認のため、ここに簡易的なHTMLを記述します。
# 実際には外部ファイルに分離することが推奨されます。

@app.cli.command('create-html')
def create_html():
    """index.html を生成するコマンド (開発用)"""
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write("""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #f7f7f9; }
        .card { transition: all 0.3s ease; }
        .card:hover { transform: translateY(-3px); box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05); }
    </style>
</head>
<body class="p-8">
    <div class="max-w-4xl mx-auto">
        <h1 class="text-4xl font-extrabold text-gray-900 mb-8">{{ title }}</h1>
        
        <!-- ステータスセクション -->
        <div class="mb-8 p-6 bg-white rounded-xl shadow-lg border-t-4 border-indigo-500">
            <h2 class="text-2xl font-semibold text-indigo-600 mb-3">現在のBOTステータス</h2>
            <div id="status-display" class="space-y-2 text-lg">
                <p>最終更新: <span id="last-updated" class="font-mono text-gray-700">{{ data.last_updated }}</span></p>
                <p>スケジューラー: <span id="scheduler-status" class="font-bold {{ 'text-green-600' if '稼働中' in data.scheduler_status else 'text-red-600' }}">{{ data.scheduler_status }}</span></p>
                <p>データ件数: <span id="data-count" class="font-bold text-blue-600">{{ data.data_count }} (日足)</span></p>
            </div>
        </div>

        <!-- 分析結果セクション -->
        <div class="grid md:grid-cols-2 gap-6 mb-8">
            <!-- 最新価格カード -->
            <div class="card p-6 bg-white rounded-xl shadow-lg border-l-4 border-green-500">
                <p class="text-sm font-medium text-gray-500">最新価格 (USD)</p>
                <p id="current-price" class="text-4xl font-extrabold mt-1 text-green-700">${{ "{:,.2f}".format(data.current_price) }}</p>
                <p class="text-sm text-gray-500 mt-2">データ件数: {{ data.data_count }}件</p>
            </div>
            
            <!-- 戦略カード -->
            <div class="card p-6 bg-white rounded-xl shadow-lg border-l-4 border-yellow-500">
                <p class="text-sm font-medium text-gray-500">推奨戦略 & トレンド</p>
                <p id="strategy" class="text-3xl font-extrabold mt-1 text-yellow-700">{{ data.strategy }}</p>
                <p id="bias" class="text-md text-gray-600 mt-2">バイアス: <span class="font-semibold">{{ data.bias }}</span></p>
            </div>
        </div>

        <!-- 詳細分析 -->
        <div class="card p-6 bg-white rounded-xl shadow-lg mb-8">
            <h2 class="text-2xl font-semibold text-gray-800 mb-3">詳細分析と予測</h2>
            <p id="predictions" class="text-gray-600 leading-relaxed">{{ data.predictions }}</p>
        </div>

        <!-- バックテスト結果 -->
        <div class="card p-6 bg-white rounded-xl shadow-lg">
            <h2 class="text-2xl font-semibold text-gray-800 mb-3">簡易バックテスト結果 (100日間)</h2>
            <div id="backtest-results" class="space-y-2 text-gray-600">
                {% if data.backtest.initial_balance is defined %}
                    <p>初期資産: <span class="font-mono">${{ "{:,.2f}".format(data.backtest.initial_balance) }}</span></p>
                    <p>最終資産: <span class="font-mono">${{ "{:,.2f}".format(data.backtest.final_value) }}</span></p>
                    <p>損益率: <span class="font-extrabold {{ 'text-green-600' if data.backtest.profit_loss_pct >= 0 else 'text-red-600' }}">{{ data.backtest.profit_loss_pct }}%</span></p>
                    <p>取引回数: <span class="font-mono">{{ data.backtest.trades_count }}回</span></p>
                {% else %}
                    <p>バックテスト結果はまだ実行されていません。</p>
                {% endif %}
            </div>
        </div>

        <p class="mt-8 text-center text-gray-500 text-sm">データは6時間ごとに更新されます。</p>
    </div>

    <script>
        // APIをポーリングしてデータを更新する
        async function fetchStatus() {
            try {
                const response = await fetch('/status');
                const data = await response.json();
                
                // 価格表示を整形する関数
                const formatPrice = (price) => '$' + parseFloat(price).toLocaleString('en-US', {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2
                });

                // バックテスト結果のパース
                let backtestData = {};
                try {
                    backtestData = JSON.parse(data.backtest);
                } catch (e) {
                    backtestData = {error: 'パースエラー'};
                }

                document.getElementById('last-updated').textContent = data.last_updated;
                document.getElementById('scheduler-status').textContent = data.scheduler_status;
                document.getElementById('current-price').textContent = formatPrice(data.current_price);
                document.getElementById('strategy').textContent = data.strategy;
                document.getElementById('bias').innerHTML = 'バイアス: <span class="font-semibold">' + data.bias + '</span>';
                document.getElementById('predictions').textContent = data.predictions;
                document.getElementById('data-count').textContent = data.data_count + ' (日足)';
                
                // バックテスト結果の更新
                const backtestElement = document.getElementById('backtest-results');
                if (backtestData.initial_balance !== undefined) {
                    backtestElement.innerHTML = \`
                        <p>初期資産: <span class="font-mono">\${formatPrice(backtestData.initial_balance)}</span></p>
                        <p>最終資産: <span class="font-mono">\${formatPrice(backtestData.final_value)}</span></p>
                        <p>損益率: <span class="font-extrabold \${backtestData.profit_loss_pct >= 0 ? 'text-green-600' : 'text-red-600'}">\${backtestData.profit_loss_pct}%</span></p>
                        <p>取引回数: <span class="font-mono">\${backtestData.trades_count}回</span></p>
                    \`;
                } else {
                    backtestElement.innerHTML = '<p>バックテスト結果はまだ実行されていません。</p>';
                }


            } catch (error) {
                console.error("ステータスAPIの取得に失敗しました:", error);
                document.getElementById('scheduler-status').textContent = 'API接続エラー';
            }
        }

        // 5秒ごとに更新
        setInterval(fetchStatus, 5000);
        
        // 初回ロード時にも実行
        document.addEventListener('DOMContentLoaded', fetchStatus);

    </script>
</body>
</html>
""")
    logging.info("templates/index.html を生成しました。")

if __name__ == '__main__':
    # 開発環境で動作させる場合（本番環境ではGunicorn等を使用）
    # Flaskのテンプレートエンジンが利用できるように、簡易HTMLを一時的に作成します。
    # 実際の運用では、`templates/index.html`を別途配置してください。
    if not os.path.exists('templates'):
        os.makedirs('templates')
    # index.htmlのテンプレートをメモリ上で生成して使用します
    @app.cli.command('run')
    def run_server():
        create_html()
        app.run(debug=True, use_reloader=False) # reloaderはスケジューラーと競合するため無効化
        
    if 'run' in os.sys.argv:
        # コマンドラインから 'flask run' が実行された場合
        create_html()
        app.run(debug=True, use_reloader=False)
    elif len(os.sys.argv) == 1:
        # スクリプトが直接実行された場合
        create_html()
        app.run(debug=True, use_reloader=False)

# 注意: 本番環境では、Flaskサーバーの起動とスケジューラーの管理を適切に行ってください。
