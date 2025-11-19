import datetime
import logging
import time
import os
import requests
from threading import Thread
import io
import random
import math
import json

# グラフ描画とデータ処理のためのインポート
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, DayLocator, HourLocator
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# 実践的な分析のための新しいライブラリ
import yfinance as yf
import pandas_ta as ta
import numpy as np 

# -----------------
# ロギング設定
# -----------------
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# -----------------
# Matplotlib 日本語フォント設定
# -----------------
# 注: 環境によっては'Noto Sans CJK JP'が利用できない場合があります。その場合はIPAexGothicなどがフォールバックされます。
try:
    plt.rcParams['font.family'] = 'sans-serif'
    # Noto Sans CJK JPは一般的な環境にインストールされている可能性が高い
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'IPAexGothic', 'Hiragino Sans GB', 'Liberation Sans', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False # マイナス記号の文字化け防止
    logging.info("日本語フォント設定を試みました。")
except Exception as e:
    logging.warning(f"日本語フォント設定に失敗しました: {e}. 英語フォントで続行します。")

# Flask関連のインポート
from flask import Flask, render_template, jsonify
from flask_apscheduler import APScheduler

# -----------------
# グローバル設定
# -----------------
TICKER = 'BTC-USD'
# yfinanceからデータを取得する際の期間と間隔
PERIOD_1Y = '1y'
INTERVAL_1D = '1d'
PERIOD_30D = '30d'
INTERVAL_4H = '4h'

# -----------------
# Telegram Bot設定 (環境変数またはデフォルト値)
# -----------------
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE')
# 例: '5890119671' (環境変数がない場合のフォールバック)
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '5890119671') 

TELEGRAM_API_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# -----------------
# Flask & Scheduler インスタンス
# -----------------
app = Flask(__name__)
scheduler = APScheduler()

# -----------------
# グローバルデータストア (レポート/ダッシュボード用)
# -----------------
# 最終分析結果を格納する辞書
global_report_data = {
    'last_updated': 'N/A',
    'status': 'Initializing...',
    'main_analysis': {},
    'stats': [],
    'chart_image_buffer': None # チャート画像のバイナリデータを保持
}
# データストアをスレッドセーフにアクセスするためのロック（今回は単純化のため省略）

# -----------------
# データ取得関数
# -----------------
def get_historical_data(ticker: str, period: str, interval: str, max_retries: int = 3) -> pd.DataFrame:
    """yfinanceから指定された銘柄の過去データを取得します。"""
    for attempt in range(max_retries):
        try:
            logging.info(f"yfinanceから{ticker}の過去データ（{period}, {interval}）を取得中... (試行 {attempt + 1}/{max_retries})")
            
            # yfinanceでデータを取得
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(period=period, interval=interval)

            if data.empty:
                raise ValueError("取得したデータが空です。")
            
            # カラム名をすべて小文字に変換（pandas_taの慣習に合わせるため）
            data.columns = [col.lower() for col in data.columns]
            data.index.name = 'date'
            
            logging.info(f"✅ 過去データ取得成功。件数: {len(data)} ({interval})")
            return data
        
        except Exception as e:
            logging.error(f"過去データ取得中にエラーが発生しました (試行 {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt) # 指数バックオフ
            else:
                return pd.DataFrame() # 最終的に失敗した場合は空のDataFrameを返す
    return pd.DataFrame()

# -----------------
# 分析関数 (テクニカル指標の計算)
# -----------------
def analyze_data(df: pd.DataFrame) -> pd.DataFrame:
    """テクニカル指標を計算し、データフレームに追加します。"""
    if df.empty:
        return df

    # pandas_ta (ta) を利用して主要なテクニカル指標を追加
    # 注: カラム名はpandas_taのデフォルト命名規則に従います (例: RSI_14, SMA_50)

    # 1. シンプル移動平均 (SMA): 50日, 200日 (長期トレンド把握)
    df.ta.sma(length=50, append=True) # -> SMA_50
    df.ta.sma(length=200, append=True) # -> SMA_200

    # 2. RSI (Relative Strength Index): 買われすぎ/売られすぎ
    df.ta.rsi(length=14, append=True) # -> RSI_14

    # 3. MACD (Moving Average Convergence Divergence)
    df.ta.macd(append=True) # -> MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
    
    # 4. VMA (Volume Moving Average) - ✨ エラー修正箇所 ✨
    # チャート描画関数が 'VMA_20' を必要としているため、明示的にこのカラム名で代入する
    # df.ta.sma(close=df['volume'], length=20, prefix='VMA', append=True) # <- この方法ではカラム名が保証されない
    vma_series = ta.sma(df['volume'], length=20)
    if vma_series is not None:
        # 結果を 'VMA_20' という名前でデータフレームに追加
        df['VMA_20'] = vma_series
    
    # 5. Stochastic Oscillator (STOCH)
    df.ta.stoch(append=True) # -> STOCHk_14_3_3, STOCHd_14_3_3

    logging.info("✅ テクニカル指標の計算完了。")
    return df

# -----------------
# バックテスト（簡易戦略）
# -----------------
def run_backtest(df: pd.DataFrame) -> dict:
    """簡易的なゴールデンクロス戦略でバックテストを実行します。"""
    if df.empty or 'sma_50' not in df.columns or 'sma_200' not in df.columns:
        return {'return': 'N/A', 'trades': 0, 'strategy': 'SMA Crossover'}

    # ゴールデンクロス (GC) とデッドクロス (DC) のシグナルを生成
    # GC: 短期線(sma_50)が長期線(sma_200)を上回った時
    df['Signal'] = 0
    df['Signal'][50:] = np.where(df['sma_50'][50:] > df['sma_200'][50:], 1, 0)
    
    # ポジションの変更点 (エントリー/エグジットポイント)
    df['Position'] = df['Signal'].diff()
    
    initial_cash = 100000  # 10万円から開始
    position = 0
    cash = initial_cash
    asset_value = initial_cash
    trades = 0

    # バックテストシミュレーション
    for i in range(1, len(df)):
        current_close = df['close'].iloc[i]
        
        # ゴールデンクロス (Position == 1.0) で購入 (全額)
        if df['Position'].iloc[i] == 1.0 and position == 0:
            buy_price = current_close
            shares = cash / buy_price
            position = shares
            cash = 0
            trades += 1
            
        # デッドクロス (Position == -1.0) で売却
        elif df['Position'].iloc[i] == -1.0 and position > 0:
            sell_price = current_close
            cash = position * sell_price
            position = 0
            trades += 1

        # 総資産価値の更新
        asset_value = cash + (position * current_close)

    # 最終リターンを計算
    final_return = ((asset_value - initial_cash) / initial_cash) * 100
    
    return {
        'return': f"{final_return:.2f}%",
        'trades': trades,
        'strategy': 'SMA Crossover (50 vs 200)'
    }

# -----------------
# チャート画像生成関数
# -----------------
def generate_chart_image(df: pd.DataFrame, title: str) -> io.BytesIO | None:
    """テクニカル指標を含むチャート画像を生成し、BytesIOとして返します。"""
    
    # 必須カラムのチェック (エラー修正後、VMA_20を含める)
    required_cols = ['close', 'volume', 'sma_50', 'sma_200', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'VMA_20']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"チャート描画に必要なカラムの一部が不足しています: {missing_cols}. analyze_dataの処理を確認してください。")
        return None

    try:
        # サブプロットを作成 (価格/出来高/RSI/MACD)
        fig = Figure(figsize=(16, 12), dpi=100)
        gs = fig.add_gridspec(4, 1, height_ratios=[4, 1, 1, 1], hspace=0.1)

        # 1. 価格チャート (メイン)
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(df.index, df['close'], label='終値', color='#0077b6', linewidth=1.5)
        ax1.plot(df.index, df['sma_50'], label='SMA 50', color='#ff6f00', linestyle='--', linewidth=1.0)
        ax1.plot(df.index, df['sma_200'], label='SMA 200', color='#b20202', linestyle='--', linewidth=1.0)
        
        ax1.set_title(f'{title} ({TICKER}) 価格とトレンド指標', fontsize=18)
        ax1.set_ylabel('価格 (USD)', fontsize=12)
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.tick_params(axis='x', labelbottom=False) # X軸ラベルは最下段のみに表示

        # 2. 出来高チャート
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.bar(df.index, df['volume'], color='#80bfff', alpha=0.6, label='出来高')
        ax2.plot(df.index, df['VMA_20'], label='VMA 20', color='#333333', linewidth=1.0) # VMA_20を使用
        ax2.set_ylabel('出来高', fontsize=12)
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend(loc='upper left', fontsize=10)
        ax2.tick_params(axis='x', labelbottom=False)

        # 3. RSIチャート
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.plot(df.index, df['RSI_14'], label='RSI 14', color='#1e8449', linewidth=1.5)
        ax3.axhline(70, linestyle='--', color='red', alpha=0.5)
        ax3.axhline(30, linestyle='--', color='green', alpha=0.5)
        ax3.set_ylabel('RSI', fontsize=12)
        ax3.set_ylim(0, 100)
        ax3.grid(True, linestyle=':', alpha=0.6)
        ax3.legend(loc='upper left', fontsize=10)
        ax3.tick_params(axis='x', labelbottom=False)

        # 4. MACDチャート
        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        ax4.plot(df.index, df['MACD_12_26_9'], label='MACD Line', color='#0077b6', linewidth=1.0)
        ax4.plot(df.index, df['MACDs_12_26_9'], label='Signal Line', color='#ff6f00', linestyle='--', linewidth=1.0)
        ax4.bar(df.index, df['MACDh_12_26_9'], color=np.where(df['MACDh_12_26_9'] >= 0, '#4CAF50', '#F44336'), alpha=0.5, label='Histogram')
        ax4.set_ylabel('MACD', fontsize=12)
        ax4.grid(True, linestyle=':', alpha=0.6)
        ax4.legend(loc='upper left', fontsize=10)
        
        # X軸の日付フォーマット設定
        if len(df) > 100:
             # 日足データなどの場合
            ax4.xaxis.set_major_formatter(DateFormatter('%Y/%m/%d'))
            ax4.xaxis.set_major_locator(DayLocator(interval=30))
        else:
             # 4時間足データなどの場合
            ax4.xaxis.set_major_formatter(DateFormatter('%m/%d %H:%M'))
            ax4.xaxis.set_major_locator(HourLocator(interval=24))

        fig.autofmt_xdate(rotation=45) # 日付ラベルの傾き

        # 画像をBytesIOバッファに保存
        canvas = FigureCanvas(fig)
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        plt.close(fig) # メモリ解放
        
        logging.info("✅ チャート画像のBytesIOバッファ生成成功。")
        return buffer
    
    except Exception as e:
        logging.error(f"チャート描画中に予期せぬエラーが発生しました: {e}")
        return None

# -----------------
# Telegram通知関数
# -----------------
def send_telegram_message(text: str, image_buffer: io.BytesIO | None = None) -> bool:
    """Telegramにメッセージと画像を送信します。"""
    if TELEGRAM_BOT_TOKEN == 'YOUR_BOT_TOKEN_HERE':
        logging.warning("Telegram Bot Tokenが設定されていません。通知をスキップしました。")
        return False
    
    if image_buffer:
        # 画像とテキストを一緒に送信 (sendPhoto)
        url = f"{TELEGRAM_API_BASE}/sendPhoto"
        files = {'photo': ('chart.png', image_buffer, 'image/png')}
        data = {
            'chat_id': TELEGRAM_CHAT_ID,
            'caption': text,
            'parse_mode': 'Markdown'
        }
        
        try:
            response = requests.post(url, data=data, files=files)
            response.raise_for_status()
            logging.info("✅ Telegramメッセージの送信成功。")
            return True
        except requests.exceptions.RequestException as e:
            logging.error(f"❌ Telegramへの画像送信中にエラー: {e}")
            return False
            
    else:
        # テキストのみ送信 (sendMessage)
        url = f"{TELEGRAM_API_BASE}/sendMessage"
        data = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': text,
            'parse_mode': 'Markdown'
        }
        
        try:
            response = requests.post(url, data=data)
            response.raise_for_status()
            logging.info("✅ Telegramメッセージの送信成功。")
            return True
        except requests.exceptions.RequestException as e:
            logging.error(f"❌ Telegramへのテキスト送信中にエラー: {e}")
            return False

# -----------------
# メインのレポート更新ロジック
# -----------------
def update_report_data():
    """
    市場データを取得、分析し、レポートを更新してTelegramに通知します。
    APSchedulerによって定期的に実行されます。
    """
    global global_report_data
    
    logging.info("スケジュールされたレポート更新タスク開始（実践分析モード）...")
    start_time = time.time()
    
    # 1. データ取得 (長期トレンド用: 1年, 1日足)
    df_1y = get_historical_data(TICKER, PERIOD_1Y, INTERVAL_1D)
    if df_1y.empty:
        global_report_data['status'] = 'データ取得失敗 (1年/1日足)'
        logging.error("❌ 長期トレンド用データ取得失敗。処理を中断します。")
        return

    # 2. データ取得 (短期動向用: 30日, 4時間足)
    df_30d = get_historical_data(TICKER, PERIOD_30D, INTERVAL_4H)
    if df_30d.empty:
        # 短期データがなくても長期データがあれば続行可能
        logging.warning("⚠️ 短期動向用データ取得失敗。長期データで分析を続行します。")

    # 3. 分析 (長期データ)
    df_1y = analyze_data(df_1y)
    current_price = df_1y['close'].iloc[-1]
    
    # 4. バックテスト実行
    backtest_result = run_backtest(df_1y)
    logging.info("✅ バックテスト完了。")

    # 5. チャート画像生成 (短期データがあれば短期を優先、なければ長期を使用)
    chart_data_df = df_30d if not df_30d.empty else df_1y
    chart_title = f"{TICKER} 最新30日 (4時間足)" if not df_30d.empty else f"{TICKER} 最新1年 (日足)"
    image_buffer = generate_chart_image(chart_data_df, chart_title)
    
    # 6. 分析結果の集計
    latest_data = df_1y.iloc[-1]
    
    # テクニカル指標の解釈
    analysis_text = f"**{TICKER} 市場レポート ({datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')})**\n\n"
    analysis_text += f"**現在の価格:** ${current_price:,.2f}\n"
    
    # RSI解釈
    rsi = latest_data.get('RSI_14')
    if rsi is not None:
        if rsi >= 70:
            rsi_status = "⚠️ 買われすぎ (RSI: {:.2f})".format(rsi)
        elif rsi <= 30:
            rsi_status = "🟢 売られすぎ (RSI: {:.2f})".format(rsi)
        else:
            rsi_status = "中立 ({:.2f})".format(rsi)
        analysis_text += f"- RSI (14): {rsi_status}\n"
    
    # MACD解釈
    macd_h = latest_data.get('MACDh_12_26_9')
    if macd_h is not None:
        if macd_h > 0:
            macd_status = "上昇トレンド (ヒストグラム: {:.4f})".format(macd_h)
        else:
            macd_status = "下降トレンド (ヒストグラム: {:.4f})".format(macd_h)
        analysis_text += f"- MACD: {macd_status}\n"

    # SMAクロス解釈
    sma_50 = latest_data.get('sma_50')
    sma_200 = latest_data.get('sma_200')
    if sma_50 is not None and sma_200 is not None:
        if sma_50 > sma_200:
            sma_status = "ゴールデンクロス継続中 (長期上昇トレンド)"
        else:
            sma_status = "デッドクロス継続中 (長期下降トレンド)"
        analysis_text += f"- SMAクロス: {sma_status}\n"
        
    analysis_text += f"\n**簡易バックテスト (SMA 50/200):**\n"
    analysis_text += f"- リターン: {backtest_result['return']}\n"
    analysis_text += f"- トレード回数: {backtest_result['trades']}回\n"

    # 7. グローバルデータストアの更新
    global_report_data.update({
        'last_updated': datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S'),
        'status': 'Analysis Complete',
        'main_analysis': {
            'price': current_price,
            'rsi': rsi,
            'macd_h': macd_h,
            'sma_50': sma_50,
            'sma_200': sma_200,
            'backtest': backtest_result
        },
        'stats': [
            {'label': '現在の価格', 'value': current_price, 'format': 'currency'},
            {'label': 'RSI (14)', 'value': rsi, 'format': 'float'},
            {'label': 'MACD ヒストグラム', 'value': macd_h, 'format': 'float'},
            {'label': 'SMA 50', 'value': sma_50, 'format': 'currency'},
            {'label': 'SMA 200', 'value': sma_200, 'format': 'currency'},
            {'label': 'バックテストリターン', 'value': backtest_result['return'], 'format': 'text'},
        ],
        'chart_image_buffer': image_buffer.getvalue() if image_buffer else None
    })

    # 8. Telegram通知
    if image_buffer:
        image_buffer.seek(0) # バッファのポインタを先頭に戻す
        send_telegram_message(analysis_text, image_buffer)
    else:
        logging.error("❌ チャート画像のバッファが空です。画像送信をスキップしました。")
        analysis_text += "\n\n⚠️ *チャート画像生成に失敗しました。*"
        send_telegram_message(analysis_text) # テキストのみ送信
        
    end_time = time.time()
    logging.info(f"レポート更新タスク完了。所要時間: {end_time - start_time:.2f}秒")

# -----------------
# Flask ルート設定
# -----------------

@app.route('/')
def index():
    """メインダッシュボードページをレンダリングします。"""
    # HTMLはPythonコード内に埋め込み (シングルファイル構成のため)
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bitcoin 自動分析ダッシュボード</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@100..900&family=Noto+Sans+JP:wght@100..900&display=swap');
        body {
            font-family: 'Noto Sans JP', 'Inter', sans-serif;
            background-color: #f4f7f9;
        }
        .container {
            max-width: 1000px;
        }
        .card {
            background-color: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stat-box {
            background-color: #e3f2fd;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            transition: transform 0.2s;
        }
        .stat-box:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15);
        }
        .stat-label {
            font-size: 0.875rem;
            color: #4a5568;
            font-weight: 500;
        }
        .stat-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #1a202c;
            margin-top: 0.25rem;
        }
    </style>
</head>
<body class="p-4 md:p-8">
    <div class="container mx-auto">
        <header class="mb-8">
            <h1 class="text-3xl font-extrabold text-gray-900 mb-2">₿ Bitcoin 自動分析ダッシュボード</h1>
            <p class="text-gray-500">Telegram Botによる市場データ（BTC-USD）の定期的なテクニカル分析結果を表示。</p>
        </header>

        <!-- ステータスと最終更新日時 -->
        <div class="card p-4 mb-8 flex justify-between items-center bg-blue-50 border-l-4 border-blue-500">
            <div>
                <span class="text-sm font-semibold text-gray-600">最終分析日時:</span>
                <span id="last-updated" class="ml-2 font-bold text-gray-800">ロード中...</span>
            </div>
            <div id="status-badge" class="px-3 py-1 text-sm font-semibold rounded-full bg-yellow-200 text-yellow-800">
                ロード中
            </div>
        </div>

        <!-- 主要統計情報グリッド -->
        <h2 class="text-xl font-semibold text-gray-700 mb-4">主要なテクニカル指標</h2>
        <div id="stats-container" class="grid grid-cols-2 md:grid-cols-3 gap-4 mb-8">
            <!-- 統計情報はJSで挿入されます -->
        </div>

        <!-- チャート画像エリア -->
        <div class="card p-6 mb-8">
            <h2 class="text-xl font-semibold text-gray-700 mb-4">最新のテクニカルチャート</h2>
            <div id="chart-area" class="w-full h-auto bg-gray-100 rounded-lg flex items-center justify-center p-4">
                <img id="chart-image" src="" alt="テクニカル分析チャート" class="w-full h-auto max-h-[600px] object-contain rounded-lg shadow-md hidden" onerror="this.classList.add('hidden'); document.getElementById('chart-placeholder').classList.remove('hidden');">
                <p id="chart-placeholder" class="text-gray-500 p-8">チャート画像をロード中、または利用できません。</p>
            </div>
            <p class="text-sm text-gray-400 mt-2 text-right">※チャートは定期的に更新されます (4時間足/日足)</p>
        </div>

        <!-- Telegram設定情報 (デバッグ用) -->
        <div class="card p-6 border-t mt-8 bg-gray-50">
            <h3 class="text-lg font-semibold text-gray-700 mb-2">システム情報 (開発者向け)</h3>
            <p class="text-sm text-gray-600">
                Telegram Chat ID: <span class="font-mono text-blue-700" id="chat-id-display">...</span>
            </p>
            <p class="text-sm text-gray-600">
                Ticker: <span class="font-mono text-blue-700">BTC-USD</span>
            </p>
        </div>
    </div>

    <script>
        const API_URL = '/data';
        const IMAGE_URL = '/chart_image';

        // 値のフォーマット関数
        function formatValue(value, format) {
            if (value === null || value === undefined || value === 'N/A') return 'N/A';
            
            if (typeof value === 'string' && value.endsWith('%')) {
                return value; // テキスト形式のパーセンテージはそのまま
            }

            const num = parseFloat(value);
            if (isNaN(num)) return value;

            switch (format) {
                case 'currency':
                    return '$' + num.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
                case 'float':
                    return num.toFixed(4);
                default:
                    return value;
            }
        }

        // ダッシュボードの更新
        async function updateDashboard() {
            try {
                const response = await fetch(API_URL);
                const data = await response.json();

                // 1. ステータスと最終更新日時を更新
                document.getElementById('last-updated').textContent = data.last_updated;
                const statusBadge = document.getElementById('status-badge');
                statusBadge.textContent = data.status;
                
                statusBadge.className = 'px-3 py-1 text-sm font-semibold rounded-full';
                if (data.status.includes('Complete')) {
                    statusBadge.classList.add('bg-green-200', 'text-green-800');
                } else if (data.status.includes('Initializing')) {
                    statusBadge.classList.add('bg-yellow-200', 'text-yellow-800');
                } else {
                    statusBadge.classList.add('bg-red-200', 'text-red-800');
                }

                // 2. 統計情報を更新
                const statsContainer = document.getElementById('stats-container');
                statsContainer.innerHTML = ''; // クリア
                
                // Telegram Chat IDをデバッグ表示エリアに設定
                document.getElementById('chat-id-display').textContent = data.telegram_chat_id;

                data.stats.forEach(item => {
                    // バックテストリターンは文字列として扱う
                    const formattedValue = item.label === 'バックテストリターン' 
                        ? item.value 
                        : formatValue(item.value, item.format);

                    const html = `
                        <div class="stat-box">
                            <div class="stat-label">${item.label}</div>
                            <div class="stat-value text-base">${formattedValue}</div>
                        </div>
                    `;
                    statsContainer.innerHTML += html;
                });
                
                // 3. チャート画像を更新 (キャッシュを避けるためにタイムスタンプを付加)
                const chartImage = document.getElementById('chart-image');
                const chartPlaceholder = document.getElementById('chart-placeholder');
                const timestamp = new Date().getTime();
                
                if (data.has_chart) {
                    chartImage.src = `${IMAGE_URL}?t=${timestamp}`;
                    chartImage.classList.remove('hidden');
                    chartPlaceholder.classList.add('hidden');
                } else {
                    chartImage.classList.add('hidden');
                    chartPlaceholder.classList.remove('hidden');
                    chartPlaceholder.textContent = 'チャート画像を生成できませんでした。ログを確認してください。';
                }

            } catch (error) {
                console.error("ダッシュボードデータの取得に失敗しました:", error);
                document.getElementById('status-badge').textContent = 'エラー';
                document.getElementById('status-badge').className = 'px-3 py-1 text-sm font-semibold rounded-full bg-red-200 text-red-800';
            }
        }

        // 初期ロードと定期更新の開始
        document.addEventListener('DOMContentLoaded', () => {
            updateDashboard(); // 初期ロード
            // 5秒ごとに更新を試みる
            setInterval(updateDashboard, 5000); 
        });
    </script>
</body>
</html>
    """

@app.route('/data')
def get_analysis_data():
    """ダッシュボード用の分析データをJSONで返します。"""
    global global_report_data
    
    # 画像バイナリデータは除外し、代わりにフラグを返す
    display_data = global_report_data.copy()
    display_data['has_chart'] = display_data['chart_image_buffer'] is not None
    del display_data['chart_image_buffer']
    
    # Telegram Chat IDをフロントエンドに渡す
    display_data['telegram_chat_id'] = TELEGRAM_CHAT_ID

    return jsonify(display_data)

@app.route('/chart_image')
def get_chart_image():
    """生成されたチャート画像をストリームとして返します。"""
    global global_report_data
    
    image_buffer_value = global_report_data.get('chart_image_buffer')
    
    if image_buffer_value:
        buffer = io.BytesIO(image_buffer_value)
        from flask import send_file
        buffer.seek(0)
        return send_file(buffer, mimetype='image/png')
    
    # 画像がない場合は、エラーメッセージを返すか、404を返す
    from flask import Response
    return Response("Chart image not available", status=404, mimetype='text/plain')


# -----------------
# スケジューラーの初期設定と開始
# -----------------
if not scheduler.running:
    app.config.update({
        'SCHEDULER_JOBSTORES': {'default': {'type': 'memory'}},
        # スケジューラのExecutor設定（今回はデフォルトを使用）
        'SCHEDULERS_EXECUTORS': {'default': {'type': 'threadpool', 'max_workers': 20}},
        'SCHEDULER_API_ENABLED': False
    })

    scheduler.init_app(app)

    # 6時間ごとにupdate_report_dataを実行
    # Render環境でのデプロイ後、初回実行が成功すれば、その後は6時間間隔で実行される
    scheduler.add_job(id='report_update_job', func=update_report_data,
                      trigger='interval', hours=6, replace_existing=True)

    scheduler.start()
    logging.info("✅ スケジューラーを開始しました。")

# アプリ起動時に初回実行をトリガー (非同期で実行)
# これにより、Webサービスが起動してからすぐに分析が開始されます。
Thread(target=update_report_data).start()

# -----------------
# サーバーの実行
# -----------------
if __name__ == '__main__':
    # Flaskの標準実行。Renderなどのデプロイ環境では通常Gunicornが実行します。
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000), debug=False)
