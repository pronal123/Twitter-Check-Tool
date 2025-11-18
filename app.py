import datetime
import logging
import time
import os
import requests 
from threading import Thread
import io 
import random 
import math

# グラフ描画とデータ処理のためのインポート
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, DayLocator

# 実践的な分析のための新しいライブラリ
import yfinance as yf 
import pandas_ta as ta

# -----------------
# Matplotlib 日本語フォント設定
# -----------------
# 注: 環境によっては'Noto Sans CJK JP'が利用できない場合があります。その場合はIPAexGothicなどがフォールバックされます。
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'IPAexGothic', 'Hiragino Sans GB', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False 

# Flask関連のインポート
from flask import Flask, render_template, jsonify
from flask_apscheduler import APScheduler 

# -----------------
# Telegram Bot設定
# -----------------
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE') 
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '5890119671') 

TELEGRAM_API_BASE_URL = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}'
TELEGRAM_API_URL_MESSAGE = f'{TELEGRAM_API_BASE_URL}/sendMessage'
TELEGRAM_API_URL_PHOTO = f'{TELEGRAM_API_BASE_URL}/sendPhoto'


# -----------------
# ロギング設定
# -----------------
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# -----------------
# アプリケーション初期化
# -----------------
app = Flask(__name__, template_folder='.') 
scheduler = APScheduler()

# グローバル状態
global_data = {
    'last_updated': 'N/A',
    'data_range': '過去10日間', 
    'data_count': 0,
    'scheduler_status': '初期化中',
    'current_price': 0,
    'strategy': 'データ処理中'
}

# -----------------
# Telegram 通知ヘルパー関数 (変更なし)
# -----------------
def send_telegram_message(message):
    """Telegramにテキストメッセージを送信します。"""
    if TELEGRAM_BOT_TOKEN == 'YOUR_BOT_TOKEN_HERE' or not TELEGRAM_CHAT_ID:
        logging.warning("⚠️ Telegram BOT TOKENまたはCHAT IDが設定されていません。通知をスキップします。")
        return

    try:
        logging.info(f"Telegramにテキストメッセージを送信中... Chat ID: {TELEGRAM_CHAT_ID}")
        response = requests.post(
            TELEGRAM_API_URL_MESSAGE,
            json={'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'},
            timeout=15
        )
        response.raise_for_status()
        logging.info("✅ Telegramテキストメッセージの送信成功。")
        
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"❌ Telegram HTTPエラーが発生しました: {http_err} - 応答: {response.text}")
    except requests.exceptions.RequestException as req_err:
        logging.error(f"❌ Telegram API接続エラーが発生しました: {req_err}")
    except Exception as e:
        logging.error(f"❌ Telegramテキストメッセージの送信中に予期せぬエラーが発生しました: {e}")

def send_telegram_photo(photo_buffer: io.BytesIO, caption: str):
    """Telegramにチャート画像を送信します。"""
    if TELEGRAM_BOT_TOKEN == 'YOUR_BOT_TOKEN_HERE' or not TELEGRAM_CHAT_ID:
        logging.warning("⚠️ Telegram BOT TOKENまたはCHAT IDが設定されていません。画像通知をスキップします。")
        return

    try:
        logging.info(f"Telegramにチャート画像を送信中... Chat ID: {TELEGRAM_CHAT_ID}")

        response = requests.post(
            TELEGRAM_API_URL_PHOTO,
            data={'chat_id': TELEGRAM_CHAT_ID, 'caption': caption, 'parse_mode': 'Markdown'},
            files={'photo': ('chart.png', photo_buffer, 'image/png')},
            timeout=30
        )
        response.raise_for_status()
        logging.info("✅ Telegramチャート画像の送信成功。")
        
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"❌ Telegram Photo HTTPエラーが発生しました: {http_err} - 応答: {response.text}")
    except requests.exceptions.RequestException as req_err:
        logging.error(f"❌ Telegram Photo API接続エラーが発生しました: {req_err}")
    except Exception as e:
        logging.error(f"❌ Telegramチャート画像の送信中に予期せぬエラーが発生しました: {e}")


# -----------------
# 🚀 実践的分析ロジック
# -----------------

def fetch_btc_ohlcv_data():
    """
    yfinanceからBTC-USDの日足データを取得します。（過去10日間）
    """
    ticker = "BTC-USD"
    period = "10d" 
    interval = "1d" 
    
    try:
        logging.info(f"yfinanceから{ticker}の過去データ（{period}）を取得中...")
        df = yf.download(ticker, period=period, interval=interval)
        
        if df.empty:
            raise ValueError("取得したデータが空です。")
            
        # インデックス名を'Date'に設定
        df.index.name = 'Date'
        # 終値 (Close) を小数点以下2桁に丸める
        df['Close'] = df['Close'].round(2)
        
        logging.info(f"✅ 過去データ取得成功。件数: {len(df)}")
        return df
        
    except Exception as e:
        logging.error(f"❌ yfinanceからデータ取得中にエラーが発生しました: {e}")
        return pd.DataFrame()


def analyze_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    取得したデータフレームにテクニカル指標（MA, RSI, MACD, BB）を追加します。
    """
    if df.empty:
        return df
        
    # --- 移動平均線 (SMA) ---
    # 中期トレンドの指標としてSMA50日を使用
    df.ta.sma(length=50, append=True) 
    
    # --- 相対力指数 (RSI) ---
    df.ta.rsi(length=14, append=True)
    
    # --- MACD (Moving Average Convergence Divergence) ---
    # MACD, MACD_H, MACD_Sの3列が追加される
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    
    # --- ボリンジャーバンド (BBANDS) ---
    # BBL, BBM, BBU, BBB, BBPの5列が追加される
    df.ta.bbands(length=20, append=True) 
    
    # 最新のデータを使用するため、NaN行を削除
    df.dropna(inplace=True)
    
    logging.info("✅ テクニカル指標の計算完了。")
    return df


def calculate_pivot_levels_from_data(H: float, L: float, C: float) -> tuple[float, float, float]:
    """
    前日のH, L, C（高値、安値、終値）から、クラシックピボットポイントのP, R1, S1を算出します。
    """
    P = (H + L + C) / 3
    R1 = 2 * P - L
    S1 = 2 * P - H
    
    # 小数点以下2桁に丸めて返却
    return round(P, 2), round(R1, 2), round(S1, 2)


def generate_strategy(df: pd.DataFrame) -> dict:
    """
    最新のテクニカル指標に基づいて、総合的な戦略と予測を決定します。
    """
    if df.empty:
        return {
            'bias': 'データ不足',
            'strategy': 'データ取得を待ってください。',
            'details': ['過去データが不足しているため、分析できません。'],
            'predictions': {'1h': 'N/A', '4h': 'N/A', '12h': 'N/A', '24h': 'N/A'}
        }

    latest = df.iloc[-1]
    prev_latest = df.iloc[-2] if len(df) >= 2 else latest

    # 最新の指標値の取得
    price = latest['Close']
    ma50 = latest['SMA_50']
    rsi = latest['RSI_14']
    macd_h = latest['MACDh_12_26_9'] # MACDヒストグラム
    
    # ピボットポイントの計算（前日のデータを使用）
    H_prev, L_prev, C_prev = latest['High'], latest['Low'], latest['Close'] 
    P, R1, S1 = calculate_pivot_levels_from_data(H_prev, L_prev, C_prev) 
    
    # 総合バイアスと戦略の決定
    bias = "中立"
    strategy = "様子見（ブレイクアウト待ち）"
    details = []
    
    # --- 1. トレンドバイアス (MA50と価格の関係) ---
    if price > ma50 * 1.005:
        bias = "強い上昇"
        details.append(f"・*MA50*: 価格がMA50 ({ma50:,.2f}) を明確に上回っており、中期的に強い強気トレンドが継続しています。")
    elif price < ma50 * 0.995:
        bias = "強い下降"
        details.append(f"・*MA50*: 価格がMA50 ({ma50:,.2f}) を明確に下回っており、中期的な弱気トレンドが優勢です。")
    else:
        bias = "レンジ"
        details.append(f"・*MA50*: 価格がMA50 ({ma50:,.2f}) 付近で推移しており、レンジ相場が想定されます。")

    # --- 2. モメンタムシグナル (MACD) ---
    # MACDヒストグラムがゼロラインを上抜け（ゴールデンクロスに近い）
    if latest['MACD_12_26_9'] > latest['MACDs_12_26_9'] and prev_latest['MACD_12_26_9'] < prev_latest['MACDs_12_26_9']:
        details.append("・*MACD*: ゴールデンクロス（買いシグナル）が確認されました。短期的なモメンタムの上昇が期待できます。")
        bias = "上昇" if bias == "中立" or bias == "レンジ" else bias
    # MACDヒストグラムがゼロラインを下抜け（デッドクロスに近い）
    elif latest['MACD_12_26_9'] < latest['MACDs_12_26_9'] and prev_latest['MACD_12_26_9'] > prev_latest['MACDs_12_26_9']:
        details.append("・*MACD*: デッドクロス（売りシグナル）が発生しました。短期的なモメンタムの低下に注意が必要です。")
        bias = "下降" if bias == "中立" or bias == "レンジ" else bias

    # --- 3. 過熱感 (RSI) ---
    if rsi > 70:
        details.append(f"・*RSI*: 70 ({rsi:,.2f}) を超え、*買われすぎ*を示唆しています。短期的な調整（利確売り）に警戒が必要です。")
        if bias == "強い上昇": strategy = "利益確定 or 逆張り売り検討"
    elif rsi < 30:
        details.append(f"・*RSI*: 30 ({rsi:,.2f}) を下回り、*売られすぎ*を示唆しています。短期的な反発（押し目買い）のチャンスです。")
        if bias == "強い下降": strategy = "押し目買い検討 or 逆張り買い検討"
    else:
        details.append(f"・*RSI*: {rsi:,.2f}で中立圏。トレンドの勢いは過熱していません。")
        
    # --- 4. 総合戦略の決定 ---
    if bias == "強い上昇":
        strategy = f"トレンドフォローの押し目買い戦略。S1 ({S1:,.2f}) やP ({P:,.2f}) への短期的な反落時が買い場。"
    elif bias == "強い下降":
        strategy = f"トレンドフォローの戻り売り戦略。R1 ({R1:,.2f}) やP ({P:,.2f}) への短期的な上昇時が売り場。"
    elif bias == "レンジ":
        # ボリンジャーバンドの幅 (BBB) が狭い場合（圧縮）はブレイクアウト待ち
        if latest['BBB_20_2.0'] < 10: # BBB < 10%はボラティリティ低下を示す
             strategy = f"ボラティリティ圧縮中。R1 ({R1:,.2f}) / S1 ({S1:,.2f}) のブレイクアウト待ち。"
        else:
             strategy = f"レンジ取引。S1 ({S1:,.2f}) 付近で買い、R1 ({R1:,.2f}) 付近で売り。"

    # --- 短期予測 (簡略化) ---
    predictions = {
        "1h": "上昇 📈" if macd_h > 0 and rsi < 60 else "レンジ ↔️",
        "4h": "下降 📉" if bias == "強い下降" and rsi < 50 else "レンジ ↔️",
        "12h": "上昇 📈" if bias == "強い上昇" and price > P else "レンジ ↔️",
        "24h": "上昇 📈" if bias == "強い上昇" else "レンジ ↔️" 
    }
    
    return {
        'price': price,
        'P': P, 'R1': R1, 'S1': S1, 'MA50': ma50, 'RSI': rsi,
        'bias': bias,
        'strategy': strategy,
        'details': details,
        'predictions': predictions
    }


def generate_chart_image(df: pd.DataFrame, analysis_result: dict) -> io.BytesIO:
    """
    終値と主要なテクニカル指標を含むチャート画像を生成します。
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100) 
    
    # --- 1. 価格ライン ---
    ax.plot(df.index, df['Close'], label='BTC 終値 (USD)', color='#059669', linewidth=2)
    
    # --- 2. テクニカル指標ラインの描画 ---
    
    # 50日移動平均線 (MA50)
    ax.plot(df.index, df['SMA_50'], label='SMA 50', color='#fbbf24', linestyle='-', linewidth=1.5, alpha=0.7)
    
    # ボリンジャーバンド (Upper/Lower Band)
    ax.plot(df.index, df['BBU_20_2.0'], label='BB Upper', color='#ef4444', linestyle=':', linewidth=1)
    ax.plot(df.index, df['BBL_20_2.0'], label='BB Lower', color='#3b82f6', linestyle=':', linewidth=1)
    
    # --- 3. 最新の主要レベルの描画 (価格分析結果から取得) ---
    price = analysis_result['price']
    P, R1, S1 = analysis_result['P'], analysis_result['R1'], analysis_result['S1']
    
    # ピボットポイント (P)
    ax.axhline(P, color='#9333ea', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.text(df.index[-1], P, f' P: ${P:,.2f}', color='#9333ea', ha='right', va='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))

    # R1/S1の線は省略し、現在価格と最新のテクニカルラインに集中させます。

    # 現在価格の点とラベル
    ax.scatter(df.index[-1], price, color='black', s=80, zorder=5) 
    ax.text(df.index[-1], price, f' 現在 ${price:,.2f}', color='black', ha='right', va='bottom', fontsize=11, weight='bold')

    # 4. グラフの装飾
    ax.set_title(f'BTC/USD 価格推移とテクニカル分析 (過去{len(df)}日間)', fontsize=16, color='#1f2937', weight='bold')
    ax.set_xlabel('日付', fontsize=12)
    ax.set_ylabel('終値 (USD)', fontsize=12)
    
    formatter = DateFormatter("%m/%d")
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(DayLocator()) # 日付を毎日表示
    
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left')
    plt.tight_layout()

    # 5. 画像をメモリ上のバイトストリームとして保存
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig) 
    
    return buf


# -----------------
# スケジューリングタスク
# -----------------
def update_report_data():
    """定期的に実行されるタスク：データ取得、分析、レポート更新の実行"""
    global global_data

    logging.info("スケジュールされたレポート更新タスク開始（実践分析モード）...")
    now = datetime.datetime.now()
    
    # 1. データ取得
    df = fetch_btc_ohlcv_data()
    
    # データが空の場合の処理
    if df.empty:
        logging.error("致命的エラー: データ取得に失敗したため、レポートを生成できません。")
        global_data['scheduler_status'] = 'エラー'
        global_data['strategy'] = 'データ取得エラー'
        # エラーメッセージを通知
        error_msg = f"❌ *BTC分析レポート生成エラー*\n\nデータ取得に失敗しました。yfinanceのステータスまたはネットワーク接続を確認してください。\n最終更新: {now.strftime('%Y-%m-%d %H:%M:%S')}"
        Thread(target=send_telegram_message, args=(error_msg,)).start()
        return

    # 2. テクニカル分析
    df_analyzed = analyze_data(df)
    
    # 3. 戦略と予測の生成
    analysis_result = generate_strategy(df_analyzed)

    # 4. グローバル状態の更新
    last_updated_str = now.strftime('%Y-%m-%d %H:%M:%S')
    global_data['last_updated'] = last_updated_str
    global_data['data_count'] = len(df)
    global_data['scheduler_status'] = '稼働中'
    global_data['current_price'] = analysis_result['price']
    global_data['strategy'] = analysis_result['strategy']
    
    # 5. レポートの整形
    price = analysis_result['price']
    P, R1, S1, ma50, rsi = analysis_result['P'], analysis_result['R1'], analysis_result['S1'], analysis_result['MA50'], analysis_result['RSI']
    bias = analysis_result['bias']
    strategy = analysis_result['strategy']
    details = analysis_result['details']
    predictions = analysis_result['predictions']
    
    # 価格をカンマ区切りにフォーマット
    formatted_current_price = f"`${price:,.2f}`"
    formatted_P = f"`${P:,.2f}`"
    formatted_R1 = f"`${R1:,.2f}`"
    formatted_S1 = f"`${S1:,.2f}`"
    formatted_MA50 = f"`${ma50:,.2f}`" 
    formatted_RSI = f"`{rsi:,.2f}`" 
    
    price_analysis = [
        f"💰 *現在価格 (BTC-USD)*: {formatted_current_price}",
        f"🟡 *ピボットポイント (P)*: {formatted_P}",
        f"🔼 *主要レジスタンス (R1)*: {formatted_R1}",
        f"🔽 *主要サポート (S1)*: {formatted_S1}",
        f"💡 *中期トレンド転換点 (MA50)*: {formatted_MA50}",
        f"🔥 *RSI (14期間)*: {formatted_RSI}"
    ]

    prediction_lines = [f"• {tf}後予測: *{predictions[tf]}*" for tf in ["1h", "4h", "12h", "24h"]]
    
    report_message = (
        f"👑 *BTC実践分析レポート (テクニカルBOT)* 👑\n\n"
        f"📅 最終データ更新: `{last_updated_str}`\n"
        f"📊 処理データ件数: *{len(df)}* 件\n"
        f"--- *主要価格帯と指標 (USD)* ---\n"
        f"{'\\n'.join(price_analysis)}\n\n" 
        f"--- *総合予測* ---\n"
        f"{'\\n'.join(prediction_lines)}\n\n"
        f"--- *動向の詳細分析と根拠* ---\n"
        f"{'\\n'.join(details)}\n\n"
        f"--- *総合戦略サマリー* ---\n"
        f"💡 *中期バイアス*: *{bias}* 傾向\n"
        f"🛡️ *推奨戦略*: *{strategy}*\n"
        f"_※ この分析は、実戦的なテクニカル分析に基づきますが、投資助言ではありません。_"
    )
    
    # 6. 画像生成と通知の実行
    try:
        logging.info("チャート画像を生成中...")
        chart_buffer = generate_chart_image(df_analyzed, analysis_result)
        
        photo_caption = (
            f"📈 *BTC実践分析チャート* 📉\n"
            f"📅 更新: `{last_updated_str}`\n"
            f"💰 現在価格: {formatted_current_price}\n"
            f"💡 *中期バイアス*: *{bias}* / 🛡️ *推奨戦略*: {strategy}\n"
            f"_詳細は別途送信されるテキストレポートをご確認ください。_"
        )
        
        # 通知はスレッドで非同期実行
        Thread(target=send_telegram_photo, args=(chart_buffer, photo_caption)).start()
        
    except Exception as e:
        logging.error(f"❌ チャート画像の生成または送信に失敗しました: {e}")

    Thread(target=send_telegram_message, args=(report_message,)).start()
    
    logging.info("レポート更新タスク完了。通知キューに追加されました。")


# -----------------
# ルート（エンドポイント）
# -----------------
@app.route('/')
def index():
    """ダッシュボードの表示"""
    return render_template('index.html', title='BTC実践テクニカル分析 BOT ダッシュボード', data=global_data)

@app.route('/status')
def status():
    """現在のステータスをJSONで返すAPIエンドポイント"""
    return jsonify(global_data)

# -----------------
# スケジューラーの初期設定と開始
# -----------------
if not scheduler.running:
    app.config.update({
        'SCHEDULER_JOBSTORES': {'default': {'type': 'memory'}},
        'SCHEDULER_EXECUTORS': {'default': {'type': 'threadpool', 'max_workers': 20}},
        'SCHEDULER_API_ENABLED': False 
    })
    
    scheduler.init_app(app)
    
    # 実践的分析のため、更新頻度を下げて日足ベースの分析に合わせる（例：6時間ごと）
    scheduler.add_job(id='report_update_job', func=update_report_data, 
                      trigger='interval', hours=6, replace_existing=True) 
    
    scheduler.start()
    logging.info("✅ スケジューラーを開始しました。")

# アプリ起動時に初回実行をトリガー
Thread(target=update_report_data).start()
