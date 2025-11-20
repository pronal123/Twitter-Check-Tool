import datetime
import logging
import time
import os
import requests
import io
import random
import math

# グラフ描画とデータ処理のためのインポート
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, DayLocator
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

# 実践的な分析のための新しいライブラリ
import yfinance as yf
import pandas_ta as ta
import numpy as np 

# Flask関連のインポート
from flask import Flask, render_template, jsonify
from flask_apscheduler import APScheduler

# -----------------
# ロギング設定 (デバッグレベルで詳細を出力)
# -----------------
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------
# Matplotlib 日本語フォント設定
# -----------------
# Canvas環境での実行を想定し、一般的な日本語フォントを使用
try:
    plt.rcParams['font.family'] = 'sans-serif'
    # Noto Sans CJK JPはCanvas環境で一般的に利用可能です
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False # マイナス記号の文字化け防止
    logging.info("✅ 日本語フォント設定を適用しました。")
except Exception as e:
    logging.warning(f"⚠️ 日本語フォント設定に失敗しました: {e}. 英語フォントで続行します。")

# -----------------
# Telegram Bot設定 (環境変数から取得)
# -----------------
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', 'YOUR_CHAT_ID_HERE')
TELEGRAM_API_BASE_URL = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}'
TELEGRAM_API_URL_MESSAGE = f'{TELEGRAM_API_BASE_URL}/sendMessage'
TELEGRAM_API_URL_PHOTO = f'{TELEGRAM_API_BASE_URL}/sendPhoto'


# -----------------
# タイムゾーン定義
# -----------------
JST = datetime.timezone(datetime.timedelta(hours=9), 'JST')


# -----------------
# グローバル定数と初期状態
# -----------------
TICKER = "BTC-USD"
LONG_INTERVAL = "1d" # 長期分析用 (日足)
SHORT_INTERVAL = "4h" # 短期分析用 (4時間足)
LONG_PERIOD = "1y" # データ取得期間 (1年)
SHORT_PERIOD = "60d" # データ取得期間 (60日)
BACKTEST_CAPITAL = 100000 # バックテストの初期資本
# 【修正】4時間ごとの通知に変更
SCHEDULER_INTERVAL_HOURS = 4 # スケジューラー実行間隔 

app = Flask(__name__)
scheduler = APScheduler()

# グローバルデータを保持する辞書
global_data = {
    'scheduler_status': '初期化中',
    'last_updated': datetime.datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S JST'),
    'next_update_time': (datetime.datetime.now(JST) + datetime.timedelta(hours=SCHEDULER_INTERVAL_HOURS)).strftime('%Y-%m-%d %H:%M:%S JST'),
    'current_price': 0.0,
    'data_count': 0,
    'strategy': 'データ取得待ち...',
    'bias': 'N/A',
    'dominance': 'N/A',
    'P': 0.0, 'R1': 0.0, 'S1': 0.0, 'MA50': 0.0, 'RSI': 0.0,
    'predictions': {'1h': 'N/A', '4h': 'N/A', '12h': 'N/A', '24h': 'N/A'},
    'backtest': {
        'final_capital': BACKTEST_CAPITAL, 'total_return': 0.0, 
        'profit_factor': 0.0, 'max_drawdown': 0.0, 'trades': 0, 'win_rate': 0.0
    }
}


# -----------------
# ヘルパー関数 (Telegram)
# -----------------

# テキストメッセージの送信
def send_telegram_message(text, parse_mode='MarkdownV2'):
    """指定されたテキストメッセージをTelegramに送信します。"""
    if TELEGRAM_BOT_TOKEN == 'YOUR_BOT_TOKEN_HERE':
        logging.warning("⚠️ Telegram BOTトークンが設定されていません。通知をスキップします。")
        return

    # MarkdownV2の特殊文字をエスケープ ('.', '-', etc.)
    safe_text = (
        text.replace('.', '\\.')
            .replace('-', '\\-')
            .replace('(', '\\(')
            .replace(')', '\\)')
            .replace('!', '\\!')
            .replace('+', '\\+')
            .replace('=', '\\=')
            .replace('|', '\\|')
            .replace('{', '\\{')
            .replace('}', '\\}')
            .replace('[', '\\[')
            .replace(']', '\\]')
            .replace('>', '\\>')
            .replace('#', '\\#')
    )
    # 太字、イタリック、インラインコード (`...`) のエスケープは残す
    safe_text = safe_text.replace('*', '(*)') # 一時的に置換
    safe_text = safe_text.replace('_', '(_)') # 一時的に置換
    safe_text = safe_text.replace('`', '(`)') # 一時的に置換
    
    # 復元
    safe_text = safe_text.replace('(*)', '*') 
    safe_text = safe_text.replace('(_)', '_') 
    safe_text = safe_text.replace('(`)', '`') 
    
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': safe_text,
        'parse_mode': parse_mode
    }
    
    try:
        response = requests.post(TELEGRAM_API_URL_MESSAGE, json=payload, timeout=10)
        response.raise_for_status()
        logging.info(f"Telegramメッセージ送信成功: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Telegramメッセージ送信失敗: {e}")

# 画像（チャート）の送信
def send_telegram_photo(image_buffer: io.BytesIO, caption: str):
    """画像データをTelegramに送信します。"""
    if TELEGRAM_BOT_TOKEN == 'YOUR_BOT_TOKEN_HERE':
        logging.warning("⚠️ Telegram BOTトークンが設定されていません。画像通知をスキップします。")
        return

    # MarkdownV2の特殊文字をエスケープ (caption用)
    safe_caption = (
        caption.replace('.', '\\.')
            .replace('-', '\\-')
            .replace('(', '\\(')
            .replace(')', '\\)')
            .replace('!', '\\!')
            .replace('+', '\\+')
            .replace('=', '\\=')
            .replace('|', '\\|')
            .replace('{', '\\{')
            .replace('}', '\\}')
            .replace('[', '\\[')
            .replace(']', '\\]')
            .replace('>', '\\>')
            .replace('#', '\\#')
    )
    safe_caption = safe_caption.replace('*', '(*)') # 一時的に置換
    safe_caption = safe_caption.replace('_', '(_)') # 一時的に置換
    safe_caption = safe_caption.replace('`', '(`)') # 一時的に置換
    
    # 復元
    safe_caption = safe_caption.replace('(*)', '*') 
    safe_caption = safe_caption.replace('(_)', '_') 
    safe_caption = safe_caption.replace('(`)', '`') 

    files = {
        'photo': ('chart.png', image_buffer.getvalue(), 'image/png')
    }
    data = {
        'chat_id': TELEGRAM_CHAT_ID,
        'caption': safe_caption,
        'parse_mode': 'MarkdownV2'
    }
    
    try:
        response = requests.post(TELEGRAM_API_URL_PHOTO, data=data, files=files, timeout=30)
        response.raise_for_status()
        logging.info(f"Telegram画像送信成功: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Telegram画像送信失敗: {e}")


# -----------------
# データ取得と分析
# -----------------

# データ取得
def fetch_btc_ohlcv_data(period: str, interval: str) -> pd.DataFrame:
    """yfinanceからBTC-USDのOHLCVデータを取得します。"""
    try:
        logging.info(f"Yfinanceから {period} 期間の {interval} 足データを取得中...")
        # progress=Falseでログ出力を抑制
        df = yf.download(TICKER, period=period, interval=interval, progress=False, auto_adjust=True, timeout=10)
        if df.empty:
            logging.warning("⚠️ 取得データが空です。")
        else:
            logging.info(f"✅ データ取得成功。件数: {len(df)}")
            df.index.name = 'Datetime' # インデックス名の統一
        return df
    except Exception as e:
        logging.error(f"❌ データ取得失敗: {e}")
        return pd.DataFrame()

# リアルタイム価格取得
def fetch_current_price() -> float:
    """
    yfinanceからBTC-USDの最新の価格をリアルタイムで取得します（リトライ付き）。
    安定性のために1時間足の最新終値を使用します。
    """
    max_retries = 3
    
    # 期間を2日、間隔を1時間に変更して安定性を向上
    INTERVAL_1H = "1h"
    PERIOD_2D = "2d" 
    
    for attempt in range(max_retries):
        try:
            logging.info(f"1時間足の最新終値を取得中 (ソース: Yfinance/{INTERVAL_1H})... (試行 {attempt + 1}/{max_retries})")
            
            # yfinance.downloadを使用して1時間足データを取得
            df_1h = yf.download(TICKER, period=PERIOD_2D, interval=INTERVAL_1H, progress=False, auto_adjust=True, timeout=5)
            
            if df_1h.empty or 'Close' not in df_1h.columns or len(df_1h) == 0:
                raise ValueError("1時間足のデータが空または不十分です。")
            
            # 最新の終値を取得 (Seriesから float 値を確実に取得)
            latest_close = df_1h['Close'].iloc[-1]
            
            # latest_close が Series の場合 (稀なケース)、float に変換
            if isinstance(latest_close, pd.Series):
                latest_close = latest_close.iloc[0]

            # 価格が float または numpy.float であることを確認し、正の値かチェック
            if isinstance(latest_close, (float, np.float_)) and latest_close > 0:
                logging.info(f"✅ 1時間足の最新終値取得成功: ${latest_close:,.2f}")
                return round(latest_close, 2)
            else:
                raise ValueError(f"取得した最新終値が不正な値です: {latest_close}")

        except Exception as e:
            # Pandasの比較エラーを含む、その他のエラーを捕捉
            logging.warning(f"⚠️ Yfinanceからの1時間足価格取得失敗 (試行 {attempt + 1}/{max_retries}): {e}")
            
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt * 2 + random.uniform(0, 1)
            time.sleep(wait_time)
            continue
        else:
            logging.error("❌ 1時間足価格取得の最大リトライ回数に達しました。0.0を返します。")
            return 0.0

# テクニカル分析の実行
def analyze_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pandas-TAを使用して、テクニカル指標を計算しDataFrameに追加します。"""
    if df.empty:
        return df

    # 移動平均線 (MA)
    df.ta.sma(length=50, append=True)
    df.ta.sma(length=200, append=True)

    # RSI (Relative Strength Index)
    df.ta.rsi(length=14, append=True)

    # MACD (Moving Average Convergence Divergence)
    df.ta.macd(append=True)

    # Bollinger Bands
    df.ta.bbands(append=True)

    # ADX (Average Directional Index)
    df.ta.adx(append=True)
    
    # NaNやInfを削除するとバックテストで問題が発生するため、fillnaで0に置換
    return df.fillna(0.0)


# ピボットレベルの計算
def calculate_pivot_levels(df: pd.DataFrame, method: str = 'Classic'):
    """指定されたピボットポイント計算メソッドに基づいてレベルを計算します。"""
    if len(df) < 2:
        # データ不足の場合は最新価格をPとして、適当なR/Sを返す
        latest_close = df['Close'].iloc[-1] if not df.empty and 'Close' in df.columns else 0.0
        return latest_close, latest_close * 1.01, latest_close * 0.99, latest_close * 1.02, latest_close * 0.98

    # ピボットは常に「前日」または「前の足」のデータを使用して計算
    prev_day = df.iloc[-2]
    H, L, C = prev_day['High'], prev_day['Low'], prev_day['Close']

    P = (H + L + C) / 3

    if method == 'Classic':
        R1 = 2 * P - L
        S1 = 2 * P - H
        R2 = P + (R1 - S1)
        S2 = P - (R1 - S1)
        
    elif method == 'Fibonacci':
        R1 = P + 0.382 * (H - L)
        S1 = P - 0.382 * (H - L)
        R2 = P + 0.618 * (H - L)
        S2 = P - 0.618 * (H - L)
        
    else: # Default to Classic
        R1 = 2 * P - L
        S1 = 2 * P - H
        R2 = P + (R1 - S1)
        S2 = P - (R1 - S1)

    return P, R1, S1, R2, S2


# -----------------
# 戦略生成と予測
# -----------------

# 戦略生成ロジック
def generate_strategy(df_long: pd.DataFrame, df_short: pd.DataFrame) -> dict:
    """
    日足と4時間足のテクニカル指標に基づいて、総合的な戦略と予測、市場の優勢度を決定します。
    """
    df_long_clean = df_long.dropna()
    df_short_clean = df_short.dropna()

    # データ不足時のエラーハンドリング
    if len(df_long_clean) < 2 or len(df_short_clean) < 2:
        price = df_long['Close'].iloc[-1] if not df_long.empty and 'Close' in df_long.columns else 0
        return {
            'price': price, 'P': price, 'R1': price * 1.01, 'S1': price * 0.99, 'MA50': price, 'RSI': 50,
            'bias': 'データ不足', 'dominance': 'N/A',
            'strategy': '分析に必要な十分な期間のデータが揃っていません。',
            'details': ['分析に必要な十分な期間のデータが揃っていません。'],
            'predictions': {'1h': 'N/A', '4h': 'N/A', '12h': 'N/A', '24h': 'N/A'}
        }

    latest = df_long_clean.iloc[-1]
    
    # 日足の指標値
    price = latest['Close'] 
    ma50 = latest['SMA_50']
    ma200 = latest['SMA_200']
    rsi = latest['RSI_14']

    # ピボットポイントの計算 (日足データでクラシックピボットを使用)
    P_long, R1_long, S1_long, _, _ = calculate_pivot_levels(df_long, 'Classic')

    # 短期（4時間足）の分析
    latest_short = df_short_clean.iloc[-1]
    _, R1_short, S1_short, _, _ = calculate_pivot_levels(df_short, 'Classic')
    
    # 【修正】SMA_50の値を取得。NaNの場合は現在の終値を代替として使用し、KeyErrorを防ぐ
    short_ma50 = latest_short.get('SMA_50', latest_short['Close']) 
    # 【修正】MACDhの値を取得。NaNの場合は0.0を代替として使用し、KeyErrorを防ぐ
    macdh_short = latest_short.get('MACDh_12_26_9', 0.0) 

    # 総合バイアスと戦略の決定
    bias = "中立"
    strategy = "様子見（ブレイクアウト待ち）"
    details = []
    bull_score = 0
    bear_score = 0

    # --- 1. 長期・中期トレンドバイアス (日足 MA) ---
    if price > ma200:
        details.append(f"• *長期トレンド*: 価格 (`{price:,.2f}`) はMA200 (`{ma200:,.2f}`) を上回り、*長期的な強気相場*です。")
        bull_score += 2
    else:
        details.append(f"• *長期トレンド*: 価格はMA200 (`{ma200:,.2f}`) の下で、長期的な弱気相場が優勢です。")
        bear_score += 2

    if price > ma50 * 1.005:
        details.append(f"• *中期トレンド*: 価格がMA50 (`{ma50:,.2f}`) を明確に上回り、中期的に強い強気トレンドです。")
        bull_score += 1
    elif price < ma50 * 0.995:
        details.append(f"• *中期トレンド*: 価格がMA50 (`{ma50:,.2f}`) を明確に下回り、中期的な弱気トレンドが優勢です。")
        bear_score += 1
    else:
        details.append(f"• *中期トレンド*: 価格はMA50 (`{ma50:,.2f}`) 付近で推移しており、レンジ相場が想定されます。")

    # --- 2. モメンタムシグナル (MACDとRSI 50ライン) ---
    MACD_COL = 'MACD_12_26_9'
    MACDs_COL = 'MACDs_12_26_9'
    if MACD_COL in latest and MACDs_COL in latest:
        # MACDの値もNaNの場合があるため、安全にチェック
        macd_val = latest.get(MACD_COL, 0.0)
        macds_val = latest.get(MACDs_COL, 0.0)
        
        if macd_val > macds_val:
            details.append("• *モメンタム*: MACDがシグナルラインの上にあり、モメンタムは*上昇*傾向です。")
            bull_score += 1
        elif macd_val < macds_val:
            details.append("• *モメンタム*: MACDがシグナルラインの下にあり、モメンタムは*下降*傾向です。")
            bear_score += 1
        else:
             details.append("• *モメンタム*: MACDとシグナルラインがクロス付近で、モメンタムは*中立*です。")

    # --- 3. 過熱感 (RSI) ---
    rsi_val = latest.get('RSI_14', 50.0)
    if rsi_val > 70:
        details.append(f"• *RSI*: 70 (`{rsi_val:,.2f}`) を超え、*買われすぎ*を示唆。短期的な調整（利確売り）に警戒。")
        bear_score += 1 
    elif rsi_val < 30:
        details.append(f"• *RSI*: 30 (`{rsi_val:,.2f}`) を下回り、*売られすぎ*を示唆。短期的な反発（押し目買い）のチャンス。")
        bull_score += 1 
    elif rsi_val > 50:
        details.append(f"• *RSI*: 50 (`{rsi_val:,.2f}`) を上回り、強いモメンタムが*維持*されています。")
    else:
        details.append(f"• *RSI*: 50 (`{rsi_val:,.2f}`) を下回り、弱いモメンタムが*継続*しています。")

    # --- 4. 総合バイアスの決定 ---
    score_diff = bull_score - bear_score
    
    if score_diff >= 3:
        dominance = "明確なロング優勢 🚀"
        bias = "強い上昇"
    elif score_diff == 2:
        dominance = "ロング優勢 📈"
        bias = "上昇"
    elif score_diff <= -3:
        dominance = "明確なショート優勢 💥"
        bias = "強い下降"
    elif score_diff == -2:
        dominance = "ショート優勢 📉"
        bias = "下降"
    else:
        dominance = "中立/レンジ ↔️"
        bias = "レンジ/中立"

    # --- 5. 総合戦略の決定 ---
    R1_long_str = f"`${R1_long:,.2f}`"
    S1_long_str = f"`${S1_long:,.2f}`"
    P_long_str = f"`${P_long:,.2f}`"
    R1_short_str = f"`${R1_short:,.2f}`"
    S1_short_str = f"`${S1_short:,.2f}`"


    if dominance in ["明確なロング優勢 🚀", "ロング優勢 📈"]:
        # 短期がMA50の上にあるかチェック (短期トレンドも強いか)
        if latest_short['Close'] > short_ma50: 
            strategy = f"🌟 *最強のロング戦略*。日足S1 ({S1_long_str}) または4h S1 ({S1_short_str}) への*押し目買い*を積極的に検討。"
        else:
            strategy = f"ロング優勢の押し目買い戦略。日足P ({P_long_str}) への短期的な反落時が主な買い場。"
    elif dominance in ["明確なショート優勢 💥", "ショート優勢 📉"]:
        # 短期がMA50の下にあるかチェック (短期トレンドも弱いか)
        if latest_short['Close'] < short_ma50: 
            strategy = f"💥 *最強のショート戦略*。日足R1 ({R1_long_str}) または4h R1 ({R1_short_str}) への*戻り売り*を積極的に検討。"
        else:
            strategy = f"ショート優勢の戻り売り戦略。日足P ({P_long_str}) への短期的な上昇時が主な売り場。"
    elif dominance == "中立/レンジ ↔️":
        # ボリンジャーバンドの幅 (BBB) がデータに存在する場合にチェック
        BBB_COL = 'BBB_20_2.0_2.0' 
        bbb = latest.get(BBB_COL, 100) 

        if bbb < 10: # ボラティリティ圧縮の基準
             strategy = f"ボラティリティ圧縮中。日足R1 ({R1_long_str}) / S1 ({S1_long_str}) の*ブレイクアウト待ち*。"
        else:
             strategy = f"レンジ取引。日足S1 ({S1_long_str}) 付近で買い、日足R1 ({R1_long_str}) 付近で売り。"

    # --- 短期予測の強化 (修正済み) ---
    predictions = {
        # 1hは短期モメンタム(4h MACD) + 4hトレンド(MA50)
        "1h": "強い上昇 🚀" if macdh_short > 0 and latest_short['Close'] > short_ma50 else "強い下降 📉" if macdh_short < 0 and latest_short['Close'] < short_ma50 else "レンジ ↔️",
        # 4hは短期トレンド(4h MA50)
        "4h": "上昇 📈" if latest_short['Close'] > short_ma50 else "下降 📉",
        # 12hは日足のピボットPに対する位置
        "12h": "上昇 📈" if latest['Close'] > P_long else "下降 📉",
        # 24hは総合バイアス
        "24h": bias
    }

    return {
        'price': price,
        'P': P_long, 'R1': R1_long, 'S1': S1_long, 'MA50': ma50, 'RSI': rsi,
        'bias': bias,
        'dominance': dominance,
        'strategy': strategy,
        'details': details,
        'predictions': predictions
    }


# -----------------
# バックテストロジック (簡易的な移動平均クロス戦略)
# -----------------

def backtest_strategy(df_analyzed: pd.DataFrame) -> dict:
    """
    分析済みの日足データを使用して、簡易的なバックテスト（MA50 vs MA200クロス）を実行します。
    """
    df = df_analyzed.copy()
    
    # NaNを削除すると取引判断ができなくなるため、バックテストに必要なカラムのみ残す
    required_cols = ['Close', 'SMA_50', 'SMA_200']
    if not all(col in df.columns for col in required_cols):
        raise ValueError("バックテストに必要なSMA50またはSMA200データが不足しています。")

    df['signal'] = 0 # 0:何もしない, 1:買い, -1:売り
    
    # SMA50がSMA200を上回ったら「買いシグナル」
    df.loc[(df['SMA_50'].shift(1) <= df['SMA_200'].shift(1)) & (df['SMA_50'] > df['SMA_200']), 'signal'] = 1
    # SMA50がSMA200を下回ったら「売りシグナル」
    df.loc[(df['SMA_50'].shift(1) >= df['SMA_200'].shift(1)) & (df['SMA_50'] < df['SMA_200']), 'signal'] = -1
    
    capital = BACKTEST_CAPITAL
    position = 0 # 保有ポジション（BTCの量）
    trade_count = 0
    win_count = 0
    
    initial_price = df['Close'].iloc[0] if not df.empty else 0.0
    initial_btc_holding = capital / initial_price if initial_price > 0 else 0
    
    trades = [] # 取引記録 [(タイプ, 価格, 資本)]
    capital_history = [capital]
    max_capital = capital
    max_drawdown = 0.0

    for i in range(1, len(df)):
        current_date = df.index[i]
        current_price = df['Close'].iloc[i]
        signal = df['signal'].iloc[i]
        
        # 1. 買いシグナル (ロングエントリー)
        if signal == 1 and position == 0:
            # 全額を投入してBTCを購入
            position = capital / current_price
            capital = 0 # 現金はゼロ
            trade_count += 1
            trades.append(('BUY', current_price, current_date))
            logging.debug(f"BUY @ {current_price:,.2f} on {current_date}")

        # 2. 売りシグナル (ロングイグジット)
        elif signal == -1 and position > 0:
            # 全てのBTCを売却
            new_capital = position * current_price
            profit = new_capital - (trades[-1][1] * position) if trades else 0 
            
            # 勝敗判定
            if new_capital > (trades[-1][1] * position if trades else BACKTEST_CAPITAL):
                 win_count += 1

            capital = new_capital
            position = 0
            trades.append(('SELL', current_price, current_date))
            logging.debug(f"SELL @ {current_price:,.2f} on {current_date}. New Capital: {capital:,.2f}")
        
        # 毎日の資本を記録 (含み益を含む)
        current_equity = capital + (position * current_price)
        capital_history.append(current_equity)
        max_capital = max(max_capital, current_equity)
        
        # ドローダウン計算
        drawdown = (max_capital - current_equity) / max_capital
        max_drawdown = max(max_drawdown, drawdown)


    # 最終的な資本（未決済ポジションを決済）
    final_capital = capital + (position * current_price)
    
    # 最終的なパフォーマンスの計算
    total_return = ((final_capital / BACKTEST_CAPITAL) - 1) * 100
    max_drawdown_percent = max_drawdown * 100
    
    # プロフィットファクター (ここでは簡易的に計算)
    # 実際のプロフィットファクターは、総利益 / 総損失 で計算されるが、ここでは単純にリターンを指標化
    profit_factor = final_capital / BACKTEST_CAPITAL if BACKTEST_CAPITAL > 0 else 0.0
    
    # 勝率の計算
    win_rate = (win_count / trade_count) * 100 if trade_count > 0 else 0.0


    return {
        'final_capital': final_capital,
        'total_return': total_return,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown_percent,
        'trades': trade_count,
        'win_rate': win_rate
    }

# -----------------
# チャート描画ロジック
# -----------------
def generate_chart_image(df: pd.DataFrame, analysis_result: dict) -> io.BytesIO:
    """日足のOHLCVデータとテクニカル指標をプロットし、画像をBytesIOオブジェクトとして返します。"""
    
    # プロットに必要なデータが十分にあるか確認
    if df.empty or len(df) < 20:
        logging.error("❌ チャート描画に十分なデータがありません。")
        return io.BytesIO()

    try:
        df_plot = df.iloc[-180:].copy() # 直近180日分をプロット (約半年)
        
        # 3つのサブプロットを作成 (価格, MACD, RSI)
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True, 
                                 gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # --- 1. 価格チャート (メイン) ---
        ax1 = axes[0]
        ax1.set_title(f'BTC-USD Price Analysis ({LONG_INTERVAL} - Last 180 periods)', fontsize=14, fontweight='bold', color='#1f2937')
        
        # ローソク足を描画する簡易的な実装 (プロットの可読性を優先し、Closeで代替)
        ax1.plot(df_plot.index, df_plot['Close'], label='Close Price', color='#4f46e5', linewidth=1.5)
        
        # 移動平均線 (MA50, MA200)
        ax1.plot(df_plot.index, df_plot['SMA_50'], label='SMA 50', color='#f97316', linestyle='--', linewidth=1.0)
        ax1.plot(df_plot.index, df_plot['SMA_200'], label='SMA 200', color='#059669', linestyle='--', linewidth=1.0)
        
        # ピボットポイント (P, R1, S1)
        P, R1, S1 = analysis_result['P'], analysis_result['R1'], analysis_result['S1']
        current_price = analysis_result['price']

        ax1.axhline(P, color='#facc15', linestyle='-', linewidth=1.0, label=f'Pivot (P: ${P:,.0f})')
        ax1.axhline(R1, color='#ef4444', linestyle=':', linewidth=1.0, label=f'R1 (${R1:,.0f})')
        ax1.axhline(S1, color='#22c55e', linestyle=':', linewidth=1.0, label=f'S1 (${S1:,.0f})')

        # 現在価格のマーカー
        ax1.axhline(current_price, color='#1e40af', linestyle='-', linewidth=2.0, alpha=0.8, label=f'Current Price (${current_price:,.0f})')
        
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.set_ylabel('Price (USD)')
        ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))


        # --- 2. MACDチャート ---
        ax2 = axes[1]
        MACD_COL = 'MACD_12_26_9'
        MACDs_COL = 'MACDs_12_26_9'
        MACDh_COL = 'MACDh_12_26_9'
        
        if MACD_COL in df_plot.columns:
            # ヒストグラム
            ax2.bar(df_plot.index, df_plot[MACDh_COL], label='MACD Histogram', color=np.where(df_plot[MACDh_COL] > 0, '#34d399', '#f87171'), alpha=0.7)
            # MACD Line
            ax2.plot(df_plot.index, df_plot[MACD_COL], label='MACD Line', color='#2563eb', linewidth=1.0)
            # Signal Line
            ax2.plot(df_plot.index, df_plot[MACDs_COL], label='Signal Line', color='#fb923c', linewidth=1.0, linestyle='--')
            
            ax2.axhline(0, color='gray', linestyle='-', linewidth=0.5)
            ax2.legend(loc='upper left', fontsize=8)
            ax2.set_ylabel('MACD')
            ax2.grid(True, linestyle=':', alpha=0.6)
        else:
            ax2.text(0.5, 0.5, 'MACD Data Not Available', transform=ax2.transAxes, ha='center', fontsize=12, color='gray')


        # --- 3. RSIチャート ---
        ax3 = axes[2]
        RSI_COL = 'RSI_14'
        if RSI_COL in df_plot.columns:
            ax3.plot(df_plot.index, df_plot[RSI_COL], label='RSI (14)', color='#8b5cf6', linewidth=1.5)
            ax3.axhline(70, color='red', linestyle=':', linewidth=1.0, label='Overbought (70)')
            ax3.axhline(30, color='green', linestyle=':', linewidth=1.0, label='Oversold (30)')
            ax3.axhline(50, color='gray', linestyle='--', linewidth=0.5)
            ax3.set_ylim(0, 100)
            ax3.legend(loc='upper left', fontsize=8)
            ax3.set_ylabel('RSI')
            ax3.grid(True, linestyle=':', alpha=0.6)
        else:
            ax3.text(0.5, 0.5, 'RSI Data Not Available', transform=ax3.transAxes, ha='center', fontsize=12, color='gray')


        # --- 共通設定 ---
        # 日付フォーマットの設定
        date_fmt = DateFormatter('%Y-%m-%d')
        ax3.xaxis.set_major_formatter(date_fmt)
        
        # 軸の回転
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # 画像をバッファに保存
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig) # メモリ解放
        buf.seek(0)
        
        logging.info("✅ チャート画像の生成が完了しました。")
        return buf

    except Exception as e:
        logging.error(f"❌ チャート描画中に深刻なエラーが発生しました: {e}", exc_info=True)
        return io.BytesIO()


# -----------------
# スケジューリングタスク (致命的なエラー対策を強化)
# -----------------
def update_report_data():
    """定期的に実行されるタスク：データ取得、分析、レポート更新、バックテストの実行"""
    global global_data

    logging.info("-" * 50)
    logging.info("🤖 レポート更新タスクを開始します...")
    
    # 【JST時刻の取得とフォーマット】
    # 現在時刻をUTCで取得し、JSTに変換
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    now_jst = now_utc.astimezone(JST)
    last_updated_str = now_jst.strftime('%Y-%m-%d %H:%M:%S JST')
    
    # 次回更新時刻の計算 (SCHEDULER_INTERVAL_HOURS時間後)
    next_run_time_utc = now_utc + datetime.timedelta(hours=SCHEDULER_INTERVAL_HOURS)
    next_run_time_jst = next_run_time_utc.astimezone(JST)
    next_run_time_fmt = next_run_time_jst.strftime('%Y-%m-%d %H:%M:%S JST') 
    global_data['next_update_time'] = next_run_time_fmt
    
    # エラーが発生した場合の通知用キャプション
    error_caption = None 
    
    # === [CRITICAL FIX] 広範なtryブロックを開始し、予期せぬエラーでスレッドが停止するのを防ぐ ===
    try: 
        # 1. 処理開始ステータスの即時更新
        global_data['scheduler_status'] = 'データ取得中' 
        global_data['last_updated'] = last_updated_str 

        # 2. データ取得 (リアルタイム、日足、4時間足)
        realtime_price = fetch_current_price() # <-- 1h足に変更して安定化
        df_long = fetch_btc_ohlcv_data(LONG_PERIOD, LONG_INTERVAL)
        df_short = fetch_btc_ohlcv_data(SHORT_PERIOD, SHORT_INTERVAL)

        # データ不足チェック
        if df_long.empty or df_short.empty:
            raise ValueError("データ取得に失敗したか、データが空です。Yfinanceの接続またはレート制限を確認してください。")
        
        # リアルタイム価格が取得できなかった場合のフォールバック処理を強化
        price_source = "OHLCV 終値 (最新の足)"
        if realtime_price <= 0 and not df_long.empty:
            # 日足データから最新の終値を取得してフォールバック
            realtime_price = df_long['Close'].iloc[-1].round(2)
            price_source = "日足データ終値 (フォールバック)"
            logging.warning(f"⚠️ リアルタイム価格取得失敗。日足終値 ${realtime_price:,.2f} を使用して続行します。")
        elif realtime_price > 0:
            # 正常に取得できた場合は、1h足の終値なのでソース名を変更
            price_source = "リアルタイム単価 (1時間足)" 
        else:
            # どちらも取得できなかった場合
            raise ValueError("価格データの取得に失敗し、フォールバックも機能しませんでした。")

            
        # 3. テクニカル分析
        global_data['scheduler_status'] = '分析実行中'
        df_long_analyzed = analyze_data(df_long)
        df_short_analyzed = analyze_data(df_short)
        
        # 4. バックテストの実行
        try:
            logging.info(f"バックテスト実行中... 期間: {LONG_PERIOD}")
            backtest_results = backtest_strategy(df_long_analyzed) 
            global_data['backtest'] = backtest_results
            logging.info("✅ バックテスト完了。")
        except Exception as e:
            logging.error(f"❌ バックテスト中にエラーが発生しました: {e}", exc_info=True)
            backtest_results = {'error': f"バックテスト失敗: {str(e)}"}
            global_data['backtest'] = backtest_results

        # 5. 戦略と予測の生成
        analysis_result = generate_strategy(df_long_analyzed, df_short_analyzed)

        # リアルタイム価格の適用
        analysis_result['price'] = realtime_price
            
        # 6. グローバル状態の最終更新
        price = analysis_result['price']
        global_data['data_count'] = len(df_long) + len(df_short) 
        global_data['scheduler_status'] = '稼働中' # 成功時
        global_data['current_price'] = price
        global_data['strategy'] = analysis_result['strategy']
        global_data['bias'] = analysis_result['bias']
        global_data['dominance'] = analysis_result['dominance']
        global_data['predictions'] = analysis_result['predictions']

        # 7. レポートの整形
        P, R1, S1, ma50, rsi = analysis_result['P'], analysis_result['R1'], analysis_result['S1'], analysis_result['MA50'], analysis_result['RSI']
        dominance = analysis_result['dominance']
        strategy = analysis_result['strategy']
        details = analysis_result['details']
        predictions = analysis_result['predictions']

        # 価格をカンマ区切りにフォーマット
        formatted_current_price = f"`${price:,.2f}`"
        
        price_analysis = [
            f"💰 *現在価格 (BTC-USD)*: {formatted_current_price} (_{price_source}_)",
            f"🟡 *ピボットポイント (P, 日足)*: {f'`${P:,.2f}`'}",
            f"🔼 *主要レジスタンス (R1, 日足)*: {f'`${R1:,.2f}`'}",
            f"🔽 *主要サポート (S1, 日足)*: {f'`${S1:,.2f}`'}",
            f"💡 *中期トレンド転換点 (MA50, 日足)*: {f'`${ma50:,.2f}`'}",
            f"🔥 *RSI (14期間, 日足)*: {f'`{rsi:,.2f}`'}"
        ]

        # 短期予測の出力を修正
        prediction_lines = [
            f"• 1h後予測: *{predictions.get('1h', 'N/A')}*",
            f"• 4h後予測: *{predictions.get('4h', 'N/A')}*",
            f"• 12h後予測: *{predictions.get('12h', 'N/A')}*",
            f"• 24h後予測: *{predictions.get('24h', 'N/A')}*"
        ]


        # バックテスト結果の構築
        backtest_results = global_data['backtest']
        if 'error' in backtest_results:
            bt_summary = f"⚠️ *バックテストエラー*: {backtest_results['error']}"
        else:
            bt_summary = (
                f"💰 *最終資本*: `${backtest_results['final_capital']:,.2f}` (初期: `${BACKTEST_CAPITAL:,.2f}`)\n"
                f"📈 *総リターン率*: *{backtest_results['total_return']:,.2f}%*\n"
                f"🏆 *プロフィットファクター*: `{backtest_results['profit_factor']:,.2f}` (1.0以上が望ましい)\n"
                f"📉 *最大ドローダウン (DD)*: `{backtest_results['max_drawdown']:,.2f}%` (リスク指標)\n"
                f"📊 *取引回数*: `{backtest_results['trades']}` (勝率: `{backtest_results['win_rate']:,.2f}%`)"
            )
            
        # --- レポートメッセージの構築 ---
        report_message = (
            f"👑 *BTC実践分析レポート (テクニカルBOT)* 👑\n\n"
            f"📅 *最終データ更新*: `{last_updated_str}`\n"
            f"🕒 **次回更新予定**: {next_run_time_fmt}\n" 
            f"📊 *処理データ件数*: *{len(df_long)}* 件 ({LONG_INTERVAL}足) + *{len(df_short)}* 件 ({SHORT_INTERVAL}足)\n\n"
            
            f"**🚀 市場の優勢 (Dominance) 🚀**\n"
            f"🚨 *総合優勢度*: *{dominance}*\n\n"
            
            f"--- *主要価格帯と指標 (USD)* ---\n"
            f"{'\n'.join(price_analysis)}\n\n"
            
            f"--- *動向の詳細分析と根拠* ---\n"
            f"{'\n'.join(details)}\n\n"
            
            f"--- *短期動向と予測* ---\n"
            f"{'\n'.join(prediction_lines)}\n\n"
            
            f"--- *総合戦略サマリー* ---\n"
            f"🛡️ *推奨戦略*: *{strategy}*\n\n"
            
            f"{chr(8212) * 20}\n"
            f"--- *バックテスト結果 ({LONG_PERIOD} / {LONG_INTERVAL}足)* ---\n"
            f"{bt_summary}\n\n"
            f"_※ この分析は、実戦的なマルチタイムフレーム分析に基づきますが、投資助言ではありません。_"
        )

        # 8. テキストメッセージの送信 (最優先で同期実行)
        # 【修正】同期実行に変更 (Threadを削除)
        send_telegram_message(report_message)
        logging.info("✅ レポートテキストメッセージの送信完了。")


        # 9. チャート描画と写真送信 
        global_data['scheduler_status'] = 'チャート描画中'
        try:
            logging.info("チャート画像を生成中...")
            chart_buffer = generate_chart_image(df_long_analyzed, analysis_result)
            
            photo_caption = (
                f"📈 *BTC実践分析チャート ({LONG_INTERVAL}足)* 📉\n"
                f"📅 更新: `{now_jst.strftime('%Y-%m-%d %H:%M:%S JST')}`\n"
                f"💰 現在価格: {formatted_current_price}\n"
                f"🚨 *優勢度*: *{dominance}*\n"
                f"🛡️ *推奨戦略*: {strategy}\n"
                f"_詳細は別途送信されたテキストレポートをご確認ください。_"
            )
            
            if chart_buffer.getbuffer().nbytes > 0:
                # 【修正】画像の送信を同期実行に変更 (Threadを削除)
                send_telegram_photo(chart_buffer, photo_caption)
                logging.info("✅ チャート画像メッセージの送信完了。")
            else:
                 logging.error("❌ チャート画像のバッファが空です。画像送信をスキップしました。")

        except Exception as e:
            logging.error(f"❌ チャート画像の生成または送信に失敗しました: {e}", exc_info=True)
            
        logging.info("レポート更新タスク完了。")


    except Exception as e:
        # メインタスク全体で例外が発生した場合のログと通知
        global_data['scheduler_status'] = 'タスク失敗 (未処理例外)'
        logging.critical(f"❌ 致命的エラー: update_report_dataタスクが未処理の例外で失敗しました: {e}", exc_info=True)
        # 失敗通知を試みる (この通知が最後の砦)
        error_msg = f"💀 **BOT致命的エラー**: メイン分析タスクが失敗しました。詳細をログで確認してください: {str(e)[:200]}..."
        # 【修正】エラー通知の送信を同期実行に変更 (Threadを削除)
        send_telegram_message(error_msg)
        
    logging.info("-" * 50)


# -----------------
# ルート（エンドポイント）
# -----------------
@app.route('/')
def index():
    """ダッシュボードの表示"""
    # テンプレートにglobal_dataを渡すことで、初回表示時に初期値を埋め込む
    return render_template('index.html', title='BTC実践テクニカル分析 BOT ダッシュボード', data=global_data)

@app.route('/status')
def status():
    """現在のステータスをJSONで返すAPIエンドポイント"""
    return jsonify(global_data)

# -----------------
# スケジューラーの初期設定と開始
# -----------------
# スケジューラーが未起動の場合のみ実行
if not scheduler.running:
    app.config.update({
        'SCHEDULER_JOBSTORES': {'default': {'type': 'memory'}},
        'SCHEDULER_EXECUTORS': {'default': {'type': 'threadpool', 'max_workers': 20}},
        'SCHEDULER_API_ENABLED': False
    })

    scheduler.init_app(app)

    # 4時間ごとにupdate_report_dataを実行するように修正
    scheduler.add_job(id='report_update_job', func=update_report_data,
                      trigger='interval', hours=SCHEDULER_INTERVAL_HOURS, replace_existing=True)

    scheduler.start()
    # 初期起動時に即座に一回実行
    update_report_data()
    logging.info("✅ スケジューラーを開始しました。初回レポート生成を実行しました。")


# -----------------
# HTML ダッシュボードテンプレート
# -----------------
# 注: Flaskアプリのテンプレートは通常、`templates/index.html`に配置されますが、
#     ここでは実行環境の制約により、簡略化されたHTMLを直接提供します。

@app.cli.command("start")
def start_app():
    # 開発サーバーの起動 (本番環境では使用されないが、ローカルテスト用)
    app.run(host='0.0.0.0', port=5000)

# index.htmlのテンプレート
# Flaskの仕様に基づき、ここで直接テンプレートをレンダリングすることはできません。
# ユーザーがダッシュボードを確認できるように、`index.html`を提供します。

# NOTE: The provided environment will automatically serve this HTML file 
# based on the content of the `index.html` file block if created. 
# Since this is a Flask app, we will assume a basic structure.
# For simplicity, I will generate the HTML content for `index.html`.

# -----------------
# HTML ダッシュボード
# -----------------
# FlaskのHTMLテンプレート（index.html）を生成します。
# このテンプレートは、/statusエンドポイントから最新のデータを取得して表示します。
