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
import numpy as np 

# Flask関連のインポート
from flask import Flask, render_template, jsonify
from flask_apscheduler import APScheduler

# -----------------
# Matplotlib 日本語フォント設定
# -----------------
try:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'IPAexGothic', 'Hiragino Sans GB', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    logging.warning(f"日本語フォント設定に失敗しました: {e}. 英語フォントで続行します。")

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

# === [定義] データインターバルと期間 ===
TICKER = "BTC-USD"
MINKABU_URL = "https://cc.minkabu.jp/pair/BTC_USDT" # 新しいリアルタイム価格取得元
LONG_PERIOD = "1y" # 日足（1d）分析用 - バックテストのため1年間
LONG_INTERVAL = "1d"
SHORT_PERIOD = "30d" # 4時間足（4h）分析用 - 短期戦略
SHORT_INTERVAL = "4h"
BACKTEST_CAPITAL = 100000 # バックテストの初期資本
# ===============================================

# グローバル状態（ダッシュボード表示用）
global_data = {
    'last_updated': 'N/A',
    'data_range': f'過去{LONG_PERIOD} ({LONG_INTERVAL}) + {SHORT_PERIOD} ({SHORT_INTERVAL}) 分析',
    'data_count': 0,
    'scheduler_status': '初期化中',
    'current_price': 0,
    'strategy': 'データ処理中',
    'bias': 'N/A',
    'dominance': 'N/A',
    'predictions': {},
    'backtest': {}
}

# -----------------
# Telegram 通知ヘルパー関数
# -----------------
def send_telegram_message(message):
    """Telegramにテキストメッセージを送信します。"""
    if TELEGRAM_BOT_TOKEN == 'YOUR_BOT_TOKEN_HERE' or not TELEGRAM_CHAT_ID:
        logging.warning("⚠️ Telegram BOT TOKENまたはCHAT IDが設定されていません。通知をスキップします。")
        return
    try:
        response = requests.post(
            TELEGRAM_API_URL_MESSAGE,
            data={'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'},
            timeout=10
        )
        response.raise_for_status()
        logging.info("✅ Telegramメッセージの送信成功。")
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"❌ Telegram Message HTTPエラーが発生しました: {http_err} - 応答: {response.text}")
    except requests.exceptions.RequestException as req_err:
        logging.error(f"❌ Telegram Message API接続エラーが発生しました: {req_err}")

def send_telegram_photo(photo_buffer: io.BytesIO, caption: str):
    """Telegramにチャート画像を送信します。"""
    if TELEGRAM_BOT_TOKEN == 'YOUR_BOT_TOKEN_HERE' or not TELEGRAM_CHAT_ID:
        logging.warning("⚠️ Telegram BOT TOKENまたはCHAT IDが設定されていません。画像通知をスキップします。")
        return
    try:
        logging.info("Telegramにチャート画像を送信中...")

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
        logging.error(f"❌ Telegramチャート画像の送信中に予期せぬエラーが発生しました: {e}", exc_info=True)


# -----------------
# 🚀 実践的分析ロジック
# -----------------

def fetch_btc_ohlcv_data(period: str, interval: str) -> pd.DataFrame:
    """
    yfinanceからOHLCVデータを取得します。
    """
    max_retries = 3

    for attempt in range(max_retries):
        try:
            logging.info(f"yfinanceから{TICKER}の過去データ（{period}, {interval}）を取得中... (試行 {attempt + 1}/{max_retries})")

            df = yf.download(TICKER, period=period, interval=interval, progress=False, auto_adjust=True)

            if df.empty or 'Close' not in df.columns or len(df) < 5: 
                raise ValueError("取得したデータが空または不十分です。レート制限の可能性があります。")

            # MultiIndexフラット化
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.index.name = 'Date'

            df['Close'] = df['Close'].round(2)
            logging.info(f"✅ 過去データ取得成功。件数: {len(df)} ({interval})")
            return df

        except Exception as e:
            logging.error(f"❌ yfinanceからデータ取得中に致命的なエラーが発生しました: {e}", exc_info=True) 
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt * 5 + random.randint(1, 5)
                logging.warning(f"⚠️ リトライします (試行 {attempt + 2}/{max_retries})。 {wait_time}秒待機。")
                time.sleep(wait_time)
                continue
            else:
                logging.error("❌ 最大リトライ回数に達しました。データ取得を中止し、空のDataFrameを返します。")
                return pd.DataFrame() # 空のDataFrameを返して呼び出し元で処理させる

# === リアルタイム価格取得関数 (リトライ機能付き) - Minkabu対応 ===
def fetch_current_price() -> float:
    """
    みんかぶ (cc.minkabu.jp) からBTC/USDTの最新の価格をリアルタイムでスクレイピングします（リトライ付き）。
    注意: スクレイピングはサイトの構造変更に弱いため、将来的に機能しなくなる可能性があります。
    """
    max_retries = 3
    # サイトにブロックされないよう、一般的なUser-Agentを設定
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for attempt in range(max_retries):
        try:
            logging.info(f"リアルタイム価格取得中 (ソース: Minkabu)... (試行 {attempt + 1}/{max_retries})")
            
            response = requests.get(MINKABU_URL, headers=headers, timeout=10)
            response.raise_for_status() # HTTPエラーチェック
            html_content = response.text
            
            # --- 価格抽出ロジック (HTML構造に依存) ---
            # Minkabuの主要な価格は、通常、<span class="stock_price">タグ内にあります。
            price_search_key = '<span class="stock_price">'
            
            if price_search_key in html_content:
                # 検索キーの次から次の'</span>'までの文字列を取得
                price_str = html_content.split(price_search_key, 1)[1].split('</span>', 1)[0].strip()
                
                # カンマを除去し、floatに変換
                if price_str:
                    current_price = float(price_str.replace(',', ''))
                    logging.info(f"✅ リアルタイム価格取得成功 (Minkabu): ${current_price:,.2f}")
                    return current_price
                    
            raise ValueError("MinkabuのHTML構造から価格を抽出できませんでした。")

        except requests.exceptions.RequestException as e:
            logging.warning(f"⚠️ Minkabu接続失敗 (試行 {attempt + 1}/{max_retries}): {e}")
        except ValueError as e:
            logging.warning(f"⚠️ 価格抽出失敗 (試行 {attempt + 1}/{max_retries}): {e}")
            
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt * 2 + random.uniform(0, 1) # 2, 4秒待機 (ランダムジッター追加)
            time.sleep(wait_time)
        else:
            logging.error("❌ リアルタイム価格取得の最大リトライ回数に達しました。0.0を返します。")
            return 0.0
# =======================================

def analyze_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    取得したデータフレームにテクニカル指標（MA, RSI, MACD, BB, Stoachastics）を追加します。
    """
    if df.empty:
        return df

    # テクニカル指標の追加
    df.ta.sma(length=50, append=True) # 中期トレンド
    df.ta.sma(length=200, append=True) # 長期トレンド
    df.ta.rsi(length=14, append=True) # 過熱感
    df.ta.macd(fast=12, slow=26, signal=9, append=True) # モメンタム
    df.ta.bbands(length=20, append=True) # ボラティリティ
    df.ta.stoch(k=14, d=3, append=True) # ストキャスティクス (短期過熱感の補完)
    # ===============================================

    logging.info("✅ テクニカル指標の計算完了。")
    return df

# === ピボットポイントの計算関数を強化 ===
def calculate_pivot_levels(df: pd.DataFrame, pivot_type: str = 'Classic') -> tuple[float, float, float, float, float]:
    """
    前日のOHLCデータから指定されたタイプのピボットポイントを算出します。
    返り値: P, R1, S1, R2, S2 (全て丸められた値)
    """
    if len(df) < 2:
        # データが不十分な場合は0を返す
        return 0, 0, 0, 0, 0

    # 最新の完成した足 (前日/前の4時間足) のデータを使用
    prev = df.iloc[-2]
    H, L, C = prev['High'], prev['Low'], prev['Close']

    if pivot_type == 'Classic':
        P = (H + L + C) / 3
        R1 = 2 * P - L
        S1 = 2 * P - H
        R2 = P + (H - L)
        S2 = P - (H - L)
    elif pivot_type == 'Fibonacci':
        # フィボナッチピボット
        P = (H + L + C) / 3
        
        R1 = P + 0.382 * (H - L)
        S1 = P - 0.382 * (H - L)
        R2 = P + 0.618 * (H - L)
        S2 = P - (H - L) 
        
    else: # デフォルトはクラシック
        P, R1, S1, R2, S2 = calculate_pivot_levels(df, 'Classic')

    return tuple(round(level, 2) for level in [P, R1, S1, R2, S2])
# ===============================================

# === バックテスト機能のコアロジック ===
def backtest_strategy(df: pd.DataFrame, initial_capital: float = BACKTEST_CAPITAL) -> dict:
    """
    データフレームに基づき、現在の戦略ロジックをバックテストします。
    """
    df_clean = df.dropna().copy()
    if df_clean.empty or len(df_clean) < 10:
        # データ不足時の処理を強化
        return {
            'trades': 0, 'wins': 0, 'win_rate': 0.0, 'profit_factor': 0.0,
            'max_drawdown': 0.0, 'total_return': 0.0, 'final_capital': initial_capital,
            'error': 'バックテストに必要なデータが不足しています。'
        }
    
    MA_COL = 'SMA_50'
    RSI_COL = 'RSI_14'
    
    capital = initial_capital
    position = 0.0 # ポジションサイズ (プラス: ロング, マイナス: ショート)
    entry_price = 0.0
    trades = []
    
    capital_history = [initial_capital]

    for i in range(1, len(df_clean)):
        current_data = df_clean.iloc[i]
        close = current_data['Close']
        
        # --- 既にポジションを持っている場合 (エグジット条件) ---
        if position > 0: # 買いポジション (ロング) の場合
            # 損切り: MA50の0.5%下を下回った場合、または利益確定: RSIが買われすぎ水準 (75) に達した場合
            if close < current_data[MA_COL] * 0.995 or current_data[RSI_COL] > 75: 
                profit = (close - entry_price) * position
                capital += profit
                trades.append({'type': 'LONG', 'entry': entry_price, 'exit': close, 'profit': profit})
                position = 0.0
        
        elif position < 0: # 売りポジション (ショート) の場合
            # 損切り: MA50の0.5%上を上回った場合、または利益確定: RSIが売られすぎ水準 (25) に達した場合
            if close > current_data[MA_COL] * 1.005 or current_data[RSI_COL] < 25:
                profit = (entry_price - close) * abs(position)
                capital += profit
                trades.append({'type': 'SHORT', 'entry': entry_price, 'exit': close, 'profit': profit})
                position = 0.0

        # --- ポジションを持っていない場合 (エントリー条件) ---
        if position == 0:
            # 買いシグナル: 終値がMA50を上回り、かつRSIが買われすぎ水準ではない
            if close > current_data[MA_COL] * 1.005 and current_data[RSI_COL] < 70:
                position = capital * 0.5 / close # 資本の50%をポジションに割り当てる
                entry_price = close
            
            # 売りシグナル: 終値がMA50を下回り、かつRSIが売られすぎ水準ではない
            elif close < current_data[MA_COL] * 0.995 and current_data[RSI_COL] > 30:
                position = - (capital * 0.5 / close) # ショートポジション
                entry_price = close
                
        capital_history.append(capital)

    # 最終的なクローズ（もしポジションがあれば）
    if position != 0:
        close = df_clean.iloc[-1]['Close']
        if position > 0: # ロング
            profit = (close - entry_price) * position
            trades.append({'type': 'LONG (Final)', 'entry': entry_price, 'exit': close, 'profit': profit})
        else: # ショート
            profit = (entry_price - close) * abs(position)
            trades.append({'type': 'SHORT (Final)', 'entry': entry_price, 'exit': close, 'profit': profit})
        capital += profit
        capital_history[-1] = capital # 最後の資本を更新

    # --- パフォーマンス指標の計算 ---
    total_trades = len(trades)
    if total_trades == 0:
        return {
            'trades': 0, 'wins': 0, 'win_rate': 0.0, 'profit_factor': 0.0,
            'max_drawdown': 0.0, 'total_return': 0.0, 'final_capital': initial_capital,
            'error': 'バックテスト期間中に取引が成立しませんでした。'
        }

    wins = sum(1 for t in trades if t['profit'] > 0)
    total_gross_profit = sum(t['profit'] for t in trades if t['profit'] > 0)
    total_gross_loss = abs(sum(t['profit'] for t in trades if t['profit'] < 0))

    win_rate = (wins / total_trades) * 100

    if total_gross_loss > 0:
        profit_factor = total_gross_profit / total_gross_loss
    else: 
        # 損失がない場合は、利益額をPFとして返す（極端な値にならないように）
        profit_factor = total_gross_profit if total_gross_profit > 0 else 0.0

    equity = pd.Series(capital_history)
    peak = equity.cummax()
    drawdown = (peak - equity) / peak
    max_drawdown = drawdown.max() * 100

    total_return = ((capital - initial_capital) / initial_capital) * 100

    return {
        'trades': total_trades,
        'wins': wins,
        'win_rate': round(win_rate, 2),
        'profit_factor': round(profit_factor, 2),
        'max_drawdown': round(max_drawdown, 2),
        'total_return': round(total_return, 2),
        'final_capital': round(capital, 2)
    }
# ===============================================

# === 戦略生成ロジック ===
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
            'price': price, 'P': price, 'R1': price * 1.01, 'S1': price * 0.99,
            'MA50': price, 'RSI': 50, 'bias': 'データ不足', 'dominance': 'N/A',
            'strategy': '分析に必要な十分な期間のデータが不足しています。', 'details': [], 'predictions': {}
        }

    # --- 1. 日足の最新データとピボットポイントの計算 ---
    latest_long = df_long_clean.iloc[-1]
    P_long, R1_long, S1_long, R2_long, S2_long = calculate_pivot_levels(df_long_clean, 'Classic')
    
    # --- 2. 4時間足の最新データとピボットポイントの計算 ---
    latest_short = df_short_clean.iloc[-1]
    P_short, R1_short, S1_short, R2_short, S2_short = calculate_pivot_levels(df_short_clean, 'Classic')
    
    long_ma50 = latest_long['SMA_50']
    long_ma200 = latest_long['SMA_200']
    rsi = latest_long['RSI_14']
    macd_hist = latest_long['MACDh_12_26_9']
    
    short_ma50 = latest_short['SMA_50']

    # --- 3. 優勢度スコアリング ---
    bull_score = 0
    bear_score = 0
    details = []

    # 長期トレンド (MA200)
    if latest_long['Close'] > long_ma200:
        bull_score += 2
        details.append(f"• *長期トレンド*: 価格はMA200 (`{long_ma200:,.2f}`) の上で、長期的な*強気相場*が優勢です。")
    else:
        bear_score += 2
        details.append(f"• *長期トレンド*: 価格はMA200 (`{long_ma200:,.2f}`) の下で、長期的な*弱気相場*が優勢です。")

    # 中期トレンド (MA50)
    if latest_long['Close'] > long_ma50:
        bull_score += 1
        details.append(f"• *中期トレンド*: 価格がMA50 (`{long_ma50:,.2f}`) を明確に上回り、中期的な*強気トレンド*が優勢です。")
    else:
        bear_score += 1
        details.append(f"• *中期トレンド*: 価格がMA50 (`{long_ma50:,.2f}`) を明確に下回り、中期的な*弱気トレンド*が優勢です。")

    # モメンタム (MACD)
    if macd_hist > 0:
        bull_score += 1
        details.append("• *モメンタム*: MACDがシグナルラインの上にあり、モメンタムは*上昇傾向*です。")
    else:
        bear_score += 1
        details.append("• *モメンタム*: MACDがシグナルラインの下にあり、モメンタムは*下降傾向*です。")

    # 過熱感 (RSI)
    if rsi > 60:
        bear_score += 1 # 買われすぎはショートのサイン
        details.append(f"• *RSI*: 60 (`{rsi:,.2f}`) を上回り、*買われすぎ*の可能性があります。")
    elif rsi < 40:
        bull_score += 1 # 売られすぎはロングのサイン
        details.append(f"• *RSI*: 40 (`{rsi:,.2f}`) を下回り、*売られすぎ*の可能性があります。")
    elif rsi > 50:
        bull_score += 0.5
        details.append(f"• *RSI*: 50 (`{rsi:,.2f}`) を上回り、強いモメンタムが*維持*されています。")
    else:
        bear_score += 0.5
        details.append(f"• *RSI*: 50 (`{rsi:,.2f}`) を下回り、弱いモメンタムが*継続*しています。")

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
        if latest_short['Close'] > short_ma50: # 短期も上向き
            strategy = f"🌟 *最強のロング戦略*。日足S1 ({S1_long_str}) または4h S1 ({S1_short_str}) への*押し目買い*を積極的に検討。"
        else:
            strategy = f"ロング優勢の押し目買い戦略。日足P ({P_long_str}) への短期的な反落時が主な買い場。"
    elif dominance in ["明確なショート優勢 💥", "ショート優勢 📉"]:
        if latest_short['Close'] < short_ma50: # 短期も下向き
            strategy = f"💥 *最強のショート戦略*。日足R1 ({R1_long_str}) または4h R1 ({R1_short_str}) への*戻り売り*を積極的に検討。"
        else:
            strategy = f"ショート優勢の戻り売り戦略。日足P ({P_long_str}) への短期的な反発時が主な売り場。"
    else:
        strategy = f"レンジ相場戦略。日足R1 ({R1_long_str}) 付近での戻り売りと、日足S1 ({S1_long_str}) 付近での押し目買いを検討。"
        
    # --- 6. 短期予測 ---
    # ランダム性を持たせたシンプルな予測
    predictions = {
        '1h後予測': random.choice(['レンジ ↔️', 'レンジ ↔️', '下降 📉', '上昇 📈']),
        '4h後予測': random.choice(['レンジ ↔️', '下降 📉', '下降 📉', '強い下降 💀'] if score_diff < 0 else ['レンジ ↔️', '上昇 📈', '上昇 📈', '強い上昇 🔥']),
        '12h後予測': random.choice(['下降 📉', '強い下降 💀'] if score_diff <= -2 else ['レンジ ↔️', '下降 📉'] if score_diff < 0 else ['レンジ ↔️', '上昇 📈']),
        '24h後予測': random.choice(['強い下降 💀'] if score_diff <= -3 else ['下降 📉'] if score_diff < 0 else ['強い上昇 🔥'] if score_diff >= 3 else ['上昇 📈'])
    }
    
    return {
        'price': latest_long['Close'], # OHLCVの最新終値を初期値として設定
        'P': P_long,
        'R1': R1_long,
        'S1': S1_long,
        'MA50': long_ma50,
        'RSI': rsi,
        'bias': bias,
        'dominance': dominance,
        'strategy': strategy,
        'details': details,
        'predictions': predictions,
        'R1_short': R1_short
    }

# === レポート作成のメインタスク ===
def update_report_data():
    """
    メインの分析タスク。データ取得、分析、バックテスト、レポート生成、通知を行います。
    """
    logging.info("-" * 50)
    logging.info("🤖 レポート更新タスクを開始します...")
    global_data['scheduler_status'] = 'データ取得中'
    
    # 1. データ取得
    df_long = fetch_btc_ohlcv_data(LONG_PERIOD, LONG_INTERVAL)
    df_short = fetch_btc_ohlcv_data(SHORT_PERIOD, SHORT_INTERVAL)

    if df_long.empty or df_short.empty:
        global_data['scheduler_status'] = 'データ取得失敗'
        logging.error("❌ データ取得に失敗しました。レポートを更新できません。")
        # エラー通知
        Thread(target=send_telegram_message, args=("❌ **BTC分析BOTエラー**: 過去データの取得に失敗しました。",)).start()
        return

    # 2. リアルタイム価格の取得 (Minkabu)
    realtime_price = fetch_current_price()
    
    # 3. テクニカル分析
    global_data['scheduler_status'] = '分析実行中'
    df_long_analyzed = analyze_data(df_long)
    df_short_analyzed = analyze_data(df_short)

    # 4. バックテスト
    try:
        backtest_results = backtest_strategy(df_long_analyzed)
    except Exception as e:
        logging.error(f"❌ バックテスト中にエラーが発生しました: {e}", exc_info=True)
        backtest_results = {'error': f"バックテスト失敗: {str(e)}"}
    global_data['backtest'] = backtest_results

    # 5. 戦略と予測の生成
    analysis_result = generate_strategy(df_long_analyzed, df_short_analyzed)

    # **リアルタイム価格の適用とソースの決定**
    price_source = "OHLCV 終値 (最新の足)"
    if realtime_price > 0:
        analysis_result['price'] = realtime_price # リアルタイム価格で上書き
        price_source = "リアルタイム単価 (みんかぶ/BTC_USDT)" # ソースをみんかぶに設定

    # 6. グローバル状態の最終更新
    price = analysis_result['price']
    global_data['last_updated'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    global_data['data_count'] = len(df_long) + len(df_short)
    global_data['scheduler_status'] = '稼働中' # 成功時
    global_data['current_price'] = price # 最新の価格（リアルタイムまたは終値）
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
    R1_short = analysis_result.get('R1_short', 0.0) # 4h R1

    # 価格をカンマ付きで整形
    price_fmt = f"{price:,.2f}"
    P_fmt = f"{P:,.2f}"
    R1_fmt = f"{R1:,.2f}"
    S1_fmt = f"{S1:,.2f}"
    ma50_fmt = f"{ma50:,.2f}"
    
    # バックテスト結果の整形
    bt_error = backtest_results.get('error')
    bt_summary = ""
    if bt_error:
        bt_summary = f"❌ *バックテストエラー*: {bt_error}"
    else:
        bt_summary = f"""
💰 最終資本: ${backtest_results['final_capital']:,.2f} (初期: ${BACKTEST_CAPITAL:,.2f})
📈 総リターン率: {backtest_results['total_return']:,.2f}%
🏆 プロフィットファクター: {backtest_results['profit_factor']:,.2f} (1.0以上が望ましい)
📉 最大ドローダウン (DD): {backtest_results['max_drawdown']:,.2f}% (リスク指標)
📊 取引回数: {backtest_results['trades']} (勝率: {backtest_results['win_rate']:,.2f}%)
        """

    report_message = f"""
👑 BTC実践分析レポート (テクニカルBOT) 👑

📅 最終データ更新: {global_data['last_updated']}
📊 処理データ件数: {len(df_long)} 件 ({LONG_INTERVAL}足) + {len(df_short)} 件 ({SHORT_INTERVAL}足)

🚀 市場の優勢 (Dominance) 🚀
🚨 総合優勢度: {dominance}

--- 主要価格帯と指標 (USD) ---
💰 現在価格 (BTC-USD): ${price_fmt} ({price_source})
🟡 ピボットポイント (P, 日足): ${P_fmt}
🔼 主要レジスタンス (R1, 日足): ${R1_fmt}
🔽 主要サポート (S1, 日足): ${S1_fmt}
💡 中期トレンド転換点 (MA50, 日足): ${ma50_fmt}
🔥 RSI (14期間, 日足): {rsi:,.2f}

--- 動向の詳細分析と根拠 ---
{'\n'.join(details)}

--- 短期動向と予測 ---
• 1h後予測: {predictions['1h後予測']}
• 4h後予測: {predictions['4h後予測']}
• 12h後予測: {predictions['12h後予測']}
• 24h後予測: {predictions['24h後予測']}

--- 総合戦略サマリー ---
🛡️ 推奨戦略: {strategy.replace(f"`${R1_short:,.2f}`", f"`${R1_short:,.2f}`")}

————————————————————
--- バックテスト結果 ({LONG_PERIOD} / {LONG_INTERVAL}足) ---
{bt_summary}

※ この分析は、実戦的なマルチタイムフレーム分析に基づきますが、投資助言ではありません。
"""

    # 8. チャート描画と送信
    global_data['scheduler_status'] = 'チャート描画中'
    try:
        chart_buffer = create_chart(df_long_analyzed, analysis_result)
        caption = f"👑 BTCテクニカル分析レポート\n\n**現在価格**: ${price_fmt}\n**優勢度**: {dominance}\n**推奨戦略**: {strategy}"
        # 画像送信を非同期で実行
        Thread(target=send_telegram_photo, args=(chart_buffer, caption)).start()
    except Exception as e:
        logging.error(f"❌ チャート描画または送信中にエラーが発生しました: {e}", exc_info=True)
        error_caption = f"⚠️ **チャート描画失敗**: {str(e)}"
        Thread(target=send_telegram_message, args=(error_caption,)).start()


    # テキストメッセージは必ず最後に送信
    Thread(target=send_telegram_message, args=(report_message,)).start()

    logging.info("レポート更新タスク完了。通知キューに追加されました。")


# -----------------
# チャート描画関数
# -----------------
def create_chart(df: pd.DataFrame, analysis_result: dict) -> io.BytesIO:
    """
    テクニカル指標と主要レベルを含む価格チャートを生成し、BytesIOバッファとして返します。
    """
    df_plot = df.iloc[-90:].copy() # 直近90日間のみをプロット

    # ボリンジャーバンドのカラム名を確認
    BBU_COL = 'BBU_20_2.0'
    BBL_COL = 'BBL_20_2.0'
    bb_cols_exist = BBU_COL in df_plot.columns and BBL_COL in df_plot.columns

    # 1. メインチャート (価格とMA, BB)
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
    
    # --- 1. 価格ライン ---
    ax.plot(df_plot.index, df_plot['Close'], label='BTC 終値 (USD)', color='#059669', linewidth=2.5)

    # --- 2. テクニカル指標ラインの描画 ---
    ax.plot(df_plot.index, df_plot['SMA_50'], label='SMA 50 (中期)', color='#fbbf24', linestyle='-', linewidth=2, alpha=0.8)
    ax.plot(df_plot.index, df_plot['SMA_200'], label='SMA 200 (長期)', color='#ef4444', linestyle='--', linewidth=1.5, alpha=0.9)
    
    # ボリンジャーバンド (カラムが存在する場合のみ描画)
    if bb_cols_exist:
        ax.plot(df_plot.index, df_plot[BBU_COL], label='BB Upper (+2σ)', color='#ef4444', linestyle=':', linewidth=1)
        ax.plot(df_plot.index, df_plot[BBL_COL], label='BB Lower (-2σ)', color='#3b82f6', linestyle=':', linewidth=1)

    # --- 3. 最新の主要レベルの描画 ---
    # analysis_result['price'] は、リアルタイム価格が取得できていればその値が設定されています。
    price = analysis_result['price']
    P = analysis_result['P']
    R1 = analysis_result['R1']
    S1 = analysis_result['S1']
    
    # ピボットポイント (P)
    ax.axhline(P, color='#9333ea', linestyle='--', linewidth=1.5, alpha=0.8, zorder=0)
    ax.text(df_plot.index[-1], P, f' P: ${P:,.2f}', color='#9333ea', ha='right', va='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
    
    # R1
    ax.axhline(R1, color='red', linestyle='-', linewidth=1, alpha=0.6, zorder=0)
    ax.text(df_plot.index[-1], R1, f' R1: ${R1:,.2f}', color='red', ha='right', va='bottom', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))

    # S1
    ax.axhline(S1, color='blue', linestyle='-', linewidth=1, alpha=0.6, zorder=0)
    ax.text(df_plot.index[-1], S1, f' S1: ${S1:,.2f}', color='blue', ha='right', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
    
    # 現在価格の点とラベル
    if len(df_plot) > 0:
        # チャートの最終データポイントの時刻を使用し、価格は最新の価格を使用
        last_data_time = df_plot.index[-1]
        ax.scatter(last_data_time, price, color='black', s=100, zorder=5)
        ax.text(last_data_time, price, f' 現在 ${price:,.2f}', color='black', ha='right', va='bottom', fontsize=12, weight='bold')

    # 4. グラフの装飾
    ax.set_title(f'{TICKER} 価格推移とテクニカル分析 ({LONG_INTERVAL}足)', fontsize=18, color='#1f2937', weight='bold')
    ax.set_xlabel('日付', fontsize=12)
    ax.set_ylabel('終値 (USD)', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # X軸の日付フォーマット設定
    ax.xaxis.set_major_formatter(DateFormatter('%m/%d'))
    ax.xaxis.set_major_locator(DayLocator(interval=10))
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 画像をメモリに保存
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close(fig) # メモリ解放
    
    return buffer

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
if not scheduler.running:
    app.config.update({
        'SCHEDULER_JOBSTORES': {'default': {'type': 'memory'}},
        'SCHEDULER_EXECUTORS': {'default': {'type': 'threadpool', 'max_workers': 20}},
        'SCHEDULER_API_ENABLED': False
    })

    scheduler.init_app(app)

    # 6時間ごとにupdate_report_dataを実行
    scheduler.add_job(id='report_update_job', func=update_report_data,
                      trigger='interval', hours=6, replace_existing=True)

    scheduler.start()
    logging.info("✅ スケジューラーを開始しました...")
