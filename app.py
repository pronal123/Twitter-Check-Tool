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

# -----------------
# Matplotlib 日本語フォント設定
# -----------------
try:
    # 注: 環境によっては'Noto Sans CJK JP'が利用できない場合があります。その場合はIPAexGothicなどがフォールバックされます。
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'IPAexGothic', 'Hiragino Sans GB', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    # 実行環境によってはフォント設定ができないため、エラーはログに記録し、続行します。
    logging.warning(f"日本語フォント設定に失敗しました: {e}. 英語フォントで続行します。")

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

# === [定義] データインターバルと期間 ===
TICKER = "BTC-USD"
LONG_PERIOD = "1y" # 日足（1d）分析用 - バックテストのため1年間
LONG_INTERVAL = "1d"
SHORT_PERIOD = "30d" # 4時間足（4h）分析用 - 短期戦略
SHORT_INTERVAL = "4h"
BACKTEST_CAPITAL = 100000 # バックテストの初期資本
NEXT_RUN_HOURS = 6 # 次回通知までの時間 (Schedulerの設定と一致させる)
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
    'dominance': 'N/A', # 新しい優勢度フィールド
    'predictions': {},
    'backtest': {} # バックテスト結果を格納
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
        # Markdownを使用 (V2ではないため、\n\nでセクション区切りを確実にする)
        response = requests.post(
            TELEGRAM_API_URL_MESSAGE,
            data={'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'},
            timeout=10
        )
        response.raise_for_status()
        logging.info("✅ Telegramメッセージの送信成功。")
    except requests.exceptions.HTTPError as http_err:
        # HTTP 400エラーの場合、Markdownのパースエラーの可能性
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
        logging.error(f"❌ Telegramチャート画像の送信中に予期せぬエラーが発生しました: {e}")


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

            # yfinanceのFutureWarningを抑制するためにauto_adjustを明示的にTrueに設定
            df = yf.download(TICKER, period=period, interval=interval, progress=False, auto_adjust=True)

            if df.empty:
                raise ValueError("取得したデータが空です。レート制限の可能性があります。")

            # MultiIndexフラット化
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.index.name = 'Date'
            if 'Close' not in df.columns:
                raise KeyError("'Close'カラムが見つかりません。")

            df['Close'] = df['Close'].round(2)
            logging.info(f"✅ 過去データ取得成功。件数: {len(df)} ({interval})")
            return df

        except Exception as e:
            logging.error(f"❌ yfinanceからデータ取得中にエラーが発生しました: {e}")
            if "Rate limited" in str(e) or "取得したデータが空です" in str(e):
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt * 5 + random.randint(1, 5)
                    logging.warning(f"⚠️ レート制限の可能性があります。{wait_time}秒待ってリトライします (試行 {attempt + 2}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    logging.error("❌ 最大リトライ回数に達しました。データ取得を中止します。")
                    return pd.DataFrame()

            return pd.DataFrame()

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
        S2 = P - 0.618 * (H - L)
        
    else: # デフォルトはクラシック
        P, R1, S1, R2, S2 = calculate_pivot_levels(df, 'Classic')

    return tuple(round(level, 2) for level in [P, R1, S1, R2, S2])
# ===============================================

# === バックテスト機能のコアロジック ===
def backtest_strategy(df: pd.DataFrame, initial_capital: float = BACKTEST_CAPITAL) -> dict:
    """
    データフレームに基づき、現在の戦略ロジックをバックテストします。
    日足データを使用し、MAとRSIに基づくトレンドフォロー戦略を適用します。
    """
    df_clean = df.dropna().copy()
    if df_clean.empty:
        return {
            'trades': 0, 'wins': 0, 'win_rate': 0.0, 'profit_factor': 0.0,
            'max_drawdown': 0.0, 'total_return': 0.0, 'final_capital': initial_capital
        }
    
    # 使用するテクニカル指標のカラム名
    MA_COL = 'SMA_50'
    RSI_COL = 'RSI_14'
    
    capital = initial_capital
    position = 0.0 # ポジションサイズ (0: ノーポジション, 正: ロング, 負: ショート)
    entry_price = 0.0
    trades = []
    
    capital_history = [initial_capital]

    for i in range(1, len(df_clean)):
        current_data = df_clean.iloc[i]
        close = current_data['Close']
        
        # --- 既にポジションを持っている場合 (エグジット条件) ---
        if position > 0: # 買いポジション (ロング) の場合
            # 売りシグナル（MA50を下にクロス、またはRSIが買われすぎ反転）でエグジット
            # 修正: MA50を下回った、またはRSIの過熱感から反転
            if close < current_data[MA_COL] * 0.995 or current_data[RSI_COL] > 75: 
                profit = (close - entry_price) * position # 利益を計算
                capital += profit
                trades.append({'type': 'LONG', 'entry': entry_price, 'exit': close, 'profit': profit})
                position = 0.0
        
        elif position < 0: # 売りポジション (ショート) の場合
            # 買いシグナル（MA50を上にクロス、またはRSIが売られすぎ反転）でエグジット
            # 修正: MA50を上回った、またはRSIの売られすぎから反転
            if close > current_data[MA_COL] * 1.005 or current_data[RSI_COL] < 25:
                profit = (entry_price - close) * abs(position) # 利益を計算 (ショートは逆算)
                capital += profit
                trades.append({'type': 'SHORT', 'entry': entry_price, 'exit': close, 'profit': profit})
                position = 0.0

        # --- ポジションを持っていない場合 (エントリー条件) ---
        if position == 0:
            # 買いシグナル: 終値がMA50を上回り、かつRSIが買われすぎ水準ではない
            if close > current_data[MA_COL] * 1.005 and current_data[RSI_COL] < 70:
                # 資本の50%をポジションに割り当てる (レバレッジなし)
                position = capital * 0.5 / close 
                entry_price = close
            
            # 売りシグナル: 終値がMA50を下回り、かつRSIが売られすぎ水準ではない
            elif close < current_data[MA_COL] * 0.995 and current_data[RSI_COL] > 30:
                position = - (capital * 0.5 / close) # ショートポジション
                entry_price = close
        
        # 各足での資本状況を記録 (未決済ポジションの含み益/含み損を考慮)
        current_equity = capital + (close - entry_price) * position if position != 0 else capital
        capital_history.append(current_equity)


    # --- パフォーマンス指標の計算 ---
    total_trades = len(trades)
    if total_trades == 0:
         # 取引がなかった場合
         return {
            'trades': 0, 'wins': 0, 'win_rate': 0.0, 'profit_factor': 0.0,
            'max_drawdown': 0.0, 'total_return': 0.0, 'final_capital': initial_capital
        }
    
    # 勝率と総利益/総損失の計算
    wins = sum(1 for t in trades if t['profit'] > 0)
    total_gross_profit = sum(t['profit'] for t in trades if t['profit'] > 0)
    total_gross_loss = abs(sum(t['profit'] for t in trades if t['profit'] < 0))
    
    win_rate = (wins / total_trades) * 100
    
    # プロフィットファクター (PF) の計算
    if total_gross_loss > 0:
        profit_factor = total_gross_profit / total_gross_loss
    else:
        profit_factor = total_gross_profit if total_gross_profit > 0 else 0.0

    # 最大ドローダウン (MDD) の計算
    equity = pd.Series(capital_history)
    peak = equity.cummax()
    drawdown = (peak - equity) / peak
    max_drawdown = drawdown.max() * 100
    
    # トータルリターンの計算
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
            'bias': 'データ不足', 'dominance': 'N/A', # 初期値
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
    # 4時間足のピボットR1, S1を再計算または取得 (ここでは日足と同じクラシックを使用し、4時間足データで計算)
    _, R1_short, S1_short, _, _ = calculate_pivot_levels(df_short, 'Classic')
    short_ma50 = latest_short['SMA_50']

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
        if latest[MACD_COL] > latest[MACDs_COL]:
            details.append("• *モメンタム*: MACDがシグナルラインの上にあり、モメンタムは*上昇*傾向です。")
            bull_score += 1
        elif latest[MACD_COL] < latest[MACDs_COL]:
            details.append("• *モメンタム*: MACDがシグナルラインの下にあり、モメンタムは*下降*傾向です。")
            bear_score += 1

    # --- 3. 過熱感 (RSI) ---
    if rsi > 70:
        details.append(f"• *RSI*: 70 (`{rsi:,.2f}`) を超え、*買われすぎ*を示唆。短期的な調整（利確売り）に警戒。")
        bear_score += 1 # 買われすぎは短期的な弱気要因
    elif rsi < 30:
        details.append(f"• *RSI*: 30 (`{rsi:,.2f}`) を下回り、*売られすぎ*を示唆。短期的な反発（押し目買い）のチャンス。")
        bull_score += 1 # 売られすぎは短期的な強気要因
    elif rsi > 50:
        details.append(f"• *RSI*: 50 (`{rsi:,.2f}`) を上回り、強いモメンタムが*維持*されています。")
    else:
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
            strategy = f"ショート優勢の戻り売り戦略。日足P ({P_long_str}) への短期的な上昇時が主な売り場。"
    elif dominance == "中立/レンジ ↔️":
        BBB_COL = 'BBB_20_2.0_2.0' 
        bbb = latest[BBB_COL] if BBB_COL in latest else 100 

        if bbb < 10: # ボラティリティ圧縮
             strategy = f"ボラティリティ圧縮中。日足R1 ({R1_long_str}) / S1 ({S1_long_str}) の*ブレイクアウト待ち*。"
        else:
             strategy = f"レンジ取引。日足S1 ({S1_long_str}) 付近で買い、日足R1 ({R1_long_str}) 付近で売り。"

    # --- 短期予測の強化 (MACD, 短期MA50, ピボット基準) ---
    predictions = {
        # 1hは短期モメンタム(4h MACD)
        "1h": "強い上昇 🚀" if latest_short['MACDh_12_26_9'] > 0 and latest_short['Close'] > short_ma50 else "強い下降 📉" if latest_short['MACDh_12_26_9'] < 0 and latest_short['Close'] < short_ma50 else "レンジ ↔️",
        # 4hは短期トレンド(4h MA50)
        "4h": "上昇 📈" if latest_short['Close'] > short_ma50 else "下降 📉",
        # 12hは日足のピボットPに対する位置
        "12h": "上昇 📈" if price > P_long else "下降 📉",
        # 24hは総合バイアス
        "24h": bias
    }

    return {
        'price': price,
        'P': P_long, 'R1': R1_long, 'S1': S1_long, 'MA50': ma50, 'RSI': rsi,
        'bias': bias,
        'dominance': dominance, # 優勢度を追加
        'strategy': strategy,
        'details': details,
        'predictions': predictions
    }


def generate_chart_image(df: pd.DataFrame, analysis_result: dict) -> io.BytesIO:
    """
    終値と主要なテクニカル指標を含むチャート画像を生成します。
    """
    # 修正: pandas_taの命名規則に合わせてカラム名を変更
    BBU_COL = 'BBU_20_2.0_2.0'
    BBL_COL = 'BBL_20_2.0_2.0'
    
    required_cols = ['Close', 'High', 'Low', 'SMA_50', 'SMA_200', BBU_COL, BBL_COL]
    
    # NaN行を削除してから描画に渡す（描画エラーを防ぐため）
    df_plot = df.dropna(subset=['Close', 'SMA_50']).copy() 
    
    # 必要なカラムが全て存在するか確認
    if not all(col in df_plot.columns for col in required_cols):
        logging.error(f"チャート描画に必要なカラムの一部が不足しています。利用可能なカラム: {df_plot.columns.tolist()}")
        return io.BytesIO()


    fig, ax = plt.subplots(figsize=(12, 7), dpi=100) # チャートサイズを少し大きく
    
    # --- 1. 価格ライン ---
    ax.plot(df_plot.index, df_plot['Close'], label='BTC 終値 (USD)', color='#059669', linewidth=2.5) # ラインを太く

    # --- 2. テクニカル指標ラインの描画 ---
    # 50日移動平均線 (MA50)
    ax.plot(df_plot.index, df_plot['SMA_50'], label='SMA 50 (中期)', color='#fbbf24', linestyle='-', linewidth=2, alpha=0.8) 
    # 200日移動平均線 (MA200) - 長期トレンド
    ax.plot(df_plot.index, df_plot['SMA_200'], label='SMA 200 (長期)', color='#ef4444', linestyle='--', linewidth=1.5, alpha=0.9)

    # ボリンジャーバンド (Upper/Lower Band)
    ax.plot(df_plot.index, df_plot[BBU_COL], label='BB Upper (+2σ)', color='#ef4444', linestyle=':', linewidth=1)
    ax.plot(df_plot.index, df_plot[BBL_COL], label='BB Lower (-2σ)', color='#3b82f6', linestyle=':', linewidth=1)

    # --- 3. 最新の主要レベルの描画 ---
    price = analysis_result['price']
    P = analysis_result['P']

    # ピボットポイント (P)
    ax.axhline(P, color='#9333ea', linestyle='--', linewidth=1.5, alpha=0.8, zorder=0)
    ax.text(df_plot.index[-1], P, f' P: ${P:,.2f}', color='#9333ea', ha='right', va='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))

    # 現在価格の点とラベル
    if len(df_plot) > 0:
        ax.scatter(df_plot.index[-1], price, color='black', s=100, zorder=5) # 点を大きく
        ax.text(df_plot.index[-1], price, f' 現在 ${price:,.2f}', color='black', ha='right', va='bottom', fontsize=12, weight='bold')

    # 4. グラフの装飾
    ax.set_title(f'{TICKER} 価格推移とテクニカル分析 ({LONG_INTERVAL}足)', fontsize=18, color='#1f2937', weight='bold')
    ax.set_xlabel('日付', fontsize=12)
    ax.set_ylabel('終値 (USD)', fontsize=12)

    formatter = DateFormatter("%m/%d")
    ax.xaxis.set_major_formatter(formatter)

    # データを間引いて表示するためにDayLocatorを設定
    if len(df_plot.index) > 15:
        # X軸ラベルが見やすくなるように間隔を調整
        ax.xaxis.set_major_locator(DayLocator(interval=math.ceil(len(df_plot.index) / 8)))
    else:
        ax.xaxis.set_major_locator(DayLocator())

    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', fontsize=10)
    plt.tight_layout()

    # 5. 画像をメモリ上のバイトストリームとして保存
    buf = io.BytesIO()
    plt.figure(fig.number)
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)

    return buf


# -----------------
# スケジューリングタスク
# -----------------
def update_report_data():
    """定期的に実行されるタスク：データ取得、分析、レポート更新、バックテストの実行"""
    global global_data

    logging.info("スケジュールされたレポート更新タスク開始（実践分析モード）...")
    now = datetime.datetime.now()
    last_updated_str = now.strftime('%Y-%m-%d %H:%M:%S')

    # --- 次回通知時間の計算 ---
    # NEXT_RUN_HOURS = 6時間 (グローバル定数を使用)
    next_run_time = now + datetime.timedelta(hours=NEXT_RUN_HOURS)
    # タイムゾーン情報がないため、JSTであることを仮定してメッセージに含める
    next_run_str = next_run_time.strftime('%Y-%m-%d %H:%M:%S JST') 
    # --------------------------

    # 1. データ取得 (日足と4時間足)
    df_long = fetch_btc_ohlcv_data(LONG_PERIOD, LONG_INTERVAL)
    df_short = fetch_btc_ohlcv_data(SHORT_PERIOD, SHORT_INTERVAL)

    # データが空の場合の処理
    if df_long.empty or df_short.empty:
        logging.error("致命的エラー: データ取得に失敗したため、レポートを生成できません。")
        global_data['scheduler_status'] = 'エラー'
        global_data['strategy'] = 'データ取得エラー'
        error_msg = f"❌ *BTC分析レポート生成エラー*\n\nデータ取得に失敗しました。ネットワーク接続を確認するか、数分後に再試行してください。\n最終更新: {last_updated_str}"
        Thread(target=send_telegram_message, args=(error_msg,)).start()
        return

    # 2. テクニカル分析
    try:
        df_long_analyzed = analyze_data(df_long)
        df_short_analyzed = analyze_data(df_short) # 短期分析も実行
    except Exception as e:
        logging.error(f"致命的エラー: テクニカル分析中にエラーが発生しました: {e}", exc_info=True)
        global_data['scheduler_status'] = 'エラー'
        error_msg = f"❌ *BTC分析レポート生成エラー*\n\nテクニカル分析中にエラーが発生しました。\n詳細: {str(e)}\n最終更新: {last_updated_str}"
        Thread(target=send_telegram_message, args=(error_msg,)).start()
        return

    # 3. バックテストの実行 (日足データを使用)
    try:
        logging.info(f"バックテスト実行中... 期間: {LONG_PERIOD}")
        backtest_results = backtest_strategy(df_long_analyzed) 
        global_data['backtest'] = backtest_results
        logging.info("✅ バックテスト完了。")
    except Exception as e:
        logging.error(f"❌ バックテスト中にエラーが発生しました: {e}", exc_info=True)
        backtest_results = {'error': f"バックテスト失敗: {str(e)}"}
        global_data['backtest'] = backtest_results

    # 4. 戦略と予測の生成 (日足と4時間足の両方を使用)
    analysis_result = generate_strategy(df_long_analyzed, df_short_analyzed)

    # 5. グローバル状態の更新
    global_data['last_updated'] = last_updated_str
    global_data['data_count'] = len(df_long) + len(df_short) 
    global_data['scheduler_status'] = '稼働中'
    global_data['current_price'] = analysis_result['price']
    global_data['strategy'] = analysis_result['strategy']
    global_data['bias'] = analysis_result['bias']
    global_data['dominance'] = analysis_result['dominance'] # 優勢度を更新
    global_data['predictions'] = analysis_result['predictions']

    # 6. レポートの整形 (改行と優勢度の強調)
    price = analysis_result['price']
    P, R1, S1, ma50, rsi = analysis_result['P'], analysis_result['R1'], analysis_result['S1'], analysis_result['MA50'], analysis_result['RSI']
    dominance = analysis_result['dominance'] # 優勢度
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

    # --- Markdown整形を強化 ---
    price_analysis = [
        f"💰 *現在価格 (BTC-USD)*: {formatted_current_price}",
        f"🟡 *ピボットポイント (P, 日足)*: {formatted_P}",
        f"🔼 *主要レジスタンス (R1, 日足)*: {formatted_R1}",
        f"🔽 *主要サポート (S1, 日足)*: {formatted_S1}",
        f"💡 *中期トレンド転換点 (MA50, 日足)*: {formatted_MA50}",
        f"🔥 *RSI (14期間, 日足)*: {formatted_RSI}"
    ]

    prediction_lines = [f"• {tf}後予測: *{predictions[tf]}*" for tf in ["1h", "4h", "12h", "24h"]]

    report_message = (
        f"👑 *BTC実践分析レポート (テクニカルBOT)* 👑\n\n"
        
        f"📅 *最終データ更新*: `{last_updated_str}`\n"
        f"⏰ *次回通知予定*: *`{next_run_str}`* (約 {NEXT_RUN_HOURS}時間後)\n"
        f"📊 *処理データ件数*: *{len(df_long)}* 件 ({LONG_INTERVAL}足) + *{len(df_short)}* 件 ({SHORT_INTERVAL}足)\n\n" 
        
        # --- 市場優勢度の強調 ---
        f"**🚀 市場の優勢 (Dominance) 🚀**\n"
        f"🚨 *総合優勢度*: *{dominance}*\n\n"
        
        f"--- *主要価格帯と指標 (USD)* ---\n"
        # FIX: リストを単一改行文字 ('\n') で結合
        f"{'\n'.join(price_analysis)}\n\n" 
        
        f"--- *動向の詳細分析と根拠* ---\n"
        # FIX: リストを単一改行文字 ('\n') で結合
        f"{'\n'.join(details)}\n\n" 
        
        f"--- *短期動向と予測* ---\n"
        # FIX: リストを単一改行文字 ('\n') で結合
        f"{'\n'.join(prediction_lines)}\n\n"
        
        f"--- *総合戦略サマリー* ---\n"
        f"🛡️ *推奨戦略*: *{strategy}*\n\n"
    )
    
    # --- バックテスト結果のレポートへの追加 ---
    if 'error' in backtest_results:
        backtest_lines = [f"⚠️ *バックテスト結果*: {backtest_results['error']}"]
    else:
        backtest_lines = [
            f"--- *戦略バックテスト結果 ({LONG_PERIOD} / {LONG_INTERVAL}足)* ---",
            f"💰 *最終資本*: `\$ {backtest_results['final_capital']:,.2f}` (初期: `\$ {BACKTEST_CAPITAL:,.2f}`)",
            f"📈 *総リターン率*: *{backtest_results['total_return']}%*",
            f"🏆 *プロフィットファクター*: `{backtest_results['profit_factor']}` (1.0以上が望ましい)",
            f"📉 *最大ドローダウン (DD)*: `{backtest_results['max_drawdown']}%` (リスク指標)",
            f"📊 *取引実績*: `{backtest_results['trades']}` 回の取引 (勝率: `{backtest_results['win_rate']}%`)"
        ]

    report_message += (
        f"{chr(8212) * 20}\n" # 区切り線
        # FIX: リストを単一改行文字 ('\n') で結合
        f"{'\n'.join(backtest_lines)}\n\n" 
        f"_※ この分析は、実戦的なマルチタイムフレーム分析に基づきますが、投資助言ではありません。_"
    )


    # 7. 画像生成と通知の実行
    try:
        logging.info("チャート画像を生成中...")
        chart_buffer = generate_chart_image(df_long_analyzed, analysis_result)
        
        photo_caption = (
            f"📈 *BTC実践分析チャート ({LONG_INTERVAL}足)* 📉\n"
            f"📅 更新: `{last_updated_str}`\n"
            f"💰 現在価格: {formatted_current_price}\n"
            f"🚨 *優勢度*: *{dominance}*\n" # 優勢度を画像キャプションにも追加
            f"🛡️ *推奨戦略*: {strategy}\n"
            f"_詳細は別途送信されるテキストレポートをご確認ください。_"
        )

        # チャートバッファが空でないことを確認してから送信
        if chart_buffer.getbuffer().nbytes > 0:
            Thread(target=send_telegram_photo, args=(chart_buffer, photo_caption)).start()
        else:
             logging.error("❌ チャート画像のバッファが空です。画像送信をスキップしました。")
             error_caption = f"⚠️ *チャート生成失敗*\n\nデータは正常に処理されましたが、チャート画像生成中にエラーが発生しました。\n最終更新: {last_updated_str}"
             Thread(target=send_telegram_message, args=(error_caption,)).start()


    except Exception as e:
        logging.error(f"❌ チャート画像の生成または送信に失敗しました: {e}", exc_info=True)
        error_caption = f"⚠️ *チャート生成失敗*\n\nデータは正常に処理されましたが、チャート画像生成中に予期せぬエラーが発生しました。\nエラー詳細: {str(e)[:100]}...\n最終更新: {last_updated_str}"
        Thread(target=send_telegram_message, args=(error_caption,)).start()


    # テキストメッセージは必ず最後に送信
    Thread(target=send_telegram_message, args=(report_message,)).start()

    logging.info("レポート更新タスク完了。通知キューに追加されました。")


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
                      trigger='interval', hours=NEXT_RUN_HOURS, replace_existing=True) # 定義した定数を使用

    scheduler.start()
    logging.info("✅ スケジューラーを開始しました。")

# アプリ起動時に初回実行をトリガー
Thread(target=update_report_data).start()

# -----------------
# サーバーの実行
# -----------------
if __name__ == '__main__':
    # 開発環境向けのデバッグモードをオフにし、本番環境向けの実行
    port = int(os.environ.get('PORT', 5000))
    logging.info(f"ローカルサーバーを {port} ポートで開始します。")
    app.run(host='0.0.0.0', port=port)
