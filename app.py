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
            if 'Volume' not in df.columns: # Volumeカラムのチェックを追加
                df['Volume'] = 0 # 存在しない場合は0埋め
            
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
    取得したデータフレームにテクニカル指標（MA, RSI, MACD, BB, Stoachastics, VMA）を追加します。
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
    
    # === NEW: 出来高分析の追加 (Volume Moving Average) ===
    # 出来高の移動平均 (20期間)
    df.ta.sma(close=df['Volume'], length=20, prefix='VMA', append=True) 
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
    # NOTE: バックテスト時はNaN値を削除して完全にデータが揃った期間で実行
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
    出来高分析を追加。
    """
    # NOTE: .iloc[-1]で最新の行を取得する際、NaN値で KeyError になるのを避けるため
    # dropna()せずに、iloc[-1]を取得し、辞書アクセスでget()を使用してデフォルト値を設定する
    if df_long.empty or df_short.empty:
        price = df_long['Close'].iloc[-1] if not df_long.empty and 'Close' in df_long.columns else 0
        return {
            'price': price, 'P': price, 'R1': price * 1.01, 'S1': price * 0.99, 'MA50': price, 'RSI': 50,
            'R2_long': price * 1.02, 'S2_long': price * 0.98, 'R1_short': price * 1.005, 'S1_short': price * 0.995,
            'MA200': price, 'BBW': 0, 'StochK_long': 50, 'StochD_long': 50,
            'ShortRSI': 50, 'ShortMACDH': 0, 'ShortStochK': 50,
            'Volume': 0, 'VMA': 0, 'VolumeRatio': 0, # NEW: 出来高情報
            'bias': 'データ不足', 'dominance': 'N/A', # 初期値
            'strategy': '分析に必要な十分な期間のデータが揃っていません。',
            'details': ['分析に必要な十分な期間のデータが揃っていません。'],
            'predictions': {'1h': 'N/A', '4h': 'N/A', '12h': 'N/A', '24h': 'N/A'}
        }


    latest = df_long.iloc[-1]
    latest_short = df_short.iloc[-1]

    # 日足の指標値 (getメソッドを使用して、KeyErrorを防ぐ)
    price = latest.get('Close', 0)
    ma50 = latest.get('SMA_50', price)
    ma200 = latest.get('SMA_200', price)
    rsi = latest.get('RSI_14', 50)
    bbw = latest.get('BBW_20_2.0_2.0', 0) # ボリンジャーバンド幅 (BBW)
    stoch_k_long = latest.get('STOCHk_14_3_3', 50)
    stoch_d_long = latest.get('STOCHd_14_3_3', 50)
    
    # === NEW: 出来高関連の指標 (KeyErrorが発生していた箇所) ===
    volume = latest.get('Volume', 0)
    vma = latest.get('VMA_20', volume if volume > 0 else 1) # VMAがない場合は現在の出来高を使用 (または1で割るのを避ける)
    # VMAが0の場合はエラーを避けるために1を使用
    volume_ratio = (volume / vma) * 100 if vma > 0 else 0
    is_volume_surge = volume_ratio > 150 # 出来高がVMAの150%を超えたら急増と判断
    # ===============================

    # ピボットポイントの計算 (日足データでクラシックピボットを使用)
    P_long, R1_long, S1_long, R2_long, S2_long = calculate_pivot_levels(df_long, 'Classic') # R2, S2も取得

    # 短期（4時間足）の分析 (getメソッドを使用して、KeyErrorを防ぐ)
    P_short, R1_short, S1_short, R2_short, S2_short = calculate_pivot_levels(df_short, 'Fibonacci')
    short_ma50 = latest_short.get('SMA_50', price)
    short_rsi = latest_short.get('RSI_14', 50) # 4h RSI
    short_macd_h = latest_short.get('MACDh_12_26_9', 0) # 4h MACD Hist
    short_stoch_k = latest_short.get('STOCHk_14_3_3', 50) # 4h Stoch K


    # 総合バイアスと戦略の決定
    bias = "中立"
    strategy = "様子見（ブレイクアウト待ち）"
    details = []
    bull_score = 0
    bear_score = 0
    
    # MACDの比較に必要な値も安全に取得
    macd_long = latest.get('MACD_12_26_9', 0)
    macds_long = latest.get('MACDs_12_26_9', 0)


    # --- 1. 長期・中期トレンドバイアス (日足 MA) ---
    if price > ma200:
        details.append(f"• *長期トレンド*: 価格 (`{price:,.2f}`) はMA200 (`{ma200:,.2f}`) を上回り、*長期的な強気相場*です。")
        bull_score += 2
    else:
        details.append(f"• *長期トレンド*: 価格はMA200 (`{ma200:,.2f}`) の下で、長期的な弱気相場が優勢です。")
        bear_score += 2

    if price > ma50 * 1.005:
        details.append(f"• *中期トレンド*: 価格がMA50 (`{ma50:,.2f}`) を明確に上回り、中期的に強い強気トレンドが優勢です。")
        bull_score += 1
    elif price < ma50 * 0.995:
        details.append(f"• *中期トレンド*: 価格がMA50 (`{ma50:,.2f}`) を明確に下回り、中期的な弱気トレンドが優勢です。")
        bear_score += 1
    else:
        details.append(f"• *中期トレンド*: 価格はMA50 (`{ma50:,.2f}`) 付近で推移しており、レンジ相場が想定されます。")

    # --- 2. モメンタムシグナル (MACDとRSI 50ライン) ---
    if macd_long > macds_long:
        details.append("• *モメンタム (日足)*: MACDがシグナルラインの上にあり、モメンタムは*上昇*傾向です。")
        bull_score += 1
    elif macd_long < macds_long:
        details.append("• *モメンタム (日足)*: MACDがシグナルラインの下にあり、モメンタムは*下降*傾向です。")
        bear_score += 1

    # --- 3. 過熱感 (RSI) ---
    if rsi > 70:
        details.append(f"• *RSI (日足)*: 70 (`{rsi:,.2f}`) を超え、*買われすぎ*を示唆。短期的な調整（利確売り）に警戒。")
        bear_score += 1 # 買われすぎは短期的な弱気要因
    elif rsi < 30:
        details.append(f"• *RSI (日足)*: 30 (`{rsi:,.2f}`) を下回り、*売られすぎ*を示唆。短期的な反発（押し目買い）のチャンス。")
        bull_score += 1 # 売られすぎは短期的な強気要因
    elif rsi > 50:
        details.append(f"• *RSI (日足)*: 50 (`{rsi:,.2f}`) を上回り、強いモメンタムが*維持*されています。")
    else:
        details.append(f"• *RSI (日足)*: 50 (`{rsi:,.2f}`) を下回り、弱いモメンタムが*継続*しています。")
        
    # --- 4. Stochastics (日足) ---
    if stoch_k_long > 80 and stoch_d_long > 80:
        details.append("• *ストキャスティクス (日足)*: 買われすぎ水準。日足の*利確売り*に注意が必要です。")
        bear_score += 0.5
    elif stoch_k_long < 20 and stoch_d_long < 20:
        details.append("• *ストキャスティクス (日足)*: 売られすぎ水準。日足の*反発の可能性*があります。")
        bull_score += 0.5
        
    # --- 5. Volatility Analysis (日足 BBW) ---
    if bbw < 5:
        details.append(f"• *ボラティリティ (日足)*: BB幅 (`{bbw:,.2f}%`) が極端に狭く、*大相場前のエネルギー蓄積*を示唆します（ブレイクアウトに注意）。")
    elif bbw > 15:
        details.append(f"• *ボラティリティ (日足)*: BB幅 (`{bbw:,.2f}%`) が広く、*ボラティリティが高止まり*しており、調整（レンジ回帰）リスクがあります。")
    else:
        details.append(f"• *ボラティリティ (日足)*: BB幅 (`{bbw:,.2f}%`) は平均的で、通常のトレンド継続またはレンジを想定します。")
        
    # --- 6. Short-term Analysis (4h) ---
    details.append(f"• *短期RSI (4h)*: `{short_rsi:,.2f}`。{( '70超えで買われすぎ' if short_rsi > 70 else '30未満で売られすぎ' if short_rsi < 30 else '中立水準')}")
    if short_macd_h > 0:
        details.append(f"• *短期モメンタム (4h MACD Hist)*: ポジティブ (`{short_macd_h:,.2f}`)。短期的には*上昇圧力が強い*です。")
        bull_score += 0.5
    elif short_macd_h < 0:
        details.append(f"• *短期モメンタム (4h MACD Hist)*: ネガティブ (`{short_macd_h:,.2f}`)。短期的には*下降圧力が強い*です。")
        bear_score += 0.5

    # --- NEW: 7. 出来高分析 (Volume) ---
    if is_volume_surge:
        volume_msg = f"• *出来高*: 過去20日の平均出来高 (`{vma:,.0f}`) に対し、*出来高が急増* (`{volume:,.0f}` / {volume_ratio:,.0f}%) しています。"
        
        # 出来高を伴う価格変動はトレンドの信頼性を高める
        if price > ma50: # 価格がMA50より上で、出来高急増 = 上昇の信頼性強化
            details.append(volume_msg + "価格がMA50を上回っているため、上昇トレンドの*信頼性が高い*です。")
            bull_score += 1.5
        elif price < ma50: # 価格がMA50より下で、出来高急増 = 下降の信頼性強化
            details.append(volume_msg + "価格がMA50を下回っているため、下降トレンドの*信頼性が高い*です。")
            bear_score += 1.5
        else:
            details.append(volume_msg + "*レンジブレイク*の兆候か、*大きなトレンド転換*を示唆します。")
            
    else:
        details.append(f"• *出来高*: 出来高は平均的 (`{volume:,.0f}` / VMA20: `{vma:,.0f}`) で、トレンドの信頼性に関する特筆すべきシグナルはありません。")

    # --- 8. 総合バイアスの決定 ---
    score_diff = bull_score - bear_score
    
    if score_diff >= 4.5:
        dominance = "明確なロング優勢 🚀"
        bias = "強い上昇"
    elif score_diff >= 1.5:
        dominance = "ロング優勢 📈"
        bias = "上昇"
    elif score_diff <= -4.5:
        dominance = "明確なショート優勢 💥"
        bias = "強い下降"
    elif score_diff <= -1.5:
        dominance = "ショート優勢 📉"
        bias = "下降"
    else:
        dominance = "中立/レンジ ↔️"
        bias = "レンジ/中立"

    # --- 9. 総合戦略の決定 (RSIと出来高を考慮) ---
    R1_long_str = f"`${R1_long:,.2f}`"
    S1_long_str = f"`${S1_long:,.2f}`"
    P_long_str = f"`${P_long:,.2f}`"
    S1_short_str = f"`${S1_short:,.2f}`"
    R1_short_str = f"`${R1_short:,.2f}`"


    if dominance in ["明確なロング優勢 🚀", "ロング優勢 📈"]:
        if is_volume_surge:
            strategy = f"🚀 *ブレイクアウト伴う最強のロング戦略*。出来高が急増し、上昇トレンドの確度が高い。日足S1 ({S1_long_str}) への押し目買いを積極的に検討。"
        # RSIが買われすぎ水準の場合、短期調整を警戒
        elif rsi > 70 or short_rsi > 70: 
            strategy = f"🚨 *短期調整警戒のロング戦略*。中期はロング優勢だが、RSIが買われすぎ水準。短期的な調整（利確売り）を警戒し、日足S1 ({S1_long_str}) での押し目買いを待つ。"
        elif latest_short.get('Close', price) > short_ma50: # 短期も上向き
            strategy = f"🌟 *ロング優勢の押し目買い戦略*。日足S1 ({S1_long_str}) または4h S1 ({S1_short_str}) への*押し目買い*を検討。"
        else:
            strategy = f"ロング優勢の押し目買い戦略。日足P ({P_long_str}) への短期的な反落時が主な買い場。"
            
    elif dominance in ["明確なショート優勢 💥", "ショート優勢 📉"]:
        if is_volume_surge:
            strategy = f"💥 *ブレイクアウト伴う最強のショート戦略*。出来高が急増し、下降トレンドの確度が高い。日足R1 ({R1_long_str}) への戻り売りを積極的に検討。"
        # RSIが売られすぎ水準の場合、短期反発を警戒 (現在のレポートの状況を反映)
        elif rsi < 30 or short_rsi < 30: 
            strategy = f"💡 *短期反発警戒のショート戦略*。中期はショート優勢だが、RSIが売られすぎ水準。短期的な反発（押し目買い）を待ってから、日足R1 ({R1_long_str}) または4h R1 ({R1_short_str}) への*戻り売り*を検討。"
        elif latest_short.get('Close', price) < short_ma50: # 短期も下向き
            strategy = f"📉 *ショート優勢の戻り売り戦略*。日足R1 ({R1_long_str}) または4h R1 ({R1_short_str}) への*戻り売り*を検討。"
        else:
            strategy = f"ショート優勢の戻り売り戦略。日足P ({P_long_str}) への短期的な上昇時が主な売り場。"
            
    elif dominance == "中立/レンジ ↔️":
        BBB_COL = 'BBB_20_2.0_2.0' 
        # BBBも安全に取得
        bbb = latest.get(BBB_COL, 100)

        if is_volume_surge: # レンジでも出来高急増
             strategy = f"🚨 *出来高を伴うブレイクアウト警戒*。日足R1 ({R1_long_str}) / S1 ({S1_long_str}) のどちらに抜けるか注意深く監視する。"
        elif bbw < 5: # ボラティリティ圧縮
             strategy = f"ボラティリティ圧縮中。日足R1 ({R1_long_str}) / S1 ({S1_long_str}) の*ブレイクアウト待ち*。"
        else:
             strategy = f"レンジ取引。日足S1 ({S1_long_str}) 付近で買い、日足R1 ({R1_long_str}) 付近で売り。"

    # --- 短期予測の強化 (MACD, 短期MA50, ピボット基準) ---
    predictions = {
        # 1hは短期モメンタム(4h MACD)
        "1h": "強い上昇 🚀" if short_macd_h > 0 and latest_short.get('Close', price) > short_ma50 else "強い下降 📉" if short_macd_h < 0 and latest_short.get('Close', price) < short_ma50 else "レンジ ↔️",
        # 4hは短期トレンド(4h MA50)
        "4h": "上昇 📈" if latest_short.get('Close', price) > short_ma50 else "下降 📉",
        # 12hは日足のピボットPに対する位置
        "12h": "上昇 📈" if price > P_long else "下降 📉",
        # 24hは総合バイアス
        "24h": bias
    }

    return {
        'price': price,
        'P': P_long, 'R1': R1_long, 'S1': S1_long, 
        'R2_long': R2_long, 'S2_long': S2_long,
        'R1_short': R1_short, 'S1_short': S1_short,
        'MA50': ma50, 'MA200': ma200, 'RSI': rsi, 'BBW': bbw,
        'StochK_long': stoch_k_long, 'StochD_long': stoch_d_long,
        'ShortRSI': short_rsi, 'ShortMACDH': short_macd_h, 'ShortStochK': short_stoch_k,
        'Volume': volume, 'VMA': vma, 'VolumeRatio': volume_ratio, # NEW: 出来高情報
        'bias': bias,
        'dominance': dominance, # 優勢度を追加
        'strategy': strategy,
        'details': details,
        'predictions': predictions
    }


def generate_chart_image(df: pd.DataFrame, analysis_result: dict) -> io.BytesIO:
    """
    終値と主要なテクニカル指標を含むチャート画像を生成します。
    出来高のサブプロットを追加。
    """
    # 修正: pandas_taの命名規則に合わせてカラム名を変更
    BBU_COL = 'BBU_20_2.0_2.0'
    BBL_COL = 'BBL_20_2.0_2.0'
    VMA_COL = 'VMA_20' # NEW: VMAカラム
    
    # 必要なカラムのリスト
    required_cols = ['Close', 'High', 'Low', 'Volume', 'SMA_50', 'SMA_200', BBU_COL, BBL_COL, VMA_COL]
    
    # 【修正1: 必要なカラムが全てdfに存在することを確認】
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        # analyze_dataの処理を確認するようログメッセージを修正
        logging.error(f"チャート描画に必要なカラムの一部が不足しています: {missing_cols}. analyze_dataの処理を確認してください。")
        return io.BytesIO()

    # 【修正2: 必要なカラムのみを選択し、SMA_50が存在する行のみを残す】
    # SMA_50がNaNではない行（つまり、データが十分に揃った行）のみを選択。
    # これによりVMA_20も確実に含まれ、かつプロットに必要なデータが揃う。
    # df_plotを作成する前に、dfからrequired_colsだけを選択し、その上でdropnaを実行します。
    df_plot = df[required_cols].dropna(subset=['SMA_50']).copy()
    
    # NaN値の行を削除した結果、データフレームが空になった場合のチェック
    if df_plot.empty:
        logging.error("❌ NaN値の行を削除した結果、プロット用のデータフレームが空になりました。チャート描画をスキップします。")
        return io.BytesIO()
        
    # === NEW: 出来高用のサブプロットを追加 (2段構成) ===
    # 出来高(Volume)用のサブプロットをax2として追加
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 9), dpi=100, sharex=True, 
                                 gridspec_kw={'height_ratios': [3, 1]}) 
    plt.subplots_adjust(hspace=0.05) # プロット間のスペースを削減
    # ===============================================

    
    # --- 1. 価格ライン (ax) ---
    ax.plot(df_plot.index, df_plot['Close'], label='BTC 終値 (USD)', color='#059669', linewidth=2.5) # ラインを太く

    # --- 2. テクニカル指標ラインの描画 (ax) ---
    # 50日移動平均線 (MA50)
    ax.plot(df_plot.index, df_plot['SMA_50'], label='SMA 50 (中期)', color='#fbbf24', linestyle='-', linewidth=2, alpha=0.8) 
    # 200日移動平均線 (MA200) - 長期トレンド
    ax.plot(df_plot.index, df_plot['SMA_200'], label='SMA 200 (長期)', color='#ef4444', linestyle='--', linewidth=1.5, alpha=0.9)

    # ボリンジャーバンド (Upper/Lower Band)
    ax.plot(df_plot.index, df_plot[BBU_COL], label='BB Upper (+2σ)', color='#ef4444', linestyle=':', linewidth=1)
    ax.plot(df_plot.index, df_plot[BBL_COL], label='BB Lower (-2σ)', color='#3b82f6', linestyle=':', linewidth=1)

    # --- 3. 最新の主要レベルの描画 (ax) ---
    price = analysis_result['price']
    P = analysis_result['P']

    # ピボットポイント (P)
    ax.axhline(P, color='#9333ea', linestyle='--', linewidth=1.5, alpha=0.8, zorder=0)
    ax.text(df_plot.index[-1], P, f' P: ${P:,.2f}', color='#9333ea', ha='right', va='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))

    # 現在価格の点とラベル
    if len(df_plot) > 0:
        ax.scatter(df_plot.index[-1], price, color='black', s=100, zorder=5) # 点を大きく
        ax.text(df_plot.index[-1], price, f' 現在 ${price:,.2f}', color='black', ha='right', va='bottom', fontsize=12, weight='bold')

    # 4. グラフの装飾 (ax)
    ax.set_title(f'{TICKER} 価格推移とテクニカル分析 ({LONG_INTERVAL}足)', fontsize=18, color='#1f2937', weight='bold')
    ax.set_ylabel('終値 (USD)', fontsize=12)
    ax.tick_params(axis='x', labelbottom=False) # 上のプロットのX軸ラベルを非表示にする
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', fontsize=10)

    # === NEW: 出来高プロットの描画 (ax2) ===
    
    # 出来高バー（前日比で色分け）
    # 出来高のローソク足の色を決定: 終値が前日より高ければ緑、低ければ赤
    # 最初のデータポイントを比較対象として使うため、インデックスが1から始まります。
    colors = ['#059669' if df_plot['Close'].iloc[i] >= df_plot['Close'].iloc[i-1] else '#ef4444' 
              for i in range(1, len(df_plot))]
    
    # 最初のバーの色は前日がないため、とりあえず緑（上昇）として扱う (最初のバーはデータ開始点)
    if len(df_plot) > 0:
        colors.insert(0, '#059669') 

    ax2.bar(df_plot.index, df_plot['Volume'], color=colors, alpha=0.7, label='出来高')
    ax2.plot(df_plot.index, df_plot[VMA_COL], color='#6b7280', linestyle='--', linewidth=1, label='VMA 20 (出来高移動平均)')
    
    ax2.set_ylabel('出来高', fontsize=10)
    ax2.set_xlabel('日付', fontsize=12) # ax2にのみX軸ラベルを表示
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, linestyle=':', alpha=0.4)
    
    # X軸のフォーマットと回転をax2に適用
    formatter = DateFormatter("%m/%d")
    ax2.xaxis.set_major_formatter(formatter)

    # データを間引いて表示するためにDayLocatorを設定
    if len(df_plot.index) > 15:
        ax2.xaxis.set_major_locator(DayLocator(interval=math.ceil(len(df_plot.index) / 8)))
    else:
        ax2.xaxis.set_major_locator(DayLocator())

    plt.sca(ax2) # X軸ラベルの回転は最後に実行
    plt.xticks(rotation=45, ha='right')
    
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
    
    # === 修正箇所: 実行時刻をUTCからJSTへ変換し、次回実行時刻もJSTで計算 ===
    # JST = UTC + 9時間として処理します。
    now_utc = datetime.datetime.now()
    now_jst = now_utc + datetime.timedelta(hours=9)
    # ---------------------------------------------
    
    last_updated_str = now_jst.strftime('%Y-%m-%d %H:%M:%S')
    
    # 次回実行時刻の計算 (JSTベースで6時間後)
    next_run_time = now_jst + datetime.timedelta(hours=6)
    next_run_time_str = next_run_time.strftime('%Y-%m-%d %H:%M:%S')

    # === 処理開始時のステータス更新 ===
    global_data['scheduler_status'] = 'データ取得・分析中...'
    global_data['last_updated'] = last_updated_str
    
    # 1. データ取得 (日足と4時間足)
    df_long = fetch_btc_ohlcv_data(LONG_PERIOD, LONG_INTERVAL)
    df_short = fetch_btc_ohlcv_data(SHORT_PERIOD, SHORT_INTERVAL)

    # データが空の場合の処理
    if df_long.empty or df_short.empty:
        logging.error("致命的エラー: データ取得に失敗したため、レポートを生成できません。")
        # エラー発生時はステータスを更新
        global_data['scheduler_status'] = 'エラー（データ取得失敗）'
        global_data['strategy'] = 'データ取得エラー'
        error_msg = f"❌ *BTC分析レポート生成エラー*\n\nデータ取得に失敗しました。ネットワーク接続を確認するか、数分後に再試行してください。\n最終更新: {last_updated_str} (JST)"
        Thread(target=send_telegram_message, args=(error_msg,)).start()
        return

    # 2. テクニカル分析
    try:
        df_long_analyzed = analyze_data(df_long)
        df_short_analyzed = analyze_data(df_short) # 短期分析も実行
    except Exception as e:
        logging.error(f"致命的エラー: テクニカル分析中にエラーが発生しました: {e}", exc_info=True)
        global_data['scheduler_status'] = 'エラー（分析失敗）'
        error_msg = f"❌ *BTC分析レポート生成エラー*\n\nテクニカル分析中にエラーが発生しました。\n詳細: {str(e)}\n最終更新: {last_updated_str} (JST)"
        Thread(target=send_telegram_message, args=(error_msg,)).start()
        return

    # 3. バックテストの実行 (日足データを使用)
    try:
        logging.info(f"バックテスト実行中... 期間: {LONG_PERIOD}")
        # バックテストはデータが揃っている部分のみを使用するため、dropna()後のデータを使用
        df_long_clean_for_backtest = df_long_analyzed.dropna() 
        backtest_results = backtest_strategy(df_long_clean_for_backtest) 
        global_data['backtest'] = backtest_results
        logging.info("✅ バックテスト完了。")
    except Exception as e:
        logging.error(f"❌ バックテスト中にエラーが発生しました: {e}", exc_info=True)
        backtest_results = {'error': f"バックテスト失敗: {str(e)}"}
        global_data['backtest'] = backtest_results

    # 4. 戦略と予測の生成 (日足と4時間足の両方を使用)
    # iloc[-1]を使用するため、dropna()していない分析済みデータフレームを渡す (KeyError対策をgenerate_strategy内で実施済み)
    analysis_result = generate_strategy(df_long_analyzed, df_short_analyzed)

    # 5. グローバル状態の更新 (正常完了時)
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
    R2_long, S2_long = analysis_result['R2_long'], analysis_result['S2_long'] 
    R1_short, S1_short = analysis_result['R1_short'], analysis_result['S1_short'] 
    ma200, bbw = analysis_result['MA200'], analysis_result['BBW'] 
    stoch_k_long, stoch_d_long = analysis_result['StochK_long'], analysis_result['StochD_long'] 
    
    # NEW: 出来高情報
    volume, vma, volume_ratio = analysis_result['Volume'], analysis_result['VMA'], analysis_result['VolumeRatio']


    dominance = analysis_result['dominance'] # 優勢度
    strategy = analysis_result['strategy']
    details = analysis_result['details'] 
    predictions = analysis_result['predictions']

    # 価格をカンマ区切りにフォーマット
    formatted_current_price = f"`${price:,.2f}`"
    formatted_P = f"`${P:,.2f}`"
    formatted_R1_long = f"`${R1:,.2f}`"
    formatted_S1_long = f"`${S1:,.2f}`"
    formatted_R2_long = f"`${R2_long:,.2f}`"
    formatted_S2_long = f"`${S2_long:,.2f}`"
    formatted_R1_short = f"`${R1_short:,.2f}`"
    formatted_S1_short = f"`${S1_short:,.2f}`"
    formatted_MA50 = f"`${ma50:,.2f}`"
    formatted_MA200 = f"`${ma200:,.2f}`"
    formatted_RSI = f"`{rsi:,.2f}`"
    formatted_BBW = f"`{bbw:,.2f}%`"

    price_analysis = [
        f"💰 *現在価格 (BTC-USD)*: {formatted_current_price}",
        f"🟡 *ピボットポイント (P, 日足)*: {formatted_P}",
        f"💡 *中期トレンド転換点 (MA50)*: {formatted_MA50}",
        f"🐻 *長期トレンド基準 (MA200)*: {formatted_MA200}",
        f"--- 日足 主要レベル (Classic Pivot) ---",
        f"🔼 R1: {formatted_R1_long}, R2: {formatted_R2_long}",
        f"🔽 S1: {formatted_S1_long}, S2: {formatted_S2_long}",
        f"--- 4h 短期主要レベル (Fibonacci Pivot) ---",
        f"⬆️ R1 (4h): {formatted_R1_short}",
        f"⬇️ S1 (4h): {formatted_S1_short}",
        f"--- 主要オシレーター指標 ---",
        f"🔥 RSI (14期間, 日足): {formatted_RSI}",
        f"📊 BB幅 (20, 日足): {formatted_BBW}",
        f"✨ Stochastics K/D (日足): K=`{stoch_k_long:,.2f}`, D=`{stoch_d_long:,.2f}`",
        f"--- 出来高情報 (Volume) ---", # NEW
        f"📈 最新出来高: `{volume:,.0f}` (平均VMA20比: `{volume_ratio:,.0f}%`)", # NEW
    ]

    prediction_lines = [f"• {tf}後予測: *{predictions[tf]}*" for tf in ["1h", "4h", "12h", "24h"]]

    # 改行を多く入れ、セクションを明確に分離
    report_message = (
        f"👑 *BTC実践分析レポート (テクニカルBOT)* 👑\n\n"
        f"📅 *最終データ更新*: `{last_updated_str}` (JST)\n"
        f"⏱️ *次回の通知予定*: `{next_run_time_str}` (JST)\n" # JSTベースの次回通知予定時刻
        f"📊 *処理データ件数*: *{len(df_long)}* 件 ({LONG_INTERVAL}足) + *{len(df_short)}* 件 ({SHORT_INTERVAL}足)\n\n" 
        
        # --- 市場優勢度の強調 ---
        f"**🚀 市場の優勢 (Dominance) 🚀**\n"
        f"🚨 *総合優勢度*: *{dominance}*\n\n"
        
        f"--- *主要価格帯と指標 (USD)* ---\n"
        # 修正: '\n' (実際の改行コード) をジョイナーとして使用
        f"{'\n'.join(price_analysis)}\n\n" 
        
        f"--- *動向の詳細分析と根拠* ---\n"
        # 修正: '\n' (実際の改行コード) をジョイナーとして使用
        f"{'\n'.join(details)}\n\n"
        
        f"--- *短期動向と予測* ---\n"
        # 修正: '\n' (実際の改行コード) をジョイナーとして使用
        f"{'\n'.join(prediction_lines)}\n\n"
        
        f"--- *総合戦略サマリー* ---\n"
        f"🛡️ *推奨戦略*: *{strategy}*\n\n"
    )
    
    # --- バックテスト結果のレポートへの追加 (分かりやすい表現に変更) ---
    if 'error' in backtest_results:
        backtest_lines = [f"⚠️ *バックテスト結果*: {backtest_results['error']}"]
    else:
        # パフォーマンスサマリーの追加
        total_return = backtest_results['total_return']
        profit_factor = backtest_results['profit_factor']
        
        # 0.05% (5ベーシスポイント) を超える利益があれば成功と見なす
        if total_return > 0.05:
            summary = f"✨ *評価*: この戦略は*{total_return}%の利益*を生み出し、*{profit_factor}* のPFを達成しました。堅調に機能しています。"
        elif total_return > -0.05:
            summary = f"💡 *評価*: この戦略はほぼ*中立*なパフォーマンスでした。大きな優位性はありません。"
        else:
            summary = f"🔻 *評価*: この戦略は*{total_return}%の損失*を計上しました。ロジックの*見直しが必要*です。"

        backtest_lines = [
            f"--- *戦略バックテスト結果 ({LONG_PERIOD} / {LONG_INTERVAL}足)* ---",
            summary,
            # $マークのエスケープを削除 (Python f-string内では\は不要だが、Markdownとして解釈させるために`$`で囲むことで安全性を高める)
            f"💰 *最終資産*: `${backtest_results['final_capital']:,.2f}` (初期: `${BACKTEST_CAPITAL:,.2f}`)",
            f"📈 *総リターン率*: *{total_return}%* (期間中の増減)",
            f"🏆 *プロフィットファクター (PF)*: `{profit_factor}` (総利益/総損失。*1.0以上*が優勢を示します)",
            f"📉 *最大ドローダウン (DD)*: `{backtest_results['max_drawdown']}%` (期間中の最大の元本割れ率)",
            f"📊 *取引実績*: `{backtest_results['trades']}` 回の取引 (勝率: `{backtest_results['win_rate']}%`)"
        ]

    report_message += (
        f"{chr(8212) * 20}\n" # 区切り線
        # 修正: '\n' (実際の改行コード) をジョイナーとして使用
        f"{'\n'.join(backtest_lines)}\n\n" 
        f"_※ この分析は、実戦的なマルチタイムフレーム分析に基づきますが、投資助言ではありません。_"
    )


    # 7. 画像生成と通知の実行
    try:
        logging.info("チャート画像を生成中...")
        chart_buffer = generate_chart_image(df_long_analyzed, analysis_result)
        
        # NEW: 出来高情報をキャプションに追加
        volume_status = "出来高急増" if analysis_result.get('VolumeRatio', 0) > 150 else "出来高平均的"
        
        photo_caption = (
            f"📈 *BTC実践分析チャート ({LONG_INTERVAL}足)* 📉\n"
            f"📅 更新: `{last_updated_str}` (JST)\n" # JSTの更新時刻
            f"💰 現在価格: {formatted_current_price}\n"
            f"🚨 *優勢度*: *{dominance}* ({volume_status})\n" # 出来高ステータスを追加
            f"🛡️ *推奨戦略*: {strategy}\n"
            f"_詳細は別途送信されるテキストレポートをご確認ください。_"
        )

        # チャートバッファが空でないことを確認してから送信
        if chart_buffer.getbuffer().nbytes > 0:
            Thread(target=send_telegram_photo, args=(chart_buffer, photo_caption)).start()
        else:
             logging.error("❌ チャート画像のバッファが空です。画像送信をスキップしました。")
             error_caption = f"⚠️ *チャート生成失敗*\n\nデータは正常に処理されましたが、チャート画像生成中にエラーが発生しました。\n最終更新: {last_updated_str} (JST)"
             Thread(target=send_telegram_message, args=(error_caption,)).start()


    except Exception as e:
        logging.error(f"❌ チャート画像の生成または送信に失敗しました: {e}", exc_info=True)
        error_caption = f"⚠️ *チャート生成失敗*\n\nデータは正常に処理されましたが、チャート画像生成中に予期せぬエラーが発生しました。\nエラー詳細: {str(e)[:100]}...\n最終更新: {last_updated_str} (JST)"
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
    # HTMLファイルの内容がないため、ここでHTMLファイルを生成します。
    html_content = generate_index_html()
    return render_template('index.html', **global_data)

@app.route('/status')
def status():
    """現在のステータスをJSONで返すAPIエンドポイント"""
    return jsonify(global_data)

def generate_index_html():
    """ダッシュボードHTMLを生成します (ファイルが見つからないエラー回避のため)"""
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BTC実践テクニカル分析 BOT ダッシュボード</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        body { font-family: 'Inter', sans-serif; background-color: #f3f4f6; color: #1f2937; }
        .card { background-color: white; border-radius: 1rem; box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1); padding: 1.5rem; }
        .stat-label { font-size: 0.875rem; color: #6b7280; font-weight: 600; margin-bottom: 0.25rem; }
        .stat-value { font-size: 1.5rem; font-weight: 700; color: #1f2937; }
        .price-value { font-size: 2.25rem; font-weight: 800; color: #059669; }
        .strategy-box { border: 2px solid #3b82f6; background-color: #eff6ff; border-radius: 0.75rem; padding: 1rem; margin-top: 1rem; }
        .strategy-title { color: #1d4ed8; font-weight: 700; }
        .prediction-item { background-color: #f9fafb; border-radius: 0.5rem; padding: 0.5rem; }
        .prediction-value { font-weight: 700; }
    </style>
</head>
<body>
    <div class="container mx-auto p-4 sm:p-8">
        <header class="text-center mb-8">
            <h1 class="text-4xl font-bold text-gray-800">₿ BTC 実践テクニカル分析 BOT ダッシュボード</h1>
            <p class="text-gray-600 mt-2" id="last-updated">最終更新: N/A</p>
        </header>

        <div class="card mb-8">
            <div class="flex flex-wrap items-center justify-between">
                <div class="mb-4 sm:mb-0">
                    <div class="stat-label">現在のBTC価格 (USD)</div>
                    <div class="price-value" id="current-price">$0.00</div>
                    <div class="text-lg font-semibold" id="dominance">N/A</div>
                </div>
                
                <div class="grid grid-cols-2 gap-4">
                    <div class="text-right">
                        <div class="stat-label">総合バイアス</div>
                        <div class="stat-value" id="bias">N/A</div>
                    </div>
                    <div class="text-right">
                        <div class="stat-label">データ範囲</div>
                        <div class="stat-value text-base" id="data-range">N/A</div>
                    </div>
                </div>
            </div>
            
            <div class="strategy-box">
                <div class="strategy-title text-xl mb-2">🛡️ 推奨戦略サマリー</div>
                <p id="strategy" class="text-gray-700 font-medium">データ処理中...</p>
            </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div class="card">
                <div class="stat-label">BOTステータス</div>
                <div class="stat-value text-sm font-medium" id="scheduler-status">初期化中...</div>
            </div>
            <div class="card">
                <div class="stat-label">バックテスト総リターン</div>
                <div class="stat-value text-red-500" id="backtest-return">0.00%</div>
            </div>
            <div class="card">
                <div class="stat-label">プロフィットファクター (PF)</div>
                <div class="stat-value" id="backtest-pf">0.00</div>
            </div>
        </div>
        
        <div class="card mb-8">
            <h2 class="text-2xl font-semibold mb-4 text-gray-800">短期予測 (Predictions)</h2>
            <div id="predictions" class="grid grid-cols-2 sm:grid-cols-4 gap-4">
                </div>
        </div>

        <div class="card mb-8">
            <h2 class="text-2xl font-semibold mb-4 text-gray-800">主要テクニカル指標とレベル</h2>
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4" id="tech-indicators">
                </div>
        </div>
        
        <footer class="text-center text-gray-500 text-sm pt-4">
            <p>※ このデータはyfinanceから取得され、6時間ごとに更新されます。</p>
            <p>※ 投資助言ではありません。</p>
        </footer>
    </div>

    <script>
        // グローバル変数
        let lastKnownData = {};

        // データを取得してダッシュボードを更新する関数
        async function updateDashboard() {
            try {
                const response = await fetch('/status');
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const data = await response.json();
                lastKnownData = data;
                
                // メインサマリーの更新
                document.getElementById('last-updated').textContent = `最終更新: ${data.last_updated} (JST)`;
                document.getElementById('data-range').textContent = data.data_range;
                
                const priceElement = document.getElementById('current-price');
                const price = parseFloat(data.current_price || 0);
                priceElement.textContent = `$${price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
                
                // 価格のカラーリング（前回の価格と比較）
                const previousPrice = parseFloat(lastKnownData.current_price || 0);
                if (price > previousPrice && previousPrice !== 0) {
                    priceElement.classList.remove('text-red-500');
                    priceElement.classList.add('text-green-600');
                } else if (price < previousPrice && previousPrice !== 0) {
                    priceElement.classList.remove('text-green-600');
                    priceElement.classList.add('text-red-500');
                } else {
                    priceElement.classList.remove('text-green-600', 'text-red-500');
                    priceElement.classList.add('text-[#059669]');
                }

                document.getElementById('dominance').textContent = `市場優勢度: ${data.dominance}`;
                document.getElementById('bias').textContent = data.bias;
                document.getElementById('strategy').textContent = data.strategy;
                document.getElementById('scheduler-status').textContent = data.scheduler_status;
                
                // バックテスト結果の更新
                const backtestReturn = parseFloat(data.backtest.total_return || 0);
                const backtestReturnEl = document.getElementById('backtest-return');
                backtestReturnEl.textContent = `${backtestReturn.toFixed(2)}%`;
                
                if (backtestReturn > 0) {
                    backtestReturnEl.classList.remove('text-red-500', 'text-gray-700');
                    backtestReturnEl.classList.add('text-green-600');
                } else if (backtestReturn < 0) {
                    backtestReturnEl.classList.remove('text-green-600', 'text-gray-700');
                    backtestReturnEl.classList.add('text-red-500');
                } else {
                    backtestReturnEl.classList.remove('text-green-600', 'text-red-500');
                    backtestReturnEl.classList.add('text-gray-700');
                }
                
                document.getElementById('backtest-pf').textContent = parseFloat(data.backtest.profit_factor || 0).toFixed(2);
                
                // 短期予測の更新
                updatePredictions(data.predictions);

                // 技術指標の更新
                updateTechIndicators(data);

            } catch (error) {
                console.error("ダッシュボードの更新に失敗しました:", error);
                document.getElementById('scheduler-status').textContent = 'エラー（通信失敗）';
            }
        }

        function updatePredictions(predictions) {
            const container = document.getElementById('predictions');
            container.innerHTML = '';
            
            const timeframes = {
                '1h': '1時間後',
                '4h': '4時間後',
                '12h': '12時間後',
                '24h': '24時間後'
            };

            for (const key in predictions) {
                const value = predictions[key];
                const tfName = timeframes[key] || key;
                
                let colorClass = 'text-gray-600';
                if (value.includes('上昇') || value.includes('ロング')) {
                    colorClass = 'text-green-600';
                } else if (value.includes('下降') || value.includes('ショート')) {
                    colorClass = 'text-red-500';
                }

                const html = `
                    <div class="prediction-item">
                        <div class="stat-label">${tfName}</div>
                        <div class="prediction-value ${colorClass}">${value}</div>
                    </div>
                `;
                container.innerHTML += html;
            }
        }

        function updateTechIndicators(data) {
            const container = document.getElementById('tech-indicators');
            container.innerHTML = '';

            const indicators = [
                { label: '日足 MA50', value: data.MA50, format: 'currency' },
                { label: '日足 MA200', value: data.MA200, format: 'currency' },
                { label: '日足 RSI', value: data.RSI, format: 'decimal' },
                { label: '日足 BBW', value: data.BBW, format: 'percent' },
                { label: '日足 P', value: data.P, format: 'currency' },
                { label: '日足 R1', value: data.R1, format: 'currency' },
                { label: '日足 S1', value: data.S1, format: 'currency' },
                { label: '日足 Volume', value: data.Volume, format: 'int' },
                { label: '日足 VMA20', value: data.VMA, format: 'int' },
                { label: '出来高比 (VMA20)', value: data.VolumeRatio, format: 'percent' },
                { label: '4h RSI', value: data.ShortRSI, format: 'decimal' },
                { label: '4h MACD Hist', value: data.ShortMACDH, format: 'decimal' },
            ];

            indicators.forEach(item => {
                let formattedValue = 'N/A';
                let rawValue = parseFloat(item.value);

                if (!isNaN(rawValue)) {
                    if (item.format === 'currency') {
                        formattedValue = `$${rawValue.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
                    } else if (item.format === 'percent') {
                        formattedValue = `${rawValue.toFixed(2)}%`;
                    } else if (item.format === 'int') {
                        formattedValue = rawValue.toLocaleString('en-US', { maximumFractionDigits: 0 });
                    } else { // decimal
                        formattedValue = rawValue.toFixed(2);
                    }
                }

                const html = `
                    <div class="stat-box">
                        <div class="stat-label">${item.label}</div>
                        <div class="stat-value text-base">${formattedValue}</div>
                    </div>
                `;
                container.innerHTML += html;
            });
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
    
# -----------------
# スケジューラーの初期設定と開始
# -----------------
if not scheduler.running:
    app.config.update({
        'SCHEDULER_JOBSTORES': {'default': {'type': 'memory'}},
        'SCHEDULERS_EXECUTORS': {'default': {'type': 'threadpool', 'max_workers': 20}},
        'SCHEDULER_API_ENABLED': False
    })

    scheduler.init_app(app)

    # 6時間ごとにupdate_report_dataを実行
    scheduler.add_job(id='report_update_job', func=update_report_data,
                      trigger='interval', hours=6, replace_existing=True)

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
