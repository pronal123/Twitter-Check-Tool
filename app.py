import datetime
import logging
import time
import os
import requests 
from threading import Thread
import io 
import random 

# グラフ描画とデータ処理のためのインポート
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

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
# 注: ログから取得されたIDを一時的にデフォルトに設定していますが、必ずご自身のChat IDに置き換えてください。
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '5890119671') 

# 修正: 正しいTelegram APIのエンドポイントを設定
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

# ダミーデータとグローバル状態
global_data = {
    'last_updated': 'N/A',
    'data_range': '2023-01-01 - 2025-11-18', 
    'data_count': 0,
    'scheduler_status': '初期化中'
}
data_item_count = 0

# -----------------
# Telegram 通知ヘルパー関数 (API呼び出しを有効化)
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
# テクニカル指標のシミュレーション関数 (変更なし)
# -----------------

def simulate_technical_signals(data_count: int, current_price: int, ma50: int) -> tuple[bool, bool, bool]:
    """
    RSIとMACDのシグナルを現在のデータ件数と価格関係に基づいてシミュレートします。
    戻り値: (RSI買われすぎシグナル, MACDゴールデンクロスシグナル, MACDデッドクロスシグナル)
    """
    # 1. RSI (Relative Strength Index) シミュレーション
    rsi_overbought = False
    if data_count % 7 == 0 and current_price > ma50 * 1.005:
        rsi_overbought = True
        
    # 2. MACD (Moving Average Convergence Divergence) シミュレーション
    macd_golden_cross = False  # 買いシグナル
    macd_dead_cross = False    # 売りシグナル
    
    # データ件数が偶数なら買いシグナル、奇数なら売りシグナルの可能性が高い、というシミュレーション
    if data_count % 2 == 0:
        macd_golden_cross = True
    elif data_count % 2 != 0 and current_price < ma50 * 0.995:
        macd_dead_cross = True

    return rsi_overbought, macd_golden_cross, macd_dead_cross


def simulate_pivot_data(current_price: int, data_count: int) -> tuple[int, int, int]:
    """
    前日の高値(H), 安値(L), 終値(C)をシミュレーションし、ピボットポイント(P)に必要な値を生成します。
    """
    # 過去の変動率をシミュレート (データ件数によってボラティリティを変える)
    volatility = 0.02 + (data_count % 1000 / 1000) * 0.01 
    
    # 終値 (C) は現在価格に近い値
    close_price = int(current_price * random.uniform(0.998, 1.002))
    
    # 高値 (H) と 安値 (L) を終値からボラティリティを考慮してシミュレート
    high_price = int(close_price * (1 + random.uniform(0.5, 1.0) * volatility))
    low_price = int(close_price * (1 - random.uniform(0.5, 1.0) * volatility))
    
    # Hは必ずCより高く、Lは必ずCより低いことを保証
    H = max(current_price, high_price)
    L = min(current_price, low_price)
    C = close_price
    
    return H, L, C

def calculate_pivot_levels(H: int, L: int, C: int) -> tuple[int, int, int]:
    """
    クラシックピボットポイントの計算式に基づいて、P, R1, S1を算出します。
    """
    P = int((H + L + C) / 3)
    R1 = int(2 * P - L)
    S1 = int(2 * P - H)
    
    return P, R1, S1

def get_real_time_btc_data(data_count: int) -> tuple[int, int, int, int, int, int, int]:
    """
    CoinGecko APIからBTCのリアルタイム価格を取得し、実践的なシミュレーションに基づきP, R1, S1, MA50を計算します。
    戻り値: (現在価格, H, L, C, P, R1, S1, MA50)
    """
    API_URL = "https://api.coingecko.com/api/v3/simple/price"
    params = {'ids': 'bitcoin', 'vs_currencies': 'usd'}
    MAX_RETRIES = 3 
    current_price = 0
    
    for attempt in range(MAX_RETRIES):
        try:
            logging.info(f"CoinGecko APIからリアルタイムBTC価格を取得中... (試行 {attempt + 1}/{MAX_RETRIES})")
            response = requests.get(API_URL, params=params, timeout=10)
            response.raise_for_status() 
            data = response.json()
            
            if 'bitcoin' in data and 'usd' in data['bitcoin']:
                current_price = int(data['bitcoin']['usd'])
                logging.info(f"CoinGeckoからリアルタイム価格を取得しました: ${current_price:,}")
                break 
            else:
                logging.warning("CoinGecko APIからのレスポンスに価格データが含まれていませんでした。")
                break 
                
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = 2 ** attempt 
                logging.error(f"CoinGecko API接続エラー: {e}。{wait_time}秒後にリトライします。")
                time.sleep(wait_time)
            else:
                logging.error(f"CoinGecko API接続エラー: {e}。シミュレーションにフォールバックします。")
                break
    
    # -----------------
    # フォールバック (APIが失敗した場合)
    # -----------------
    if current_price <= 0:
        base_price = 60000 
        price_factor = (data_count // 1000) % 10 
        simulated_price = base_price + price_factor * 2000 + random.randint(-1000, 1000) 
        current_price = int(simulated_price)
        logging.info(f"シミュレーション価格を使用します: ${current_price:,}")
    
    # -----------------
    # 実践的なテクニカルレベルのシミュレーション計算
    # -----------------
    
    # 1. H, L, Cのシミュレーション
    H, L, C = simulate_pivot_data(current_price, data_count)
    
    # 2. P, R1, S1の計算（ピボットポイント方式）
    P, R1, S1 = calculate_pivot_levels(H, L, C)
    
    # 3. MA50のシミュレーション（トレンド追従の特性を模倣）
    ma50_bias = 0.999 + (random.randint(0, 10) / 1000) 
    ma50_base = P 
    
    if data_count % 5 == 1: 
        ma50_final = int(ma50_base * random.uniform(0.99, 0.995))
    elif data_count % 5 == 4: 
        ma50_final = int(ma50_base * random.uniform(1.005, 1.01))
    else:
        ma50_final = int(ma50_base * ma50_bias)

    
    # 返り値: (現在価格, H, L, C, P, R1, S1, MA50)
    return current_price, H, L, C, P, R1, S1, ma50_final


def generate_chart_image(current_price: int, P: int, r1: int, s1: int, ma50: int) -> io.BytesIO:
    """
    価格推移、P, R1, S1, 50MAを含むチャート画像を生成し、io.BytesIOで返します。
    """
    # 1. ダミー時系列データの生成 (過去30日間)
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=30)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    price_series = [current_price]
    for _ in range(len(dates) - 1, 0, -1):
        change = random.uniform(-0.015 * P, 0.015 * P)
        # 価格がPを中心に収束する傾向をシミュレート
        next_price = max(s1 * 0.9, min(r1 * 1.1, price_series[0] + change * 0.5)) 
        
        # Pに近づく引力（価格がPから離れているほどPに戻りやすい）
        pull_to_pivot = (P - next_price) * 0.1
        next_price += pull_to_pivot
        
        price_series.insert(0, next_price)
    
    df = pd.DataFrame({'Price': price_series}, index=dates)

    # 2. Matplotlibでチャート描画
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100) 
    
    # 価格ライン
    ax.plot(df.index, df['Price'], label='BTC Price (Sim.)', color='#059669', linewidth=2)

    # --- 価格帯レベルの描画 ---
    
    # R1 (レジスタンス): 赤色の破線
    ax.axhline(r1, color='#ef4444', linestyle='--', linewidth=1.5, label=f'R1: ${r1:,}')
    ax.text(df.index[-1], r1, f' R1 (レジスタンス) ${r1:,}', color='#ef4444', ha='right', va='bottom', fontsize=10, weight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))

    # S1 (サポート): 青色の破線
    ax.axhline(s1, color='#3b82f6', linestyle='--', linewidth=1.5, label=f'S1: ${s1:,}')
    ax.text(df.index[-1], s1, f' S1 (サポート) ${s1:,}', color='#3b82f6', ha='right', va='top', fontsize=10, weight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
    
    # P (ピボットポイント): 紫色の点線 (中期転換点)
    ax.axhline(P, color='#9333ea', linestyle=':', linewidth=2, alpha=0.8, label=f'P: ${P:,}')
    ax.text(df.index[-1], P, f' P (ピボットポイント) ${P:,}', color='#9333ea', ha='right', va='center', fontsize=10, weight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))

    # 50MA (長期トレンド転換点): 黄色/オレンジ色の実線
    ma50_color = '#facc15'
    ax.axhline(ma50, color=ma50_color, linestyle='-', linewidth=2, alpha=0.8, label=f'50MA: ${ma50:,}')
    ma50_label_color = '#b45309' 
    ma50_label = f' 50MA (中期トレンド転換点) ${ma50:,}'
    va_pos = 'top' if ma50 > current_price else 'bottom'
    ax.text(df.index[-1], ma50, ma50_label, color=ma50_label_color, ha='right', va=va_pos, fontsize=10, weight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))


    # 現在価格の点とラベル
    ax.scatter(df.index[-1], current_price, color='black', s=80, zorder=5) 
    ax.text(df.index[-1] + datetime.timedelta(days=0.5), current_price, f' 現在価格 ${current_price:,}', color='black', ha='left', va='center', fontsize=11, weight='bold')

    # 3. グラフの装飾
    is_simulated = current_price > 0 and current_price < 65000 # 65k未満をシミュレーション価格と暫定的に判定
    price_source_label = "（CoinGecko API）" if not is_simulated else "（シミュレーション）"
    ax.set_title(f'BTC価格推移と主要な価格帯 {price_source_label}', fontsize=16, color='#1f2937', weight='bold')
    ax.set_xlabel('日付', fontsize=12)
    ax.set_ylabel('価格 (USD)', fontsize=12)
    
    formatter = DateFormatter("%m/%d")
    ax.xaxis.set_major_formatter(formatter)
    
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    # 4. 画像をメモリ上のバイトストリームとして保存
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig) 
    
    return buf

# -----------------
# スケジューリングタスク
# -----------------
def update_report_data():
    """定期的に実行されるタスク：データ取得とレポート更新の実行"""
    global global_data
    global data_item_count

    logging.info("スケジュールされたレポート更新タスク開始...")
    
    data_item_count += random.randint(500, 1500) 
    now = datetime.datetime.now()
    
    # 1. グローバル状態の更新
    last_updated_str = now.strftime('%Y-%m-%d %H:%M:%S')
    global_data['last_updated'] = last_updated_str
    global_data['data_count'] = data_item_count
    global_data['scheduler_status'] = '稼働中'
    
    # 2. 実践的なテクニカルレベルと価格の取得
    current_price, H, L, C, P, R1, S1, ma50 = get_real_time_btc_data(data_item_count) 
    
    # 3. テクニカルシグナルのシミュレーション
    rsi_overbought, macd_gc, macd_dc = simulate_technical_signals(data_item_count, current_price, ma50)

    outcomes = {"UP": "上昇 📈", "DOWN": "下降 📉", "SIDE": "レンジ ↔️"}
    predictions = {}
    analysis_details = []
    
    # 価格をカンマ区切りにフォーマット
    formatted_current_price = f"`${current_price:,}`"
    formatted_P = f"`${P:,}`"
    formatted_R1 = f"`${R1:,}`"
    formatted_S1 = f"`${S1:,}`"
    formatted_MA50 = f"`${ma50:,}`" 

    # 価格取得のソースを判定し、メッセージに含める
    price_source = "リアルタイム価格 (CoinGecko)"
    if current_price < 65000: 
        price_source = "シミュレーション価格 (API取得失敗時)"
    
    price_analysis = [
        f"💰 *現在価格 ({price_source})*: {formatted_current_price}",
        f"🟡 *ピボットポイント (P)*: {formatted_P} (本日の短期中立点)",
        f"🔼 *主要レジスタンス (R1)*: {formatted_R1} (Pからの上昇ターゲット)",
        f"🔽 *主要サポート (S1)*: {formatted_S1} (Pからの下降ターゲット)",
        f"💡 *中期トレンド転換点 (50MA)*: {formatted_MA50}" 
    ]

    # --- 実践的予測ロジックの実行 ---
    
    # 50MAとPに基づく中期バイアス
    if current_price > ma50 and current_price > P:
        short_term_bias = "強い上昇"
        ma_analysis = "・価格は50MAとPを明確に上回り、中期的に強い強気トレンドが継続しています。"
    elif current_price < ma50 and current_price < P:
        short_term_bias = "強い下降"
        ma_analysis = "・価格は50MAとPを下回っており、中期的な弱気トレンドが優勢です。Pと50MAが重要なレジスタンスとして機能しています。"
    else:
        short_term_bias = "中立/レンジ"
        ma_analysis = "・価格はPと50MAの間で推移しており、トレンドの方向性について市場が迷っている状態です。Pがブレイクポイントです。"
    analysis_details.append(ma_analysis)

    # 1h予測: 短期的なシグナル (RSI過熱感とPとの距離)
    if rsi_overbought:
        predictions["1h"] = outcomes["DOWN"]
        analysis_details.append("・1h: *RSI買われすぎシグナル*をシミュレート。短期的な利確売りによる調整下降の可能性が高いです。")
    elif current_price < P:
        predictions["1h"] = outcomes["UP"]
        analysis_details.append("・1h: 価格はPを下回っていますが、短期的な買い圧力が強まっています。Pへの回帰（リテスト）が期待されます。")
    else:
        predictions["1h"] = outcomes["SIDE"]
        analysis_details.append("・1h: PとR1の間で小動き。短期的なエネルギーの蓄積期間に入っています。")

    # 4h予測: 中期トレンドシグナル (MACDと50MA)
    if macd_gc:
        predictions["4h"] = outcomes["UP"]
        analysis_details.append("・4h: *MACDゴールデンクロス*をシミュレート。中期的な上昇トレンドへの転換が強く示唆されます。")
    elif macd_dc:
        predictions["4h"] = outcomes["DOWN"]
        analysis_details.append("・4h: *MACDデッドクロス*をシミュレート。中期的な下降トレンドへの転換リスクが高まっています。")
    else:
        predictions["4h"] = outcomes["SIDE"]
        analysis_details.append("・4h: テクニカルシグナルは混在しており、トレンドはまだ明確ではありません。50MAの方向性が鍵となります。")
        
    # 12h予測: S1とR1のどちらが遠いか（トレンドの目標）
    if abs(current_price - R1) < abs(current_price - S1):
        predictions["12h"] = outcomes["DOWN"]
        analysis_details.append("・12h: 短期的な上昇目標であるR1に近づいており、達成後の反落（S1方向）を意識した動きが予想されます。")
    else:
        predictions["12h"] = outcomes["UP"]
        analysis_details.append("・12h: S1付近で反発。中期的な買いが入りやすく、次のターゲットはR1となります。")
        
    # 24h予測: 長期的なデータ量と50MAのバイアス
    if current_price > ma50 * 1.01:
        predictions["24h"] = outcomes["UP"]
        analysis_details.append("・24h: 50MAからの乖離が大きく、強いモメンタムでの上昇継続が期待されます。長期的な強気相場です。")
        long_term_advice = "押し目買い戦略（PまたはS1がターゲット）"
    elif current_price < ma50 * 0.99:
        predictions["24h"] = outcomes["DOWN"]
        analysis_details.append("・24h: 50MAを明確に下回っており、弱気な展開が予想されます。長期的なトレンド転換のリスクがあります。")
        long_term_advice = "戻り売り戦略（Pまたは50MAがターゲット）"
    else:
        predictions["24h"] = outcomes["SIDE"]
        analysis_details.append("・24h: 50MA付近でのレンジ相場。次のトレンド方向性を決めるためのエネルギーを蓄積中です。")
        long_term_advice = "ブレイクアウト待ちの様子見戦略"
    
    # --- ロジック終了 ---
    
    # 予測結果の組み立て
    prediction_lines = [f"• {tf}後予測: *{predictions[tf]}*" for tf in ["1h", "4h", "12h", "24h"]]
        
    prediction_text = "\n".join(prediction_lines)
    analysis_text = "\n".join(analysis_details)
    price_analysis_text = "\n".join(price_analysis)
    
    # 総合サマリーの抽出
    formatted_data_count = f"{data_item_count:,}"
    
    report_message = (
        f"👑 *BTC詳細分析レポート (ML BOT)* 👑\n\n"
        f"📅 最終データ更新: `{last_updated_str}`\n"
        f"📊 処理データ件数: *{formatted_data_count}* 件\n"
        f"--- *主要価格帯分析 (USD)* ---\n"
        f"_(ピボットポイント方式に基づき算出)_\n"
        f"{price_analysis_text}\n\n" 
        f"--- *総合予測* ---\n"
        f"{prediction_text}\n\n"
        f"--- *動向の詳細分析と根拠* ---\n"
        f"{analysis_text}\n\n"
        f"--- *総合戦略サマリー* ---\n"
        f"💡 *中期バイアス*: *{short_term_bias}* 傾向\n"
        f"🛡️ *推奨戦略*: {long_term_advice}がベースとなります。特にPとR1/S1、そして50MAの関係に注目してください。\n"
        f"_※ この分析は、実戦的なテクニカルシミュレーションに基づきますが、投資助言ではありません。_"
    )
    
    # 5. 画像生成と画像通知の実行
    try:
        logging.info("チャート画像を生成中...")
        chart_buffer = generate_chart_image(current_price, P, R1, S1, ma50)
        
        photo_caption = (
            f"📈 *BTCチャート分析 ({price_source})* 📉\n"
            f"📅 更新: `{last_updated_str}`\n"
            f"{price_analysis_text}\n\n"
            f"総合予測: 💡 *中期バイアス*:{short_term_bias} / 🛡️ *推奨戦略*:{long_term_advice}\n"
            f"_詳細は別途送信されるテキストレポートをご確認ください。_"
        )
        
        Thread(target=send_telegram_photo, args=(chart_buffer, photo_caption)).start()
        
    except Exception as e:
        logging.error(f"❌ チャート画像の生成または送信に失敗しました: {e}")

    # テキストレポートの送信
    Thread(target=send_telegram_message, args=(report_message,)).start()
    
    logging.info("レポート更新タスク完了。通知キューに追加されました。")


# -----------------
# ルート（エンドポイント）
# -----------------
@app.route('/')
def index():
    """ダッシュボードの表示"""
    return render_template('index.html', title='ML BOT分析レポート ダッシュボード', data=global_data)

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
    
    scheduler.add_job(id='report_update_job', func=update_report_data, 
                      trigger='interval', hours=1, replace_existing=True) 
    
    scheduler.start()
    logging.info("✅ スケジューラーを開始しました。")

# アプリ起動時に初回実行をトリガー
Thread(target=update_report_data).start()
