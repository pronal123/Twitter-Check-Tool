import datetime
import logging
import time
import os
import requests 
from threading import Thread
import io # 画像データをメモリ上で扱うために使用
import random # ダミーデータ生成に使用

# グラフ描画とデータ処理のためのインポート
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# Flask関連のインポート
from flask import Flask, render_template, jsonify
from flask_apscheduler import APScheduler # スケジューラーをインポート

# -----------------
# Telegram Bot設定
# -----------------
# 🚨 実際のBotトークンとチャットIDに置き換えてください
# 環境変数から取得。設定がない場合はダミー値を使用します。
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE') 
# チャットIDは通常マイナス値です（グループの場合）。
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '-1234567890') 
TELEGRAM_API_URL_MESSAGE = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
TELEGRAM_API_URL_PHOTO = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto'

# -----------------
# ロギング設定
# -----------------
# ログ形式を設定
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# -----------------
# アプリケーション初期化
# -----------------
# Flaskアプリのインスタンスを作成
# template_folderをカレントディレクトリ('.')に指定し、
# app.pyと同じ階層の index.html をテンプレートとして読み込むように設定
app = Flask(__name__, template_folder='.') 
# スケジューラーのインスタンスを作成
scheduler = APScheduler()

# ダミーデータとグローバル状態
global_data = {
    'last_updated': 'N/A',
    'data_range': '2023-01-01 - 2025-11-18', # 初期ダミー期間
    'data_count': 0,
    'scheduler_status': '初期化中'
}
data_item_count = 0

# -----------------
# データ取得・分析関数 (将来的にAPI呼び出しに置き換えるモック)
# -----------------
def get_real_time_btc_data(data_count: int) -> tuple[int, int, int]:
    """
    BTCの現在価格、R1、S1をシミュレーションで取得します。
    
    🚨 ユーザーはここを、実際の金融API（例: CoinGecko, Binance, Yahoo Financeなど）
       を呼び出す実践的なロジックに置き換える必要があります。
    """
    # --- 現在はダミー価格生成 ---
    # データ件数に基づいて価格の基準を変動させる
    base_price = 60000 
    price_factor = (data_count // 1000) % 10 
    
    # 価格にランダムな変動を加える
    simulated_price = base_price + price_factor * 2000 + random.randint(-700, 700) 
        
    current_price = int(simulated_price)
    
    # サポート/レジスタンスレベルの計算（現在価格の±1.5%）
    r1 = int(current_price * 1.015)  
    s1 = int(current_price * 0.985)  
    
    # 返り値: (現在価格, R1, S1)
    return current_price, r1, s1

# -----------------
# Telegram通知関数
# -----------------
def send_telegram_message(message: str):
    """
    指定されたメッセージをTelegramチャットに送信します。
    """
    if TELEGRAM_BOT_TOKEN == 'YOUR_BOT_TOKEN_HERE' or not TELEGRAM_CHAT_ID or TELEGRAM_CHAT_ID == '-1234567890':
        logging.warning("Telegram BotトークンまたはチャットIDが設定されていません。通知をスキップします。")
        return

    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown'
    }
    
    try:
        response = requests.post(TELEGRAM_API_URL_MESSAGE, data=payload, timeout=10)
        response.raise_for_status() 
        logging.info("Telegramテキスト通知を送信しました。")
    except requests.exceptions.RequestException as e:
        logging.error(f"Telegramテキスト通知の送信に失敗しました: {e}")
        if 'response' in locals() and response.text:
            logging.error(f"Telegram APIレスポンス: {response.text}")


def send_telegram_photo(photo_bytes: io.BytesIO, caption: str):
    """
    指定された画像をTelegramチャットに送信します。
    """
    if TELEGRAM_BOT_TOKEN == 'YOUR_BOT_TOKEN_HERE' or not TELEGRAM_CHAT_ID or TELEGRAM_CHAT_ID == '-1234567890':
        logging.warning("Telegram BotトークンまたはチャットIDが設定されていません。画像通知をスキップします。")
        return

    # ファイルをバイトストリームとして辞書に追加
    files = {'photo': ('chart.png', photo_bytes.getvalue(), 'image/png')}
    data = {
        'chat_id': TELEGRAM_CHAT_ID,
        'caption': caption,
        'parse_mode': 'Markdown'
    }

    try:
        # sendPhoto APIエンドポイントを使用
        response = requests.post(TELEGRAM_API_URL_PHOTO, data=data, files=files, timeout=20)
        response.raise_for_status() 
        logging.info("Telegram画像通知を送信しました。")
    except requests.exceptions.RequestException as e:
        logging.error(f"Telegram画像通知の送信に失敗しました: {e}")
        if 'response' in locals() and response.text:
            logging.error(f"Telegram APIレスポンス: {response.text}")


def generate_chart_image(current_price: int, r1: int, s1: int) -> io.BytesIO:
    """
    価格推移、R1, S1を含むチャート画像を生成し、io.BytesIOで返します。
    """
    # 1. ダミー時系列データの生成 (過去30日間)
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=30)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # 過去価格をシミュレート (現在の価格を基準にランダムな変動を加える)
    price_series = [current_price]
    for _ in range(len(dates) - 1, 0, -1):
        # 過去に行くほどランダムに変動させ、S1とR1の間に収まりやすくする
        change = random.uniform(-0.015 * current_price, 0.015 * current_price)
        # 価格が極端に外れないようにS1/R1近辺に収束させるシミュレーション
        next_price = max(s1 * 0.9, min(r1 * 1.1, price_series[0] + change * 0.5)) 
        price_series.insert(0, next_price)
    
    df = pd.DataFrame({'Price': price_series}, index=dates)

    # 2. Matplotlibでチャート描画
    # チャートのスタイルを設定
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100) # figsizeを調整
    
    # 価格ライン
    ax.plot(df.index, df['Price'], label='BTC Price (Sim.)', color='#059669', linewidth=2)

    # --- サポート/レジスタンスラインの描画 ---
    
    # R1 (レジスタンス): 赤色の破線
    ax.axhline(r1, color='#ef4444', linestyle='--', linewidth=1.5, label=f'R1: ${r1:,}')
    # R1にラベルを付与
    ax.text(df.index[-1], r1, f' R1 (抵抗線) ${r1:,}', color='#ef4444', ha='right', va='bottom', fontsize=10, weight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))

    # S1 (サポート): 青色の破線
    ax.axhline(s1, color='#3b82f6', linestyle='--', linewidth=1.5, label=f'S1: ${s1:,}')
    # S1にラベルを付与
    ax.text(df.index[-1], s1, f' S1 (支持線) ${s1:,}', color='#3b82f6', ha='right', va='top', fontsize=10, weight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
    
    # 現在価格の点とラベル
    ax.scatter(df.index[-1], current_price, color='black', s=80, zorder=5) # 現在価格を強調
    ax.text(df.index[-1] + datetime.timedelta(days=0.5), current_price, f' Now ${current_price:,}', color='black', ha='left', va='center', fontsize=11, weight='bold')

    # 3. グラフの装飾
    ax.set_title('BTC Price Action with Key Levels (30 Days Sim.)', fontsize=16, color='#1f2937', weight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    
    # 日付フォーマットの設定
    formatter = DateFormatter("%m/%d")
    ax.xaxis.set_major_formatter(formatter)
    
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    # 4. 画像をメモリ上のバイトストリームとして保存
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig) # メモリを解放
    
    return buf

# -----------------
# スケジューリングタスク
# -----------------
def update_report_data():
    """定期的に実行されるタスク：データ取得とレポート更新のシミュレーション"""
    global global_data
    global data_item_count

    logging.info("スケジュールされたレポート更新タスク開始...")
    
    # 1. データ取得のシミュレーション 
    logging.info("ダミーのデータ取得と分析計算をシミュレート...")
    
    # 2. ダミーデータの更新
    data_item_count += random.randint(500, 1500) # 毎回ランダムにデータ件数を増加
    now = datetime.datetime.now()
    
    # 3. グローバル状態の更新
    last_updated_str = now.strftime('%Y-%m-%d %H:%M:%S')
    
    global_data['last_updated'] = last_updated_str
    global_data['data_count'] = data_item_count
    global_data['scheduler_status'] = '稼働中'
    
    logging.info(f"データ取得が成功しました。期間: {global_data['data_range']}, 件数: {global_data['data_count']:,}")
    
    # 4. Telegram通知の実行
    
    # BTC予測のシミュレーション (実践的なシミュレーションロジックを導入)
    data_count = global_data['data_count']
    
    # --- 主要価格帯のシミュレーション (get_real_time_btc_data関数で処理) ---
    current_price, r1, s1 = get_real_time_btc_data(data_count)

    outcomes = {"UP": "上昇 📈", "DOWN": "下降 📉", "SIDE": "レンジ ↔️"}
    predictions = {}
    analysis_details = []
    
    # 価格をカンマ区切りにフォーマット
    formatted_current_price = f"`${current_price:,}`"
    formatted_r1 = f"`${r1:,}`"
    formatted_s1 = f"`${s1:,}`"
    
    price_analysis = [
        f"💰 *現在価格 (シミュレート)*: {formatted_current_price}",
        f"🔼 *主要レジスタンス (R1)*: {formatted_r1} (ブレイクで強い上昇トレンド開始)",
        f"🔽 *主要サポート (S1)*: {formatted_s1} (維持で反発、割れると下降加速)"
    ]

    # --- 実践的なシミュレーションロジック ---
    # 1h予測: データ件数に基づいた短期的なモメンタム
    if (data_count % 3) == 0:
        predictions["1h"] = outcomes["UP"]
        short_term_bias = "上昇"
        analysis_details.append("・1h: 短期RSIは40台で推移しており、上値トライの余地があります。強い下降シグナルは出ていません。")
    elif (data_count % 3) == 1:
        predictions["1h"] = outcomes["DOWN"]
        short_term_bias = "下降"
        analysis_details.append("・1h: 短期移動平均線が短期トレンドラインを下回り、短期的な売り圧力が強まっています。")
    else:
        predictions["1h"] = outcomes["SIDE"]
        short_term_bias = "レンジ"
        analysis_details.append("・1h: 短期的な値動きのエネルギーが低下し、主要なサポート・レジスタンスラインの間で価格が膠着しています。")
        
    # 4h予測: データ件数の偶奇による中期的なトレンド
    if data_count % 2 != 0:
        predictions["4h"] = outcomes["DOWN"]
        analysis_details.append("・4h: MACDラインがシグナルラインを上から下にクロスしており、中期的な下降トレンドへの転換リスクが高まっています。")
    else:
        predictions["4h"] = outcomes["UP"]
        analysis_details.append("・4h: MACDラインがシグナルラインを下から上にクロスし、強い買いシグナルを発生させています。中期的なトレンドは上向きに転換しつつあります。")
        
    # 12h予測: 価格がR1に近いかS1に近いかで判断
    if current_price > (r1 + s1) / 2:
        predictions["12h"] = outcomes["UP"]
        analysis_details.append("・12h: 主要なフィボナッチリトレースメントの0.618ラインを突破し、次のレジスタンスを目指す動きが確認されています。")
    else:
        predictions["12h"] = outcomes["SIDE"]
        analysis_details.append("・12h: 長期的なレジスタンスラインに近づいており、大きな売り注文が集中していることが示唆されます。レンジに留まる可能性が高いです。")
        
    # 24h予測: 長期的なデータ量によるバイアス
    if data_count > 5000:
        predictions["24h"] = outcomes["UP"]
        analysis_details.append("・24h: 長期移動平均線（200MA）の傾きが明確に上向きであり、強い長期上昇トレンドが継続中です。")
        long_term_advice = "長期的な押し目買い戦略"
    else:
        predictions["24h"] = outcomes["SIDE"]
        analysis_details.append("・24h: 長期移動平均線はフラットで、明確な長期トレンドは確立されていません。市場は次の大きなカタリストを待っている状態です。")
        long_term_advice = "次のカタリストまでの様子見戦略"
    # --- ロジック終了 ---
    
    # 予測結果の組み立て
    prediction_lines = []
    timeframes = ["1h", "4h", "12h", "24h"]
    for tf in timeframes:
        prediction_lines.append(f"• {tf}後予測: *{predictions[tf]}*") 
        
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
        f"{price_analysis_text}\n\n" # 新しい価格分析セクション
        f"--- *総合予測* ---\n"
        f"{prediction_text}\n\n"
        f"--- *動向の詳細分析と根拠* ---\n"
        f"{analysis_text}\n\n"
        f"--- *総合戦略サマリー* ---\n"
        f"💡 *短期（1h）バイアス*: *{short_term_bias}* 傾向\n"
        f"🛡️ *推奨戦略*: {long_term_advice}がベースとなります。短期的な値動き（1h/4h）は、分析詳細で触れたモメンタム指標に大きく依存します。\n"
        f"_※ この予測は、現在のデータセットに基づくシミュレーションであり、投資助言ではありません。_"
    )
    
    # 5. 画像生成と画像通知の実行
    try:
        logging.info("チャート画像を生成中...")
        chart_buffer = generate_chart_image(current_price, r1, s1)
        
        # 画像キャプションとしてレポートサマリーの一部を使用
        photo_caption = (
            f"📈 *BTCチャート分析 (Sim.)* 📉\n"
            f"📅 更新: `{last_updated_str}`\n"
            f"{price_analysis_text}\n\n"
            f"総合予測: 💡 *短期バイアス*:{short_term_bias} / 🛡️ *推奨戦略*:{long_term_advice}\n"
            f"_詳細は別途送信されるテキストレポートをご確認ください。_"
        )
        
        # スレッド化して画像通知関数を呼び出し
        Thread(target=send_telegram_photo, args=(chart_buffer, photo_caption)).start()
        
    except Exception as e:
        logging.error(f"チャート画像の生成または送信に失敗しました: {e}")
        # 画像送信失敗時もテキストレポートは必ず送信する

    # テキストレポートの送信
    Thread(target=send_telegram_message, args=(report_message,)).start()
    
    logging.info("レポート更新タスク完了。通知キューに追加されました。")


# -----------------
# ルート（エンドポイント）
# -----------------
@app.route('/')
def index():
    """ダッシュボードの表示"""
    # テンプレート名を参照 (ルートにある index.html を使用)
    return render_template('index.html', title='ML BOT分析レポート ダッシュボード', data=global_data)

@app.route('/status')
def status():
    """現在のステータスをJSONで返すAPIエンドポイント"""
    return jsonify(global_data)

# -----------------
# スケジューラーの初期設定と開始
# -----------------
# Gunicorn環境でscheduler.runningのチェックは非常に重要です
if not scheduler.running:
    # スケジューラー設定
    app.config.update({
        'SCHEDULER_JOBSTORES': {
            'default': {'type': 'memory'}
        },
        'SCHEDULER_EXECUTORS': {
            'default': {'type': 'threadpool', 'max_workers': 20}
        },
        'SCHEDULER_API_ENABLED': False # API経由での制御を無効化
    })
    
    # アプリケーションにスケジューラーを登録
    scheduler.init_app(app)
    
    # 1時間間隔でジョブを追加
    scheduler.add_job(id='report_update_job', func=update_report_data, 
                      trigger='interval', hours=1, replace_existing=True) 
    
    # スケジューラーを開始
    scheduler.start()
    logging.info("✅ スケジューラーを開始しました。")

# 最初のデータロードを強制的に実行し、初期表示に備える
# スケジューラースレッドとは独立して実行します
Thread(target=update_report_data).start()


# -----------------
# サーバー起動 (Gunicorn/本番環境では app.run() は不要です)
# -----------------
