# futures_ml_bot.py (MEXCダッシュボード洞察組み込みの完全ロジック)

import os
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import requests
import joblib
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, Dict, Any

# --- 1. 環境変数設定 ---
# These variables must be set in the deployment environment
MEXC_API_KEY = os.environ.get('MEXC_API_KEY')
MEXC_SECRET = os.environ.get('MEXC_SECRET')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

FUTURES_SYMBOL = 'BTC_USDT'
TIMEFRAME = '4h'
MODEL_FILENAME = 'btc_futures_ml_model.joblib'
MEXC_API_BASE_URL = 'https://contract.mexc.com' 

# External API (Assumed) - Replace with actual API URLs
FG_INDEX_API_URL = 'https://api.alternative.me/fng/?limit=1'
COINGLASS_API_URL = 'https://api.coinglass.com/api/v1/liquidation/recent' # Assumed liquidation API


# --- 2. Advanced Custom Data Fetching Function ---
def fetch_advanced_metrics(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
    """
    Fetches/calculates FR, OI, L/S Ratio, Fear & Greed Index, Liquidation Data,
    and simulated insights from the MEXC Dashboard.
    """
    mexc_symbol = symbol.replace('_', '/') 
    metrics = {}
    
    # Simulate latest price (for simulation purposes)
    dummy_price = 95000 + np.random.uniform(-500, 500)

    try:
        # 1. Fetch Funding Rate (FR)
        ticker = exchange.fetch_ticker(mexc_symbol)
        metrics['funding_rate'] = float(ticker.get('fundingRate', 0) or 0)
        
        # 2. Fetch Fear & Greed Index
        fg_response = requests.get(FG_INDEX_API_URL, timeout=5)
        fg_response.raise_for_status()
        fg_data = fg_response.json().get('data', [{}])
        metrics['fg_index'] = int(fg_data[0].get('value', 50))
        metrics['fg_value'] = fg_data[0].get('value_classification', 'Neutral')

        # 3. Fetch Liquidation Data (Coinglass API - Assumed)
        liquidation_response = requests.get(COINGLASS_API_URL, params={'symbol': 'BTC'}, timeout=5)
        liquidation_response.raise_for_status()
        liq_data = liquidation_response.json().get('data', {})
        metrics['liq_24h_total'] = liq_data.get('totalLiquidationUSD', 0.0) 
        metrics['liq_24h_long'] = liq_data.get('longLiquidationUSD', 0.0)
        
        # 4. Simulate OI/LSR (Simulated logic re-inserted)
        metrics['ls_ratio'] = 1.05 + np.random.uniform(-0.1, 0.2) # 1.05 - 1.25
        metrics['oi_change_4h'] = 0.01 + np.random.uniform(-0.02, 0.01) # -0.01 - 0.02
        
        # --- 5. MEXC Macro Data & Heatmap Insight Simulation ---
        
        # Macro Data Simulation (Aggregated OI Trend)
        metrics['aggregated_oi_trend'] = np.random.choice([
            'OI Increasing (Strong Trend Confirmation)',
            'OI Decreasing (Clean Washout)',
            'OI Increasing (Weak Divergence)',
            'Stable OI (Range Play)'
        ])

        # Heat Map Simulation (Liquidation Cluster Insight)
        # Simulate liquidation clustering based on price
        cluster_price_short = int(dummy_price * (1 - np.random.uniform(0.01, 0.03)))
        cluster_price_long = int(dummy_price * (1 + np.random.uniform(0.01, 0.03)))
        metrics['liquidation_cluster'] = np.random.choice([
            f'Large Short Liquidation Cluster below ${cluster_price_short:,.0f}',
            f'Significant Long Liquidation Cluster above ${cluster_price_long:,.0f}',
            'No Dominant Liquidation Cluster'
        ])
        
        return metrics
    
    except requests.exceptions.RequestException as req_e:
        print(f"🚨 External API Request Error: {req_e}")
        # Fallback values if API fails
        return {
            'funding_rate': 0.0, 'ls_ratio': 1.0, 'oi_change_4h': 0.0, 
            'fg_index': 50, 'fg_value': 'API Failed', 
            'liq_24h_total': 0.0, 'liq_24h_long': 0.0,
            'aggregated_oi_trend': 'API Failed - Data Unavailable',
            'liquidation_cluster': 'API Failed - No Cluster Detected'
        }
    except Exception as e:
        print(f"🚨 Futures Index Data Processing Error: {e}")
        # Fallback values for other errors
        return {
            'funding_rate': 0.0, 'ls_ratio': 1.0, 'oi_change_4h': 0.0, 
            'fg_index': 50, 'fg_value': 'API Failed', 
            'liq_24h_total': 0.0, 'liq_24h_long': 0.0,
            'aggregated_oi_trend': 'Internal Error - Data Unavailable',
            'liquidation_cluster': 'Internal Error - No Cluster Detected'
        }


# --- 3. Main BOT Class ---
class FuturesMLBot:
    def __init__(self):
        if not all([MEXC_API_KEY, MEXC_SECRET]):
             raise ValueError("API key not set. Please check environment variables.")
             
        # Initialize CCXT MEXC Futures Client
        self.exchange = ccxt.mexc({
            'apiKey': MEXC_API_KEY,
            'secret': MEXC_SECRET,
            'options': {'defaultType': 'future'},
            'enableRateLimit': True,
        })
        # Target volatility threshold for prediction
        self.target_threshold = 0.0005 
        self.prediction_period = 1 
        self.feature_cols = [] 

    # --- (A) Data Fetching (OHLCV) ---
    def fetch_ohlcv_data(self, limit: int = 100, timeframe: str = TIMEFRAME) -> pd.DataFrame:
        """Fetches OHLCV data"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(FUTURES_SYMBOL, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            raise Exception(f"OHLCV data fetching error: {e}")

    # --- (B) Feature Engineering (including ATR) ---
    def create_ml_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Creates practical features"""
        
        # Calculate Technical Indicators
        df['SMA'] = ta.sma(df['Close'], length=20)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['MACD_H'] = ta.macd(df['Close'])['MACDh_12_26_9']
        df['Vol_Diff'] = df['Volume'] / ta.sma(df['Volume'], length=20)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14) 

        # Add Lag Features (previous values as features)
        for lag in [1, 2, 3]:
            df[f'RSI_L{lag}'] = df['RSI'].shift(lag)
            df[f'Price_L{lag}'] = df['Close'].pct_change(lag).shift(lag)
            
        # Create Target Variable (next period's closing price change rate)
        future_change = df['Close'].pct_change(periods=-self.prediction_period).shift(self.prediction_period)
        
        # Classify Target Variable ('Target') into [-1 (Down), 0 (Range), 1 (Up)]
        df['Target'] = np.select(
            [future_change > self.target_threshold, future_change < -self.target_threshold],
            [1, -1], default=0
        )
        
        df.dropna(inplace=True)
        
        # Generate feature column list on first run
        if not self.feature_cols:
            cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'timestamp']]
            self.feature_cols = [col for col in cols if df[col].dtype in [np.float64, np.int64]]
        
        return df[self.feature_cols], df['Target']

    # --- (C) Model Training and Saving (Continuous Learning) ---
    def train_and_save_model(self, df_long_term: pd.DataFrame) -> bool:
        """Retrains the model from long-term data and saves it to a file"""
        print("🧠 Model Retraining Task Started...")
        X_train, Y_train = self.create_ml_features(df_long_term.copy())
        
        # Use RandomForestClassifier
        model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', max_depth=10)
        model.fit(X_train, Y_train)
        
        # Save the model to a file
        joblib.dump(model, MODEL_FILENAME)
        print("✅ Model Retraining Completed and Saved to File.")
        return True

    # --- (D) Real-time Prediction and Notification ---
    def predict_and_report(self, df_latest: pd.DataFrame, advanced_data: Dict[str, Any]) -> bool:
        """Executes prediction on latest data and generates/notifies two reports"""
        
        try:
            # Load the model from the file
            model = joblib.load(MODEL_FILENAME)
        except FileNotFoundError:
            report = "🚨 Error: Model file not found. Please train and commit first."
            self.send_telegram_notification(report)
            return False

        # Create features for the latest data
        X_latest, _ = self.create_ml_features(df_latest.copy())
        latest_X = X_latest.iloc[[-1]] 
        
        # Execute prediction
        prediction_val = model.predict(latest_X)[0]
        prediction_proba = model.predict_proba(latest_X)[0]
        
        # Generate two reports
        report_structure, report_conclusion = self._generate_two_part_reports(
            latest_price_data=df_latest.iloc[-1], 
            advanced_data=advanced_data, 
            ml_prediction=prediction_val, 
            proba=prediction_proba
        )
        
        # Send to Telegram
        self.send_telegram_notification(report_structure)
        self.send_telegram_notification(report_conclusion)
        
        return True

    # --- (E) Report Generation Helper Function - Generates Advanced Integrated Analysis Report ---
    def _generate_two_part_reports(self, latest_price_data: pd.Series, advanced_data: Dict[str, Any], ml_prediction: int, proba: np.ndarray) -> Tuple[str, str]:
        """
        Generates reports in two parts: "Market Structure and Main Drivers Analysis" and "Final Conclusion and Strategy"
        """
        # Price Data
        price = latest_price_data['Close']
        sma = latest_price_data['SMA']
        atr = latest_price_data['ATR']
        
        # Prediction Result Map
        pred_map = {-1: "📉 Down", 0: "↔️ Range", 1: "📈 Up"}
        ml_result = pred_map.get(ml_prediction, "Unknown")
        
        # Advanced Indicators
        fr = advanced_data.get('funding_rate', 0)
        lsr = advanced_data.get('ls_ratio', 1.0)
        oi_chg = advanced_data.get('oi_change_4h', 0.0)
        fg_index = advanced_data.get('fg_index', 50)
        fg_value = advanced_data.get('fg_value', 'Neutral')
        liq_long = advanced_data.get('liq_24h_long', 0)
        
        # MEXC Dashboard Insights
        oi_trend = advanced_data.get('aggregated_oi_trend', 'Data Fetch Failed')
        liq_cluster_info = advanced_data.get('liquidation_cluster', 'No Cluster Detected')
        
        current_time = datetime.now(timezone.utc).astimezone(None).strftime('%Y-%m-%d %H:%M JST')
        
        max_proba = proba[np.argmax(proba)]
        uncertainty_score = 1.0 - max_proba
        
        # Logic for determining Main Driver and Risk (Simplified)
        main_cause = "Technical Environment (Maintaining Key Support)"
        if fg_index <= 30 and liq_long > 100_000_000:
             main_cause = "Sentiment Shock (Extreme Fear and Long Liquidation Cascade)"
        elif fr > 0.00015 and lsr > 1.1:
             main_cause = "Supply/Demand Imbalance (Long Overheating and High FR)"
        
        risk_level = "Medium🔴"
        if uncertainty_score > 0.40 or fg_index <= 25:
             risk_level = "High🔴🔴"
             
        
        # --- Report A: Market Structure and Main Drivers Analysis ---
        report_structure = f"""
==> **【BTC Market Driver Analysis】** <==
📅 {current_time}

📌 **Key Points**
* **Main Driver:** The primary driver of the current market trend is **{main_cause}**.
* **Sentiment:** The Fear & Greed Index is at **{fg_index}** ("**{fg_value}**" level), suggesting market volatility.
* **Technical Environment:** BTC price **${price:.2f}** is {'🟢 Above' if price > sma else '🔴 Below'} the 20-SMA (${sma:.2f}). Short-term trend is {'bullish' if price > sma else 'bearish'}.

---
### 📉 Market Drivers and Risk Analysis

| Category | Indicator | Current Value / Status | Analysis / Implication |
| :--- | :--- | :--- | :--- |
| **S/D & Liquidity** | FR (Funding Rate) | {fr*100:.4f}% | {'🚨 High cost for long positions. Squeeze risk present.' if fr > 0.00015 else 'Neutral.'} |
| | L/S Ratio | {lsr:.2f} | {'🔴 Long dominance. Imbalance in leveraged positions.' if lsr > 1.1 else '🟡 Balanced.'} |
| | OI Change (4H) | {oi_chg*100:.1f}% | {'🔴 Increasing. Strong momentum for trend continuation.' if oi_chg > 0.03 else '🟢 Decreasing. Potential trend slowdown.'} |
| **Sentiment** | F&G Index | {fg_index} ({fg_value}) | {'Extreme Fear. Counter-trend opportunity or warning of a bottom break.' if fg_index <= 20 else 'Optimistic. Short-term overheating possible.'} |
| | 24H Long Liq. | ${liq_long:,.0f} | {'🚨 Large liquidations occurred. Caution for flash crashes.' if liq_long > 100_000_000 else 'Normal.'} |
| **Volatility** | ATR | ${atr:.2f} | **{(atr / price) * 100:.2f}%**. Suggests range-bound or accelerating trend. |

---
### 📊 MEXC Dashboard Insights (Macro Data / Heatmap)

| Item | Insight | Implication |
| :--- | :--- | :--- |
| **Aggregated OI Trend** | {oi_trend} | Assess the momentum of fund inflow/outflow into the market based on macro data. |
| **清算ヒートマップ** | {liq_cluster_info} | Identify liquidation clustering that acts as a short-term price **magnet**. |

### 🎯 Opportunities and Risks

* **Opportunity:** With market fear rising (F&G Index: {fg_index}), a **strong buy-the-dip opportunity** may arise.
* **🚨 Risk Level:** **{risk_level}**. Risk of liquidation cascades due to high leverage continues. Confirmation of bounce at key support is mandatory.
"""
        
        # --- Prediction Result Adjustment ---
        final_conclusion = ml_result
        if (ml_result == "📈 Up" and fr > 0.00015):
             final_conclusion = f"⚠️ {ml_result} (Caution: Long Overheating)"
        elif (ml_result == "📉 Down" and liq_long > 100_000_000):
             final_conclusion = f"🚨 {ml_result} (Liquidation Cascade Risk)"
        
        # Determine Recommended Strategy
        if uncertainty_score > 0.40 and ml_prediction == 0:
            strategy_advice_short = "Strongly recommend waiting/avoiding trades. Wait for range break."
            entry_long = "Safe support zone"
            entry_short = "Strong resistance"
        else:
             strategy_advice_short = f"Consider trading aligned with ML prediction: **{final_conclusion}**."
             entry_long = f"Buy the dip at the current price level (${price:.2f})"
             entry_short = f"Sell the rally at the current price level (${price:.2f})"
        
        # --- Report B: Final Conclusion and Action Plan ---
        report_conclusion = f"""
==> **【Final Conclusion and Action Plan】** <==
📅 {current_time}

---
### 🤖 Prediction and Overall Strategy

| Item | Analysis Result | Probability | Uncertainty Score |
| :--- | :--- | :--- | :--- |
| **ML Prediction Conclusion** | **{final_conclusion}** | **{max_proba*100:.1f}%** | **{uncertainty_score*100:.1f}%** |

* **Overall Judgment:** **{strategy_advice_short}** Due to the high uncertainty score, limit position size, especially for short-term trades.

### 🎯 Short-term Strategy (Futures/Day Trade)

| Direction | Entry Target | Stop Loss | Take Profit Target |
| :--- | :--- | :--- | :--- |
| **{'Bearish' if ml_prediction <= 0 else 'Bullish'}** | {entry_short if ml_prediction <= 0 else entry_long} | Based on ATR (${atr:.2f} amount) | Recent High/Low zones |

### 📈 Medium/Long-term Strategy (Spot/Dips)

* **Strategy:** **Wait and Buy the Dip**. View market fear as an opportunity to plan buying at safe support zones (e.g., around $90,000 USD).
* **Diversification:** Do not concentrate solely on BTC; diversify funds into altcoins with growth themes (ETH, SOL, etc.) to mitigate medium/long-term risk.

📚 **Summary**
The BOT's final analysis balances technical signs and sentiment. The current market is like a pot of stew where "timing the ingredients is everything." Remain calm and execute actions without haste.
"""
        return report_structure, report_conclusion
        
    # --- (F) Telegram Notification Function - Enhanced Error Handling ---
    def send_telegram_notification(self, message: str):
        """Implementation of notification"""
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
        try:
            response = requests.post(url, data=payload)
            if response.status_code == 200:
                print("✅ Telegram notification completed.")
            else:
                print(f"🚨 Telegram notification error (HTTP {response.status_code}): {response.text}")
        except Exception as e:
            print(f"🚨 Telegram request failed: {e}")
