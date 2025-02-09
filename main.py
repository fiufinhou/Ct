import time
import json
import numpy as np
import requests
import telebot
import tensorflow as tf
import xgboost as xgb
import simpleaudio as sa
import websocket
import re
import random
import datetime
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# ✅ إعداد متصفح Selenium لسحب بيانات اللعبة
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
driver.get("https://www.1xbit.com")

# ✅ إعداد Telegram Bot للإشعارات
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# ✅ قائمة لتخزين بيانات الانهيار وتحليلها
crash_data = []
time_intervals = []
market_trends = []
player_activity = []
chat_messages = []
scaler = MinMaxScaler(feature_range=(0, 1))
prev_time = time.time()

# ✅ اتصال WebSocket لجمع نشاط اللاعبين
def on_message(ws, message):
    data = json.loads(message)
    if "player" in data:
        player_activity.append(data["player"])
    if "chat" in data:
        chat_messages.append(data["chat"])

ws = websocket.WebSocketApp("wss://game-server.example.com", on_message=on_message)
ws.run_forever()

# ✅ تحليل السوق وجمع بيانات تقلبات العملات الرقمية
def get_market_trend():
    response = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd")
    if response.status_code == 200:
        market_data = response.json()
        return (market_data["bitcoin"]["usd"], market_data["ethereum"]["usd"])
    return None

# ✅ إنشاء نموذج Reinforcement Learning باستخدام DQN
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Dense(24, activation='relu', input_dim=self.state_size),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

agent = DQNAgent(state_size=3, action_size=3)

# ✅ تدريب نموذج LSTM + XGBoost مع تحليل السوق ونشاط اللاعبين
def train_models():
    if len(crash_data) < 50:
        return None, None

    df = np.array(crash_data).reshape(-1, 1)
    df_scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(df_scaled) - 10):
        X.append(df_scaled[i:i+10, 0])
        y.append(df_scaled[i+10, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # ✅ نموذج LSTM
    lstm_model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X, y, epochs=50, batch_size=16, verbose=1)

    # ✅ نموذج XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    xgb_model.fit(X.reshape(X.shape[0], -1), y)

    return lstm_model, xgb_model

# ✅ استخراج بيانات اللعبة وتحليل تأثير نشاط اللاعبين
def scrape_multiplier():
    global prev_time
    try:
        element = driver.find_element(By.CLASS_NAME, "crash-multiplier")
        multiplier = float(element.text.replace("x", ""))

        # ✅ تحليل الزمن بين الجولات
        current_time = time.time()
        interval = current_time - prev_time
        prev_time = current_time
        time_intervals.append(interval)

        return multiplier
    except:
        return None

# ✅ تحليل نصوص الدردشة لاستخلاص مؤشرات الرهان
def analyze_chat():
    chat_text = " ".join(chat_messages[-20:]).lower()
    if re.search(r"(crash|pump|dump)", chat_text):
        return True
    return False

# ✅ تشغيل النظام لتحليل البيانات
while True:
    multiplier = scrape_multiplier()
    if multiplier:
        crash_data.append(multiplier)
        market_data = get_market_trend()
        is_chat_suspicious = analyze_chat()

        # ✅ تدريب النماذج عند توفر بيانات كافية
        lstm_model, xgb_model = train_models()
        if lstm_model and xgb_model:
            X_pred = np.array([[multiplier]])
            X_pred = scaler.transform(X_pred)
            X_pred = np.reshape(X_pred, (1, 1, 1))

            predicted_lstm = lstm_model.predict(X_pred)[0][0]
            predicted_lstm = scaler.inverse_transform([[predicted_lstm]])[0][0]

            predicted_xgb = xgb_model.predict(X_pred.reshape(1, -1))[0]

            # ✅ استخدام Reinforcement Learning لاختيار القرار الأفضل
            state = np.array([[multiplier, market_data[0], market_data[1]]])
            action = agent.act(state)

            if action == 0 or is_chat_suspicious:
                advice = "🚨 تجنب المراهنة (انهيار مبكر محتمل)"
            elif action == 1:
                advice = "✅ فرصة متوسطة، انسحب مبكرًا"
            else:
                advice = "🚀 فرصة لرهان أطول، لكن كن حذرًا!"

            print(advice)
            bot.send_message(TELEGRAM_CHAT_ID, advice)

            # ✅ تنبيه صوتي عند انهيارات خطيرة
            if predicted_lstm < 1.5 or predicted_xgb < 1.5:
                wave_obj = sa.WaveObject.from_wave_file("alert.wav")
                play_obj = wave_obj.play()

    time.sleep(2)
