import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import time
import asyncio
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor

# 🔹 رابط بيانات كراش (يجب تعديله حسب الموقع)
CRASH_HISTORY_URL = "https://1xbit.com/fr/allgamesentrance/crash"

# 🔹 بيانات Telegram Bot
TELEGRAM_BOT_TOKEN = "6768620123:AAE8-vG9GKbLI9LvP0yqOaFLNThtPgrG5kE"
CHAT_ID = "1722250078"

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher(bot)

# 🔍 دالة لجلب بيانات كراش
def get_crash_history():
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(CRASH_HISTORY_URL, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        crash_results = soup.find_all("div", class_="crash-result")
        crash_values = [float(result.text.strip()) for result in crash_results[:1000]]  # تجميع 1000 جولة

        return crash_values[::-1]  # ترتيب البيانات من الأقدم للأحدث
    else:
        print("❌ فشل في جلب البيانات")
        return []

# 📊 تحليل الإحصائيات والأنماط
def analyze_patterns(crash_values):
    df = pd.DataFrame({"Multiplier": crash_values})

    mean_value = np.mean(crash_values)
    std_dev = np.std(crash_values)
    min_value = np.min(crash_values)
    max_value = np.max(crash_values)

    print(f"📊 **إحصائيات كراش:**")
    print(f"✔️ المتوسط الحسابي: {round(mean_value, 2)}X")
    print(f"📉 أقل قيمة انهيار: {min_value}X")
    print(f"📈 أعلى قيمة انهيار: {max_value}X")
    print(f"📊 التباين: {round(std_dev, 2)}")

# 🔍 تحليل البيانات بصريًا
def visualize_data(crash_values):
    plt.figure(figsize=(10, 5))
    sns.histplot(crash_values, bins=30, kde=True, color="blue")
    plt.title("🔍 توزيع قيم Crash")
    plt.xlabel("Multiplier")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

# 🔮 بناء نموذج LSTM لتحليل الأنماط والتوقع
def train_lstm_model(crash_values):
    if len(crash_values) < 50:
        return None, "❌ بيانات غير كافية للتدريب."

    df = pd.DataFrame({"crash_value": crash_values})

    # تطبيع البيانات
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df)

    # تحضير البيانات للنموذج
    X, y = [], []
    sequence_length = 10  # استخدام آخر 10 جولات للتوقع

    for i in range(len(df_scaled) - sequence_length):
        X.append(df_scaled[i:i + sequence_length])
        y.append(df_scaled[i + sequence_length])

    X, y = np.array(X), np.array(y)

    # بناء نموذج LSTM
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=20, batch_size=16, verbose=1)

    return model, scaler

# 🔮 توقع الجولة القادمة باستخدام LSTM
def predict_next_crash_lstm(crash_values, model, scaler):
    if model is None:
        return "❌ لا يمكن التنبؤ بسبب نقص البيانات."

    last_values = np.array(crash_values[-10:]).reshape(1, 10, 1)
    last_values_scaled = scaler.transform(last_values.reshape(-1, 1)).reshape(1, 10, 1)

    predicted_scaled = model.predict(last_values_scaled)
    predicted_value = scaler.inverse_transform(predicted_scaled.reshape(-1, 1))[0][0]

    return f"🔮 **توقع الجولة القادمة (LSTM):** {round(predicted_value, 2)}X"

# 📢 إرسال إشعار تيليجرام عند توقع جولة جيدة
async def send_telegram_alert(prediction):
    message = f"🚀 **إشعار كراش** 🚀\n{prediction}"
    await bot.send_message(CHAT_ID, message)

# 🔄 تسجيل البيانات وتحليلها بشكل دوري
def record_crash_data():
    while True:
        crash_data = get_crash_history()

        if crash_data:
            print(f"📊 آخر 10 نتائج: {crash_data[-10:]}")
            analyze_patterns(crash_data)
            visualize_data(crash_data)

            model, scaler = train_lstm_model(crash_data)
            prediction = predict_next_crash_lstm(crash_data, model, scaler)

            print(prediction)
            asyncio.run(send_telegram_alert(prediction))

        time.sleep(10)  # تحديث كل 10 ثوانٍ

# تشغيل التحليل
if __name__ == "__main__":
    record_crash_data()
