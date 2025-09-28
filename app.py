import streamlit as st
import pandas as pd
import re
import os
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------
# 📂 Wczytanie danych CSV
# -------------------------------
@st.cache_data
def load_data():
    df1 = pd.read_csv("halfmarathon_wroclaw_2023__final.csv")
    df2 = pd.read_csv("halfmarathon_wroclaw_2024__final.csv")
    df = pd.concat([df1, df2], ignore_index=True)
    return df

data = load_data()

# -------------------------------
# 🏃 Funkcja predykcji półmaratonu
# -------------------------------
def predict_halfmarathon(time_text):
    """
    Przewiduje czas półmaratonu na podstawie czasu 5 km.
    Obsługuje format np. '23 minuty' albo '25:30'.
    """
    match = re.search(r'(\d+)\s*min', time_text.lower())
    if match:
        minutes = int(match.group(1))
        seconds = 0
    else:
        parts = re.findall(r'\d+', time_text)
        if len(parts) == 2:  # mm:ss
            minutes, seconds = int(parts[0]), int(parts[1])
        elif len(parts) == 3:  # hh:mm:ss
            hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
            minutes += hours * 60
        else:
            return None

    # czas na 5 km w sekundach
    t1 = minutes * 60 + seconds
    d1 = 5
    d2 = 21.097

    # Riegel formula
    t2 = t1 * (d2 / d1) ** 1.06

    # Konwersja na hh:mm:ss
    hours = int(t2 // 3600)
    minutes = int((t2 % 3600) // 60)
    seconds = int(t2 % 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# -------------------------------
# 🌐 Funkcja do analizy z GPT
# -------------------------------
def analyze_with_gpt(user_text, dataframe_head):
    prompt = f"""
    Użytkownik podał następujące informacje o sobie:
    {user_text}

    Oto przykładowe dane z maratonów Wrocławskich:
    {dataframe_head.to_string(index=False)}

    Na podstawie tego porównaj użytkownika z danymi
    i napisz wnioski w kilku zdaniach:
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",   # zamiast text-davinci-003
        messages=[
            {"role": "system", "content": "Jesteś pomocnym asystentem analizującym biegi."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.7
    )

    return response.choices[0].message.content.strip()

# -------------------------------
# 🖥️ Interfejs Streamlit
# -------------------------------
st.set_page_config(page_title="Analiza biegu – Wrocław Marathon", layout="centered")

st.title("🏃 Analiza Twojego Biegu")
st.write("Podaj swoje dane (płeć, wiek, czas na 5 km). Aplikacja porówna Cię z danymi maratonów Wrocławskich i przewidzi Twój czas półmaratonu.")

# Formularz użytkownika
user_input = st.text_area(
    "Wpisz dane (np. 'Jestem mężczyzną, mam 37 lat, mój czas na 5 km to 23 minuty.')"
)

if st.button("Analizuj"):
    if user_input.strip() == "":
        st.warning("❗ Proszę wprowadzić swoje dane.")
    else:
        # Walidacja danych
        missing = []
        if not re.search(r'\b(mężczyzna|kobieta)\b', user_input.lower()):
            missing.append("płeć")
        if not re.search(r'\d+\s*(lat|roku|lata)', user_input.lower()):
            missing.append("wiek")
        if not re.search(r'(\d+\s*min)|(\d+:\d+)', user_input.lower()):
            missing.append("czas na 5 km")

        if missing:
            st.error(f"⚠️ Brakuje danych: {', '.join(missing)}. Uzupełnij je, aby przeprowadzić pełną analizę.")
        else:
            st.success("✅ Dane zapisane. Analizuję…")

            # 🔹 Analiza GPT
            result = analyze_with_gpt(user_input, data.head(10))
            st.subheader("📊 Wynik analizy")
            st.write(result)

            # 🔹 Kalkulator półmaratonu
            predicted = predict_halfmarathon(user_input)
            if predicted:
                st.subheader("🏅 Szacowany czas półmaratonu")
                st.write(f"Twój przewidywany czas: **{predicted}**")
            else:
                st.info("Nie udało się rozpoznać Twojego czasu na 5 km. Podaj np. '23 minuty' albo '25:30'.")