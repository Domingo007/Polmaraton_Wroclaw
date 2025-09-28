import streamlit as st
import pandas as pd
import re
import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional, Dict, Tuple, List


load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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

def parse_user_input(user_text: str) -> Tuple[Dict[str, Optional[str]], List[str]]:
    """
    Parsuje płeć, wiek i czas na 5 km z wolnego tekstu.
    Zwraca (dane, brakujące_pola)
    dane: {"sex": "M"/"K", "age": int lub None, "time_str": "mm:ss" lub "X min Y s"}
    """
    text = user_text.lower().strip()

    # PŁEĆ
    sex = None
    if re.search(r'\b(mężczyzn(?:a|ą)|facet|mezczyzna)\b', text):
        sex = "M"
    elif re.search(r'\b(kobiet(?:a|ą)|dziewczyna|baba)\b', text):
        sex = "K"

    # WIEK
    age = None
    m_age = re.search(r'(\d{1,3})\s*(lat|lata|roku|rz)\b', text)
    if m_age:
        try:
            age = int(m_age.group(1))
        except Exception:
            age = None

    # CZAS 5 KM — akceptuj różne formy:
    #  - "23:15", "25:5", "00:22:59"
    #  - "23 min", "23min", "23 min 15 s", "23m 15s", "23 m 15 sek"
    #  - dopuszczamy przecinek zamiast dwukropka: "23,15" -> "23:15"
    ttext = text.replace(',', ':')
    time_seconds = None
    time_str = None

    # Najpierw hh:mm:ss lub mm:ss
    m_hms = re.search(r'\b(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?\b', ttext)
    if m_hms:
        mm = int(m_hms.group(1))
        ss = int(m_hms.group(2))
        hh = int(m_hms.group(3)) if m_hms.group(3) else 0
        if hh > 0:
            total = hh*3600 + mm*60 + ss
        else:
            total = mm*60 + ss
        time_seconds = total
        time_str = f"{mm:02d}:{ss:02d}" if hh == 0 else f"{hh:02d}:{mm:02d}:{ss:02d}"
    else:
        # Formy słowne
        # np. "23 min 15 s", "23min", "23 m", "23 m 5 sek"
        m_min_only = re.search(r'\b(\d{1,3})\s*(?:m|min|min\.|minuty|minut|minutę)\b', text)
        m_sec = re.search(r'\b(\d{1,2})\s*(?:s|sek|sek\.|sekundy|sekund)\b', text)
        if m_min_only and m_sec:
            mm = int(m_min_only.group(1))
            ss = int(m_sec.group(1))
            time_seconds = mm*60 + ss
            time_str = f"{mm:02d}:{ss:02d}"
        elif m_min_only:
            mm = int(m_min_only.group(1))
            time_seconds = mm*60
            time_str = f"{mm:02d}:00"

    # Zbierz braki
    missing = []
    if sex is None:
        missing.append("płeć")
    if age is None:
        missing.append("wiek")
    if time_seconds is None:
        missing.append("czas na 5 km")

    # Zwróć
    data = {
        "sex": sex,
        "age": age,
        "time_str": time_str,          # zachowujemy tekstowy format
        "time_seconds": time_seconds,  # i wersję w sekundach, gdybyś potrzebował
    }
    return data, missing

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
        parsed, missing = parse_user_input(user_input)

        if missing:
            st.error(
                "⚠️ Brakuje danych do pełnej analizy: "
                + ", ".join(missing)
                + ".\n\n"
                "Przykłady poprawnych formatów:\n"
                "- płeć: 'Jestem mężczyzną' / 'Jestem kobietą'\n"
                "- wiek: '37 lat'\n"
                "- czas: '23:15' lub '23 min 15 s' lub '23 min'"
            )
        else:
            st.success("✅ Dane rozpoznane. Analizuję…")

            # 🔹 Analiza GPT
            try:
                result = analyze_with_gpt(user_input, data.head(10))
                st.subheader("📊 Wynik analizy (GPT)")
                st.write(result)
            except Exception as e:
                st.error("❌ Błąd podczas wywołania modelu. Sprawdź klucz API i sieć.")
                st.exception(e)

            # 🔹 Kalkulator półmaratonu z rozpoznanego czasu
            predicted = predict_halfmarathon(parsed["time_str"] or user_input)
            if predicted:
                st.subheader("🏅 Szacowany czas półmaratonu")
                st.write(f"Twój przewidywany czas: **{predicted}**")
            else:
                st.info("Nie udało się zinterpretować Twojego czasu na 5 km. Podaj np. '23:15' lub '23 min 15 s'.")

