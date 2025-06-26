import streamlit as st
import requests
import pandas as pd
import os

# --- Pagina Configuratie ---
st.set_page_config(
    page_title="RealEstateGPT",
    page_icon="🏘️",
    layout="wide"
)

# --- Backend API URL ---
# Zorg ervoor dat je FastAPI-server draait op http://127.0.0.1:8000
API_URL = "http://127.0.0.1:8000/chat"

# --- Applicatie Titel ---
st.title("🏘️🤖 RealEstateGPT")
st.markdown("Chat met je vastgoedportefeuille. Stel vragen over je panden, de waarde, leegstand of huuropbrengsten.")

# --- Weergave van de Portfolio Data ---
st.subheader("Huidige Vastgoedportefeuille")
DATA_FILE = os.path.join("data", "portfolio.csv")

try:
    df = pd.read_csv(DATA_FILE)
    st.dataframe(df)
except FileNotFoundError:
    st.error(f"Bestand niet gevonden: `{DATA_FILE}`. Zorg ervoor dat het bestand in de juiste map staat.")
    st.info("Je moet eerst de data laden met `python app/data_loader.py --csv data/portfolio.csv` voordat je de backend start.")
    st.stop()


# --- Chat Interface ---
st.subheader("Chat")

# Initialiseer chat geschiedenis in de sessie state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Toon eerdere berichten
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Vraag om input van de gebruiker
if prompt := st.chat_input("Stel een vraag, bijvoorbeeld: 'Wat is de totale waarde van de panden in Amsterdam?'"):
    # Voeg gebruikersbericht toe aan de geschiedenis en toon het
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Toon het antwoord van de assistent
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            # Roep de backend API aan
            payload = {"question": prompt}
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()  # Genereert een error bij een slechte status code (bv. 404, 500)
            
            answer = response.json().get("answer", "Sorry, ik kon geen antwoord vinden.")
            full_response = answer

        except requests.exceptions.RequestException as e:
            full_response = f"Fout: Kon geen verbinding maken met de backend. Draait de server? Details: {e}"
        
        message_placeholder.markdown(full_response)

    # Voeg het antwoord van de assistent toe aan de geschiedenis
    st.session_state.messages.append({"role": "assistant", "content": full_response})