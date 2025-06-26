import streamlit as st
import pandas as pd
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings # We gebruiken dit lokaal
from langchain_groq import ChatGroq # Aangepast voor Groq
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# --- PAGINA CONFIGURATIE ---
st.set_page_config(
    page_title="RealEstateGPT",
    page_icon="🏘️",
    layout="wide"
)

# --- FUNCTIES (Logica van de oude backend) ---

# Gebruik caching om te voorkomen dat de chain elke keer opnieuw wordt geladen
@st.cache_resource
def get_chain(_groq_api_key):
    """
    Maakt de volledige RAG-keten aan en laadt deze in het geheugen.
    """
    prompt_template = """Gebruik de volgende context om de vraag aan het einde te beantwoorden.
    Als je het antwoord niet weet, zeg dan dat je het niet weet, probeer geen antwoord te verzinnen.
    Het datumformaat in de context is YYYY-MM-DD.

    Context:
    {context}

    Vraag: {question}
    Antwoord:"""

    QA_PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    try:
        # Voor de embeddings gebruiken we nog steeds de LOKALE Ollama.
        # Dit gebeurt eenmalig wanneer de app start of de data verandert.
        # De Streamlit Cloud zal dit ook cachen.
        ollama_embeddings = OllamaEmbeddings(model="mistral")

        vectordb = Chroma(
            persist_directory='vectorstore',
            embedding_function=ollama_embeddings
        )

        # Voor het CHATTEN gebruiken we de SNELLE GROQ API
        llm = ChatGroq(
            temperature=0.0,
            model_name="llama3-8b-8192", # Llama3 is een geweldig model op Groq
            groq_api_key=_groq_api_key
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectordb.as_retriever(),
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT}
        )
        return chain
    except Exception as e:
        st.error(f"Fout bij het laden van de chain: {e}")
        return None

# --- HOOFD APPLICATIE ---
st.title("🏘️🤖 RealEstateGPT")
st.markdown("Chat met je vastgoedportefeuille.")

# Haal de API-sleutel op uit de Streamlit Secrets
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except FileNotFoundError:
    st.warning("Secrets-bestand niet gevonden. Vul hieronder je Groq API-sleutel in voor lokale tests.")
    groq_api_key = st.text_input("Voer je Groq API Key in (gsk_...):", type="password")

if not groq_api_key:
    st.info("Voer alsjeblieft je Groq API key in om de app te starten.")
    st.stop()

# Laad de chain
chain = get_chain(groq_api_key)

if not chain:
    st.stop()

# Weergave van de Portfolio Data
st.subheader("Huidige Vastgoedportefeuille")
DATA_FILE = os.path.join("data", "portfolio.csv")

try:
    df = pd.read_csv(DATA_FILE)
    st.dataframe(df)
except FileNotFoundError:
    st.error(f"Bestand niet gevonden: {DATA_FILE}")
    st.stop()

# Chat Interface
st.subheader("Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Stel een vraag..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Antwoord wordt voorbereid..."):
            result = chain.invoke({
                "question": prompt,
                "chat_history": st.session_state.get("chat_history", [])
            })
            response = result.get("answer", "Sorry, ik kon geen antwoord genereren.")
            st.markdown(response)
            
            # Update chat history for the chain
            st.session_state["chat_history"] = st.session_state.get("chat_history", []) + [(prompt, response)]

    st.session_state.messages.append({"role": "assistant", "content": response})