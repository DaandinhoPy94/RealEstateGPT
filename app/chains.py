from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

def get_chain():
    """
    Returns a ConversationalRetrievalChain instance using Mistral via Ollama.
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
        # Gebruik Ollama met het Mistral model voor de embeddings
        ollama_embeddings = OllamaEmbeddings(model="mistral") # <-- AANGEPAST

        vectordb = Chroma(
            persist_directory='vectorstore',
            embedding_function=ollama_embeddings
        )

        # Gebruik het Mistral chatmodel via Ollama
        llm = ChatOllama(model="mistral", temperature=0.0) # <-- AANGEPAST

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectordb.as_retriever(),
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT}
        )
        print("Chain loaded successfully using Mistral.")
        return chain
    except Exception as e:
        print(f"Error loading chain: {e}")
        return None