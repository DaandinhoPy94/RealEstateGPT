import argparse
import os
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document # <-- DE NIEUWE IMPORT

def load_and_prepare_df(csv_path: str) -> pd.DataFrame:
    """
    Laadt het CSV-bestand, voegt een ID-kolom toe en maakt het schoon.
    """
    df = pd.read_csv(csv_path)
    df.insert(0, 'id', df.index)
    df.columns = df.columns.str.lower().str.strip()
    return df.fillna("")

def df_to_docs(df: pd.DataFrame) -> list[Document]: # <-- LET OP: geeft nu een lijst met Documenten terug
    """
    Converteert elke rij van het DataFrame naar een beschrijvend Document-object.
    """
    docs = []
    for _, row in df.iterrows():
        doc_string = (
            f"Dit is een pand met ID-nummer {row['id']}. "
            f"Het adres is {row['address']}. "
            f"Het is een pand van het type '{row['type']}'. "
            f"De geschatte waarde is {row['value']} euro. "
            f"De leegstand is momenteel {row['vacancyrate'] * 100:.0f} procent. "
            f"De jaarlijkse inkomsten zijn {row['anualincome']} euro. "
            f"Het huidige huurcontract loopt af op {row['endlease']}."
        )
        # Stop de tekst (doc_string) in een Document-object "envelop"
        docs.append(Document(page_content=doc_string)) # <-- DE CRUCIALE AANPASSING
    
    print("\nVoorbeeld van een nieuw document (inhoud):")
    print(docs[0].page_content if docs else "Geen documenten om te tonen.")
    
    return docs

def main(csv_path: str, vs_path: str = "vectorstore"):
    """Laadt de CSV, converteert het naar documenten en creëert de vectorstore."""
    df = load_and_prepare_df(csv_path)
    docs = df_to_docs(df)

    print("\nEmbeddings creëren met een lokaal model (all-MiniLM-L6-v2). Dit kan even duren...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=vs_path
    )
    vectordb.persist()
    print(f"\nVectorstore opgeslagen in '{vs_path}'. Dit was de laatste stap van de voorbereiding!")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Pad naar portfolio CSV")
    main(p.parse_args().csv)