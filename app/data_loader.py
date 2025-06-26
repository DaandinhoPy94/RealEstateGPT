import argparse
import os
import pandas as pd
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Vult lege cellen in het DataFrame."""
    return df.fillna("")

def df_to_docs(df: pd.DataFrame) -> list[str]:
    """
    Converteert elke rij van het DataFrame naar een beschrijvende,
    natuurlijke taaltekst die beter is voor de AI.
    """
    # Geef de eerste kolom een duidelijke naam
    df = df.rename(columns={df.columns[0]: 'ID'})
    
    docs = []
    for _, row in df.iterrows():
        # Creëer een volledige zin voor elke rij
        doc_string = (
            f"Dit is een pand met ID-nummer {row['ID']}. "
            f"Het adres is {row['adress']}. "
            f"Het is een pand van het type '{row['type']}'. "
            f"De geschatte waarde is {row['Value']} euro. "
            f"De leegstand is momenteel {row['VacancyRate'] * 100:.0f} procent. "
            f"De jaarlijkse inkomsten zijn {row['AnualIncome']} euro. "
            f"Het huidige huurcontract loopt af op {row['EndLease']}."
        )
        docs.append(doc_string)
    
    print("\nVoorbeeld van een nieuw document:")
    print(docs[0] if docs else "Geen documenten om te tonen.")
    
    return docs

def main(csv_path: str, vs_path: str = "vectorstore"):
    """Laadt de CSV, converteert het naar documenten en creëert de vectorstore."""
    df = clean(pd.read_csv(csv_path))
    docs = df_to_docs(df)

    print("\nEmbeddings creëren met Mistral. Dit kan even duren...")
    ollama_embeddings = OllamaEmbeddings(model="mistral")

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=ollama_embeddings,
        persist_directory=vs_path
    )
    vectordb.persist()
    print(f"\nVectorstore opgeslagen in '{vs_path}' met de nieuwe document-structuur.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Pad naar portfolio CSV")
    main(p.parse_args().csv)