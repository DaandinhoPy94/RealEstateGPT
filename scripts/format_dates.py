import pandas as pd
import os

# Pad naar het CSV-bestand
CSV_FILE_PATH = os.path.join('data', 'portfolio.csv')

def format_lease_dates():
    """
    Leest het portfolio.csv-bestand, verandert het 'EndLease'-datumformaat
    van DD-MM-YYYY naar YYYY-MM-DD, en overschrijft het bestand.
    """
    try:
        print(f"Bestand wordt gelezen: {CSV_FILE_PATH}")
        df = pd.read_csv(CSV_FILE_PATH)

        print("\nOriginele 'EndLease' datums (eerste 5):")
        print(df['EndLease'].head())

        # Converteer de 'EndLease'-kolom naar datetime-objecten
        # We specificeren het originele formaat om fouten te voorkomen
        df['EndLease'] = pd.to_datetime(df['EndLease'], format='%d-%m-%Y')

        # Formatteer de datums terug naar strings in het nieuwe 'YYYY-MM-DD'-formaat
        df['EndLease'] = df['EndLease'].dt.strftime('%Y-%m-%d')

        # Sla het aangepaste dataframe op, en overschrijf het originele bestand
        # index=False voorkomt dat er een extra indexkolom wordt toegevoegd
        df.to_csv(CSV_FILE_PATH, index=False)

        print("\nNieuwe 'EndLease' datums (eerste 5):")
        print(df['EndLease'].head())
        
        print(f"\nDatumformaat succesvol aangepast in {CSV_FILE_PATH}")

    except FileNotFoundError:
        print(f"Fout: Het bestand {CSV_FILE_PATH} kon niet worden gevonden.")
    except Exception as e:
        print(f"Er is een fout opgetreden: {e}")

if __name__ == "__main__":
    format_lease_dates()