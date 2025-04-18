import os
import pandas as pd
from tqdm import tqdm
from datetime import timedelta

RAW_DIR = "data/raw/hhblock_dataset"
OUTPUT_FILE = "data/processed/merged_clean.csv"

# === Lecture uniquement des fichiers (0 à 3) ===
all_series = []
files_to_keep = {"block_0.csv", "block_1.csv", "block_2.csv", "block_3.csv"}

for file in tqdm(sorted(os.listdir(RAW_DIR)), desc="Traitement blocs 0 à 3"):
    if file not in files_to_keep:
        continue

    path = os.path.join(RAW_DIR, file)
    try:
        df = pd.read_csv(path)
        if not {"LCLid", "day"}.issubset(df.columns):
            print(f"⚠️ {file} ignoré : colonnes LCLid ou day absentes")
            continue

        hh_cols = [col for col in df.columns if col.startswith("hh_")]
        df_long = df.melt(id_vars=["LCLid", "day"], value_vars=hh_cols,
                          var_name="half_hour", value_name="value")

        df_long["half_hour"] = df_long["half_hour"].str.extract(r"(\d+)").astype(int)
        df_long["datetime"] = pd.to_datetime(df_long["day"], errors="coerce") +                               pd.to_timedelta(df_long["half_hour"] * 30, unit="min")

        df_long = df_long.drop(columns=["day", "half_hour"])
        df_long = df_long.rename(columns={"LCLid": "client_id"})
        df_long = df_long.dropna(subset=["datetime", "value"])
        all_series.append(df_long)

    except Exception as e:
        print(f" Erreur dans {file} : {e}")

if not all_series:
    raise ValueError(" Aucune série valide à concaténer.")

df_all = pd.concat(all_series, ignore_index=True)
df_all = df_all.sort_values(by=["client_id", "datetime"])
os.makedirs("data/processed", exist_ok=True)
df_all.to_csv(OUTPUT_FILE, index=False)

print(f" Données transformées et sauvegardées dans : {OUTPUT_FILE}")
