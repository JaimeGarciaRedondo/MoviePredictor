import pandas as pd
import ast
import os
import numpy as np

# Rutas de archivos
BASE_DIR = 'Datasets/csv/'
PATH_META = os.path.join(BASE_DIR, 'movies_metadata.csv')
PATH_CREDITS = os.path.join(BASE_DIR, 'credits.csv')
PATH_KEYWORDS = os.path.join(BASE_DIR, 'keywords.csv')
OUTPUT_FILE = 'movies_dataset_unificado.csv'

def extraer_director(crew_str):
    try:
        crew_list = ast.literal_eval(crew_str)
        for member in crew_list:
            if member['job'] == 'Director':
                return member['name']
        return np.nan
    except:
        return np.nan

def crear_dataset_unificado():
    # 1. CARGAR METADATA
    print(f"Cargando {PATH_META}...")
    cols_meta = ['id', 'budget', 'genres', 'production_countries', 
                 'title', 'spoken_languages', 'overview']
    
    df_meta = pd.read_csv(PATH_META, usecols=cols_meta, low_memory=False)
    
    # Limpieza ID
    df_meta['id'] = pd.to_numeric(df_meta['id'], errors='coerce')
    df_meta = df_meta.dropna(subset=['id'])
    df_meta['id'] = df_meta['id'].astype(int)

    # Limpieza Budget (Numérico, sin nulos y MAYOR que 0)
    df_meta['budget'] = pd.to_numeric(df_meta['budget'], errors='coerce').fillna(0)
    df_meta = df_meta[df_meta['budget'] > 0]

    # 2. CARGAR KEYWORDS
    print(f"Cargando {PATH_KEYWORDS}...")
    df_kwd = pd.read_csv(PATH_KEYWORDS)
    df_kwd['id'] = df_kwd['id'].astype(int)

    # 3. CARGAR CREDITS
    print(f"Cargando {PATH_CREDITS}...")
    df_credits = pd.read_csv(PATH_CREDITS)
    df_credits['id'] = df_credits['id'].astype(int)
    
    # Extracción de directores
    print("Procesando directores (esto puede tardar)...")
    df_credits['director'] = df_credits['crew'].apply(extraer_director)
    df_credits_clean = df_credits[['id', 'cast', 'director']]

    # 4. MERGE
    print("Mergeando archivos...")
    df_merge = df_meta.merge(df_kwd, on='id', how='left')
    df_final = df_merge.merge(df_credits_clean, on='id', how='left')

    # 5. GUARDAR
    print(f"Guardando resultado en {OUTPUT_FILE}...")
    df_final.to_csv(OUTPUT_FILE, index=False)
    
    print("Listo.")
    
    # --- INFORMACIÓN DE EJEMPLO ---
    print("\n--- RESUMEN DEL DATASET ---")
    print(f"Total de filas: {len(df_final)}")
    print("\nEjemplo (Primeras 5 películas):")
    print(df_final[['title', 'director', 'genres']].head(5).to_string(index=False))

if __name__ == "__main__":
    crear_dataset_unificado()