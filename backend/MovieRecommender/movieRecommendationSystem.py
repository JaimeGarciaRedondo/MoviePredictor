from sentence_transformers import SentenceTransformer
import pandas as pd
import ast
import numpy as np
import os

PATH_DATA ="../Data/Datasets/movies_dataset_unificado.csv"
PATH_EMBEDDINGS = "../Data/Embeddings/embeddings.npy"

def embeddingsMatrix():
    print("Cargando datos...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    df = pd.read_csv(PATH_DATA)
    df['final'] = df['title'] + '. ' + df['overview'] + '. ' + df['keywords']
    df['final'] = df['final'].fillna('')
    print("Generando embeddings...")
    embeddings = model.encode(df['final'])
    print("Guardando embeddings...")
    
    directory = os.path.dirname(PATH_EMBEDDINGS) # Obtiene "../Data/Embeddings"
    if not os.path.exists(directory):
        print(f"Creando carpeta: {directory}")
        os.makedirs(directory) # Crea la carpeta (y subcarpetas si faltan)
        
    print(f"Guardando embeddings en: {PATH_EMBEDDINGS}")
    np.save(PATH_EMBEDDINGS, embeddings)

def metadataMatrix():
    print("Cargando datos...")
    df = pd.read_csv(PATH_DATA)
    
    def get_top_3_actors(x):
        try:
            return [i['name'] for i in ast.literal_eval(x)][:3]
        except:
            return []
            
    df['cast'] = df['cast'].apply(get_top_3_actors)
    
    

if __name__ == "__main__":
    metadataMatrix() 


    