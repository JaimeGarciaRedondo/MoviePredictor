import pandas as pd
import ast
import os
from pathlib import Path
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Rutas de archivos
current_dir = Path(__file__).resolve().parent
PATH_DATA = "../Datasets/movies_dataset_unificado.csv"

def nameSanitization(df):
    df['director'] = df['director'].str.lower() 
    df['director'] = df['director'].str.replace(' ', '')
    df['cast'] = df['cast'].str.lower() 
    df['cast'] = df['cast'].str.replace(' ', '')
    df['genres'] = df['genres'].str.lower() 
    df['genres'] = df['genres'].str.replace(' ', '')
    return df

def minMaxBudgetNormalization(df):
    scaler = MinMaxScaler()
    df['budget'] = scaler.fit_transform(df[['budget']])
    return df

if __name__ == "__main__":
    df = pd.read_csv(PATH_DATA)
    print("Procesando archivos...")
    df = nameSanitization(df)
    df = minMaxBudgetNormalization(df)
    df.to_csv("../Datasets/movies_dataset_unificado.csv", index=False) 
    print("Listo.") 
