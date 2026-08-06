import os
import pandas as pd
from preprocessamento import execute_preprocessing_pipeline

def main():
    print("Iniciando pre-processamento dos dados...")
    
    raw_train_path = os.path.join("data", "raw", "KDDTrain+.txt")
    raw_test_path = os.path.join("data", "raw", "KDDTest+.txt")
    processed_train_path = os.path.join("data", "processed", "train_processed.csv")
    processed_test_path = os.path.join("data", "processed", "test_processed.csv")
    
    # Executa o pre-processamento inicial
    df_train = execute_preprocessing_pipeline(raw_train_path)
    df_test = execute_preprocessing_pipeline(raw_test_path)
    
    # Junta as duas bases para converter categorias em numeros (get_dummies)
    df_combined = pd.concat([df_train, df_test], keys=["train", "test"])
    df_combined = pd.get_dummies(df_combined, columns=["protocol_type", "service", "flag"], dtype=int)
    
    # Separa novamente em treino e teste
    df_train_encoded = df_combined.xs("train")
    df_test_encoded = df_combined.xs("test")
    
    # Remove colunas desnecessarias
    colunas_para_remover = ["class", "difficulty_level", "src_bytes", "dst_bytes"]
    df_train_final = df_train_encoded.drop(columns=colunas_para_remover, errors="ignore")
    df_test_final = df_test_encoded.drop(columns=colunas_para_remover, errors="ignore")
    
    os.makedirs(os.path.dirname(processed_train_path), exist_ok=True)
    
    # Salva os arquivos CSV
    df_train_final.to_csv(processed_train_path, index=False)
    df_test_final.to_csv(processed_test_path, index=False)
    
    print(f"Treino salvo: {processed_train_path} ({df_train_final.shape[0]} linhas)")
    print(f"Teste salvo: {processed_test_path} ({df_test_final.shape[0]} linhas)")
    print("Pre-processamento concluido!")

if __name__ == "__main__":
    main()
