import pandas as pd
import numpy as np

# Nomes das colunas do dataset NSL-KDD
OFFICIAL_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", 
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", 
    "logged_in", "num_compromised", "root_shell", "su_attempted", 
    "num_root", "num_file_creations", "num_shells", "num_access_files", 
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count", 
    "srv_count", "srv_serror_rate", "srv_rerror_rate", "serror_rate", 
    "rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", 
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", 
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", 
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", 
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", 
    "dst_host_srv_rerror_rate", "class", "difficulty_level"
]

def load_raw_dataset(file_path):
    """Carrega o arquivo CSV do NSL-KDD e define o nome das colunas."""
    if not pd.io.common.file_exists(file_path):
        raise FileNotFoundError(f"Arquivo nao localizado: {file_path}")
        
    df = pd.read_csv(file_path, header=None, names=OFFICIAL_COLUMNS)
    return df

def binarize_target(df):
    """Converte a coluna class em 0 para normal e 1 para ataque."""
    df_temp = df.copy()
    
    def converter_classe_para_binario(nome_classe):
        if nome_classe == "normal":
            return 0
        else:
            return 1
            
    df_temp["target"] = df_temp["class"].apply(converter_classe_para_binario)
    return df_temp

def apply_log_transformations(df):
    """Aplica escala logaritmica nos valores de bytes."""
    df_temp = df.copy()
    
    # Soma 1 para evitar log de zero
    df_temp["log_src_bytes"] = np.log10(df_temp["src_bytes"] + 1)
    df_temp["log_dst_bytes"] = np.log10(df_temp["dst_bytes"] + 1)
    return df_temp

def drop_correlated_features(df):
    """Remove colunas repetidas ou altamente correlacionadas."""
    colunas_para_remover = [
        "num_root", 
        "srv_serror_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "srv_rerror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
    ]
    return df.drop(columns=colunas_para_remover, errors='ignore')

def execute_preprocessing_pipeline(file_path):
    """Executa o pre-processamento dos dados."""
    df = load_raw_dataset(file_path)
    df = binarize_target(df)
    df = apply_log_transformations(df)
    df = drop_correlated_features(df)
    return df
