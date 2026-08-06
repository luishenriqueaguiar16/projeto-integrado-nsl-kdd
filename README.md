# Detecção de Intrusões em Redes de Computadores

Projeto desenvolvido para a disciplina de Projeto Integrado com o objetivo de identificar acessos maliciosos em redes de computadores usando Aprendizado de Máquina. O sistema auxilia a equipe de segurança na filtragem de conexões.

## Sobre o Projeto

O trabalho utiliza a base de dados pública NSL-KDD da Universidade de New Brunswick (UNB). Trata-se de uma base aberta de tráfego de rede sem dados pessoais.

O modelo escolhido foi o Random Forest (Floresta Aleatória), treinado para classificar cada conexão entre normal e ataque. A meta principal é manter a taxa de falsos alarmes abaixo de 5%.

## Como Executar

1. Clonar o repositório e acessar a pasta do projeto:
```bash
git clone https://github.com/luishenriqueaguiar16/projeto-integrado-nsl-kdd.git
cd "Projeto Integrado"
```

2. Criar e ativar o ambiente virtual:
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

3. Instalar as dependências:
```bash
pip install -r requisitos.txt
```

4. Executar o painel visual no Streamlit:
```bash
streamlit run app/app.py
```

Para reprocessar os dados brutos ou executar o treinamento:
```bash
python src/salvar_dados_processados.py
python src/executar_treinamento.py
```

## Resultados Obtidos

Na validação cruzada, o modelo DummyClassifier obteve F1-Score de 0%, pois apenas previa a classe mais frequente. O modelo de Regressão Logística obteve F1-Score de 96,90%.

O modelo final Random Forest alcançou 99,86% de F1-Score na validação cruzada e apresentou uma taxa de falsos alarmes de 2,72% na base de teste final, cumprindo a meta do projeto.
