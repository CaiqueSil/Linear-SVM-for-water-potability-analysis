# src/config.py
import os

# --- Configuração Global ---
RANDOM_STATE = 42
N_SPLITS = 200 # Número de folds usado na StratifiedKFold

# --- Configuração de Dados ---
TARGET_COLUMN = 'Potability'
# Features utilizadas na análise original
FEATURE_COLUMNS = [
    'ph',
    'Chloramines',
    'Sulfate',
    'Conductivity',
    'Trihalomethanes',
    'Turbidity',
]

# --- Configuração de Caminhos ---
# Define o caminho base como o diretório raiz do projeto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Diretórios de saída
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')
BEST_MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'best.pkl')

FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
RESULTS_CSV_PATH = os.path.join(REPORTS_DIR, 'results.csv')
