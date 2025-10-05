from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.config import RANDOM_STATE

def create_full_pipeline() -> Pipeline:
    """
    Cria e retorna o Pipeline completo de ML: Imputação, Escalonamento e LinearSVC.
    """
    pipeline = Pipeline([
        # 1. Imputação de valores faltantes (usando a mediana)
        ('imputer', SimpleImputer(strategy='median')),

        # 2. Escalonamento dos dados (Padronização)
        ('scaler', StandardScaler()),

        # 3. Modelo Linear Support Vector Classification (SVC)
        ('linear_svc', LinearSVC(
            C=1,
            loss='hinge',
            max_iter=500000,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ))
    ])
    return pipeline
