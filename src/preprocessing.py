import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
class EliminarColumnas(BaseEstimator, TransformerMixin):
    """Elimina columnas con muchos nulos o irrelevantes."""
    def __init__(self, columnas=None):
        self.columnas = columnas or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(columns=[c for c in self.columnas if c in X.columns], errors='ignore')
        return X

class ImputarEdad(BaseEstimator, TransformerMixin):
    """Imputa la edad con la mediana."""
    def fit(self, X, y=None):
        if 'edad' in X.columns:
            self.mediana_edad = X['edad'].median()
        else:
            self.mediana_edad = None
        return self

    def transform(self, X):
        X = X.copy()
        if self.mediana_edad is not None and 'edad' in X.columns:
            X['edad'] = X['edad'].fillna(self.mediana_edad)
        return X

class ReemplazarCategoricos(BaseEstimator, TransformerMixin):
    """Reemplaza los valores nulos de variables categóricas con 'Otro'."""
    def __init__(self, columnas=None):
        self.columnas = columnas or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.columnas:
            if c in X.columns:
                X[c] = X[c].fillna('Otro')
        return X

class ReemplazarNumericos(BaseEstimator, TransformerMixin):
    """Reemplaza NaN en columnas numéricas con 0."""
    def __init__(self, excluir=None):
        self.excluir = excluir or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in X.columns:
            if c not in self.excluir and pd.api.types.is_numeric_dtype(X[c]):
                X[c] = X[c].fillna(0)
        return X
class ConvertirTipos(BaseEstimator, TransformerMixin):
    """Convierte columnas a tipos definidos (int32 o float64)."""
    def __init__(self, columnas_int=None, columnas_float=None):
        self.columnas_int = columnas_int or []
        self.columnas_float = columnas_float or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.columnas_int:
            if c in X.columns:
                X[c] = X[c].astype('int32', errors='ignore')
        for c in self.columnas_float:
            if c in X.columns:
                X[c] = X[c].astype('float64', errors='ignore')
        return X
class CodificarCategoricas(BaseEstimator, TransformerMixin):
    """Aplica OneHotEncoding a variables categóricas."""
    def __init__(self):
        self.encoder = None
        self.columnas_categoricas = None

    def fit(self, X, y=None):
        self.columnas_categoricas = X.select_dtypes(include=['object']).columns.tolist()
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        if len(self.columnas_categoricas) > 0:
            self.encoder.fit(X[self.columnas_categoricas])
        return self

    def transform(self, X):
        X = X.copy()
        if len(self.columnas_categoricas) == 0:
            return X
        encoded = self.encoder.transform(X[self.columnas_categoricas])
        encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(self.columnas_categoricas))
        X = X.drop(columns=self.columnas_categoricas)
        X = pd.concat([X.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
        return X

class EscalarNumericos(BaseEstimator, TransformerMixin):
    """Estandariza las variables numéricas con StandardScaler."""
    def __init__(self):
        self.scaler = StandardScaler()
        self.columnas_numericas = None

    def fit(self, X, y=None):
        self.columnas_numericas = X.select_dtypes(include=[np.number]).columns.tolist()
        self.scaler.fit(X[self.columnas_numericas])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.columnas_numericas] = self.scaler.transform(X[self.columnas_numericas])
        return X

def load_and_split_data(path: str):
    df = pd.read_csv(path)
    df['p_codmes'] = df['p_codmes'].astype(str).str.replace('.0', '', regex=False)

    # Definir periodos
    train_periods = ['201909', '201910', '201911']
    test_period = '201912'

    train = df[df['p_codmes'].isin(train_periods)].copy()
    test = df[df['p_codmes'] == test_period].copy()

    os.makedirs('data/processed', exist_ok=True)
    train.to_csv('data/processed/train_raw.csv', index=False)
    test.to_csv('data/processed/test_raw.csv', index=False)
    print("Versión RAW guardada.")

    # Columnas según tu definición
    columnas_drop = [
        'prm_saltotrdpj03m','prm_saltotrdpj12m','dsv_diasatrrdpj12m',
        'max_pctsalimpago12m','prm_diasatrrdpj03m','key_value'
    ]
    columnas_cat = ['ubigeo_buro','grp_camptot06m','grp_campecs06m','region','grp_riesgociiu']

    columnas_int = [
        'target','ctd_prod_rccsf_m01','ctd_actrccsf_6m','ctd_flgact_rccsf_m01',
        'flg_lintcrripsaga','flg_svtcrsrcf','flg_sdtcrripsaga',
        'flg_sttcrsrcf','flg_svltcrsrcf','max_camptottlv06m','min_camptottlv06m',
        'frc_camptottlv06m','rec_camptottlv06m','ctd_camptottlv06m','max_campecstlv06m','min_campecstlv06m',
        'frc_campecstlv06m','rec_campecstlv06m','ctd_campecstlv06m',
        'max_camptot06m','min_camptot06m','frc_camptot06m','rec_camptot06m','ctd_camptot06m',
        'max_campecs06m','min_campecs06m','frc_campecs06m','rec_campecs06m','ctd_campecs06m'
    ]

    columnas_float = [
        'monto','prom_salvig_entprinc_pp_rccsf_03m','max_usotcrrstsf06m','max_usotcrrstsf03m',
        'prm_lintcribksf06m','lin_tcribksf03m','lin_tcribksf06m','cre_salvig_pp_rccsf_m02',
        'prm_usotcrrstsf06m','prm_lintcribksf03m','prm_usotcrrstsf03m','ratuso_tcrrstsf_m13',
        'promctdprodrccsf3m','ratpct_saldopprcc_m13','promctdprodrccsf6m','sld_ep1ppeallsfm01',
        'prm_sldvigrstsf12m','sldtot_tcrsrcf','sldvig_tcrsrcf','prm_camptottlv06m',
        'prm_campecstlv06m','prm_camptot06m','prm_campecs06m'
    ]

    # Crear el pipeline
    pipeline = Pipeline([
        ('eliminar_columnas', EliminarColumnas(columnas_drop)),
        ('imputar_edad', ImputarEdad()),
        ('reemplazar_categoricos', ReemplazarCategoricos(columnas_cat)),
        ('reemplazar_numericos', ReemplazarNumericos(excluir=['edad'] + columnas_cat)),
        ('convertir_tipos', ConvertirTipos(columnas_int, columnas_float)),
        ('codificar_categoricas', CodificarCategoricas()),
        ('escalar_numericos', EscalarNumericos())
    ])

    # Aplicar pipeline
    train_clean = pipeline.fit_transform(train)
    test_clean = pipeline.transform(test)

    train_clean.to_csv('data/processed/train_clean.csv', index=False)
    test_clean.to_csv('data/processed/test_clean.csv', index=False)
    print("Versión CLEAN guardada.")

if __name__ == "__main__":
    load_and_split_data('data/raw/Data_CU_venta.csv')
