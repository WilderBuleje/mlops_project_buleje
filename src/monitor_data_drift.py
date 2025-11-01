import os
import pandas as pd
import numpy as np
np.float_ = np.float64
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def monitor_drift():
    # Rutas correctas
    train_path = 'data/processed/train_clean.csv'
    test_path = 'data/processed/test_clean.csv'
    output_path = 'reports/data_drift_report.html'

    # Crear carpeta de salida
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Cargar datasets
    print("Cargando datasets...")
    reference = pd.read_csv(train_path)
    current = pd.read_csv(test_path)

    # Crear y ejecutar el reporte
    print("Generando reporte de drift...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    # Guardar resultado
    report.save_html(output_path)
    print(f"✅ Reporte de data drift guardado en: {output_path}")

if __name__ == "__main__":
    monitor_drift()

