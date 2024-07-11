import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
import pywt

# Lee el archivo CSV con codificación latin1
csv_file = 'data_test1.csv'  # Cambia esto por el nombre de tu archivo CSV
data = pd.read_csv(csv_file, encoding='latin1')

# Convertir la columna 'Timestamp' a tipo datetime
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Filtrar los datos para el intervalo entre el segundo 5 y 10
start_time = data['Timestamp'].min() + pd.Timedelta(seconds=5)
end_time = start_time + pd.Timedelta(seconds=5)
filtered_data = data[(data['Timestamp'] >= start_time) & (data['Timestamp'] <= end_time)]

# Seleccionar una ventana de 100 ms (50 muestras) del intervalo filtrado
window_data = filtered_data.iloc[:50]

# Extraer las señales de los sensores
signals = window_data[['Signal1', 'Signal2', 'Signal3']].to_numpy()

# Aplicar FastICA
ica = FastICA(n_components=3, max_iter=1000, tol=0.001)
sources = ica.fit_transform(signals)

# Aplicar mDWT de Haar (db1) a las señales de FastICA, nivel 1
coeffs = [pywt.wavedec(source, 'db1', level=1) for source in sources.T]

# Crear una figura para graficar los coeficientes de mDWT
fig, axs = plt.subplots(3, 2, figsize=(15, 10))

# Graficar los coeficientes de mDWT para cada fuente
for i in range(3):
    axs[i, 0].plot(coeffs[i][0])
    axs[i, 0].set_title(f'Source {i+1} - Approximation Coeff')
    axs[i, 0].set_xlabel('Sample Index')
    axs[i, 0].set_ylabel('Amplitude')
    
    axs[i, 1].plot(coeffs[i][1])
    axs[i, 1].set_title(f'Source {i+1} - Detail Coeff')
    axs[i, 1].set_xlabel('Sample Index')
    axs[i, 1].set_ylabel('Amplitude')

# Ajustar la disposición de las gráficas
plt.tight_layout()

# Mostrar la gráfica de coeficientes de mDWT
plt.show()
