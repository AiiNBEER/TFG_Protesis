import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from tensorflow.keras.regularizers import l2

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import joblib

def fast_ica(X, n_components, max_iter=200, tol=1e-4, func='tanh'):
    # Convert X to float64
    X = X.astype(np.float64)

    # Center the data
    X -= X.mean(axis=0)
    
    # Whitening the data
    cov = np.cov(X, rowvar=False)
    # Add a small regularization term to avoid issues with zero eigenvalues
    cov += np.eye(cov.shape[0]) * 1e-5
    eigvals, eigvecs = np.linalg.eigh(cov)
    X_white = np.dot(X, eigvecs / np.sqrt(eigvals + 1e-5))

    # Define nonlinear functions and their derivatives
    def g_tanh(x):
        return np.tanh(x)

    def g_prime_tanh(x):
        return 1 - np.tanh(x) ** 2

    def g_logcosh(x):
        return np.tanh(x)

    def g_prime_logcosh(x):
        return 1 - np.tanh(x) ** 2

    def g_exp(x):
        return x * np.exp(-x ** 2 / 2)

    def g_prime_exp(x):
        return (1 - x ** 2) * np.exp(-x ** 2 / 2)

    if func == 'tanh':
        g = g_tanh
        g_prime = g_prime_tanh
    elif func == 'logcosh':
        g = g_logcosh
        g_prime = g_prime_logcosh
    elif func == 'exp':
        g = g_exp
        g_prime = g_prime_exp
    else:
        raise ValueError("Unknown function type. Supported types: 'tanh', 'logcosh', 'exp'.")

    W = np.zeros((n_components, X_white.shape[1]), dtype=float)

    for i in range(n_components):
        w = np.random.rand(X_white.shape[1])
        
        for _ in range(max_iter):
            w_new = np.dot(X_white.T, g(np.dot(X_white, w))) / X_white.shape[0] - np.mean(g_prime(np.dot(X_white, w))) * w
            w_new /= np.linalg.norm(w_new)
            
            if np.abs(np.abs(np.dot(w_new, w)) - 1) < tol:
                break
            w = w_new
        
        W[i, :] = w

        # Decorrelate components
        for j in range(i):
            W[i, :] -= np.dot(W[i, :], W[j, :]) * W[j, :]

        W[i, :] /= np.linalg.norm(W[i, :])

    S = np.dot(X, W.T)
    return S, W

def filter_action_periods(df, start_sec=1, end_sec=4):
    # Convertir la columna 'Timestamp' a tipo datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Crear una columna 'Relative_Time' que contenga el tiempo relativo desde el inicio de cada acción
    df['Relative_Time'] = df.groupby('Command')['Timestamp'].transform(lambda x: (x - x.min()).dt.total_seconds())
    
    # Filtrar los datos para mantener solo aquellos en los segundos 1 a 4 de cada movimiento
    df['DROP'] = 0
    df['DROP'].loc[(df['Relative_Time'] >= start_sec) & (df['Relative_Time'] <= end_sec)] = 1
    
    return df

def create_windows(data, window_size):
    windows = []
    for end in range(window_size, len(data) + 1):
        start = end - window_size
        windows.append(data[start:end])
    return np.array(windows)

def estandarizar_datos(X):
    # Calcular la media y la desviación estándar
    media = np.mean(X)
    desviacion_estandar = np.std(X)
    
    # Estandarizar los datos
    X_estandarizado = (X - media) / desviacion_estandar
    
    return X_estandarizado

def db_coefficients(N):
    """
    Compute the Daubechies wavelet coefficients (h0, h1) for a given N.
    
    Parameters:
    - N: int, the order of the Daubechies wavelet (e.g., 1, 2, 3).
    
    Returns:
    - h0, h1: numpy arrays, the low-pass (scaling) and high-pass (wavelet) coefficients.
    """
    if N == 1:
        a = (1 + np.sqrt(3)) / (4 * np.sqrt(2))
        h0 = np.array([1, 1]) * a
        h1 = np.array([-1, 1]) * a
    else:
        # Calculate h0 using the standard formula
        a = (1 + np.sqrt(3)) / (4 * np.sqrt(2))
        h0 = np.zeros(2 * N)
        h0[:N] = (1 + np.sqrt(3)) / (4 * np.sqrt(2))
        h0[N:2 * N] = (1 - np.sqrt(3)) / (4 * np.sqrt(2))
        h1 = np.zeros_like(h0)
        for k in range(2 * N):
            h1[k] = ((-1) ** (N - k - 1)) * h0[N - 1 - k]
    
    return h0, h1

def dwt(signal, wavelet='db1'):
    """
    Perform one level of Discrete Wavelet Transform (DWT) on the given signal.
    
    Parameters:
    - signal: 1D numpy array, the input signal to be decomposed.
    - wavelet: string, the type of wavelet to use (default is 'db1' - Daubechies 1).
    
    Returns:
    - cA: Approximation coefficients.
    - cD: Detail coefficients.
    """
    # Determine N from wavelet type 'dbN'
    N = int(wavelet[2:])
    
    # Calculate Daubechies coefficients
    h0, h1 = db_coefficients(N)
    
    # Pad the signal to ensure the length is divisible by 2
    pad_len = len(signal) % 2
    padded_signal = np.pad(signal, (0, pad_len), mode='constant')
    
    # Perform convolution and downsampling
    cA = np.convolve(padded_signal, h0, mode='valid')[::2]
    cD = np.convolve(padded_signal, h1, mode='valid')[::2]
    
    return cA, cD, h0, h1

def mdwt(signal, wavelet='db1', max_level=1):
    """
    Perform multi-level Discrete Wavelet Transform (mDWT) on the given signal.
    
    Parameters:
    - signal: 1D numpy array, the input signal to be decomposed.
    - wavelet: string, the type of wavelet to use (default is 'db1' - Daubechies 1).
    - level: int, the number of decomposition levels (default is 1).
    
    Returns:
    - coeffs: list of tuples, each tuple contains approximation and detail coefficients at each level.
    """
    coeffs = []
    approx = signal.astype(float)  # Start with the original signal
    
    for _ in range(max_level):
        cA, cD, h0, h1 = dwt(approx, wavelet=wavelet)

        coeffs.extend((cA, cD))

        approx = cA  # Update approximation coefficients for the next level
    
    return coeffs

def calculate_wmav(data_array, start_weight = 0, end_weight = 1):

    weights_array = np.linspace(start_weight, end_weight, len(data_array)+1)[1:] # Since the first can be a 0, we are dropping that
    
    # Calculate the weighted moving average
    wmav = np.sum(data_array * weights_array) / np.sum(weights_array)
    
    return wmav

def load_and_preprocess_data_v1(csv_file, window_size=50):
    # Cargar el archivo CSV con los datos filtrados (ajusta la ruta según sea necesario)
    df = pd.read_csv(csv_file, encoding='latin1')
    df = filter_action_periods(df)
    print(df)

    # Mezclar los datos
    #df = shuffle(df, random_state=42)

    # Preparar las características (X) y las etiquetas (y)
    X = df[['Signal1', 'Signal2', 'Signal3']].values  # Ajusta según tus columnas
    y = df['Command'].values  # La columna con las etiquetas de clase
    drop = df['DROP'].values

    # Create windows of data
    X_windows = create_windows(X, window_size)

    X_windows = [window for window, indicator in zip(X_windows, drop) if indicator == 1]
    y = [y for y, indicator in zip(y, drop) if indicator == 1]

    # Apply Fast ICA to each window
    X_ica_windows = []
    for window in X_windows:
        S, W = fast_ica(window, n_components=3)  # Adjust n_components as needed
        X_ica_windows.append(S)

    X_ica_windows = np.array(X_ica_windows)
    y = np.array(y)

    # Perform Windows mDWT and extra derivations
    ValuesList = []

    for window in X_windows:

        window_features = []
        higher_coefficients_array = []

        for x in range(len(window[0])):
            ind_coefficients = []
            max_level = int('db1'[2:])

            ind_coefficients = mdwt(window[:,x], wavelet='db1', max_level=max_level) # cA, cD

            higher_coefficients_array.append(ind_coefficients)

        higher_coefficients_array = np.array(higher_coefficients_array)

        #### COMPUTE STD VARIANCE AND WMAV
        # Initialize arrays to hold values
        std_matrix = np.zeros((higher_coefficients_array.shape[0], higher_coefficients_array.shape[1]))
        var_matrix = np.zeros((higher_coefficients_array.shape[0], higher_coefficients_array.shape[1]))
        wmav_matrix = np.zeros((higher_coefficients_array.shape[0], higher_coefficients_array.shape[1]))

        # Compute std for each cA and cD separately
        for a in range(higher_coefficients_array.shape[0]):
            for b in range(higher_coefficients_array.shape[1]):
                std_matrix[a, b] = np.std(higher_coefficients_array[a, b])
                var_matrix[a, b] = std_matrix[a, b] ** 2  # Calculate variance using square of std
                wmav_matrix[a, b] = calculate_wmav(higher_coefficients_array[a, b])

        # Divide each element in the array by the sum of its corresponding column
        wmav_matrix = wmav_matrix / (np.sum(wmav_matrix, axis=0) + 1e-9)  # Small constant to avoid division by zero

        for a in range(len(std_matrix)): # Num EMG Sensors
            window_features.append(std_matrix[a, 0])
            window_features.append(std_matrix[a, 1])

            window_features.append(var_matrix[a, 0])
            window_features.append(var_matrix[a, 1])

            window_features.append(wmav_matrix[a, 0])
            window_features.append(wmav_matrix[a, 1])

        ValuesList.append(window_features)

    ValuesList = np.array(ValuesList)
    print(ValuesList, len(ValuesList))
    print(y, len(y))
    input("INPUT")

    '''
    # Flatten the windows and their corresponding labels for model training
    X_ica_flat = X_ica_windows.reshape(X_ica_windows.shape[0], -1)
    y_flat = y
    '''

    # Shuffle the data
    ValuesList, y = shuffle(ValuesList, y, random_state=42)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(ValuesList, y, test_size=0.2, random_state=42)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Replace NaN and infinite values in X_train
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=-1.0)

    # Replace NaN and infinite values in X_train
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1.0, neginf=-1.0)
    y_test = np.nan_to_num(y_test, nan=0.0, posinf=1.0, neginf=-1.0)

    # Escalar las características
    X_train = estandarizar_datos(X_train)
    X_test = estandarizar_datos(X_test)
    
    return X_train, X_test, y_train, y_test

def load_and_preprocess_data_v2(csv_file, window_size=50):
    # Cargar el archivo CSV con los datos filtrados (ajusta la ruta según sea necesario)
    df = pd.read_csv(csv_file, encoding='latin1')
    df = filter_action_periods(df)

    # Mezclar los datos
    #df = shuffle(df, random_state=42)

    # Preparar las características (X) y las etiquetas (y)
    X = df[['Signal1', 'Signal2', 'Signal3']].values  # Ajusta según tus columnas
    y = df['Command'].values  # La columna con las etiquetas de clase
    drop = df['DROP'].values

    # Create windows of data
    X_windows = create_windows(X, window_size)

    X_windows = [window for window, indicator in zip(X_windows, drop) if indicator == 1]
    y = [y for y, indicator in zip(y, drop) if indicator == 1]

    # Apply Fast ICA to each window
    X_ica_windows = []
    for window in X_windows:
        S, W = fast_ica(window, n_components=3)  # Adjust n_components as needed
        X_ica_windows.append(S)

    X_ica_windows = np.array(X_ica_windows)
    y = np.array(y)

    # Perform Windows mDWT and extra derivations
    ValuesList = []

    for window in X_windows:

        window_features = []
        higher_coefficients_array = []

        for x in range(len(window[0])):
            ind_coefficients = []
            max_level = int('db1'[2:])

            ind_coefficients = mdwt(window[:,x], wavelet='db1', max_level=max_level) # cA, cD

            higher_coefficients_array.append(ind_coefficients)

        higher_coefficients_array = np.array(higher_coefficients_array)

        for x in range(len(higher_coefficients_array)):
            for i in range(len(higher_coefficients_array[x])):
                # Root Mean Square (RMS)
                window_features.append(np.sqrt(np.mean(higher_coefficients_array[x][i]**2)))

                # Mean Absolute Value (MAV)
                window_features.append(np.mean(np.abs(higher_coefficients_array[x][i])))

                # Waveform Length (WL)
                window_features.append(np.sum(np.abs(np.diff(higher_coefficients_array[x][i]))))
                
                # Variance (VAR)
                window_features.append(np.var(higher_coefficients_array[x][i]))

        ValuesList.append(window_features)

    ValuesList = np.array(ValuesList)

    '''
    # Flatten the windows and their corresponding labels for model training
    X_ica_flat = X_ica_windows.reshape(X_ica_windows.shape[0], -1)
    y_flat = y
    '''

    # Shuffle the data
    ValuesList, y = shuffle(ValuesList, y, random_state=42)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(ValuesList, y, test_size=0.2, random_state=42)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Replace NaN and infinite values in X_train
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=-1.0)

    # Replace NaN and infinite values in X_train
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1.0, neginf=-1.0)
    y_test = np.nan_to_num(y_test, nan=0.0, posinf=1.0, neginf=-1.0)

    # Escalar las características
    X_train = estandarizar_datos(X_train)
    X_test = estandarizar_datos(X_test)
    
    return X_train, X_test, y_train, y_test

def build_and_train_model(X_train, y_train):
    # Definir el modelo
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(len(np.unique(y_train)), activation='softmax')  # Clasificación multiclase
    ])

    # Compilar el modelo
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # RMSprop
    #model.compile(optimizer=RMSprop(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # SGD con momentum
    #model.compile(optimizer=SGD(learning_rate=1e-3, momentum=0.9), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # AdamW
    #model.compile(optimizer=Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo
    model.fit(X_train, y_train, epochs=20, validation_split=0.2)

    knn = KNeighborsClassifier(n_neighbors=3)

    # Crear el modelo SVM con el kernel lineal (puedes probar otros kernels como 'rbf', 'poly', 'sigmoid')
    svm_model = SVC(kernel='linear')

    # Entrenar el modelo
    svm_model.fit(X_train, y_train)

    # Entrenar el modelo
    knn.fit(X_train, y_train)

    # Entrenar y evaluar el modelo Random Forest
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    return model, knn, svm_model, rf_model

def convert_to_tflite(model, tflite_model_path='model.tflite', test_number = None):

    if test_number != None:
        tflite_model_path = tflite_model_path.split('.')[0] + str(test_number) + '.' + tflite_model_path.split('.')[-1]

    # Convertir el modelo a TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Guardar el modelo TensorFlow Lite
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Modelo TensorFlow Lite guardado como '{tflite_model_path}'.")

def main():

    test_number = input("Introduce numero de test: ")
    csv_file = f'data_test{test_number}.csv'  # Nombre del archivo CSV donde se guardarán los datos

    # Cargar y preprocesar los datos
    X_train, X_test, y_train, y_test = load_and_preprocess_data_v2(csv_file)

    # Construir y entrenar el modelo
    model, knn, svm_model, rf_model = build_and_train_model(X_train, y_train)

    # Guardar el modelo
    joblib.dump(knn, f'knn_model{test_number}.pkl')
    joblib.dump(svm_model, f'svm_model{test_number}.pkl')
    joblib.dump(rf_model, f'rf_model{test_number}.pkl')

    # Evaluar el modelo en los datos de prueba
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Loss: {loss}, Accuracy: {accuracy}')

    print(len(X_test[0]), y_test)

    # Hacer predicciones en el conjunto de prueba
    y_pred = knn.predict(X_test)

    # Evaluar el modelo
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy:.2f}')
    print('Classification Report:')
    print(report)

    # Calcular los centroides para cada clase
    centroids = []
    for class_label in np.unique(y_train):
        class_points = X_train[y_train == class_label]
        centroid = class_points.mean(axis=0)
        centroids.append((centroid, class_label))

    # Guardar los centroides en un archivo
    centroids = np.array(centroids, dtype=object)
    np.save(f'centroids{test_number}.npy', centroids)

    # Hacer predicciones en el conjunto de prueba
    y_pred = svm_model.predict(X_test)

    # Evaluar el modelo
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy:.2f}')
    print('Classification Report:')
    print(report)

    # Hacer predicciones
    y_pred = rf_model.predict(X_test)

    # Evaluar el modelo
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy:.2f}')
    print('Classification Report:')
    print(report)

    # Convertir el modelo a TensorFlow Lite
    convert_to_tflite(model, test_number = test_number)

if __name__ == "__main__":
    main()