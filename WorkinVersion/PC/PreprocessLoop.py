import os
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
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

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues, normalized=False):
    if normalized:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalized else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main():
    # Identificar todos los archivos data_test{}.csv disponibles
    csv_files = [f for f in os.listdir() if f.startswith('data_test') and f.endswith('.csv')]
    
    # Inicializar variables para almacenar resultados
    all_y_test = []
    all_y_pred_ann = []
    all_y_pred_knn = []
    all_y_pred_svm = []
    all_y_pred_rf = []
    
    # Loop para procesar cada archivo CSV
    for csv_file in csv_files:
        print(f"Procesando {csv_file}...")
        
        # Cargar y preprocesar los datos
        X_train, X_test, y_train, y_test = load_and_preprocess_data_v2(csv_file)
        
        # Construir y entrenar el modelo
        model, knn, svm_model, rf_model = build_and_train_model(X_train, y_train)
        
        # Evaluar el modelo en los datos de prueba
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f'Modelo ANN - Loss: {loss}, Accuracy: {accuracy}')
        
        # Realizar predicciones
        y_pred_ann = np.argmax(model.predict(X_test), axis=-1)
        y_pred_knn = knn.predict(X_test)
        y_pred_svm = svm_model.predict(X_test)
        y_pred_rf = rf_model.predict(X_test)
        
        # Almacenar resultados
        all_y_test.extend(y_test)
        all_y_pred_ann.extend(y_pred_ann)
        all_y_pred_knn.extend(y_pred_knn)
        all_y_pred_svm.extend(y_pred_svm)
        all_y_pred_rf.extend(y_pred_rf)
    
    # Calcular matrices de confusión
    cm_ann = confusion_matrix(all_y_test, all_y_pred_ann)
    cm_knn = confusion_matrix(all_y_test, all_y_pred_knn)
    cm_svm = confusion_matrix(all_y_test, all_y_pred_svm)
    cm_rf = confusion_matrix(all_y_test, all_y_pred_rf)
    
    # Graficar matrices de confusión con colores
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    plt.subplot(221)
    plot_confusion_matrix(cm_ann, classes=np.unique(all_y_test), title='Confusion Matrix - ANN', normalized=True, cmap=plt.cm.Blues)
    
    plt.subplot(222)
    plot_confusion_matrix(cm_knn, classes=np.unique(all_y_test), title='Confusion Matrix - KNN', normalized=True, cmap=plt.cm.Blues)
    
    plt.subplot(223)
    plot_confusion_matrix(cm_svm, classes=np.unique(all_y_test), title='Confusion Matrix - SVM', normalized=True, cmap=plt.cm.Blues)
    
    plt.subplot(224)
    plot_confusion_matrix(cm_rf, classes=np.unique(all_y_test), title='Confusion Matrix - Random Forest', normalized=True, cmap=plt.cm.Blues)
    
    plt.tight_layout()
    plt.show()
    
    # Graficar matrices de confusión con números
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    plt.subplot(221)
    plot_confusion_matrix(cm_ann, classes=np.unique(all_y_test), title='Confusion Matrix - ANN', normalized=False, cmap=plt.cm.Blues)
    
    plt.subplot(222)
    plot_confusion_matrix(cm_knn, classes=np.unique(all_y_test), title='Confusion Matrix - KNN', normalized=False, cmap=plt.cm.Blues)
    
    plt.subplot(223)
    plot_confusion_matrix(cm_svm, classes=np.unique(all_y_test), title='Confusion Matrix - SVM', normalized=False, cmap=plt.cm.Blues)
    
    plt.subplot(224)
    plot_confusion_matrix(cm_rf, classes=np.unique(all_y_test), title='Confusion Matrix - Random Forest', normalized=False, cmap=plt.cm.Blues)
    
    plt.tight_layout()
    plt.show()
    
    # Calcular y mostrar la precisión de cada modelo
    accuracy_ann = accuracy_score(all_y_test, all_y_pred_ann)
    accuracy_knn = accuracy_score(all_y_test, all_y_pred_knn)
    accuracy_svm = accuracy_score(all_y_test, all_y_pred_svm)
    accuracy_rf = accuracy_score(all_y_test, all_y_pred_rf)
    
    print(f"Accuracy ANN: {accuracy_ann:.2f}")
    print(f"Accuracy KNN: {accuracy_knn:.2f}")
    print(f"Accuracy SVM: {accuracy_svm:.2f}")
    print(f"Accuracy RF: {accuracy_rf:.2f}")
    
    # Calcular y mostrar el informe de clasificación para cada modelo
    report_ann = classification_report(all_y_test, all_y_pred_ann)
    report_knn = classification_report(all_y_test, all_y_pred_knn)
    report_svm = classification_report(all_y_test, all_y_pred_svm)
    report_rf = classification_report(all_y_test, all_y_pred_rf)
    
    print(f"Classification Report ANN:\n{report_ann}")
    print(f"Classification Report KNN:\n{report_knn}")
    print(f"Classification Report SVM:\n{report_svm}")
    print(f"Classification Report RF:\n{report_rf}")

if __name__ == "__main__":
    main()