import serial

import joblib
import numpy as np

# Configura el puerto serie y la velocidad (debe coincidir con la configuración del ESP32)
port = input("Introduce Número del puerto COM del Serial: ")  # Cambia 'COM3' por el puerto que esté utilizando tu ESP32
port = f'COM{port}'
input_number = input("Introduce Número del archivo modelo: ")
baud_rate = 250000
nombre_modelo = f'svm_model{input_number}.pkl'  # Nombre del archivo CSV donde se guardarán los datos

knn_model = joblib.load(nombre_modelo)

def read_from_serial():
    try:
        with serial.Serial(port, baud_rate, timeout=1) as ser:
            print(f'Conectado a {port} a {baud_rate} baudios.')

            while True:
                if ser.in_waiting > 0:
                    data = ser.readline().decode('utf-8').strip()
                    values = list(data.split(','))
                    
                    values = list(map(float, values[0].split()))
                
                    values = np.array([values])
                    #print(values)
                    
                    y_pred = knn_model.predict(np.array(values))
                    print(y_pred)
                    

                    #print(values)
                    
    except serial.SerialException as e:
        print(f'Error al conectar con el puerto serie: {e}')

if __name__ == "__main__":
    read_from_serial()