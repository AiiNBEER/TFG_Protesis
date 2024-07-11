import serial
import csv
import time
import random
import threading

# Configura el puerto serie y la velocidad (debe coincidir con la configuración del ESP32)
port = input("Introduce Número del puerto COM del Serial: ")  # Cambia 'COM3' por el puerto que esté utilizando tu ESP32
port = f'COM{port}'
baud_rate = 250000
test_number = input("Introduce numero de test: ")
csv_file = f'data_test{test_number}.csv'  # Nombre del archivo CSV donde se guardarán los datos

# Lista de acciones a realizar
actions = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
random.shuffle(actions)

# Diccionario para mapear los números a las acciones
action_map = {
    0: "Relax",
    1: "Indice",
    2: "Palma",
    3: "Puño",
    4: "Pinza",
    5: "Victoria"
}

# Variable global para almacenar el comando actual
current_command = 0
current_action_name = "Relax"
action_index = 0
relax_period = True

def show_command():
    global current_command, current_action_name, action_index, relax_period
    while action_index < len(actions):
        if relax_period:
            current_command = 0
            current_action_name = action_map[current_command]
            print(f"0 Relax durante 5 segundos. Próximo movimiento: {action_map[actions[action_index]]}")
            for i in range(5, 0, -1):
                print(f"Relax: {i} segundos restantes", end='\r')
                time.sleep(1)
            relax_period = False
        else:
            current_command = actions[action_index]
            current_action_name = action_map[current_command]
            print(f"Acción {current_action_name} durante 5 segundos.")
            for i in range(5, 0, -1):
                print(f"{current_action_name}: {i} segundos restantes", end='\r')
                time.sleep(1)
            action_index += 1
            relax_period = True
    print("Todas las acciones completadas.")

def read_from_serial():
    global current_command, current_action_name
    try:
        with serial.Serial(port, baud_rate, timeout=1) as ser, open(csv_file, mode='w', newline='') as file:
            #print(f'Conectado a {port} a {baud_rate} baudios.')
            csv_writer = csv.writer(file)
            csv_writer.writerow(['Timestamp', 'Signal1', 'Signal2', 'Signal3', 'Command', 'Action'])  # Escribe la cabecera del CSV

            while action_index < len(actions):
                if ser.in_waiting > 0:
                    data = ser.readline().decode('utf-8').strip()
                    #print(f'Recibido: {data}')
                    values = data.split(',')
                    if len(values) == 3:
                        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                        csv_writer.writerow([timestamp] + values + [current_command, current_action_name])
                        file.flush()  # Asegura que los datos se escriben inmediatamente
                    else:
                        print("Datos recibidos en formato incorrecto.")
    except serial.SerialException as e:
        print(f'Error al conectar con el puerto serie: {e}')
    except PermissionError as e:
        print(f'Permiso denegado para acceder al puerto serie: {e}')

if __name__ == "__main__":
    port = input("Introduce el número del puerto COM del Serial: ")
    port = f'COM{port}'
    
    # Iniciar un hilo para mostrar comandos
    command_thread = threading.Thread(target=show_command)
    command_thread.daemon = True
    command_thread.start()
    
    # Iniciar la lectura del puerto serie
    read_from_serial()