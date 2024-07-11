import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import glob

# Directorio y patrón de archivos
file_pattern = 'data_test*.csv'

# Lista de DataFrames para almacenar todos los datos
data_frames = []

# Leer y concatenar todos los archivos CSV que coinciden con el patrón
for file in glob.glob(file_pattern):
    data = pd.read_csv(file, encoding='latin1')
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data_frames.append(data)

# Concatenar todos los DataFrames
all_data = pd.concat(data_frames).reset_index(drop=True)

# Mapea los comandos a colores
color_map = {
    0: 'blue',     # Relax
    1: 'green',    # Indice
    2: 'red',      # Palma
    3: 'purple',   # Puño
    4: 'orange',   # Pinza
    5: 'cyan'      # Victoria
}

# Crear la figura y los ejes
fig, ax = plt.subplots(figsize=(15, 8))

# Graficar cada señal con el color correspondiente usando scatter para mantener todos los puntos visibles
for command in color_map.keys():
    mask = all_data['Command'] == command
    ax.scatter(all_data.index[mask], all_data['Signal1'][mask], color=color_map[command], s=5, label=f'Signal1 - {command}')
    ax.scatter(all_data.index[mask], all_data['Signal2'][mask], color=color_map[command], s=5, label=f'Signal2 - {command}')
    ax.scatter(all_data.index[mask], all_data['Signal3'][mask], color=color_map[command], s=5, label=f'Signal3 - {command}')

# Crear la leyenda personalizada
legend_elements = [mlines.Line2D([0], [0], color=color, lw=2, label=action) 
                   for action, color in color_map.items()]
ax.legend(handles=legend_elements, title="Comandos", loc='upper right')

# Ajustes de la gráfica
ax.set_xlabel('Sample Index')
ax.set_ylabel('Signal Value')
ax.set_title('Señales EMG con cambios de color según el comando (Ensayo completo)')
plt.tight_layout()

# Mostrar la gráfica
plt.show()
