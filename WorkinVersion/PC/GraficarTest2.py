import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Lee el archivo CSV con codificación latin1
csv_file = 'data_test1.csv'  # Cambia esto por el nombre de tu archivo CSV
data = pd.read_csv(csv_file, encoding='latin1')

# Convertir la columna 'Timestamp' a tipo datetime
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Resetear el índice para usarlo en el gráfico
data = data.reset_index()

# Mapea los comandos a colores
color_map = {
    0: 'blue',     # Relax
    1: 'green',    # Indice
    2: 'red',      # Palma
    3: 'purple',   # Puño
    4: 'orange',   # Pinza
    5: 'cyan'      # Victoria
}

action_map = {
    0: "Relax",
    1: "Indice",
    2: "Palma",
    3: "Puño",
    4: "Pinza",
    5: "Victoria"
}

# Crear la figura y los ejes
fig, ax = plt.subplots(figsize=(15, 8))

# Graficar cada señal con el color correspondiente usando scatter para mantener todos los puntos visibles
for command in color_map.keys():
    mask = data['Command'] == command
    ax.scatter(data.index[mask], data['Signal1'][mask], color=color_map[command], s=5, label=f'Signal1 - {command}')
    ax.scatter(data.index[mask], data['Signal2'][mask], color=color_map[command], s=5, label=f'Signal2 - {command}')
    ax.scatter(data.index[mask], data['Signal3'][mask], color=color_map[command], s=5, label=f'Signal3 - {command}')

# Crear la leyenda personalizada
legend_elements = [mlines.Line2D([0], [0], color=color, lw=2, label=action_map[action]) 
                   for action, color in color_map.items()]
ax.legend(handles=legend_elements, title="Gestos", loc='upper right')

# Ajustes de la gráfica
ax.set_xlabel('Sample Index')
ax.set_ylabel('Signal Value')
ax.set_title('Señales EMG con cambios de color según el gesto (Ensayo completo)')
plt.tight_layout()

# Mostrar la gráfica
plt.show()
