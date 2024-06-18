import os
import shutil

def main():

    # Especifica la ruta de la carpeta de origen y la carpeta de destino
    ruta_origen = '../Data/NinaPro/DB1'
    

    for i in range(27):
        nombre_archivo = 'S' + str(i+1) + '_'  # Parte del nombre del archivo que deseas mover // Importante incluir _ Si movería el S11 en el S1 por ejemplo

        ruta_destino = '../Data/NinaPro/DB1/DB1_s' + str(i+1)

        # Crear la carpeta de destino si no existe
        if not os.path.exists(ruta_destino):
            os.makedirs(ruta_destino)

        # Buscar y mover archivos
        for archivo in os.listdir(ruta_origen):
            if nombre_archivo in archivo:
                ruta_archivo_origen = os.path.join(ruta_origen, archivo)
                ruta_archivo_destino = os.path.join(ruta_destino, archivo)
                shutil.move(ruta_archivo_origen, ruta_archivo_destino)
                print(f'Movido: {archivo}')

if __name__ == '__main__':
    main()