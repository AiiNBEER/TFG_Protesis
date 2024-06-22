import os
import shutil

import pandas as pd
import numpy as np

import scipy

import time

nina_folder = '../Data/NinaPro'
data_folder = '../Data'

db1_hz = 100
db2_3_hz = 2000
window_size = 0.1 # seconds

def archivos_por_patron(ruta_base, patron = '.mat'):

    lista_rutas = []
 
    # Obtener todos los elementos (archivos y carpetas) de la ruta_base
    elementos = os.listdir(ruta_base)

    for elemento in elementos:

        if elemento == 'ZIPS':
            continue

        ruta_elemento = os.path.join(ruta_base, elemento)

        if os.path.isdir(ruta_elemento):
            # Si es una carpeta, llamar recursivamente a la función
            lista_rutas.extend(archivos_por_patron(ruta_elemento, patron))

        elif os.path.isfile(ruta_elemento):
            # Si es un archivo, comprobar el patrón en su nombre
            if patron in elemento:
                lista_rutas.append(ruta_elemento)

    return lista_rutas

def procesar_archivo(ruta):

    if 'DB1' in ruta:
        hz = db1_hz

    if 'DB2' in ruta or 'DB3' in ruta:
        hz = db2_3_hz

    data = scipy.io.loadmat(ruta)

    emg          = data['emg']
    stimulus     = data['stimulus']
    subject      = data['subject']
    exercise     = data['exercise']
    repetition   = data['repetition']
    restimulus   = data['restimulus']
    rerepetition = data['rerepetition']

    window_length  = int(hz * window_size)
    num_iterations = len(emg) - window_length + 1

    emg_windows          = np.array([emg[i:(i + window_length)] for i in range(num_iterations)])
    restimulus_windows   = restimulus[window_length-1:]
    rerepetition_windows = rerepetition[window_length-1:]
    exercise_windows     = np.repeat(exercise, num_iterations, axis=0)
    subject_windows      = np.repeat(subject, num_iterations, axis=0)

    # Obtener dimensiones de las ventanas
    num_ventanas, filas, columnas = emg_windows.shape

    # Crear un nuevo array para almacenar las ventanas transformadas
    ventanas_transformadas = np.empty((num_ventanas, columnas, filas))

    # Transformar filas en columnas y columnas en filas para cada ventana
    for i in range(num_ventanas):
        ventanas_transformadas[i] = emg_windows[i].transpose()

def main():

    lista_rutas = archivos_por_patron(nina_folder, patron = '.mat')

    for ruta in lista_rutas:
        print(ruta)
        procesar_archivo(ruta)

if __name__ == '__main__':

    inicio = time.time()
    main()
    final = time.time()
    tiempo_transcurrido = final - inicio
    print(tiempo_transcurrido/60)