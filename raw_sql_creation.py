import os
import shutil

import pandas as pd
import numpy as np

import sqlite3
import scipy
import time

nina_folder = '../Data/NinaPro'

db1_hz = 100
db2_3_hz = 2000

exercise1_restimulus = [1,2,3,4,9,10,11,12]
exercise2_restimulus = [1,2,3,4,5,6,7,8]
exercise3_restimulus = [1,2,3,5,6,10,12,16,20]

fingers_dict = {
    'pulgar':{
        'ex1':{
            '1':[10,12],
            '-1':[9,11]
        },
        'ex2':{
            '1':[2,4,6,7],
            '-1':[1,3,5,8]
        },
        'ex3':{
            '1':[1,2,5,6,10,12,16,20],
            '-1':[3]
        }
    },
    'indice':{
        'ex1':{
            '1':[1],
            '-1':[2]
        },
        'ex2':{
            '1':[1,6],
            '-1':[2,3,4,5,7,8]
        },
        'ex3':{
            '1':[1,2,3,5,6,10,12,16,20],
            '-1':[]
        }
    },
    'corazon':{
        'ex1':{
            '1':[3],
            '-1':[4]
        },
        'ex2':{
            '1':[1,6,7],
            '-1':[2,3,4,5,8]
        },
        'ex3':{
            '1':[1,2,3,5,10,12,16,20],
            '-1':[]
        }
    }
}

def check_table_exists(conn, cursor, table_name):

    conn.commit()

    cursor.execute('''
    SELECT name FROM sqlite_master WHERE type='table' AND name=?
    ''', (table_name,))
    return cursor.fetchone() is not None

def create_raw_data_table(conn, cursor):

    # Create a table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS raw_data (
        id INTEGER PRIMARY KEY,
        db INTEGER NOT NULL,
        hz INTEGER NOT NULL,
        subject INTEGER NOT NULL,
        exercise INTEGER NOT NULL,
        restimulus INTEGER NOT NULL,
        rerepetition INTEGER NOT NULL,
        emg1 REAL NOT NULL,
        emg2 REAL NOT NULL,
        emg3 REAL NOT NULL,
        emg4 REAL NOT NULL,
        emg5 REAL NOT NULL,
        emg6 REAL NOT NULL,
        emg7 REAL NOT NULL,
        emg8 REAL NOT NULL,
        emg9 REAL NOT NULL,
        emg10 REAL NOT NULL,
        pulgar INTEGER NOT NULL,
        indice INTEGER NOT NULL,
        corazon INTEGER NOT NULL
    )
    ''')

    conn.commit()

def archivos_por_patron(ruta_base, patron = '.mat'):

    lista_rutas = []
 
    # Obtener todos los elementos (archivos y carpetas) de la ruta_base
    elementos = os.listdir(ruta_base)

    for elemento in elementos:

        if elemento in ['ZIPS', 'SQL']:
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

def insert_into_table(conn, cursor, data, chunk_size=500):

    num_keys = len(data.keys())

    # Prepare the query prefix for insertion
    insert_query_prefix = f"INSERT INTO raw_data ({', '.join(data.keys())}) VALUES "

    # Extract values from data_dict and format them into a list of tuples
    data = list(zip(*data.values()))

    # Insert data in chunks
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        
        # Prepare the query for this chunk
        insert_query = insert_query_prefix + ', '.join(['(' + ', '.join(['?']*num_keys) + ')'] * len(chunk))
        
        # Flatten chunk values
        flat_chunk = [item for sublist in chunk for item in sublist]

        # Execute the insertion query
        cursor.execute(insert_query, flat_chunk)

        # Commit changes to the database
        conn.commit()

def procesar_archivo(ruta):

    if 'DB1' in ruta:
        hz = db1_hz
        db = 1

    elif 'DB2' in ruta:
        hz = db2_3_hz
        db = 2

    elif 'DB3' in ruta:
        hz = db2_3_hz
        db = 3

    else:
        print("FAIL NO DB", ruta)
        input("INPUT")

    data = scipy.io.loadmat(ruta)

    insert_dict = {}

    length = len(data['emg'])

    insert_dict['restimulus']   = data['restimulus'].flatten()
    insert_dict['rerepetition'] = data['rerepetition'].flatten()

    insert_dict['exercise'] = np.repeat(data['exercise'], length, axis=0).flatten()
    insert_dict['subject']  = np.repeat(data['subject'], length, axis=0).flatten()
    insert_dict['db'] = np.full(length, db, dtype=np.float64)
    insert_dict['hz'] = np.full(length, hz, dtype=np.float64)

    # Separate each column into a separate emg array
    insert_dict.update({f"emg{i+1}": data['emg'][:, i] for i in range(data['emg'].shape[1]) if i < 10})

    # Find the minimum number of rows across all columns
    min_rows = min(len(col) for col in insert_dict.values())

    # Trim each column to the minimum length
    for key in insert_dict.keys():
        insert_dict[key] = insert_dict[key][:min_rows]

    # New Finger Columns
        # Delete Non Usable Columns
    filt = ((insert_dict['exercise'] == 1) & (np.isin(insert_dict['restimulus'], exercise1_restimulus))) |\
           ((insert_dict['exercise'] == 2) & (np.isin(insert_dict['restimulus'], exercise2_restimulus))) |\
           ((insert_dict['exercise'] == 3) & (np.isin(insert_dict['restimulus'], exercise3_restimulus)))

    # Update the dictionary by keeping only the rows that satisfy the filter condition
    insert_dict = {key: value[filt] for key, value in insert_dict.items()}

    # New columns based on finger direction
        # Initialize the new columns with default values (e.g., 0)
    pulgar_array  = np.zeros(insert_dict['db'].shape)
    indice_array  = np.zeros(insert_dict['db'].shape)
    corazon_array = np.zeros(insert_dict['db'].shape)

    # Change Values with Filters
    filt =  ((insert_dict['exercise'] == 1) & (np.isin(insert_dict['restimulus'], fingers_dict['pulgar']['ex1']['1']))) |\
            ((insert_dict['exercise'] == 2) & (np.isin(insert_dict['restimulus'], fingers_dict['pulgar']['ex2']['1']))) |\
            ((insert_dict['exercise'] == 3) & (np.isin(insert_dict['restimulus'], fingers_dict['pulgar']['ex3']['1'])))

    pulgar_array[filt] = 1

    # Change Values with Filters
    filt =  ((insert_dict['exercise'] == 1) & (np.isin(insert_dict['restimulus'], fingers_dict['pulgar']['ex1']['-1']))) |\
            ((insert_dict['exercise'] == 2) & (np.isin(insert_dict['restimulus'], fingers_dict['pulgar']['ex2']['-1']))) |\
            ((insert_dict['exercise'] == 3) & (np.isin(insert_dict['restimulus'], fingers_dict['pulgar']['ex3']['-1'])))

    pulgar_array[filt] = -1

    # Change Values with Filters
    filt =  ((insert_dict['exercise'] == 1) & (np.isin(insert_dict['restimulus'], fingers_dict['indice']['ex1']['1']))) |\
            ((insert_dict['exercise'] == 2) & (np.isin(insert_dict['restimulus'], fingers_dict['indice']['ex2']['1']))) |\
            ((insert_dict['exercise'] == 3) & (np.isin(insert_dict['restimulus'], fingers_dict['indice']['ex3']['1'])))

    indice_array[filt] = 1

    # Change Values with Filters
    filt =  ((insert_dict['exercise'] == 1) & (np.isin(insert_dict['restimulus'], fingers_dict['indice']['ex1']['-1']))) |\
            ((insert_dict['exercise'] == 2) & (np.isin(insert_dict['restimulus'], fingers_dict['indice']['ex2']['-1']))) |\
            ((insert_dict['exercise'] == 3) & (np.isin(insert_dict['restimulus'], fingers_dict['indice']['ex3']['-1'])))

    indice_array[filt] = -1

    # Change Values with Filters
    filt =  ((insert_dict['exercise'] == 1) & (np.isin(insert_dict['restimulus'], fingers_dict['corazon']['ex1']['1']))) |\
            ((insert_dict['exercise'] == 2) & (np.isin(insert_dict['restimulus'], fingers_dict['corazon']['ex2']['1']))) |\
            ((insert_dict['exercise'] == 3) & (np.isin(insert_dict['restimulus'], fingers_dict['corazon']['ex3']['1'])))

    corazon_array[filt] = 1

    # Change Values with Filters
    filt =  ((insert_dict['exercise'] == 1) & (np.isin(insert_dict['restimulus'], fingers_dict['corazon']['ex1']['-1']))) |\
            ((insert_dict['exercise'] == 2) & (np.isin(insert_dict['restimulus'], fingers_dict['corazon']['ex2']['-1']))) |\
            ((insert_dict['exercise'] == 3) & (np.isin(insert_dict['restimulus'], fingers_dict['corazon']['ex3']['-1'])))

    corazon_array[filt] = -1

    # Append to dict
    insert_dict['pulgar']  = pulgar_array
    insert_dict['indice']  = indice_array
    insert_dict['corazon'] = corazon_array

    return insert_dict

def main():
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect(nina_folder + '/SQL/RawData.db')
    cursor = conn.cursor()

    if not check_table_exists(conn, cursor, "raw_data"):
        create_raw_data_table(conn, cursor)

    lista_rutas = archivos_por_patron(nina_folder, patron = '.mat')

    for ruta in lista_rutas:
        print(ruta)
        data = procesar_archivo(ruta)
        insert_into_table(conn, cursor, data, chunk_size=500)

if __name__ == '__main__':
    main()