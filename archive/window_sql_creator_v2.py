import os
import sys
import shutil

import pandas as pd
import numpy as np

import sqlite3
import scipy
import time

from multiprocessing import Pool, cpu_count

import asyncio
import aiosqlite

nina_folder = '../Data/NinaPro'

batch_size = 1200000

db1_hz = 100
db2_3_hz = 2000

ica_types = {'tanh': 'tanhICAmWDT'} # {'tanh': 'tanhICAmWDT', 'logcosh': 'logcoshICAmWDT', 'exp': 'expICAmWDT', 'none': 'mWDT'}
tables = ["tanhICAmWDT"] # ["tanhICAmWDT", "logcoshICAmWDT", "expICAmWDT", "mWDT"]

wavelet_iterations = ['db1'] #['db1', 'db2', 'db3']

timeframes = ['100ms'] # ['100ms', '200ms']
timeframe_values = {100: '100ms'} # {100: '100ms', 200:'200ms'}

timeframe_value = 100

new_table_columns = ["id", "db", "subject", "exercise", "restimulus", "emg0_std_cA", "emg0_std_cD", "emg0_var_cA", "emg0_var_cD", "emg0_wmav_cA", "emg0_wmav_cD", "emg1_std_cA", "emg1_std_cD", "emg1_var_cA", "emg1_var_cD", "emg1_wmav_cA", "emg1_wmav_cD", "emg2_std_cA", "emg2_std_cD", "emg2_var_cA", "emg2_var_cD", "emg2_wmav_cA", "emg2_wmav_cD", "emg3_std_cA", "emg3_std_cD", "emg3_var_cA", "emg3_var_cD", "emg3_wmav_cA", "emg3_wmav_cD", "emg4_std_cA", "emg4_std_cD", "emg4_var_cA", "emg4_var_cD", "emg4_wmav_cA", "emg4_wmav_cD", "emg5_std_cA", "emg5_std_cD", "emg5_var_cA", "emg5_var_cD", "emg5_wmav_cA", "emg5_wmav_cD", "emg6_std_cA", "emg6_std_cD", "emg6_var_cA", "emg6_var_cD", "emg6_wmav_cA", "emg6_wmav_cD", "emg7_std_cA", "emg7_std_cD", "emg7_var_cA", "emg7_var_cD", "emg7_wmav_cA", "emg7_wmav_cD", "emg8_std_cA", "emg8_std_cD", "emg8_var_cA", "emg8_var_cD", "emg8_wmav_cA", "emg8_wmav_cD", "emg9_std_cA", "emg9_std_cD", "emg9_var_cA", "emg9_var_cD", "emg9_wmav_cA", "emg9_wmav_cD", "pulgar", "indice", "corazon"]

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

def fast_ica(X, n_components, max_iter=200, tol=1e-4, func='tanh'):
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

def check_table_exists(conn, cursor, table_name):

    conn.commit()

    cursor.execute('''
    SELECT name FROM sqlite_master WHERE type='table' AND name=?
    ''', (table_name,))
    return cursor.fetchone() is not None

def create_table(conn, cursor, table):

    def emg_feaures_str():

        return_str = ''

        for i in range(10):
            return_str = return_str + f'emg{i}_std_cA REAL NOT NULL,\n'
            return_str = return_str + f'emg{i}_std_cD REAL NOT NULL,\n '
            return_str = return_str + f'emg{i}_var_cA REAL NOT NULL,\n '
            return_str = return_str + f'emg{i}_var_cD REAL NOT NULL,\n '
            return_str = return_str + f'emg{i}_wmav_cA REAL NOT NULL,\n '
            return_str = return_str + f'emg{i}_wmav_cD REAL NOT NULL,\n '

        return return_str

    # Create a table
    cursor.execute(f'''
    CREATE TABLE IF NOT EXISTS {table} (
        _id INTEGER PRIMARY KEY,
        id INTEGER NOT NULL,
        db INTEGER NOT NULL,
        subject INTEGER NOT NULL,
        exercise INTEGER NOT NULL,
        restimulus INTEGER NOT NULL,
        {emg_feaures_str()}
        pulgar INTEGER NOT NULL,
        indice INTEGER NOT NULL,
        corazon INTEGER NOT NULL
    )
    ''')

    conn.commit()

#def insert_into_table(conn, cursor, table, columns, values):
#
#    columns_str = ', '.join(columns)
#    placeholders = ', '.join(['?' for _ in columns])
#
#    values_tuple = tuple(values[col] for col in columns)
#
#    # Create a table
#   cursor.execute(f'''INSERT INTO  {table} ({columns_str}) VALUES ({placeholders});''', values_tuple)
#    conn.commit()

def insert_into_table(conn, cursor, table, columns, values):

    columns_str = ', '.join(columns)
    #placeholders = ', '.join(['?' for _ in columns])

    #values_tuple = tuple([values[col][i] for col in columns] for i in range(len(values[columns[0]])))

    values_str = ''
    for i in range(len(values[columns[0]])):

        if i != 0:
            values_str = values_str + ', '

        elements = ', '.join([str(values[col][i]) for col in columns])
        values_str = values_str + f'({elements})'

    # Create a table
    cursor.execute(f'''INSERT INTO  {table} ({columns_str}) VALUES {values_str};''')
    conn.commit()

async def asyncio_insert_into_table(database, table, data):

    async with aiosqlite.connect(database) as db:

        # Prepare the data for batch insertion
        rows = zip(*data.values())

        columns = ', '.join(data.keys())

        # Insert rows in batches
        batch_size = 1000  # Adjust the batch size based on your needs
        batch = []
        
        for row in rows:
            values = ', '.join([f"'{v}'" for v in row])
            batch.append(f"({values})")
            
            if len(batch) == batch_size:
                sql = f'INSERT INTO {table} ({columns}) VALUES {", ".join(batch)}'
                await db.execute(sql)
                await db.commit()
                batch = []

        # Insert remaining rows in the last batch
        if len(batch) != 0:
            sql = f'INSERT INTO {table} ({columns}) VALUES {", ".join(batch)}'
            await db.execute(sql)
            await db.commit()

def make_window_db_connections():

    connections_dict = {}

    for timeframe in timeframes:

        # Check if a directory (folder) exists
        directory_path = nina_folder + '/SQL/' + timeframe
        if not os.path.exists(directory_path):
            print(f"Directory '{directory_path}' does not exist.")
            os.makedirs(directory_path)

        # Connect to SQLite database
        conn = sqlite3.connect(directory_path + '/WindowData.db')
        cursor = conn.cursor()

        connections_dict[timeframe] = {'conn': conn, 'cursor': cursor}

        for table in tables:
            if not check_table_exists(conn, cursor, table):
                create_table(conn, cursor, table)

    return connections_dict

def get_unique_column_values(conn, cursor, table, column, where=''):

    # Query to get unique values from a specific column
    query = f"SELECT DISTINCT {column} FROM {table}"

    if where != '':
        query = query + ' WHERE ' + where

    else:
        query = query + ';'

    # Execute the query
    cursor.execute(query)

    # Fetch all the results as a list
    rows = cursor.fetchall()

    # Process fetched values
    unique_values = []
    for row in rows:
        value = row[0]
        if isinstance(value, bytes):
            decoded_value = int.from_bytes(value, byteorder='big')
            unique_values.append(decoded_value)
        else:
            # Handle integer values (or other types) as needed
            unique_values.append(value)  # Convert to string

    return unique_values

def count_table_rows(conn, cursor, table,  where=''):

     # Query to get unique values from a specific column
    query = f"SELECT COUNT(*) FROM {table}"

    if where != '':
        query = query + ' WHERE ' + where

    # Execute the query
    cursor.execute(query)

    # Fetch all the results as a list
    count = cursor.fetchone()[0]

    return count

def get_rows_by_id_with_previous(conn, cursor, table, target_id, num_previous, where = ''):

    if where != '':
        where = 'and ' + where
    
    # Construct the query to get the target row and the previous rows
    query = f"""
        SELECT * FROM (
            SELECT * FROM {table}
            WHERE id <= ? {where}
            ORDER BY id DESC
            LIMIT ?
        ) subquery
        ORDER BY id ASC;
    """
    
    # Execute the query
    cursor.execute(query, (target_id, num_previous + 1))  # +1 to include the target row
    rows = cursor.fetchall()
    
    return np.array(rows)

def get_table_schema(conn, cursor, table):
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table})")
    schema_info = cursor.fetchall()
    schema = [(col[1], col[2].upper()) for col in schema_info]
    return schema

def decode_value(value, dtype):
    if isinstance(value, bytes):
        value = value.decode()  # Decode the byte string to a regular string
    if dtype == 'INTEGER':
        return int(value)
    elif dtype == 'REAL':
        return float(value)
    return value  # For other types (e.g., TEXT), return as is

def select_from_table(conn, cursor, table, columns = '', where = ''):

    # Construct the query to get the target row and the previous rows

    if columns == '':
        columns = '*'

    query = f"SELECT {columns} FROM {table}"

    if where != '':
        query = query + ' WHERE ' + where
    
    # Execute the query
    cursor.execute(query)  # +1 to include the target row
    rows = cursor.fetchall()
    
    return np.array(rows)

def get_rows_by_ids(conn, cursor, table, id_list, id_name = 'id'):
    """
    Retrieve rows from a table where the id is in the given id_list.

    Parameters:
    - db_path: str, the path to the SQLite database file.
    - table_name: str, the name of the table to query.
    - id_list: list, a list of IDs to retrieve rows for.

    Returns:
    - rows: list of tuples, each tuple represents a row from the table.
    """
    
    # Convert the list of IDs to a tuple for the SQL query
    ids_tuple = tuple(id_list)
    
    # Create the SQL query
    #query = f"SELECT * FROM {table} WHERE id IN ({','.join('?' * len(ids_tuple))})"
    query = f"SELECT * FROM {table} WHERE {id_name} IN ({','.join(map(str, ids_tuple))})"
    
    # Execute the query and fetch the results
    #cursor.execute(query, ids_tuple)
    cursor.execute(query)
    rows = cursor.fetchall()

    return np.array(rows)

def query_max_value(conn, cursor, table, column):

    query = f"SELECT max({column}) FROM {table}"

    cursor.execute(query)
    rows = cursor.fetchall()

    return rows[0][0]

def calculate_wmav(data_array, start_weight = 0, end_weight = 1):

    weights_array = np.linspace(start_weight, end_weight, len(data_array)+1)[1:] # Since the first can be a 0, we are dropping that
    
    # Calculate the weighted moving average
    wmav = np.sum(data_array * weights_array) / np.sum(weights_array)
    
    return wmav

def process_ind_window(window):    

    # CORE VALUES
    _id        = window[-1][0]
    db         = window[-1][1]
    hz         = window[-1][2]
    subject    = window[-1][3]
    exercise   = window[-1][4]
    restimulus = window[-1][5]
    pulgar     = window[-1][-3]
    indice     = window[-1][-2]
    corazon    = window[-1][-1]

    ### RETURN DICT
    values_dict = {"id": float(_id), "db": float(db), "subject": float(subject), "exercise": float(exercise), "restimulus": float(restimulus),
                    "pulgar": float(pulgar), "indice": float(indice), "corazon": float(corazon)}

    for y in range(10): # Num EMG Sensors
        values_dict[f'emg{y}_std_cA'] = float(0)
        values_dict[f'emg{y}_std_cD'] = float(0)

        values_dict[f'emg{y}_var_cA'] = float(0)
        values_dict[f'emg{y}_var_cD'] = float(0)

        values_dict[f'emg{y}_wmav_cA'] = float(0)
        values_dict[f'emg{y}_wmav_cD'] = float(0)

    # NUM ROWS
    hz_rows = int((hz * timeframe_value) / 1000)
    
    # REAL WINDOW
    window = window[-hz_rows:] # Only EMG

    # Not enough Rows
    if len(window) < hz_rows:
        values_dict['id']= -1
        return values_dict

    # Check different DB or Subject
    if np.all(window[:,1] != window[:,1][0]) or np.all(window[:,3] != window[:,3][0]): # IF SUBJECT OR DB DIFF
        values_dict['id']= -1
        return values_dict

    # Check if it is prediction
    if window[-1][-1] == 2:
        values_dict['id']= -1
        return values_dict

    # ONLY EMG
    window = window[:, 7:17]
    
    # Num Columns
    num_columns = len(window[0])

    # Perform ICA
    window, W = fast_ica(window, num_columns, max_iter=10, func='tanh')

    # Perform mDWT
    higher_coefficients_array = []

    for x in range(num_columns):
        ind_coefficients = []
        max_level = int(wavelet_iterations[0][2:])

        ind_coefficients = mdwt(window[:,x], wavelet=wavelet_iterations[0], max_level=max_level) # cA, cD

        higher_coefficients_array.append(ind_coefficients)

    higher_coefficients_array = np.array(higher_coefficients_array)

    #### COMPUTE STD VARIANCE AND WMAV
    # Initialize arrays to hold values
    std_matrix = np.zeros((higher_coefficients_array.shape[0], higher_coefficients_array.shape[1]))
    var_matrix = np.zeros((higher_coefficients_array.shape[0], higher_coefficients_array.shape[1]))
    wmav_matrix = np.zeros((higher_coefficients_array.shape[0], higher_coefficients_array.shape[1]))

    # Compute std for each cA and cD separately
    for y in range(higher_coefficients_array.shape[0]):
        for z in range(higher_coefficients_array.shape[1]):
            std_matrix[y, z] = np.std(higher_coefficients_array[y, z])
            var_matrix[y, z] = std_matrix[y, z] ** 2  # Calculate variance using square of std
            wmav_matrix[y, z] = calculate_wmav(higher_coefficients_array[y, z])

    # Divide each element in the array by the sum of its corresponding column
    wmav_matrix = wmav_matrix / (np.sum(wmav_matrix, axis=0) + 1e-9)  # Small constant to avoid division by zero

    for y in range(len(std_matrix)): # Num EMG Sensors
        values_dict[f'emg{y}_std_cA'] = std_matrix[y, 0]
        values_dict[f'emg{y}_std_cD'] = std_matrix[y, 1]

        values_dict[f'emg{y}_var_cA'] = var_matrix[y, 0]
        values_dict[f'emg{y}_var_cD'] = var_matrix[y, 1]

        values_dict[f'emg{y}_wmav_cA'] = wmav_matrix[y, 0]
        values_dict[f'emg{y}_wmav_cD'] = wmav_matrix[y, 1]

    return values_dict

def process_windows(raw_conn, raw_cursor):

    #### GETTING CONNECTIONS ####
    connections_dict = make_window_db_connections()

    max_timeframe_value = max(timeframe_values.keys())

    time_conn = connections_dict[timeframe_values[max_timeframe_value]]['conn']
    time_cursor = connections_dict[timeframe_values[max_timeframe_value]]['cursor']

    #### MAX ID FOR THE WINDOW DB ####
    max__id = query_max_value(time_conn, time_cursor, ica_types['tanh'], '_id')
    rows = get_rows_by_ids(time_conn, time_cursor, ica_types['tanh'], [max__id], id_name = '_id')
    win_schema = get_table_schema(time_conn, time_cursor, ica_types['tanh'])
    rows = np.array([[decode_value(value, dtype) for value, (col_name, dtype) in zip(row, win_schema)] for row in rows])
    min_id = rows[0][1]

    #### MAX ID FOR RAW DB ###
    max_id = query_max_value(raw_conn, raw_cursor, 'raw_data', 'id')

    #### CREATE INSERTING DICT ####
    schema = get_table_schema(raw_conn, raw_cursor, 'raw_data')

    ### INCREMENTAL ARRAY ###
    array = np.arange(min_id + 1, max_id + 1)

    ### WINDOW SIZE FOR PREV ROWS ####
    max_window_size = int((db2_3_hz * max_timeframe_value) / 1000)
   
    ### START PROCESSING BATCHES ###
    for b in range(0, len(array), batch_size):
        print(b)
        batch = array[b:b + batch_size]

        # Prev X rows ids
        whole_batch = np.arange(batch[0] - max_window_size + 1, batch[0])  # Array from x-z to x-1

        # Concatenate arrays
        whole_batch = np.concatenate((whole_batch, batch))

        # GET BIG WINDOW
        big_window = get_rows_by_ids(raw_conn, raw_cursor, 'raw_data', whole_batch)
        big_window = np.array([[decode_value(value, dtype) for value, (col_name, dtype) in zip(row, win_schema)] for row in big_window])

        # GET BIGGEST INDIVIDUAL WINDOWS
        num_windows = big_window.shape[0] - max_window_size + 1
        windows = [big_window[i:i + max_window_size] for i in range(num_windows)]

        # Use multiprocessing to process windows in parallel
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(process_ind_window, windows)

        # Combine into 1 dict to insert into db
        values_dict = {}

        for d in results:
            if d.get('id') == -1:
                continue  # Skip dictionaries where 'id' equals -1

            for key, value in d.items():
                if key in values_dict:
                    values_dict[key].append(value)
                else:
                    values_dict[key] = [value]

        if not values_dict:
            continue

        #insert_into_table(time_conn, time_cursor, ica_types["tanh"], new_table_columns, values_dict)
        # Running the async function
        asyncio.run(asyncio_insert_into_table(nina_folder + "/SQL/100ms/WindowData.db", ica_types["tanh"], values_dict))
        #input("INPUT")

        values_dict = {}

    '''
    #rows = select_from_table(raw_conn, raw_cursor, 'raw_data', columns='id, db, subject, hz', where='corazon != 2')
  
    #db_list = get_unique_column_values(raw_conn, raw_cursor, 'raw_data', 'db', 'corazon != 2')
    
    

    #for db in np.unique(rows[:, 1]):
    #for db in [1]:

        #subjects_list = get_unique_column_values(raw_conn, raw_cursor, 'raw_data', 'subject', f'db = {db} and corazon != 2')

        #for subject in np.unique(rows[:, 2]):
        #for subject in [1]:
            #where = f'db = {db} AND subject = {subject}'
            #count = count_table_rows(raw_conn, raw_cursor, 'raw_data',  where=where)

        id_array = rows[(rows[:, 1] == db) & (rows[:, 2] == subject)][:, 0]
        hz_array = rows[(rows[:, 1] == db) & (rows[:, 2] == subject)][:, 3]

        for i, hz in zip(id_array, hz_array):
        #for i, hz in zip([441], [100]):

            max_hz_rows = int((hz * max_timeframe_value) / 1000)

            #window = get_rows_by_id_with_previous(raw_conn, raw_cursor, 'raw_data', i, max_hz_rows-1, where=f'db = {db} AND subject = {subject}')

            id_list = [i - z for z in range(max_hz_rows, -1, -1)]
            window = get_rows_by_ids(raw_conn, raw_cursor, 'raw_data', id_list)

            if len(window) < max_hz_rows:
                continue

            #check_window = window[:, [1,3]]

            if np.all(window[:,1] != window[:,1][0]) or np.all(window[:,3] != window[:,3][0]): # IF SUBJECT OR DB DIFF
                continue
            
            # Decode the byte strings to the appropriate data types according to the schema
            #window = [[decode_value(value, dtype) for value, (col_name, dtype) in zip(row, schema)]for row in window]
            window = np.array([[decode_value(value, dtype) for value, (col_name, dtype) in zip(row, schema)] for row in window])

            # Out of our predictions data
            if window[-1][-1] == 2:
                continue

            exercise = window[-1][4] # Store values befores dropping them
            restimulus = window[-1][5]
            pulgar = window[-1][-3]
            indice = window[-1][-2]
            corazon = window[-1][-1]

            for timeframe_value in timeframe_values.keys():

                hz_rows = int((hz * timeframe_value) / 1000)
                # Not big enough to form window
                if len(window) < hz_rows:
                    continue

                temp_window = window[-hz_rows:, 7:17] # Only EMG

                num_columns = len(temp_window[0])

                for ica in ica_types.keys():

                    if ica != 'none':
                        temp_window, W = fast_ica(temp_window, num_columns, max_iter=10, func=ica)

                    higher_coefficients_array = []

                    for x in range(num_columns):
                        ind_coefficients = []
                        for wavelet in wavelet_iterations:
                            max_level = int(wavelet[2:])

                            ind_coefficients = mdwt(temp_window[:,x], wavelet=wavelet, max_level=max_level) # cA, cD

                        higher_coefficients_array.append(ind_coefficients)

                    higher_coefficients_array = np.array(higher_coefficients_array)

                    # Initialize arrays to hold std values
                    std_matrix = np.zeros((higher_coefficients_array.shape[0], higher_coefficients_array.shape[1]))
                    var_matrix = np.zeros((higher_coefficients_array.shape[0], higher_coefficients_array.shape[1]))
                    wmav_matrix = np.zeros((higher_coefficients_array.shape[0], higher_coefficients_array.shape[1]))

                    # Compute std for each cA and cD separately
                    for y in range(higher_coefficients_array.shape[0]):
                        for z in range(higher_coefficients_array.shape[1]):
                            std_matrix[y, z] = np.std(higher_coefficients_array[y, z])
                            var_matrix[y, z] = std_matrix[y, z] ** 2  # Calculate variance using square of std
                            wmav_matrix[y, z] = calculate_wmav(higher_coefficients_array[y, z])

                    # Divide each element in the array by the sum of its corresponding column
                    wmav_matrix = wmav_matrix / (np.sum(wmav_matrix, axis=0) + 1e-9)  # Small constant to avoid division by zero

                    values_dict["id"].append(float(i))
                    values_dict["db"].append(float(db))
                    values_dict["subject"].append(float(subject))
                    values_dict["exercise"].append(float(exercise))
                    values_dict["restimulus"].append(float(restimulus))
                    values_dict["pulgar"].append(float(pulgar))
                    values_dict["indice"].append(float(indice))
                    values_dict["corazon"].append(float(corazon))

                    for y in range(len(std_matrix)): # Num EMG Sensors
                        values_dict[f'emg{y}_std_cA'].append(std_matrix[y, 0])
                        values_dict[f'emg{y}_std_cD'].append(std_matrix[y, 1])

                        values_dict[f'emg{y}_var_cA'].append(var_matrix[y, 0])
                        values_dict[f'emg{y}_var_cD'].append(var_matrix[y, 1])

                        values_dict[f'emg{y}_wmav_cA'].append(wmav_matrix[y, 0])
                        values_dict[f'emg{y}_wmav_cD'].append(wmav_matrix[y, 1])

                    if len(values_dict['id']) == 1000:
                        insert_into_table(connections_dict[timeframe_values[timeframe_value]]['conn'], connections_dict[timeframe_values[timeframe_value]]['cursor'], ica_types[ica], new_table_columns, values_dict)

                        values_dict = {"id": [], "db": [], "subject": [], "exercise": [], "restimulus": [],
                                        "pulgar": [], "indice": [], "corazon": []}

                        for y in range(10): # Num EMG Sensors
                            values_dict[f'emg{y}_std_cA'] = []
                            values_dict[f'emg{y}_std_cD'] = []

                            values_dict[f'emg{y}_var_cA'] = []
                            values_dict[f'emg{y}_var_cD'] = []

                            values_dict[f'emg{y}_wmav_cA'] = []
                            values_dict[f'emg{y}_wmav_cD'] = []

    if len(values_dict['id']) > 0:
        insert_into_table(connections_dict[timeframe_values[timeframe_value]]['conn'], connections_dict[timeframe_values[timeframe_value]]['cursor'], ica_types[ica], new_table_columns, values_dict)

    '''
def main():

    inicio = time.time()

    file_path = nina_folder + '/SQL/Raw/RawData.db'
    if not os.path.exists(file_path):
        print(f"File '{file_path}' does not exist.")
        sys.exit(1)
    
    # Connect to SQLite database
    raw_conn = sqlite3.connect(file_path)
    raw_cursor = raw_conn.cursor()

    process_windows(raw_conn, raw_cursor)

    final = time.time()

    resta = final - inicio
    print(resta)



if __name__ == '__main__':
    main()