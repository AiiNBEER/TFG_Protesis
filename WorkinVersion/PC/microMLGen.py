from micromlgen import port
import joblib
import numpy as np

# Cargar el modelo guardado
test_number = input("Introduce test Number: ")
knn_model = joblib.load(f'knn_model{test_number}.pkl')

centroids = np.load(f'centroids{test_number}.npy', allow_pickle=True)

with open(f'centroids.h', 'w') as f:
    f.write(f'const int NUM_CENTROIDS = {len(centroids)};\n')
    f.write('const int FEATURE_SIZE = 24;\n\n')
    
    f.write('const float centroids[][24] = {\n')
    for centroid, label in centroids:
        f.write('    {' + ', '.join(map(str, centroid)) + '},\n')
    f.write('};\n\n')
    
    f.write('const int labels[] = {\n')
    for _, label in centroids:
        f.write(f'    {label},\n')
    f.write('};\n')