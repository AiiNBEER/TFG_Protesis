#include <HardwareSerial.h>
#include <Arduino.h>
#include <vector>

// Variables globales
HardwareSerial NanoSerial(2); // Utiliza el puerto UART2 del ESP32
std::vector<float> window;
const int windowSize = 50; // Longitud de la ventana
const int numValues = 3; // Número de valores que llegan a la vez (columnas)

// Declarar las matrices y otros parámetros globales aquí para simplicidad
const int rows = windowSize; // Longitud de la ventana
const int cols = numValues;  // Número de valores que llegan a la vez
const int n_components = 3;
float X[rows * cols];  // Datos para ICA
float X_white[rows * cols]; // Datos blanqueados
float S[rows * n_components];
float W[n_components * cols];
volatile bool dataReady = false; // Indicador de datos listos para el segundo núcleo

// Funciones de utilidad
void mean_center(float* X, int rows, int cols);
void whitening(float* X, float* X_white, int rows, int cols);
void matmul(float* A, float* B, float* C, int aRows, int aCols, int bCols);
void vector_tanh(float* x, float* result, int length);
void vector_tanh_prime(float* x, float* result, int length);
void normalize(float* x, int length);
float dot_product(float* a, float* b, int length);
float mean(float* x, int length);

void fast_ica(float* X, int rows, int cols, int n_components, float* S, float* W, int max_iter = 200, float tol = 1e-4) {
    float* w = new float[cols];
    float* w_new = new float[cols];
    float* gw = new float[rows];
    float* g_w_prime = new float[rows];
    float* temp = new float[cols];

    Serial.println("Starting FastICA");

    // Main FastICA loop
    for (int i = 0; i < n_components; i++) {
        for (int j = 0; j < cols; j++) {
            w[j] = static_cast<float>(rand()) / RAND_MAX; // Initialize w randomly
        }

        for (int iter = 0; iter < max_iter; iter++) {
            matmul(X_white, w, gw, rows, cols, 1);
            vector_tanh(gw, gw, rows);
            vector_tanh_prime(gw, g_w_prime, rows);
            
            matmul(X_white, gw, temp, rows, cols, 1);
            for (int j = 0; j < cols; j++) {
                w_new[j] = temp[j] / rows - mean(g_w_prime, rows) * w[j];
            }

            normalize(w_new, cols);
            
            if (fabs(fabs(dot_product(w_new, w, cols)) - 1.0) < tol) {
                break;
            }

            for (int j = 0; j < cols; j++) {
                w[j] = w_new[j];
            }
        }

        for (int j = 0; j < cols; j++) {
            W[i * cols + j] = w[j];
        }
    }

    matmul(X, W, S, rows, cols, n_components);

    Serial.println("FastICA completed");

    delete[] w;
    delete[] w_new;
    delete[] gw;
    delete[] g_w_prime;
    delete[] temp;
}

// Tarea para el segundo núcleo
void fastIcaTask(void *pvParameters) {
    while (true) {
        if (dataReady) {
            Serial.println("Task Ready"); // Imprimir cuando dataReady es true
            
            // Ejecutar FastICA
            fast_ica(X, rows, cols, n_components, S, W);
            
            // Enviar los resultados por Serial
            for (int i = 0; i < rows * n_components; ++i) {
                Serial.print(S[i]);
                Serial.print(" ");
                if ((i + 1) % n_components == 0) {
                    Serial.println();
                }
            }
            
            // Resetear el indicador de datos listos
            dataReady = false;
        }
        vTaskDelay(100 / portTICK_PERIOD_MS);  // Retraso para evitar bucle ocupado
    }
}

// Tarea para el primer núcleo
void serialReadTask(void *pvParameters) {
    static String data = "";
    while (true) {
        while (NanoSerial.available()) {
            char c = NanoSerial.read();
            if (c == '\n') {
                // Parsear los datos y agregar a la ventana
                float value;
                char dataCopy[data.length() + 1];
                data.toCharArray(dataCopy, data.length() + 1);
                
                char* token = strtok(dataCopy, ",");
                while (token != NULL) {
                    value = atof(token);
                    window.push_back(value);
                    token = strtok(NULL, ",");
                }

                Serial.print("Window size: ");
                Serial.println(window.size());

                if (window.size() >= windowSize * numValues) {
                    // Copiar la ventana al array X
                    for (int i = 0; i < rows * cols; ++i) {
                        X[i] = window[i];
                    }
                    window.clear();

                    // Centrar y blanquear los datos
                    mean_center(X, rows, cols);
                    whitening(X, X_white, rows, cols);

                    // Indicar que los datos están listos para el segundo núcleo
                    dataReady = true;
                    Serial.println("Data ready for ICA");
                }
                data = "";  // Reiniciar la cadena de datos
            } else {
                data += c;  // Acumular el carácter en la cadena de datos
            }
        }
        vTaskDelay(10 / portTICK_PERIOD_MS);  // Retraso para evitar bucle ocupado
    }
}

void setup() {
    // Configurar la frecuencia del CPU a 240 MHz
    setCpuFrequencyMhz(240);

    Serial.begin(250000); // Comunicación con la PC
    NanoSerial.begin(250000, SERIAL_8N1, 16, 17); // RX en GPIO 16, TX en GPIO 17

    // Crear la tarea para el primer núcleo
    xTaskCreatePinnedToCore(
        serialReadTask,  // Función de la tarea
        "serialReadTask", // Nombre de la tarea
        32000,           // Tamaño del stack en palabras (casi al máximo)
        NULL,            // Parámetro de entrada
        1,               // Prioridad de la tarea
        NULL,            // Manejador de la tarea
        0                // Núcleo donde se ejecutará la tarea (0 o 1)
    );

    // Crear la tarea para el segundo núcleo
    xTaskCreatePinnedToCore(
        fastIcaTask,     // Función de la tarea
        "fastIcaTask",   // Nombre de la tarea
        32000,           // Tamaño del stack en palabras (casi al máximo)
        NULL,            // Parámetro de entrada
        1,               // Prioridad de la tarea
        NULL,            // Manejador de la tarea
        1                // Núcleo donde se ejecutará la tarea (0 o 1)
    );
}

void loop() {
    // No se necesita nada aquí, todo se maneja en las tareas
}

// Definiciones de funciones de utilidad
void mean_center(float* X, int rows, int cols) {
    for (int j = 0; j < cols; j++) {
        float mean = 0;
        for (int i = 0; i < rows; i++) {
            mean += X[i * cols + j];
        }
        mean /= rows;
        for (int i = 0; i < rows; i++) {
            X[i * cols + j] -= mean;
        }
    }
}

void whitening(float* X, float* X_white, int rows, int cols) {
    // Implementar la función de blanqueo (whitening) de manera simplificada
    // Para un ESP32, esto puede necesitar simplificación significativa debido a restricciones de memoria
    for (int i = 0; i < rows * cols; i++) {
        X_white[i] = X[i];  // Copia los datos por simplicidad
    }
}

void matmul(float* A, float* B, float* C, int aRows, int aCols, int bCols) {
    for (int i = 0; i < aRows; i++) {
        for (int j = 0; j < bCols; j++) {
            C[i * bCols + j] = 0;
            for (int k = 0; k < aCols; k++) {
                C[i * bCols + j] += A[i * aCols + k] * B[k * bCols + j];
            }
        }
    }
}

void vector_tanh(float* x, float* result, int length) {
    for (int i = 0; i < length; i++) {
        result[i] = tanh(x[i]);
    }
}

void vector_tanh_prime(float* x, float* result, int length) {
    for (int i = 0; i < length; i++) {
        result[i] = 1 - tanh(x[i]) * tanh(x[i]);
    }
}

void normalize(float* x, int length) {
    float norm = 0;
    for (int i = 0; i < length; i++) {
        norm += x[i] * x[i];
    }
    norm = sqrt(norm);
    for (int i = 0; i < length; i++) {
        x[i] /= norm;
    }
}

float dot_product(float* a, float* b, int length) {
    float result = 0;
    for (int i = 0; i < length; i++) {
        result += a[i] * b[i];
    }
    return result;
}

float mean(float* x, int length) {
    float sum = 0;
    for (int i = 0; i < length; i++) {
        sum += x[i];
    }
    return sum / length;
}
