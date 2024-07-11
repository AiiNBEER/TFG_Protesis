#include <Arduino.h>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <esp32-hal-cpu.h>

HardwareSerial NanoSerial(2); // Utiliza el puerto UART2 del ESP32

#define ARRAY_SIZE 50
#define MAX_ITER 200
#define TOL 1e-4

std::vector<float> array1;
std::vector<float> array2;
std::vector<float> array3;

std::vector<float> preprocess_array1(ARRAY_SIZE);
std::vector<float> preprocess_array2(ARRAY_SIZE);
std::vector<float> preprocess_array3(ARRAY_SIZE);

std::vector<std::vector<float>> components(3, std::vector<float>(ARRAY_SIZE));

SemaphoreHandle_t xSemaphore = NULL;

void updateArray(std::vector<float>& arr, float value) {
    if (arr.size() < ARRAY_SIZE) {
        arr.push_back(value);
    } else {
        arr.erase(arr.begin());
        arr.push_back(value);
    }
}

float dotProduct(const std::vector<float>& a, const std::vector<float>& b) {
    float result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

void matrixMultiply(const std::vector<std::vector<float>>& matrix, const std::vector<float>& vec, std::vector<float>& result) {
    for (size_t i = 0; i < matrix.size(); ++i) {
        result[i] = dotProduct(matrix[i], vec);
    }
}

void normalize(std::vector<float>& vec) {
    float norm = sqrt(dotProduct(vec, vec));
    if (norm != 0) {
        for (size_t i = 0; i < vec.size(); ++i) {
            vec[i] /= norm;
        }
    }
}

void fastICA(const std::vector<std::vector<float>>& X, std::vector<std::vector<float>>& components, int max_iter, float tol) {
    size_t n_components = components.size();
    size_t n_features = X[0].size();
    
    std::vector<float> w(n_features);
    std::vector<float> w_prev(n_features);
    std::vector<float> gw(n_features, 0.0);
    std::vector<float> g_w(n_features, 0.0);

    for (size_t p = 0; p < n_components; ++p) {
        for (size_t i = 0; i < n_features; ++i) {
            components[p][i] = static_cast<float>(rand()) / RAND_MAX;
        }

        for (int iter = 0; iter < max_iter; ++iter) {
            w_prev = components[p];
            
            std::fill(gw.begin(), gw.end(), 0.0);
            std::fill(g_w.begin(), g_w.end(), 0.0);

            for (size_t i = 0; i < X.size(); ++i) {
                float dot = dotProduct(components[p], X[i]);
                float tanh_dot = tanh(dot);
                for (size_t j = 0; j < n_features; ++j) {
                    gw[j] += X[i][j] * tanh_dot;
                    g_w[j] += X[i][j] * (1 - tanh_dot * tanh_dot);
                }
            }

            for (size_t i = 0; i < n_features; ++i) {
                components[p][i] = gw[i] / X.size() - g_w[i] * components[p][i] / X.size();
            }

            normalize(components[p]);

            for (size_t j = 0; j < p; ++j) {
                float dot = dotProduct(components[p], components[j]);
                for (size_t k = 0; k < n_features; ++k) {
                    components[p][k] -= dot * components[j][k];
                }
            }

            normalize(components[p]);

            float diff = 0.0;
            for (size_t i = 0; i < n_features; ++i) {
                diff += fabs(components[p][i] - w_prev[i]);
            }

            if (diff < tol) {
                break;
            }
        }
    }
}

void serialReadTask(void *pvParameters) {
    static String data = "";
    while (true) {
        if (NanoSerial.available()) {
            char c = NanoSerial.read();
            if (c == '\n') {
                // Parsear los datos y agregar a los arrays correspondientes
                float values[3];
                int index = 0;
                char dataCopy[data.length() + 1];
                data.toCharArray(dataCopy, data.length() + 1);
                
                char* token = strtok(dataCopy, ",");
                while (token != NULL && index < 3) {
                    values[index] = atof(token);
                    token = strtok(NULL, ",");
                    index++;
                }

                if (index == 3) { // Asegurarse de que tenemos exactamente 3 valores
                    updateArray(array1, values[0]);
                    updateArray(array2, values[1]);
                    updateArray(array3, values[2]);

                    // Preprocesar las ventanas si están llenas
                    if (array1.size() == ARRAY_SIZE && array2.size() == ARRAY_SIZE && array3.size() == ARRAY_SIZE) {
                        // Los datos ya están centrados y blanqueados en array1, array2 y array3
                        preprocess_array1 = array1;
                        preprocess_array2 = array2;
                        preprocess_array3 = array3;

                        // Liberar el semáforo para indicar que los datos están listos
                        xSemaphoreGive(xSemaphore);
                    }
                }

                data = "";  // Reiniciar la cadena de datos
            } else {
                data += c;  // Acumular el carácter en la cadena de datos
            }
        }
        vTaskDelay(1 / portTICK_PERIOD_MS);  // Retraso para evitar bucle ocupado
    }
}

void sendICAVectors() {
    Serial.println("START"); // Delimitador para indicar el comienzo de los datos
    for (size_t i = 0; i < components.size(); ++i) {
        for (size_t j = 0; j < components[i].size(); ++j) {
            Serial.print(components[i][j], 6);
            if (i < components.size() - 1 || j < components[i].size() - 1) {
                Serial.print(","); // Separador entre valores
            }
        }
    }
    Serial.println(); // Asegurarse de que "END" está en una línea separada
    Serial.println("END"); // Delimitador para indicar el fin de los datos
}

void printTask(void *pvParameters) {
    while (true) {
        // Esperar hasta que se libere el semáforo
        if (xSemaphoreTake(xSemaphore, portMAX_DELAY) == pdTRUE) {
            // Construir la matriz X a partir de los arrays preprocesados
            std::vector<std::vector<float>> X = {preprocess_array1, preprocess_array2, preprocess_array3};

            // Ejecutar FastICA
            fastICA(X, components, MAX_ITER, TOL);

            // Imprimir los componentes independientes
            /*
            Serial.println("Independent Components:");
            for (size_t i = 0; i < components.size(); ++i) {
                Serial.print("Component ");
                Serial.print(i);
                Serial.print(": ");
                for (size_t j = 0; j < components[i].size(); ++j) {
                    Serial.print(components[i][j], 6);
                    Serial.print(" ");
                }
                Serial.println();
            }
            */
        }
        // Enviar los vectores procesados por ICA a través del serial
        sendICAVectors();
    }
}

void setup() {
    // Ajustar la frecuencia de los núcleos
    setCpuFrequencyMhz(240); // Ajustar a la frecuencia máxima soportada de 240 MHz

    // Inicializar Serial y Serial2
    Serial.begin(250000);
    NanoSerial.begin(250000, SERIAL_8N1, 16, 17); // RX en GPIO 16, TX en GPIO 17

    // Crear el semáforo binario
    xSemaphore = xSemaphoreCreateBinary();
    if (xSemaphore == NULL) {
        Serial.println("Failed to create semaphore");
        while (1); // Si falla la creación del semáforo, detener la ejecución
    }

    // Crear la tarea en el primer núcleo
    xTaskCreatePinnedToCore(
        serialReadTask,   // Función de la tarea
        "SerialReadTask", // Nombre de la tarea
        4096,             // Tamaño de la pila en bytes
        NULL,             // Parámetros de entrada (no se usan en este caso)
        1,                // Prioridad de la tarea
        NULL,             // Manejador de la tarea (opcional)
        0                 // Núcleo donde se ejecutará la tarea (0 = primer núcleo)
    );

    // Crear la tarea en el segundo núcleo
    xTaskCreatePinnedToCore(
        printTask,        // Función de la tarea
        "PrintTask",      // Nombre de la tarea
        8192,             // Tamaño de la pila en bytes
        NULL,             // Parámetros de entrada (no se usan en este caso)
        1,                // Prioridad de la tarea
        NULL,             // Manejador de la tarea (opcional)
        1                 // Núcleo donde se ejecutará la tarea (1 = segundo núcleo)
    );

    // Otros posibles inicializaciones
}

void loop() {
    // El loop puede estar vacío o puede contener otras funciones que se ejecuten en el segundo núcleo
}
