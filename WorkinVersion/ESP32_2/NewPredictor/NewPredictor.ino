#include <Arduino.h>
#include <vector>
#include <cmath>
#include <esp32-hal-cpu.h>
#include <cfloat>  // Incluir para FLT_MAX
#include "centroids.h"  // Incluir el archivo generado para el modelo KNN

HardwareSerial ESPSerial(2); // Utiliza el puerto UART2 del ESP32

#define TOTAL_SIZE 150
#define ARRAY_SIZE 50
#define NUM_COMPONENTS 3
#define DWT_SIZE (ARRAY_SIZE / 2)
#define FEATURE_SIZE 32

std::vector<float> receivedData(TOTAL_SIZE);
std::vector<std::vector<float>> receivedComponents(NUM_COMPONENTS, std::vector<float>(ARRAY_SIZE));
std::vector<std::vector<float>> dwtApproximation(NUM_COMPONENTS, std::vector<float>(DWT_SIZE));
std::vector<std::vector<float>> dwtDetail(NUM_COMPONENTS, std::vector<float>(DWT_SIZE));

std::vector<float> windowFeatures(FEATURE_SIZE);
std::vector<std::vector<float>> valuesList;

SemaphoreHandle_t xSemaphore = NULL;

// Filtros db1
const float h0 = 1.0 / sqrt(2.0);
const float h1 = 1.0 / sqrt(2.0);
const float g0 = 1.0 / sqrt(2.0);
const float g1 = -1.0 / sqrt(2.0);

void serialReceiveTask(void *pvParameters) {
    static String data = "";
    static bool receiving = false;
    static int dataIndex = 0;

    while (true) {
        while (ESPSerial.available()) {
            char c = ESPSerial.read();
            if (c == '\n' || c == '\r') { // Considerar tanto \n como \r como posibles fin de línea
                if (data.length() > 0) { // Asegurarse de que la línea no esté vacía

                    if (data == "START") {
                        receiving = true;
                        dataIndex = 0;
                        // Resetear los datos recibidos
                        receivedData.assign(TOTAL_SIZE, 0.0);
                    } else if (data == "END") {
                        receiving = false;
                        
                        if (dataIndex == TOTAL_SIZE) {
                            // Dividir los datos en componentes solo si hemos recibido todos los datos
                            for (size_t i = 0; i < NUM_COMPONENTS; ++i) {
                                for (size_t j = 0; j < ARRAY_SIZE; ++j) {
                                    receivedComponents[i][j] = receivedData[i * ARRAY_SIZE + j];
                                }
                            }

                            // Aplicar DWT db1 a cada componente
                            for (size_t i = 0; i < NUM_COMPONENTS; ++i) {
                                dwt(receivedComponents[i], dwtApproximation[i], dwtDetail[i]);
                            }

                            // Liberar el semáforo para indicar que los datos están listos
                            xSemaphoreGive(xSemaphore);
                        }
                    } else if (receiving) {
                        int index = 0;
                        char dataCopy[data.length() + 1];
                        data.toCharArray(dataCopy, data.length() + 1);

                        char* token = strtok(dataCopy, ",");
                        while (token != NULL && dataIndex < TOTAL_SIZE) {
                            receivedData[dataIndex] = atof(token);
                            token = strtok(NULL, ",");
                            dataIndex++;
                        }
                    }
                    data = ""; // Reiniciar la cadena de datos después de procesarla
                }
            } else {
                data += c; // Acumular el carácter en la cadena de datos
            }
        }
        vTaskDelay(10 / portTICK_PERIOD_MS); // Aumentar el retraso para evitar pérdida de datos
    }
}

void dwt(const std::vector<float>& input, std::vector<float>& approx, std::vector<float>& detail) {
    int halfSize = input.size() / 2;
    for (int i = 0; i < halfSize; ++i) {
        approx[i] = h0 * input[2 * i] + h1 * input[2 * i + 1];       // Low-pass filter (approximation)
        detail[i] = g0 * input[2 * i] + g1 * input[2 * i + 1];       // High-pass filter (detail)
    }
}

float calculateRMS(const std::vector<float>& data) {
    float sum = 0.0;
    for (const auto& value : data) {
        sum += value * value;
    }
    return sqrt(sum / data.size());
}

float calculateMAV(const std::vector<float>& data) {
    float sum = 0.0;
    for (const auto& value : data) {
        sum += fabs(value);
    }
    return sum / data.size();
}

float calculateWL(const std::vector<float>& data) {
    float sum = 0.0;
    for (size_t i = 1; i < data.size(); ++i) {
        sum += fabs(data[i] - data[i - 1]);
    }
    return sum;
}

float calculateVAR(const std::vector<float>& data) {
    float mean = 0.0;
    for (const auto& value : data) {
        mean += value;
    }
    mean /= data.size();

    float variance = 0.0;
    for (const auto& value : data) {
        variance += (value - mean) * (value - mean);
    }
    return variance / data.size();
}

void extractFeatures(const std::vector<std::vector<float>>& approx, const std::vector<std::vector<float>>& detail, std::vector<float>& features) {
    features.clear();
    for (size_t i = 0; i < approx.size(); ++i) {
        // RMS
        features.push_back(calculateRMS(approx[i]));
        // RMS
        features.push_back(calculateRMS(detail[i]));
        
        // MAV
        features.push_back(calculateMAV(approx[i]));
        // MAV
        features.push_back(calculateMAV(detail[i]));

        // WL
        features.push_back(calculateWL(approx[i]));
        // WL
        features.push_back(calculateWL(detail[i]));

        // VAR
        features.push_back(calculateVAR(approx[i]));
        // VAR
        features.push_back(calculateVAR(detail[i]));
    }
}

float calculateDistance(const float* a, const float* b, int length) {
    float distance = 0.0;
    for (int i = 0; i < length; i++) {
        distance += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(distance);
}

int knnPredict(const float* input) {
    float min_distance = FLT_MAX;
    int prediction = -1;

    for (int i = 0; i < 6; i++) {
        float distance = calculateDistance(input, centroids[i], FEATURE_SIZE);
        Serial.print("Distance to centroid ");
        Serial.print(i);
        Serial.print(": ");
        Serial.println(distance);

        if (distance < min_distance) {
            min_distance = distance;
            prediction = labels[i];
        }
    }

    Serial.print("Predicted label: ");
    Serial.println(prediction);

    return prediction;
}

void featureTask(void *pvParameters) {
    while (true) {
        // Esperar hasta que se libere el semáforo
        if (xSemaphoreTake(xSemaphore, portMAX_DELAY) == pdTRUE) {
            // Extraer características y añadirlas a la lista de valores
            extractFeatures(dwtApproximation, dwtDetail, windowFeatures);
            valuesList.push_back(windowFeatures);

            // Imprimir las características extraídas
            Serial.println("Extracted Features:");
            for (size_t i = 0; i < windowFeatures.size(); ++i) {
                Serial.print(windowFeatures[i], 6);
                Serial.print(" ");
            }
            Serial.println();

            // Realizar predicción con KNN
            int prediction = knnPredict(windowFeatures.data());
            Serial.print("Prediction: ");
            Serial.println(prediction);
        }
    }
}

void setup() {
    // Ajustar la frecuencia de los núcleos
    setCpuFrequencyMhz(240); // Ajustar a la frecuencia máxima soportada de 240 MHz

    // Inicializar Serial y ESPSerial
    Serial.begin(250000);
    ESPSerial.begin(250000, SERIAL_8N1, 16, 17); // RX en GPIO 16, TX en GPIO 17

    // Crear el semáforo binario
    xSemaphore = xSemaphoreCreateBinary();
    if (xSemaphore == NULL) {
        Serial.println("Failed to create semaphore");
        while (1); // Si falla la creación del semáforo, detener la ejecución
    }

    // Crear la tarea en el primer núcleo con más memoria
    xTaskCreatePinnedToCore(
        serialReceiveTask,   // Función de la tarea
        "SerialReceiveTask", // Nombre de la tarea
        8192,                // Tamaño de la pila en bytes, aumentado
        NULL,                // Parámetros de entrada (no se usan en este caso)
        1,                   // Prioridad de la tarea
        NULL,                // Manejador de la tarea (opcional)
        0                    // Núcleo donde se ejecutará la tarea (0 = primer núcleo)
    );

    // Crear la tarea en el segundo núcleo
    xTaskCreatePinnedToCore(
        featureTask,         // Función de la tarea
        "FeatureTask",       // Nombre de la tarea
        8192,                // Tamaño de la pila en bytes
        NULL,                // Parámetros de entrada (no se usan en este caso)
        1,                   // Prioridad de la tarea
        NULL,                // Manejador de la tarea (opcional)
        1                    // Núcleo donde se ejecutará la tarea (1 = segundo núcleo)
    );
}

void loop() {
    // El loop puede estar vacío o puede contener otras funciones que se ejecuten en el segundo núcleo
}
