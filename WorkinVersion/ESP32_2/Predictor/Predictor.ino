#include <Arduino.h>
#include <Chirale_TensorFlowLite.h>
#include <HardwareSerial.h>
#include <vector>
#include <esp_task_wdt.h>
#include "model.h" // El modelo .h generado 

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Variables globales
HardwareSerial ICASerial(2); // Utiliza el puerto UART2 del ESP32
std::vector<float> window;
const int windowSize = 50; // Longitud de la ventana
const int numValues = 3; // Número de valores en cada ventana

// Datos recibidos para ICA
float S[windowSize * numValues]; 
float dwtOutput[windowSize * numValues];

// Pesos para la Wavelet db1
const float h0 = 0.70710678118; // 1/sqrt(2)
const float h1 = 0.70710678118; // 1/sqrt(2)
const float g0 = h1; // 1/sqrt(2)
const float g1 = -h0; // -1/sqrt(2)

// Función para aplicar la DWT usando wavelet db1
void applyDWT(float* data, int length, float* output) {
    int halfLength = length / 2;
    for (int i = 0; i < halfLength; i++) {
        output[i] = h0 * data[2 * i] + h1 * data[2 * i + 1]; // Low-pass filter
        output[halfLength + i] = g0 * data[2 * i] + g1 * data[2 * i + 1]; // High-pass filter
    }
}

// Declaración del manejador de la tarea para el segundo núcleo
TaskHandle_t handleSecondCoreTask = NULL;

void readICAData(void *pvParameters) {
    while (true) {
        static String data = "";
        while (ICASerial.available()) {
            char c = ICASerial.read();
            if (c == '\n') {
                // Parsear los datos recibidos
                char dataCopy[data.length() + 1];
                data.toCharArray(dataCopy, data.length() + 1);
                int index = 0;
                char* token = strtok(dataCopy, " ");
                while (token != NULL) {
                    if (index < windowSize * numValues) {
                        S[index] = atof(token);
                        index++;
                    }
                    token = strtok(NULL, " ");
                }

                // Verificar si se recibieron suficientes datos
                if (index == windowSize * numValues) {
                    for (int i = 0; i < numValues; i++) {
                        applyDWT(&S[i * windowSize], windowSize, &dwtOutput[i * windowSize]);
                    }

                    // Indicar que los datos están listos para el segundo núcleo
                    xTaskNotifyGive(handleSecondCoreTask);
                }
                data = "";  // Reiniciar la cadena de datos
            } else {
                data += c;  // Acumular el carácter en la cadena de datos
            }
        }
        vTaskDelay(10 / portTICK_PERIOD_MS);  // Retraso para evitar bucle ocupado
    }
}

void calculateFeatures(float* data, int length, float* features) {
    for (int i = 0; i < numValues; i++) {
        float sum = 0, sumSq = 0, wl = 0, mav = 0;
        for (int j = 0; j < length; j++) {
            sum += data[i * length + j];
            sumSq += data[i * length + j] * data[i * length + j];
            if (j > 0) wl += abs(data[i * length + j] - data[i * length + j - 1]);
            mav += abs(data[i * length + j]);
        }
        features[i * 4 + 0] = sqrt(sumSq / length); // RMS
        features[i * 4 + 1] = mav / length; // MAV
        features[i * 4 + 2] = wl; // WL
        features[i * 4 + 3] = (sumSq / length) - (sum / length) * (sum / length); // VAR
    }
}

// TensorFlow Lite setup
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
uint8_t tensor_arena[10 * 1024]; // Ajustar según sea necesario

void setupModel() {
    static tflite::ErrorReporter micro_error_reporter;
    tflite::ErrorReporter* error_reporter = &micro_error_reporter;

    const tflite::Model* model = tflite::GetModel(model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        error_reporter->Report("Model provided is schema version not equal to supported version.");
        return;
    }

    static tflite::MicroMutableOpResolver<10> resolver; // Ajustar el número de operadores
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddReshape();

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, sizeof(tensor_arena), error_reporter, nullptr, nullptr); // Ajuste aquí
    interpreter = &static_interpreter;

    interpreter->AllocateTensors();
    input = interpreter->input(0);
    output = interpreter->output(0);
}

float runInference(float* features) {
    // Asumiendo que el modelo espera una entrada de longitud igual a numValues * 4
    for (int i = 0; i < numValues * 4; i++) {
        input->data.f[i] = features[i];
    }

    interpreter->Invoke();
    return output->data.f[0]; // Asumiendo una sola salida
}

void secondCoreTask(void *pvParameters) {
    setupModel(); // Configurar el modelo al inicio

    while (true) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY); // Espera notificación del primer núcleo
        
        float features[numValues * 4];
        calculateFeatures(dwtOutput, windowSize, features);

        float prediction = runInference(features);

        // Enviar los resultados por Serial
        Serial.print("Predicción: ");
        Serial.println(prediction);

        vTaskDelay(10 / portTICK_PERIOD_MS);  // Retraso para evitar bucle ocupado
    }
}

void setup() {
    // Configurar la frecuencia del CPU a 240 MHz
    setCpuFrequencyMhz(240);

    Serial.begin(250000); // Comunicación con la PC
    ICASerial.begin(250000, SERIAL_8N1, 16, 17); // RX en GPIO 16, TX en GPIO 17

    Serial.println("Setup completed, waiting for ICA data...");

    // Crear la tarea para el primer núcleo
    xTaskCreatePinnedToCore(
        readICAData,  // Función de la tarea
        "readICAData", // Nombre de la tarea
        32000,         // Tamaño del stack en palabras (casi al máximo)
        NULL,          // Parámetro de entrada
        1,             // Prioridad de la tarea
        NULL,          // Manejador de la tarea
        0              // Núcleo donde se ejecutará la tarea (0)
    );

    // Crear la tarea para el segundo núcleo
    xTaskCreatePinnedToCore(
        secondCoreTask,  // Función de la tarea
        "secondCoreTask", // Nombre de la tarea
        32000,            // Tamaño del stack en palabras (casi al máximo)
        NULL,             // Parámetro de entrada
        1,                // Prioridad de la tarea
        &handleSecondCoreTask, // Manejador de la tarea
        1                 // Núcleo donde se ejecutará la tarea (1)
    );
}

void loop() {
    // No se necesita nada aquí, todo se maneja en las tareas
}
