#include <HardwareSerial.h>

HardwareSerial NanoSerial(2); // Utiliza el puerto UART2 del ESP32

void setup() {
  Serial.begin(250000); // Comunicación con la PC
  NanoSerial.begin(250000, SERIAL_8N1, 16, 17); // RX en GPIO 16, TX en GPIO 17
}

void loop() {
  if (NanoSerial.available()) {
    String data = NanoSerial.readStringUntil('\n'); // Lee la línea de datos del Nano
    Serial.println(data); // Envía los datos recibidos a la PC
  }
}
