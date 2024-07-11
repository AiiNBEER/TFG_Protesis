unsigned long previousMicros = 0; // Variable para almacenar el tiempo anterior
const unsigned long interval = 2000; // Intervalo de tiempo en microsegundos (2 ms)

void setup() {
  Serial.begin(250000); // Configura el puerto serie a 115200 bps
}

void loop() {
  unsigned long currentMicros = micros();

  if (currentMicros - previousMicros >= interval) {
    previousMicros = currentMicros; // Actualiza el tiempo anterior

    int value1 = analogRead(A0); // Lee el valor del pin analógico A0
    int value2 = analogRead(A1); // Lee el valor del pin analógico A1
    int value3 = analogRead(A2); // Lee el valor del pin analógico A2

    // Envía los tres valores separados por comas
    Serial.print(value1);
    Serial.print(",");
    Serial.print(value2);
    Serial.print(",");
    Serial.println(value3);
  }
}
