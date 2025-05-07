const int numSensors = 5;
int sensorPins[numSensors] = {A0, A1, A2, A3, A4};

void setup() {
  Serial.begin(9600);
}

void loop() {
  for (int i = 0; i < numSensors; i++) {
    int sensorValue = analogRead(sensorPins[i]);
    Serial.print(sensorValue);
    if (i < numSensors - 1)
      Serial.print(",");
  }
  Serial.println();
  delay(50);
}
