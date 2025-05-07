import numpy as np
import cv2
import serial
import time


# Arduino connection setup
def setup_arduino(port='/dev/ttyUSB0', baud_rate=9600):
    """
    Establish connection with Arduino
    Returns serial connection object or None if connection fails
    """
    try:
        arduino = serial.Serial(port, baud_rate, timeout=1)
        time.sleep(2)  # Allow time for connection to establish
        print(f"Connected to Arduino on {port}")
        return arduino
    except Exception as e:
        print(f"Failed to connect to Arduino: {e}")
        return None


# Image processing function (from your original code)
def process_image(path, threshold_value=70):
    """
    Process image with adaptive thresholding
    Returns the processed binary image
    """
    frame = cv2.imread(path)
    if frame is None:
        print(f"Error: Could not read image from {path}")
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)

    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 11, 2)
    ret, res = cv2.threshold(th3, threshold_value, 255,
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return res


# Read flex sensor data from Arduino
def read_flex_sensor(arduino):
    """
    Read flex sensor data from Arduino
    Returns the flex sensor value as an integer
    """
    if arduino is None:
        return None

    try:
        arduino.write(b'R')  # Send request for data
        time.sleep(0.1)  # Wait for Arduino to process

        if arduino.in_waiting > 0:
            data = arduino.readline().decode('utf-8').strip()
            try:
                return int(data)
            except ValueError:
                print(f"Could not convert Arduino data to integer: {data}")
                return None
    except Exception as e:
        print(f"Error reading from Arduino: {e}")
        return None

    return None


# Adjust threshold based on flex sensor reading
def adjust_threshold(flex_value, min_thresh=50, max_thresh=150):
    """
    Map flex sensor value to threshold range
    Returns a threshold value between min_thresh and max_thresh
    """
    if flex_value is None:
        return 70  # Default value

    # Assuming flex_value ranges from 0-1023 (standard Arduino analog reading)
    # Map it to threshold range
    mapped_value = min_thresh + (flex_value / 1023.0) * (max_thresh - min_thresh)
    return int(mapped_value)


# Main function that ties everything together
def main():
    arduino = setup_arduino()
    camera = cv2.VideoCapture(0)  # Use default camera

    if not camera.isOpened():
        print("Error: Could not open camera")
        return

    try:
        while True:
            # Read flex sensor data
            flex_value = read_flex_sensor(arduino)
            if flex_value is not None:
                print(f"Flex sensor value: {flex_value}")

                # Adjust threshold based on flex sensor
                threshold = adjust_threshold(flex_value)

                # Capture frame from camera
                ret, frame = camera.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    continue

                # Save frame temporarily
                temp_path = "temp_frame.jpg"
                cv2.imwrite(temp_path, frame)

                # Process the image with the threshold
                processed = process_image(temp_path, threshold)

                if processed is not None:
                    # Display original and processed frames
                    cv2.imshow("Original", frame)
                    cv2.imshow("Processed", processed)

            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.1)  # Small delay to reduce CPU usage

    finally:
        # Clean up
        if arduino is not None:
            arduino.close()
        camera.release()
        cv2.destroyAllWindows()


# Arduino sketch (to be uploaded to Arduino)
"""
// Arduino code for flex sensor reading
const int FLEX_PIN = A0; // Pin connected to voltage divider output

void setup() {
  Serial.begin(9600);
  pinMode(FLEX_PIN, INPUT);
}

void loop() {
  // Check if there's a request from Python
  if (Serial.available() > 0) {
    char request = Serial.read();
    if (request == 'R') {
      // Read the flex sensor value
      int flexValue = analogRead(FLEX_PIN);
      // Send the value back to Python
      Serial.println(flexValue);
    }
  }
  delay(10);
}
"""

if __name__ == "__main__":
    main()