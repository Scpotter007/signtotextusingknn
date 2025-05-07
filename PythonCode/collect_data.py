import numpy as np
import os
import string
import serial
import time
import cv2  # Still needed for image processing and visualization

# Create the directory structure
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("data/train"):
    os.makedirs("data/train")
if not os.path.exists("data/test"):
    os.makedirs("data/test")
for i in range(3):
    if not os.path.exists("data/train/" + str(i)):
        os.makedirs("data/train/" + str(i))
    if not os.path.exists("data/test/" + str(i)):
        os.makedirs("data/test/" + str(i))

for i in string.ascii_uppercase:
    if not os.path.exists("data/train/" + i):
        os.makedirs("data/train/" + i)
    if not os.path.exists("data/test/" + i):
        os.makedirs("data/test/" + i)

# Initialize serial connection to Arduino
try:
    arduino = serial.Serial('COM9', 9600, timeout=1)
    print("Connected to Arduino on COM9")
    time.sleep(2)  # Allow time for connection to establish
except Exception as e:
    print(f"Error connecting to Arduino: {e}")
    exit()

# Train or test mode
mode = 'train'
directory = 'data/' + mode + '/'

# Create a blank frame for visualization
frame_width, frame_height = 640, 480
frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255  # White background


# Function to read sensor values from Arduino
def read_sensors():
    arduino.write(b'R')  # Send request for sensor data
    time.sleep(0.1)
    if arduino.in_waiting:
        try:
            line = arduino.readline().decode('utf-8').strip()
            values = [int(val) for val in line.split(',')]
            if len(values) == 4:  # Expecting 4 sensor values
                return values
            else:
                print(f"Warning: Expected 4 values, got {len(values)}")
                return None
        except Exception as e:
            print(f"Error reading sensor data: {e}")
            return None
    return None


# Function to visualize flex sensor data
def visualize_flex_data(sensor_values):
    if sensor_values is None:
        return np.copy(frame)

    vis_frame = np.copy(frame)

    # Draw sensor bars
    bar_width = 50
    bar_spacing = 20
    max_bar_height = 300
    start_x = 100

    for i, value in enumerate(sensor_values):
        # Normalize value between 0-100
        height = int((value / 1023) * max_bar_height)

        # Draw bar
        x = start_x + i * (bar_width + bar_spacing)
        cv2.rectangle(vis_frame,
                      (x, frame_height - 100 - height),
                      (x + bar_width, frame_height - 100),
                      (0, 0, 255), -1)

        # Add value text
        cv2.putText(vis_frame, f"{value}",
                    (x, frame_height - 70),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

        # Add sensor label
        cv2.putText(vis_frame, f"Sensor {i + 1}",
                    (x, frame_height - 50),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

    # Draw ROI box for visualization
    cv2.rectangle(vis_frame, (220 - 1, 9), (620 + 1, 419), (255, 0, 0), 1)

    return vis_frame


# Function to save sensor data to a file
def save_sensor_data(folder, filename, sensor_values):
    file_path = os.path.join(directory, folder, filename)
    np.save(file_path, sensor_values)
    print(f"Saved to {file_path}")


# Get counts of existing files
count = {
    'zero': len(os.listdir(directory + "/0")),
    'one': len(os.listdir(directory + "/1")),
    'two': len(os.listdir(directory + "/2")),
    'a': len(os.listdir(directory + "/A")),
    'b': len(os.listdir(directory + "/B")),
    'c': len(os.listdir(directory + "/C")),
    'd': len(os.listdir(directory + "/D")),
    'e': len(os.listdir(directory + "/E")),
    'f': len(os.listdir(directory + "/F")),
    'g': len(os.listdir(directory + "/G")),
    'h': len(os.listdir(directory + "/H")),
    'i': len(os.listdir(directory + "/I")),
    'j': len(os.listdir(directory + "/J")),
    'k': len(os.listdir(directory + "/K")),
    'l': len(os.listdir(directory + "/L")),
    'm': len(os.listdir(directory + "/M")),
    'n': len(os.listdir(directory + "/N")),
    'o': len(os.listdir(directory + "/O")),
    'p': len(os.listdir(directory + "/P")),
    'q': len(os.listdir(directory + "/Q")),
    'r': len(os.listdir(directory + "/R")),
    's': len(os.listdir(directory + "/S")),
    't': len(os.listdir(directory + "/T")),
    'u': len(os.listdir(directory + "/U")),
    'v': len(os.listdir(directory + "/V")),
    'w': len(os.listdir(directory + "/W")),
    'x': len(os.listdir(directory + "/X")),
    'y': len(os.listdir(directory + "/Y")),
    'z': len(os.listdir(directory + "/Z"))
}

try:
    while True:
        # Read sensor values from Arduino
        sensor_values = read_sensors()

        # Create visualization
        vis_frame = visualize_flex_data(sensor_values)

        # Display counts for first few categories (to avoid cluttering the screen)
        cv2.putText(vis_frame, "ZERO : " + str(count['zero']), (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        cv2.putText(vis_frame, "ONE : " + str(count['one']), (10, 80), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        cv2.putText(vis_frame, "TWO : " + str(count['two']), (10, 90), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        cv2.putText(vis_frame, "A : " + str(count['a']), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        cv2.putText(vis_frame, "B : " + str(count['b']), (10, 110), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

        # Instructions
        cv2.putText(vis_frame, "Press key (0-2, A-Z) to save current position", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 0, 0), 1)
        cv2.putText(vis_frame, "Press ESC to exit", (10, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

        # Show the frame
        cv2.imshow("Flex Sensor Data", vis_frame)

        # Check for key presses
        interrupt = cv2.waitKey(10)
        if interrupt & 0xFF == 27:  # ESC key
            break

        # Only save data if we have valid sensor readings
        if sensor_values is not None:
            # Handle number keys
            if interrupt & 0xFF == ord('0'):
                save_sensor_data('0', f"{count['zero']}.npy", sensor_values)
                count['zero'] += 1
            elif interrupt & 0xFF == ord('1'):
                save_sensor_data('1', f"{count['one']}.npy", sensor_values)
                count['one'] += 1
            elif interrupt & 0xFF == ord('2'):
                save_sensor_data('2', f"{count['two']}.npy", sensor_values)
                count['two'] += 1

            # Handle letter keys
            for letter in string.ascii_lowercase:
                if interrupt & 0xFF == ord(letter):
                    upper_letter = letter.upper()
                    save_sensor_data(upper_letter, f"{count[letter]}.npy", sensor_values)
                    count[letter] += 1
                    break

except KeyboardInterrupt:
    print("Program interrupted by user")
finally:
    # Clean up
    arduino.close()
    cv2.destroyAllWindows()
    print("Program ended")