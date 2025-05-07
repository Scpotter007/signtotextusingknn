from PIL import Image, ImageTk
import tkinter as tk
import cv2
import os
import numpy as np
import serial
import time
from keras.models import model_from_json
import operator
import sys
import matplotlib.pyplot as plt
import hunspell
from string import ascii_uppercase


class Application:
    def __init__(self):
        self.directory = 'model/'
        try:
            self.hs = hunspell.HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')
        except:
            print("Hunspell dictionary not found. Spell check will be disabled.")
            self.hs = None

        # Connect to Arduino
        try:
            self.arduino = serial.Serial('COM9', 9600, timeout=1)
            print("Connected to Arduino on COM9")
            time.sleep(2)  # Allow time for connection to establish
        except Exception as e:
            print(f"Error connecting to Arduino: {e}")
            self.arduino = None

        self.current_image = None
        self.current_image2 = None

        # Load models
        try:
            self.json_file = open(self.directory + "model-bw.json", "r")
            self.model_json = self.json_file.read()
            self.json_file.close()
            self.loaded_model = model_from_json(self.model_json)
            self.loaded_model.load_weights(self.directory + "model-bw.h5")

            self.json_file_dru = open(self.directory + "model-bw_dru.json", "r")
            self.model_json_dru = self.json_file_dru.read()
            self.json_file_dru.close()
            self.loaded_model_dru = model_from_json(self.model_json_dru)
            self.loaded_model_dru.load_weights(self.directory + "model-bw_dru.h5")

            self.json_file_tkdi = open(self.directory + "model-bw_tkdi.json", "r")
            self.model_json_tkdi = self.json_file_tkdi.read()
            self.json_file_tkdi.close()
            self.loaded_model_tkdi = model_from_json(self.model_json_tkdi)
            self.loaded_model_tkdi.load_weights(self.directory + "model-bw_tkdi.h5")

            self.json_file_smn = open(self.directory + "model-bw_smn.json", "r")
            self.model_json_smn = self.json_file_smn.read()
            self.json_file_smn.close()
            self.loaded_model_smn = model_from_json(self.model_json_smn)
            self.loaded_model_smn.load_weights(self.directory + "model-bw_smn.h5")
            print("Loaded models from disk")
        except Exception as e:
            print(f"Error loading models: {e}")

        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0
        for i in ascii_uppercase:
            self.ct[i] = 0

        # Initialize the UI
        self.root = tk.Tk()
        self.root.title("Sign Language to Text Converter")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("900x1100")

        # Create visualization frame
        self.panel = tk.Label(self.root)
        self.panel.place(x=135, y=10, width=640, height=640)

        # Processed data visualization
        self.panel2 = tk.Label(self.root)
        self.panel2.place(x=460, y=95, width=310, height=310)

        # Application title
        self.T = tk.Label(self.root)
        self.T.place(x=31, y=17)
        self.T.config(text="Sign Language to Text", font=("courier", 40, "bold"))

        # Current symbol display
        self.panel3 = tk.Label(self.root)
        self.panel3.place(x=500, y=640)
        self.T1 = tk.Label(self.root)
        self.T1.place(x=10, y=640)
        self.T1.config(text="Character:", font=("Courier", 40, "bold"))

        # Word display
        self.panel4 = tk.Label(self.root)
        self.panel4.place(x=220, y=700)
        self.T2 = tk.Label(self.root)
        self.T2.place(x=10, y=700)
        self.T2.config(text="Word:", font=("Courier", 40, "bold"))

        # Sentence display
        self.panel5 = tk.Label(self.root)
        self.panel5.place(x=350, y=760)
        self.T3 = tk.Label(self.root)
        self.T3.place(x=10, y=760)
        self.T3.config(text="Sentence:", font=("Courier", 40, "bold"))

        # Suggestions
        self.T4 = tk.Label(self.root)
        self.T4.place(x=250, y=820)
        self.T4.config(text="Suggestions", fg="red", font=("Courier", 40, "bold"))

        # About button
        self.btcall = tk.Button(self.root, command=self.action_call, height=0, width=0)
        self.btcall.config(text="About", font=("Courier", 14))
        self.btcall.place(x=825, y=0)

        # Suggestion buttons
        self.bt1 = tk.Button(self.root, command=self.action1, height=0, width=0)
        self.bt1.place(x=26, y=890)

        self.bt2 = tk.Button(self.root, command=self.action2, height=0, width=0)
        self.bt2.place(x=325, y=890)

        self.bt3 = tk.Button(self.root, command=self.action3, height=0, width=0)
        self.bt3.place(x=625, y=890)

        self.bt4 = tk.Button(self.root, command=self.action4, height=0, width=0)
        self.bt4.place(x=125, y=950)

        self.bt5 = tk.Button(self.root, command=self.action5, height=0, width=0)
        self.bt5.place(x=425, y=950)

        # Add Clear button
        self.clear_button = tk.Button(self.root, command=self.clear_text, height=2, width=10)
        self.clear_button.config(text="Clear", font=("Courier", 14))
        self.clear_button.place(x=700, y=950)

        # Initialize variables
        self.str = ""
        self.word = ""
        self.current_symbol = "Empty"
        self.photo = "Empty"
        self.sensor_data = [0, 0, 0, 0]  # Placeholder for flex sensor values

        # Create empty visualization image
        self.create_empty_visualization()

        # Start the main loop
        self.data_loop()

    def create_empty_visualization(self):
        """Create an empty visualization for the flex sensors"""
        # Create a blank white image
        width, height = 640, 640
        img = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Draw rectangle for ROI
        cv2.rectangle(img, (220 - 1, 9), (620 + 1, 419), (255, 0, 0), 1)

        # Convert to PIL format
        self.current_image = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=self.current_image)
        self.panel.imgtk = imgtk
        self.panel.config(image=imgtk)

    def read_sensors(self):
        """Read sensor values from Arduino"""
        if self.arduino is None:
            return False, [0, 0, 0, 0]

        try:
            self.arduino.write(b'R')  # Send request for sensor data
            time.sleep(0.1)
            if self.arduino.in_waiting:
                line = self.arduino.readline().decode('utf-8').strip()
                values = [int(val) for val in line.split(',')]
                if len(values) == 4:
                    return True, values
        except Exception as e:
            print(f"Error reading sensor data: {e}")

        return False, [0, 0, 0, 0]

    def visualize_flex_data(self, sensor_values):
        """Create visualization for flex sensor data"""
        width, height = 640, 640
        img = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Draw sensor bars
        bar_width = 50
        bar_spacing = 20
        max_bar_height = 300
        start_x = 100

        for i, value in enumerate(sensor_values):
            # Normalize value between 0-100 (assuming Arduino analog values 0-1023)
            norm_value = value / 1023.0
            height = int(norm_value * max_bar_height)

            # Draw bar
            x = start_x + i * (bar_width + bar_spacing)
            cv2.rectangle(img,
                          (x, 400 - height),
                          (x + bar_width, 400),
                          (0, 0, 255), -1)

            # Add value text
            cv2.putText(img, f"{value}",
                        (x, 420),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

            # Add sensor label
            cv2.putText(img, f"Flex {i + 1}",
                        (x, 440),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

        # Draw ROI box for visualization
        cv2.rectangle(img, (220 - 1, 9), (620 + 1, 419), (255, 0, 0), 1)

        return img

    def create_processed_image(self, sensor_values):
        """Create a binary image representation from sensor values for model input"""
        # This is a simplified conversion - you'll need to adjust based on your actual data
        width, height = 300, 300
        img = np.ones((height, width), dtype=np.uint8) * 255

        # Create a visual representation of the sensor data as a binary image
        bar_width = 50
        bar_spacing = 10
        max_bar_height = 250
        start_x = 20

        for i, value in enumerate(sensor_values):
            # Normalize value between 0-100
            norm_value = value / 1023.0
            bar_height = int(norm_value * max_bar_height)

            # Draw black bar representing sensor value
            x = start_x + i * (bar_width + bar_spacing)
            cv2.rectangle(img,
                          (x, height - 30 - bar_height),
                          (x + bar_width, height - 30),
                          0, -1)  # Black filled rectangle

        # Apply some processing to make it similar to the webcam processed images
        blur = cv2.GaussianBlur(img, (5, 5), 2)
        _, binary = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV)

        return binary

    def data_loop(self):
        """Main loop to read and process sensor data"""
        ok, sensor_values = self.read_sensors()

        if ok:
            self.sensor_data = sensor_values

            # Create visualization
            vis_frame = self.visualize_flex_data(sensor_values)

            # Convert to PIL format for display
            self.current_image = Image.fromarray(vis_frame)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            # Create processed binary image
            processed_img = self.create_processed_image(sensor_values)

            # Display processed image
            self.current_image2 = Image.fromarray(processed_img)
            imgtk2 = ImageTk.PhotoImage(image=self.current_image2)
            self.panel2.imgtk = imgtk2
            self.panel2.config(image=imgtk2)

            # Predict the symbol
            self.predict(processed_img)

            # Update UI elements
            self.panel3.config(text=self.current_symbol, font=("Courier", 50))
            self.panel4.config(text=self.word, font=("Courier", 40))
            self.panel5.config(text=self.str, font=("Courier", 40))

            # Update suggestions if hunspell is available
            if self.hs:
                predicts = self.hs.suggest(self.word)

                if len(predicts) > 0:
                    self.bt1.config(text=predicts[0], font=("Courier", 20))
                else:
                    self.bt1.config(text="")

                if len(predicts) > 1:
                    self.bt2.config(text=predicts[1], font=("Courier", 20))
                else:
                    self.bt2.config(text="")

                if len(predicts) > 2:
                    self.bt3.config(text=predicts[2], font=("Courier", 20))
                else:
                    self.bt3.config(text="")

                if len(predicts) > 3:
                    self.bt4.config(text=predicts[3], font=("Courier", 20))
                else:
                    self.bt4.config(text="")

                if len(predicts) > 4:
                    self.bt5.config(text=predicts[4], font=("Courier", 20))
                else:
                    self.bt5.config(text="")

        # Call this method again after 30ms
        self.root.after(30, self.data_loop)

    def predict(self, test_image):
        """Predict the sign from the processed image"""
        test_image = cv2.resize(test_image, (128, 128))

        try:
            result = self.loaded_model.predict(test_image.reshape(1, 128, 128, 1))
            result_dru = self.loaded_model_dru.predict(test_image.reshape(1, 128, 128, 1))
            result_tkdi = self.loaded_model_tkdi.predict(test_image.reshape(1, 128, 128, 1))
            result_smn = self.loaded_model_smn.predict(test_image.reshape(1, 128, 128, 1))

            prediction = {}
            prediction['blank'] = result[0][0]
            inde = 1
            for i in ascii_uppercase:
                prediction[i] = result[0][inde]
                inde += 1

            # LAYER 1
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = prediction[0][0]

            # LAYER 2
            if self.current_symbol == 'D' or self.current_symbol == 'R' or self.current_symbol == 'U':
                prediction = {}
                prediction['D'] = result_dru[0][0]
                prediction['R'] = result_dru[0][1]
                prediction['U'] = result_dru[0][2]
                prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
                self.current_symbol = prediction[0][0]

            if self.current_symbol == 'D' or self.current_symbol == 'I' or self.current_symbol == 'K' or self.current_symbol == 'T':
                prediction = {}
                prediction['D'] = result_tkdi[0][0]
                prediction['I'] = result_tkdi[0][1]
                prediction['K'] = result_tkdi[0][2]
                prediction['T'] = result_tkdi[0][3]
                prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
                self.current_symbol = prediction[0][0]

            if self.current_symbol == 'M' or self.current_symbol == 'N' or self.current_symbol == 'S':
                prediction1 = {}
                prediction1['M'] = result_smn[0][0]
                prediction1['N'] = result_smn[0][1]
                prediction1['S'] = result_smn[0][2]
                prediction1 = sorted(prediction1.items(), key=operator.itemgetter(1), reverse=True)
                if prediction1[0][0] == 'S':
                    self.current_symbol = prediction1[0][0]
                else:
                    self.current_symbol = prediction[0][0]

            # Handle the blank symbol and counter logic
            if self.current_symbol == 'blank':
                for i in ascii_uppercase:
                    self.ct[i] = 0

            self.ct[self.current_symbol] += 1

            if self.ct[self.current_symbol] > 60:
                for i in ascii_uppercase:
                    if i == self.current_symbol:
                        continue
                    tmp = self.ct[self.current_symbol] - self.ct[i]
                    if tmp < 0:
                        tmp *= -1
                    if tmp <= 20:
                        self.ct['blank'] = 0
                        for i in ascii_uppercase:
                            self.ct[i] = 0
                        return

                self.ct['blank'] = 0
                for i in ascii_uppercase:
                    self.ct[i] = 0

                if self.current_symbol == 'blank':
                    if self.blank_flag == 0:
                        self.blank_flag = 1
                        if len(self.str) > 0:
                            self.str += " "
                        self.str += self.word
                        self.word = ""
                else:
                    if len(self.str) > 16:
                        self.str = ""
                    self.blank_flag = 0
                    self.word += self.current_symbol
        except Exception as e:
            print(f"Error in prediction: {e}")

    def clear_text(self):
        """Clear the current text"""
        self.str = ""
        self.word = ""
        self.current_symbol = "Empty"

        # Update UI
        self.panel3.config(text=self.current_symbol)
        self.panel4.config(text=self.word)
        self.panel5.config(text=self.str)

    def action1(self):
        """Select first suggestion"""
        if self.hs:
            predicts = self.hs.suggest(self.word)
            if len(predicts) > 0:
                self.word = ""
                self.str += " "
                self.str += predicts[0]

    def action2(self):
        """Select second suggestion"""
        if self.hs:
            predicts = self.hs.suggest(self.word)
            if len(predicts) > 1:
                self.word = ""
                self.str += " "
                self.str += predicts[1]

    def action3(self):
        """Select third suggestion"""
        if self.hs:
            predicts = self.hs.suggest(self.word)
            if len(predicts) > 2:
                self.word = ""
                self.str += " "
                self.str += predicts[2]

    def action4(self):
        """Select fourth suggestion"""
        if self.hs:
            predicts = self.hs.suggest(self.word)
            if len(predicts) > 3:
                self.word = ""
                self.str += " "
                self.str += predicts[3]

    def action5(self):
        """Select fifth suggestion"""
        if self.hs:
            predicts = self.hs.suggest(self.word)
            if len(predicts) > 4:
                self.word = ""
                self.str += " "
                self.str += predicts[4]

    def destructor(self):
        """Clean up resources when closing the application"""
        print("Closing Application...")
        if self.arduino:
            self.arduino.close()
        self.root.destroy()
        cv2.destroyAllWindows()

    def destructor1(self):
        """Close the About window"""
        print("Closing About Window...")
        self.root1.destroy()

    def action_call(self):
        """Show the About window"""
        self.root1 = tk.Toplevel(self.root)
        self.root1.title("About")
        self.root1.protocol('WM_DELETE_WINDOW', self.destructor1)
        self.root1.geometry("900x900")

        self.tx = tk.Label(self.root1)
        self.tx.place(x=330, y=20)
        self.tx.config(text="Flex Sensor Sign Language Recognition", fg="red", font=("Courier", 20, "bold"))

        # Additional information can be added here
        self.tx_info = tk.Label(self.root1)
        self.tx_info.place(x=50, y=100)
        self.tx_info.config(text="This application uses 4 flex sensors connected to an Arduino\n"
                                 "to recognize sign language gestures and convert them to text.",
                            font=("Courier", 15))


# Start the application
if __name__ == "__main__":
    print("Starting Application...")
    app = Application()
    app.root.mainloop()