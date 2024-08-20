import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
import os
import numpy as np
import joblib

model_name = 'hand_landmark_model_1'
word_list_name = "alphabet"

def get_quadratic_bbox_coordinates_with_padding(handLandmark, image_shape, padding=15):
    all_x, all_y = [], []
    for hnd in mp_hands.HandLandmark:
        all_x.append(int(handLandmark.landmark[hnd].x * image_shape[1]))
        all_y.append(int(handLandmark.landmark[hnd].y * image_shape[0]))

    xmin, ymin, xmax, ymax = min(all_x), min(all_y), max(all_x), max(all_y)
    width, height = xmax - xmin, ymax - ymin
    side_length = max(width, height)
    center_x, center_y = xmin + width // 2, ymin + height // 2
    new_xmin = max(center_x - side_length // 2 - padding, 0)
    new_ymin = max(center_y - side_length // 2 - padding, 0)
    new_xmax = min(center_x + side_length // 2 + padding, image_shape[1])
    new_ymax = min(center_y + side_length // 2 + padding, image_shape[0])
    return new_xmin, new_ymin, new_xmax, new_ymax

# Define the neural network model (MLP)
class HandGestureMLP(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_classes=24):
        super(HandGestureMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the trained model
model = HandGestureMLP(num_classes=24)
model.load_state_dict(torch.load(f'../resources/models/{model_name}.pth', map_location=torch.device('cpu')))
model.eval()

# Load the LabelEncoder
label_encoder = joblib.load(f'../resources/models/label_encoder_{model_name}.pkl')

# Define the gestures list based on LabelEncoder classes
gestures = label_encoder.classes_.tolist()

# Define the gestures
letters = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S','T',
    'U', 'V', 'W', 'X', 'Y'
]

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def process_gesture(hand_landmarks):
    # Flatten hand_landmarks and convert to torch tensor
    hand_landmarks = torch.tensor(hand_landmarks.flatten(), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(hand_landmarks)
        probabilities = F.softmax(logits, dim=1)
        predicted_class = probabilities.argmax(dim=1).item()
    letter_index = label_encoder.inverse_transform([predicted_class])[0]
    return letters[letter_index]

class HandTrackingApp:
    def __init__(self, root):
        self.canProcess = True
        self.root = root
        self.root.title("Sign Language Spelling Bee")
        self.count = 0

        # Set the initial window state
        self.is_fullscreen = False
        self.toggle_fullscreen()

        # Bind keys
        self.root.bind('<Escape>', self.toggle_fullscreen)
        self.root.bind('z', self.quit_app)
        self.root.bind('j', self.toggle_verbose)

        self.verbose = False

        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.target_width = self.screen_width // 2
        self.target_height = self.screen_height // 2

        self.canvas = tk.Canvas(root, width=self.screen_width, height=self.screen_height, bg="black")
        self.canvas.pack()

        # Load words from a text file (modify this to your path)
        self.word_list = self.load_words(f'../resources/{word_list_name}.txt')
        self.current_word_index = 0
        self.current_word = self.word_list[self.current_word_index]
        self.current_word_label = tk.Label(root, text=self.current_word, font=("Helvetica", 50), fg="white", bg="black", borderwidth=0)
        self.canvas.create_window(self.screen_width // 2, self.screen_height - 250, window=self.current_word_label)

        # Load help image for the first letter of the word
        self.help_img = self.get_current_help_img(self.current_word[0])
        self.help_img = cv2.resize(self.help_img, (400, 400))

        self.placeholder = self.create_placeholder()
        self.current_word_placeholder = tk.Label(root, text=self.placeholder, font=("Helvetica", 50), fg="white", bg="black", borderwidth=0)
        self.canvas.create_window(self.screen_width // 2, self.screen_height - 100, window=self.current_word_placeholder)

        self.last_input_label = tk.Label(root, font=("Helvetica", 50), fg="white", bg="black", borderwidth=0)
        self.canvas.create_window(self.screen_width // 2, self.screen_height - 175, window=self.last_input_label)

        self.cap = cv2.VideoCapture(0)
        self.image_label = tk.Label(root, borderwidth=0)
        self.canvas.create_window(0, self.screen_height - self.target_height, window=self.image_label, anchor='sw')

        # Convert help_img to a format suitable for Tkinter
        help_img_rgb = cv2.cvtColor(self.help_img, cv2.COLOR_BGR2RGB)
        help_img_pil = Image.fromarray(help_img_rgb)
        help_img_tk = ImageTk.PhotoImage(image=help_img_pil)

        self.help_label = tk.Label(root, image=help_img_tk, borderwidth=0)
        self.help_label.image = help_img_tk

        # Create the window with adjusted coordinates
        self.canvas.create_window(self.screen_width - 150, self.screen_height - self.target_height + 20, window=self.help_label, anchor='se')

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5)
        
        self.current_word_letter_index = 0  # Variable to keep track of the current index in the word
        self.root.bind('<Key>', self.process_key_input)  # Bind keyboard events to process user input
        
        self.update()

    def load_words(self, filename):
        with open(filename, 'r') as file:
            return file.read().splitlines()

    def get_current_help_img(self, current_letter):
        # Modify the path to the help images directory as needed
        images_directory = "../resources/help-images/"
        image_path = os.path.join(images_directory, "letter-" + current_letter + ".png")

        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                return img
            else:
                print("Failed to read the image.")
                return None
        else:
            print("Image not found for letter", current_letter)
            return None 

    def update(self):
        success, frame = self.cap.read()
        if success:
            # Flip the frame image horizontally
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (self.target_width, self.target_height))
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    if self.verbose:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                                  mp_drawing_styles.get_default_hand_landmarks_style(),
                                                  mp_drawing_styles.get_default_hand_connections_style())
                    bbox = get_quadratic_bbox_coordinates_with_padding(hand_landmarks, frame.shape[:2])
                    
                    # Convert hand landmarks to a numpy array for prediction
                    hand_landmarks_array = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark])
                    
                    if self.canProcess:
                        recognized_gesture = process_gesture(hand_landmarks_array)
                        self.process_recognized_gesture(recognized_gesture)

                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.image_label.config(image=imgtk)
            self.image_label.image = imgtk

        self.root.after(10, self.update)

    def toggle_fullscreen(self, event=None):
        self.is_fullscreen = not self.is_fullscreen
        if self.is_fullscreen:
            self.root.attributes('-fullscreen', True)
        else:
            self.root.attributes('-fullscreen', False)
            self.root.geometry('1280x720')

    def quit_app(self, event=None):
        self.root.destroy()
    
    def toggle_verbose(self, event=None):
        self.verbose = not self.verbose 

    def process_recognized_gesture(self, recognized_gesture):
        if recognized_gesture.isalpha() and self.current_word_letter_index < len(self.current_word):
            current_letter = self.current_word[self.current_word_letter_index]

            if recognized_gesture.upper() == current_letter.upper():
                placeholder_text = self.current_word_placeholder.cget("text")
                new_placeholder_text = placeholder_text[:self.current_word_letter_index] + current_letter + placeholder_text[self.current_word_letter_index + 2:]
                self.current_word_placeholder.config(text=new_placeholder_text)

                self.last_input_label.config(text=current_letter, fg="green")
                self.canProcess = False
                self.root.after(2000, self.clear_last_input_and_proceed)

            else:
                self.last_input_label.config(text=recognized_gesture.upper(), fg="red")

    def clear_last_input_and_proceed(self):
        self.last_input_label.config(text="", bg="black")
        self.current_word_letter_index += 1
        if self.current_word_letter_index < len(self.current_word):
            next_letter = self.current_word[self.current_word_letter_index]
            next_help_img = self.get_current_help_img(next_letter)
            next_help_img = cv2.resize(next_help_img, (400, 400))

            next_help_img_rgb = cv2.cvtColor(next_help_img, cv2.COLOR_BGR2RGB)
            next_help_img_pil = Image.fromarray(next_help_img_rgb)
            next_help_img_tk = ImageTk.PhotoImage(image=next_help_img_pil)

            self.help_label.config(image=next_help_img_tk)
            self.help_label.image = next_help_img_tk
        else:
            self.current_word_index += 1
            if self.current_word_index < len(self.word_list):
                self.current_word = self.word_list[self.current_word_index]
                self.current_word_placeholder.config(text=self.create_placeholder())
                self.current_word_label.config(text=self.current_word)
                self.current_word_letter_index = 0
            else:
                self.quit_app()

        self.canProcess = True

    def process_key_input(self, event):
        key = event.char.upper()
        if key.isalpha() and self.canProcess:
            self.process_recognized_gesture(key)
        
    def create_placeholder(self):
        return '_ ' * len(self.current_word)

    def on_closing(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = HandTrackingApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
