import cv2
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
import os
from word_generator import load_words, generate_words

# Sign Language Alphabet images source: https://www.flaticon.com/search?author_id=1686&style_id=&type=standard&word=sign+language Letter icons created by Valeria - Flaticon
images_directory = "../resources/help-images/"

word_list = load_words('../resources/words.txt')
random_words = generate_words(175, word_list)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def process_gesture(hand_subimage):
    # Process the hand_subimage with the CNN and return the recognized letter as a string!
    # recognized_gesture = cnn.process(hand_subimage)
    recognized_gesture = "recognized gesture letter" # hardcoded now for testing purposes until CNN is implemented
    return recognized_gesture

def get_current_help_img(current_letter):
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

def get_quadratic_bbox_coordinates_with_padding(handLandmark, image_shape, padding=30):
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

def extract_and_preprocess_hand_subimage(image, bbox, size=(28, 28)):
    xmin, ymin, xmax, ymax = bbox
    hand_subimage = image[ymin:ymax, xmin:xmax]
    hand_subimage = cv2.resize(hand_subimage, size)
    hand_subimage = cv2.cvtColor(hand_subimage, cv2.COLOR_BGR2GRAY)
    # We need to flatten the image into a 1D array
    hand_subimage = hand_subimage.flatten()
    # we also need to transpose it to get it in the right format before passing it to the CNN
    preprocessed_hand_subimage = hand_subimage.T

    return preprocessed_hand_subimage

class HandTrackingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Spelling Bee")

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

        self.current_word_index = 0
        self.current_word = random_words[self.current_word_index]
        self.current_word_label = tk.Label(root, text=self.current_word, font=("Helvetica", 50), fg="white", bg="black", borderwidth=0)
        self.canvas.create_window(self.screen_width // 2, self.screen_height - 250, window=self.current_word_label)

        self.help_img = get_current_help_img(self.current_word[0])
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

                    hand_subimage = extract_and_preprocess_hand_subimage(frame, bbox)
                    # Pass the hand_subimage to the CNN to detect the sign language gesture!
                    recognized_gesture = process_gesture(hand_subimage)
                    self.process_recognized_gesture(recognized_gesture)

                    # Draw bounding box on the frame
                    if self.verbose:
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
        # TODO: Implement similar to process_key_input!
        print(f"Processing: '{recognized_gesture}'")

    def process_key_input(self, event):
        if event.char.isalpha() and self.current_word_letter_index < len(self.current_word):
            current_letter = self.current_word[self.current_word_letter_index]

            # Check if the input letter matches the corresponding letter in the current word
            if event.char.upper() == current_letter.upper():
                # Replace the underscore with the correct letter in the placeholder
                placeholder_text = self.current_word_placeholder.cget("text")
                new_placeholder_text = placeholder_text[:self.current_word_letter_index] + current_letter + placeholder_text[self.current_word_letter_index + 2:]
                self.current_word_placeholder.config(text=new_placeholder_text)

                # Move to the next index in the word
                self.current_word_letter_index += 1
                if self.current_word_letter_index < len(self.current_word):
                    next_letter = self.current_word[self.current_word_letter_index]
                    next_help_img = get_current_help_img(next_letter)
                    next_help_img = cv2.resize(next_help_img, (400, 400))

                    # Convert the next help image to a format suitable for Tkinter
                    next_help_img_rgb = cv2.cvtColor(next_help_img, cv2.COLOR_BGR2RGB)
                    next_help_img_pil = Image.fromarray(next_help_img_rgb)
                    next_help_img_tk = ImageTk.PhotoImage(image=next_help_img_pil)

                    # Update the help label with the new image
                    self.help_label.config(image=next_help_img_tk)
                    self.help_label.image = next_help_img_tk

                # Update last input label
                self.last_input_label.config(text=current_letter, fg="green")
            else:
                # Update last input label
                self.last_input_label.config(text=event.char.upper(), fg="red")

            # Check if the word is completed
            if self.current_word_letter_index == len(self.current_word):
                self.last_input_label.config(text="")
                self.current_word_letter_index = 0  # Reset index for the next word
                self.set_new_current_word()

    def set_new_current_word(self):
        # Update to the next word in the list
        self.current_word_index += 1
        if self.current_word_index < len(random_words):
            self.current_word = random_words[self.current_word_index]
            self.current_word_label.config(text=self.current_word)
            self.placeholder = self.create_placeholder()
            self.current_word_placeholder.config(text=self.placeholder)

            # Update help image for the first letter of the new word
            first_letter = self.current_word[0]
            first_help_img = get_current_help_img(first_letter)
            first_help_img = cv2.resize(first_help_img, (400, 400))

            # Convert the first help image to a format suitable for Tkinter
            first_help_img_rgb = cv2.cvtColor(first_help_img, cv2.COLOR_BGR2RGB)
            first_help_img_pil = Image.fromarray(first_help_img_rgb)
            first_help_img_tk = ImageTk.PhotoImage(image=first_help_img_pil)

            # Update the help label with the new image
            self.help_label.config(image=first_help_img_tk)
            self.help_label.image = first_help_img_tk

    def create_placeholder(self):
        word_text = self.current_word
        placeholder = ""
        for i, char in enumerate(word_text):
            if i == 0:
                placeholder += "_"
            else:
                placeholder += " _"
        return placeholder

if __name__ == "__main__":
    root = tk.Tk()
    app = HandTrackingApp(root)
    root.mainloop()
