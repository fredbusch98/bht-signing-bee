import cv2
import mediapipe as mp
import os
import csv

####
# This script preprocesses the data coming from those 2 datasets: 
# https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset?select=ASL_Alphabet_Dataset (for the training data) and https://www.kaggle.com/datasets/grassknoted/asl-alphabet?select=asl_alphabet_train (for the test data)
# It tries to detect hand_landmarks in every image of the data set and writes them into a CSV file with the respective letter label as the prefix in the first column.
# Depending on the dataset, the CSV file will be called hand_landmarks_train.csv or hand_landmarks_test.csv. This needs to bet set manually before starting the script.
# NOTE: The data sets need to be downloaded and added to the project locally! They are not added to the git repository because they are multiple gigabytes of size.
####

# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands

# Define the list of letters
letters = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'X', 'Y'
]

missing_landmark_counter = 0
landmark_counter = 0

# Define the path to the folder containing the dataset (switch between train and test in name depending on what you want to generate)
asl_alphabet_train_path = "../resources/data/ASL_Alphabet_Dataset/asl_alphabet_train"

# Define the output CSV file path (switch between train and test in name depending on what you want to generate)
output_csv_path = "../resources/data/hand_landmarks_train.csv"

# Initialize MediaPipe Hands
with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
    
    # Open the CSV file for writing
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header row: label followed by 21 landmarks with x, y, and z coordinates
        header = ['label'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)]
        writer.writerow(header)
        
        # Loop through each subfolder in the main directory
        for letter in letters:
            label = letters.index(letter)  # Get the index of the letter as the label
            letter_folder_path = os.path.join(asl_alphabet_train_path, letter)
            
            # Check if the folder exists
            if not os.path.isdir(letter_folder_path):
                print(f"Folder not found: {letter_folder_path}")
                continue
            
            # Process each image in the subfolder
            for image_name in os.listdir(letter_folder_path):
                image_path = os.path.join(letter_folder_path, image_name)
                
                # Read the image using OpenCV
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Could not read image: {image_path}")
                    continue
                
                # Convert the image to RGB as required by MediaPipe
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Process the image to detect hand landmarks
                results = hands.process(image_rgb)
                
                # If landmarks are detected, extract them and write to the CSV
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Extract the landmarks as a flattened list of x, y, z coordinates
                        landmarks = [landmark for lm in hand_landmarks.landmark for landmark in (lm.x, lm.y, lm.z)]
                        # Write the label and landmarks to the CSV
                        writer.writerow([label] + landmarks)
                        landmark_counter += 1
                else:
                    missing_landmark_counter += 1
                    print(f"No hand landmarks detected in {image_path}")
    
    print(f"Hand landmarks have been written to {output_csv_path}")
    print(f"Total number of collected hand landmarks: {landmark_counter}")
    print(f"Missing hand landmarks: {missing_landmark_counter}")
