import csv
from collections import Counter

####
# This script counts the number of occurrences of each label in our preprocessed hand landmarks data.
# It can be useful to check if the dataset is balanced or not.
####

def count_labels(csv_file_path):
    # Use the provided list of letters for indexing
    letters = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L",
        "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W",
        "X", "Y"
    ]
    
    # Initialize a Counter to keep track of label counts
    label_counter = Counter()
    
    # Open and read the CSV file
    with open(csv_file_path, mode='r', newline='') as file:
        csv_reader = csv.reader(file)
        
        # Iterate over each row in the CSV file
        for row in csv_reader:
            if row and row[0] != "label":  # Ensure the row is not empty and not the header "label"
                try:
                    # Convert numeric label to the corresponding letter using the provided list
                    alphabetical_label = letters[int(row[0])]
                    numerical_label = int(row[0])
                    label = f'{alphabetical_label}_{numerical_label}'
                    # Increment the count for this label
                    label_counter[label] += 1
                except (ValueError, IndexError):
                    # Handle the case where the label is not an integer or out of expected range
                    continue
    
    # Print out the counts for each label in alphabetical order (A to Y)
    for label in sorted(label_counter):
        print(f"Label {label}: {label_counter[label]}")

# Example usage
csv_file_path = '../resources/data/hand_landmarks_test.csv'  # switch between train and test in name depending on what you want to count
count_labels(csv_file_path)
