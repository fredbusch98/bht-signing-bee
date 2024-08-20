import csv
from collections import Counter

def count_labels(csv_file_path):
    # Initialize a Counter to keep track of label counts
    label_counter = Counter()
    
    # Open and read the CSV file
    with open(csv_file_path, mode='r', newline='') as file:
        csv_reader = csv.reader(file)
        
        # Iterate over each row in the CSV file
        for row in csv_reader:
            if row:  # Ensure the row is not empty
                label = row[0]
                # Increment the count for this label
                label_counter[label] += 1
    
    # Print out the counts for each label
    for label, count in sorted(label_counter.items()):
        print(f"Label {label}: {count}")

# Example usage
csv_file_path = 'hand_landmarks_test.csv'  # switch between train and test in name depending on what you want to count
count_labels(csv_file_path)
