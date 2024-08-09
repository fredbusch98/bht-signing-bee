import pandas as pd
import numpy as np

def augment_brightness(data, offsets):
    augmented_data = []

    for row_idx, row in enumerate(data):
        label = row[0]
        pixels = row[1:]  # Extract pixel values starting from the second column
        
        if len(pixels) == 0:
            print(f"Warning: Row {row_idx} has no pixel data.")
            continue
        
        if len(pixels) != 28 * 28:  # Assuming 28x28 images
            print(f"Warning: Row {row_idx} does not have 784 pixels, but has {len(pixels)} pixels.")
            continue

        pixels = np.array(pixels, dtype=np.int32)  # Ensure pixel values are integers
        augmented_data.append(row.tolist())  # Original row
        
        for offset in offsets:
            new_pixels = np.clip(pixels + offset, 0, 255)
            augmented_row = [label] + new_pixels.tolist()
            augmented_data.append(augmented_row)
        
        # Debugging: Print the original and augmented pixel values for the first few rows
        if row_idx < 5:  # Print only for the first few rows to avoid too much output
            print(f"Row {row_idx}:")
            print(f"Original pixels: {pixels[:10]}")  # Print the first 10 pixels for brevity
            for offset in offsets:
                new_pixels = np.clip(pixels + offset, 0, 255)
                print(f"Offset {offset}: {new_pixels[:10]}")  # Print the first 10 pixels for brevity
    
    return augmented_data

# Load the CSV file
input_file = 'sign_mnist_train.csv'
data = pd.read_csv(input_file, header=None)

# Extract header and data separately
header = data.iloc[0].tolist()
data = data.values[1:]

# Define brightness offsets (example: -60, -40, -20, 20, 40, 60)
brightness_offsets = [-60, -40, -20, 20, 40, 60]

# Augment the data
augmented_data = augment_brightness(data, brightness_offsets)

# Convert augmented data back to DataFrame
augmented_df = pd.DataFrame(augmented_data, columns=header)

# Save the augmented data to a new CSV file
output_file = 'augmented_file.csv'
augmented_df.to_csv(output_file, index=False)

print(f"Data augmentation completed. Augmented data saved to {output_file}.")