import pandas as pd
from PIL import Image
import numpy as np
import os

# Define file paths
csv_file_path = 'test.csv'
output_directory = '../resources/test-d'

# Create output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Load CSV data into a pandas DataFrame
df = pd.read_csv(csv_file_path)

# Filter rows with label '3'
label = 3
filtered_df = df[df.iloc[:, 0] == label]

# Iterate over filtered rows and save images
for index, row in filtered_df.iterrows():
    # Extract pixel values
    pixels = row[1:].values.astype(np.uint8)
    
    # Reshape to 28x28 and convert to image
    image_array = pixels.reshape((28, 28))
    image = Image.fromarray(image_array, mode='L')  # 'L' mode for grayscale
    
    # Save image to file
    image_file_path = os.path.join(output_directory, f'image_{index}.png')
    image.save(image_file_path)

print(f"Images saved in {output_directory}")
