import os
import numpy as np
import pandas as pd
from PIL import Image

# Output CSV file
output_csv = 'output.csv'

# Get the current working directory
current_directory = os.getcwd()

# Initialize an empty list to store image data
image_data = []

label = 4

# Process each image in the current directory
for filename in os.listdir(current_directory):
    if filename.endswith('.jpg'):
        # Open the image and convert it to a numpy array
        img_path = os.path.join(current_directory, filename)
        img = Image.open(img_path).convert('L')  # 'L' mode for grayscale
        img_array = np.array(img)
        
        # Flatten the image array and prepend the label
        img_flattened = img_array.flatten()
        img_data = np.insert(img_flattened, 0, label)
        
        # Append the image data to the list
        image_data.append(img_data)

# Convert the list of image data to a pandas DataFrame
df = pd.DataFrame(image_data)

# Save the DataFrame to a CSV file
df.to_csv(output_csv, index=False, header=False)

print(f"CSV file '{output_csv}' created successfully.")
