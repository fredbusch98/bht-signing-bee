# data_collector.py
import os
import numpy as np
import pandas as pd

# Output CSV file
output_csv = 'output.csv'

def save_image_data_to_csv(image, label):
    # Flatten the image array and prepend the label
    img_flattened = image.flatten()
    img_data = np.insert(img_flattened, 0, label)
    
    # Convert the image data to a pandas DataFrame
    df = pd.DataFrame([img_data])

    # Append the DataFrame to the CSV file
    if not os.path.isfile(output_csv):
        df.to_csv(output_csv, index=False, header=False)
    else:
        df.to_csv(output_csv, mode='a', index=False, header=False)

    print(f"Data appended to CSV file '{output_csv}' successfully.")
