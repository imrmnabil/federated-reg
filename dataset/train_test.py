import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file
file_path = "Tetuan City power consumption.csv"  # Replace with your CSV file path
data = pd.read_csv(file_path)

# Split into train and test
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Save the train and test sets to separate files
train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)

print("Train and test files created!")