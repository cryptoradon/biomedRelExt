import numpy as np

def split_data(file_path, train_ratio=0.995, validation_ratio=0.005):
    # Read the lines from the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Shuffle the data
    np.random.shuffle(lines)
    
    # Calculate split indices
    total_lines = len(lines)
    train_end = int(total_lines * train_ratio)
    validation_end = train_end + int(total_lines * validation_ratio)
    
    # Split the data
    train_data = lines[:train_end]
    validation_data = lines[train_end:validation_end]
    test_data = lines[validation_end:]
    
    # Save the splits to files
    with open('./Knowledge_Representation/tripletsAllTrain.txt', 'w') as f:
        f.writelines(train_data)
    with open('./Knowledge_Representation/tripletsAllValidation.txt', 'w') as f:
        f.writelines(validation_data)
    with open('./Knowledge_Representation/tripletsAllTest.txt', 'w') as f:
        f.writelines(test_data)
    
    return 'Data split into train.txt, validation.txt, and test.txt'

# Example usage
file_path = './Knowledge_Representation/tripletsAll.txt'  # Update this path to where your file is located
split_data(file_path)
