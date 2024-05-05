import pickle
from transformers import BertTokenizerFast
from preprocess3 import Input
from torch.utils.data import DataLoader, Dataset
import torch

# Set device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom dataset for efficiently handling input from a single document
class CustomDataset(Dataset):
    def __init__(self, inputs, tokenizer):
        self.inputs = inputs                        # Store input data
        self.tokenizer = tokenizer                  # Store tokenizer

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input = self.inputs[idx]                    # Get input data by index

        # Create the input sequence with [CLS] and [SEP] tokens
        input_sequence = f"[CLS] {input.query} [SEP] {input.context} [SEP]"

        # Tokenize the input sequence
        tokens = self.tokenizer.encode_plus(
            input_sequence,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_offsets_mapping=True,            # Include offset mappings
            return_tensors='pt'
        )

        # Find the starting and ending token indices for the ground truth span
        start_token_index = self.find_token_index(input.groundTruthStart, tokens['offset_mapping'].squeeze(0))
        end_token_index = self.find_token_index(input.groundTruthEnd, tokens['offset_mapping'].squeeze(0))

        return (
            tokens['input_ids'].squeeze(0).to(device),          # Input IDs
            tokens['attention_mask'].squeeze(0).to(device),     # Attention mask
            input.chemicalID,                                   # Chemical ID
            input.diseaseID,                                    # Disease ID
            torch.tensor(start_token_index).to(device),         # Ground truth start token index
            torch.tensor(end_token_index).to(device)            # Ground truth end token index
        )

    # Helper function to find the index of the token that corresponds to the given character position
    def find_token_index(self, char_position, offset_mapping):
        for i, (start, end) in enumerate(offset_mapping):
            if start <= char_position <= end:
                return i
        return -1

# Function to tokenize input data and save the results to a file
def tokenize_inputs_and_save(docs_inputs, tokenizer, output_file_path, batch_size=16):
    i = 0
    for inputs in docs_inputs:
        dataset = CustomDataset(inputs, tokenizer)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Write the tokenized data to the output file
        with open(output_file_path, 'ab') as outputFile:
            for input_ids, attention_masks, chemicalIDs, diseaseIDs, groundTruthStarts, groundTruthEnds in data_loader:
                # Move tensors to CPU before saving them to pickle
                input_ids = input_ids.cpu()
                attention_masks = attention_masks.cpu()
                groundTruthStarts = groundTruthStarts.cpu()
                groundTruthEnds = groundTruthEnds.cpu()
                pickle.dump((input_ids, attention_masks, chemicalIDs, diseaseIDs, groundTruthStarts, groundTruthEnds), outputFile)
        print("Input #" + str(i) + " done.")
        i += 1

if __name__ == '__main__':
    # Load the inputs that were produced by the third stage
    with open('./Preprocessed/CDRTest/input.pkl', 'rb') as inputFile:
        docs_inputs = pickle.load(inputFile)

    # Initialize tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('dmis-lab/biobert-base-cased-v1.2')

    # Clear the output file first to avoid appending to an old file
    with open('./Preprocessed/CDRTest/tokenizedInputs.pkl', 'wb') as outputFile:
        pass

    # Tokenize inputs and save, maintaining the 2D list structure
    tokenize_inputs_and_save(docs_inputs, tokenizer, './Preprocessed/CDRTest/tokenizedInputs.pkl')
