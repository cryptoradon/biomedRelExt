import pickle
from transformers import BertTokenizerFast
from preprocessCDR3 import Input
from torch.utils.data import DataLoader, Dataset
import torch

class CustomDataset(Dataset):
    """ Custom dataset for efficiently handling input from a single document """
    def __init__(self, inputs, tokenizer):
        self.inputs = inputs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input = self.inputs[idx]
        input_sequence = f"[CLS] {input.query} [SEP] {input.context} [SEP]"
        tokens = self.tokenizer.encode_plus(
            input_sequence,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_offsets_mapping=True,  # Include offset mappings
            return_tensors='pt'
        )
        start_token_index = self.find_token_index(input.groundTruthStart, tokens['offset_mapping'].squeeze(0))
        end_token_index = self.find_token_index(input.groundTruthEnd, tokens['offset_mapping'].squeeze(0))
        return (
            tokens['input_ids'].squeeze(0),
            tokens['attention_mask'].squeeze(0),
            input.chemicalID,
            input.diseaseID,
            torch.tensor(start_token_index),
            torch.tensor(end_token_index),
        )
    def find_token_index(self, char_position, offset_mapping):
        for i, (start, end) in enumerate(offset_mapping):
            if start <= char_position <= end:
                return i
        print("Char_position: ", char_position)
        for i, (start, end) in enumerate(offset_mapping):
            print(start, end)
        return -1

def tokenize_inputs_and_save(docs_inputs, tokenizer, batch_size=16, output_file_path='./Preprocessed/CDRTraining/tokenizedInputs.pkl'):
    i = 0
    for inputs in docs_inputs:    
        dataset = CustomDataset(inputs, tokenizer)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        with open(output_file_path, 'ab') as outputFile:
            for input_ids, attention_masks, chemicalIDs, diseaseIDs, groundTruthStarts, groundTruthEnds in data_loader:
                pickle.dump((input_ids, attention_masks, chemicalIDs, diseaseIDs, groundTruthStarts, groundTruthEnds), outputFile)
        print("Input #" + str(i) + " done.")
        i += 1


if __name__ == '__main__':
    # Load the inputs that were produced by the third stage
    with open('./Preprocessed/CDRTraining/input.pkl', 'rb') as inputFile:
        docs_inputs = pickle.load(inputFile)
    
    # Initialize tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('dmis-lab/biobert-base-cased-v1.2')
    
    # Clear the output file first to avoid appending to an old file
    with open('./Preprocessed/CDRTraining/tokenizedInputs.pkl', 'wb') as outputFile:
        pass

    # Tokenize inputs and save, maintaining the 2D list structure
    tokenize_inputs_and_save(docs_inputs, tokenizer)
