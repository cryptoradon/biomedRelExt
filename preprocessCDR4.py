import pickle
from transformers import BertTokenizer
from preprocessCDR3 import Input

# Function to tokenize the inputs for the bioBERT model
def tokenizeInputs(inputs):
    inputIDs = []                   # List to store the token IDs 
    attentionMasks = []             # List to store the attention masks

    # Initialize the BioBERT tokenizer for tokenizing the inputs
    tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.2')

    for inputsInDoc in inputs:
        inputIDsInDoc = []          # List to stoe the token IDs in specific doc
        attentionMasksInDoc = []    # List to store the attention masks in specific doc

        for input in inputsInDoc:
            # Construct the input sequence by appending [CLS] and [SEP] tokens
            inputSequence = f"[CLS] {input.query} [SEP] {input.context} [SEP]"
            
            # Tokenize the constructed sequence and generate the corresponding attention masks
            tokens = tokenizer.encode_plus(
                inputSequence, 
                max_length=512, 
                truncation=True, 
                padding='max_length', 
                return_tensors='pt'
            )

            # Append the token IDs and attention masks to their respective lists
            inputIDsInDoc.append(tokens['input_ids'][0])
            attentionMasksInDoc.append(tokens['attention_mask'][0])
        
        inputIDs.append(inputIDsInDoc)
        attentionMasks.append(attentionMasksInDoc)

if __name__ == '__main__':
    # Load the inputs that were produced by the third stage
    with open('./Preprocessed/CDRTraining/input.pkl', 'rb') as inputFile:
        inputs = pickle.load(inputFile)
    
    # Tokenize the inputs
    inputIDs, attentionMasks = tokenizeInputs(inputs)

    # Open a file to write the output
    with open('./Preprocessed/CDRTraining/tokenizedInputs.pkl', 'wb') as outputFile:
        pickle.dump((inputIDs, attentionMasks), outputFile)

    # Close the file after writing
    outputFile.close()