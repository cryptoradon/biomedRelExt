from bioc import pubtator
import spacy
import pickle
import copy
import os

# A class to represent sentences within the document
class Sentence:
    def __init__(self, text, start, end, annotations=None):
        self.text = text                                        # Text content of the sentence
        self.start = start                                      # Starting index of the sentence within the document
        self.end = end                                          # Ending index of the sentence within the document
        self.annotations = annotations if annotations else []   # List of annotations (chemicals) within the sentence

    def addAnnotation(self, annotation):
        self.annotations.append(annotation)

# A class for chemical-disease pairs with context and metadata
class Pair:
    def __init__(self, chemical1, chemical2, context, pmid, pairType, groundTruthStart, groundTruthEnd):
        self.chemical1 = chemical1                              # Chemical 1
        self.chemical2 = chemical2                              # Chemical 2
        self.context = context                                  # Context in which the pair occurs, with masking applied
        self.pmid = pmid                                        # PubMed ID of the article
        self.pairType = pairType                                # Specifies whether the pair is intra-sentential or inter-sentential
        self.groundTruthStart = groundTruthStart                # Starting position of the disease that we want to predict, it serves as the label of the datapoint
        self.groundTruthEnd = groundTruthEnd                    # Ending position of the disease that we want to predict, it serves as the label of the datapoint

    # Representation of a Pair object when printed
    def __repr__(self):
        return f"Pair(chemical1={self.chemical1.text}, chemical2={self.chemical2.text}, context=[{self.context[:30]}...], pmid={self.pmid}, pair_type={self.pairType})"

# Function to mask all chemicals in the context except for the target chemical
def maskOtherChemicalsInContext(context, targetChemicalAnnotation1, targetChemicalAnnotation2, allAnnotations):
    MASK_TOKEN = "[***]"                    # Token used to replace masked diseases
    maskedContext = context                 # Copy of the context to apply masking to
    annotationString = ""                   # String to store formatted annotations

    # Keep track of adjustments in indexing due to masking
    offset = 0
    for ann in allAnnotations:
        # Apply masking to chemicals that are not the target
        if ann.type == 'ChemMet' and ann.start != targetChemicalAnnotation1.start and ann.end != targetChemicalAnnotation1.end and ann.start != targetChemicalAnnotation2.start and ann.end != targetChemicalAnnotation2.end:
            maskedContext = maskedContext[:ann.start - offset] + MASK_TOKEN + maskedContext[ann.end - offset:]
            offset += len(ann.text) - len(MASK_TOKEN)
        else:
            # Accumulate annotations for inclusion in the context
            annotationString += f"{ann.pmid} {ann.start} {ann.end} {ann.text} {ann.type} {ann.id}\n"

    # Return the masked context concatenated with the annotation string
    return maskedContext + "\n" + annotationString

# Function to create chemical-disease pairs from a list of documents
def instanceConstruction(docs):
    pairs = []                              # List to store all the pairs
    nlp = spacy.load("en_core_sci_scibert") # Load a SpaCy model specialized in biomedical text

    for doc in docs[:5]:
        pairsInDoc = []                     # Pairs in specific doc
        relationsInDoc = []                 # Relations in specific doc
        annotations = doc.annotations       # Extract annotations from the document
        text = doc.text                     # Document text
        pmid = doc.pmid                     # PubMed ID of the document

        sentences = []                      # List to store Sentence objects
        sentenceTexts = nlp(text)           # Use SpaCy to tokenize the document into sentences

        # Fill the relations list
        for relation in doc.relations:
            relationsInDoc.append((relation.id1, relation.id2))

        start = 0
        # Fill the sentences list
        for senText in sentenceTexts.sents:
            if len(senText.text) <= 1:
                continue
            start = text.find(senText.text, start)
            end = start + len(senText.text)
            sentences.append(Sentence(senText.text, start, end))
            start = end

        # Process annotations for each sentence
        for ann in annotations:
            for sen in sentences:
                if ann.start >= sen.start and ann.end <= sen.end:
                    if ann.id == '-1':
                        continue
                    sen.addAnnotation(ann)


        # Generate intra-sentential pairs
        for sen in sentences:
            chemicals = [ann for ann in sen.annotations if ann.type == 'ChemMet']
            for chem1 in chemicals:
                for chem2 in chemicals:
                    if chem1.id != chem2.id:
                        if (chem1.id, chem2.id) in relationsInDoc:
                            pairsInDoc.append(Pair(chem1, chem2, maskOtherChemicalsInContext(text, chem1, chem2, annotations), pmid, "intra", chem2.start, chem2.end))
                        else:
                            pairsInDoc.append(Pair(chem1, chem2, maskOtherChemicalsInContext(text, chem1, chem2, annotations), pmid, "intra", 0, 0))

        # Initialize the start and end indices for a window of sentences
        start = 0
        end = min(3, len(sentences) - 1)    # Window size limited to 3 or fewer sentences
        
        # Generate a list of annotations for sentences within the window
        annsIn3Sentences = [[ann for ann in sen.annotations] for sen in sentences[start:end + 1]]

        # Iterate through sentences to find intersentential pairs
        for sen in sentences[:-1]:
            for j in range(1, end - start + 1):
                # Extract chemicals from the current sentence and chemicals from the next within the window
                currentChemicals = [ann for ann in annsIn3Sentences[0] if ann.type == 'ChemMet']
                nextChemicals = [ann for ann in annsIn3Sentences[j] if ann.type == 'ChemMet']

                # For each chemical-chemical pair found, create a Pair object and add to the pairs list.
                for chem1 in currentChemicals:
                    for chem2 in nextChemicals:
                        if chem1.id != chem2.id:
                            if (chem1.id, chem2.id) in relationsInDoc:
                                pairsInDoc.append(Pair(chem1, chem2, maskOtherChemicalsInContext(text, chem1, chem2, annotations), pmid, "inter", chem2.start, chem2.end))
                            else:
                                pairsInDoc.append(Pair(chem1, chem2, maskOtherChemicalsInContext(text, chem1, chem2, annotations), pmid, "inter", 0, 0))

            # Slide the window one sentence forward.
            start += 1
            annsIn3Sentences = annsIn3Sentences[1:]
            if end < len(sentences) - 1:
                end += 1
                annsIn3Sentences.append(sentences[end].annotations)
        
        pairs.append(pairsInDoc)

    return pairs


if __name__ == '__main__':
    # Load the document data from a file
    with open('./CHR_Data/CHR_corpus/train.pubtator', 'r', encoding='utf-8') as fp:
        docs = pubtator.load(fp)

    # Create pairs from the loaded documents
    pairs = instanceConstruction(docs)

    # The directory that the code will store its outputs in
    output_dir = './Preprocessed/CHRTraining'
    
    # Ensure that the directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open a file to write the output
    output_path = os.path.join(output_dir, 'pairs1.pkl')
    with open(output_path, 'wb') as outputFile:
        pickle.dump(pairs, outputFile)