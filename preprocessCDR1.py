from bioc import pubtator
import spacy
import pickle

# A class to represent sentences within the document
class Sentence:
    def __init__(self, text, start, end, annotations=None):
        self.text = text                                        # Text content of the sentence
        self.start = start                                      # Starting index of the sentence within the document
        self.end = end                                          # Ending index of the sentence within the document
        self.annotations = annotations if annotations else []   # List of annotations (chemicals and diseases) within the sentence

    def addAnnotation(self, annotation):
        self.annotations.append(annotation)

# A class for chemical-disease pairs with context and metadata
class Pair:
    def __init__(self, chemical, disease, context, pmid, pair_type):
        self.chemical = chemical                                # Chemical
        self.disease = disease                                  # Disease
        self.context = context                                  # Context in which the pair occurs, with masking applied
        self.pmid = pmid                                        # PubMed ID of the article
        self.pair_type = pair_type                              # Specifies whether the pair is intra-sentential or inter-sentential

    # Representation of a Pair object when printed
    def __repr__(self):
        return f"Pair(chemical={self.chemical.text}, disease={self.disease.text}, context=[{self.context[:30]}...], pmid={self.pmid}, pair_type={self.pair_type})"

# Function to mask all diseases in the context except for the target disease
def maskOtherDiseasesInContext(context, targetDiseaseAnnotation, allAnnotations):
    MASK_TOKEN = "[***]"                    # Token used to replace masked diseases
    maskedContext = context                 # Copy of the context to apply masking to
    annotationString = ""                   # String to store formatted annotations

    # Keep track of adjustments in indexing due to masking
    offset = 0
    for ann in allAnnotations:
        # Apply masking to diseases that are not the target
        if ann.type == 'Disease' and ann.start != targetDiseaseAnnotation.start and ann.end != targetDiseaseAnnotation.end:
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

    for doc in docs:
        pairsInDoc = []                     # Pairs in specific doc
        annotations = doc.annotations       # Extract annotations from the document
        text = doc.text                     # Document text
        pmid = doc.pmid                     # PubMed ID of the document

        sentences = []                      # List to store Sentence objects
        sentenceTexts = nlp(text)           # Use SpaCy to tokenize the document into sentences

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
                    sen.addAnnotation(ann)

        # Generate intra-sentential pairs
        for sen in sentences:
            chemicals = [ann for ann in sen.annotations if ann.type == 'Chemical']
            diseases = [ann for ann in sen.annotations if ann.type == 'Disease']
            for chem in chemicals:
                for dis in diseases:
                    pairsInDoc.append(Pair(chem, dis, maskOtherDiseasesInContext(text, dis, annotations), pmid, "intra"))

        # Initialize the start and end indices for a window of sentences
        start = 0
        end = min(3, len(sentences) - 1)    # Window size limited to 3 or fewer sentences
        
        # Generate a list of annotations for sentences within the window
        annsIn3Sentences = [[ann for ann in sen.annotations] for sen in sentences[start:end + 1]]

        # Iterate through sentences to find intersentential pairs
        for sen in sentences[:-1]:
            for j in range(1, end - start + 1):
                # Extract chemicals from the current sentence and diseases from the next within the window
                currentChemicals = [ann for ann in annsIn3Sentences[0] if ann.type == 'Chemical']
                nextDiseases = [ann for ann in annsIn3Sentences[j] if ann.type == 'Disease']

                # For each chemical-disease pair found, create a Pair object and add to the pairs list.
                for chem in currentChemicals:
                    for dis in nextDiseases:
                        context = " ".join(s.text for s in sentences[start:start+j+1])
                        pairsInDoc.append(Pair(chem, dis, maskOtherDiseasesInContext(context, dis, annotations), pmid, "inter"))

                # Repeat the process for diseases in the current sentence and chemicals in the next.
                currentDiseases = [ann for ann in annsIn3Sentences[0] if ann.type == 'Disease']
                nextChemicals = [ann for ann in annsIn3Sentences[j] if ann.type == 'Chemical']
                
                for dis in currentDiseases:
                    for chem in nextChemicals:
                        context = " ".join(s.text for s in sentences[start:start+j+1])
                        pairsInDoc.append(Pair(chem, dis, maskOtherDiseasesInContext(context, chem, annotations), pmid, "inter"))

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
    with open('./CDR_Data/CDR.Corpus.v010516/CDR_TrainingSet.PubTator.txt', 'r') as fp:
        docs = pubtator.load(fp)

    # Create pairs from the loaded documents
    pairs = instanceConstruction(docs)

    # Open a file to write the output
    with open('./Preprocessed/CDRTraining/pairs1.pkl', 'wb') as outputFile:
        pickle.dump(pairs, outputFile)

    # Close the file after writing
    outputFile.close()