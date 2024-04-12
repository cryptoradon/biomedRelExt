import pickle
import xml.etree.ElementTree as ET
from preprocessCDR1 import Pair
from bioc.pubtator.datastructure import PubTatorAnn

# Function to remove the instances containing annotations that have more specific annotations in the document
def hypernymFiltering(pairs, descriptorMap):
    newPairs = []                           # List to store the result of this step

    for pairsInDoc in pairs:
        eliminatedIDs = set()               # Set to store the ids of annotations already eliminated 
        newPairsInDoc = []                  # List to store new pairs in specific doc
        
        for i, currentPair in enumerate(pairsInDoc):
            # Remove the pair if the chemical or disease is previously eliminated
            if currentPair.chemical.id in eliminatedIDs or currentPair.disease.id in eliminatedIDs:
                continue
            
            eliminated = False
            # Compare with the chemical and diseases found later in the document and remove currentPair if more general 
            # (Replaced the name TreeNumber with Index)
            for tempPair in pairsInDoc[i + 1:]:
                # Check for chemical
                if currentPair.chemical.id != '-1' and tempPair.chemical.id != '-1':
                    for currentIndex in descriptorMap[currentPair.chemical.id]:
                        for tempIndex in descriptorMap[tempPair.chemical.id]:
                            if currentIndex in tempIndex and currentIndex != tempIndex:
                                eliminatedIDs.append(currentPair.chemical.id)
                                eliminated = True
                                # No need to check other temp indices
                                break
                        # No need to check other current indices
                        if eliminated:
                            break
                    # No need to check other temp pairs
                    if eliminated:
                        break
                
                # Same check for disease
                if currentPair.disease.id != '-1' and tempPair.disease.id != '-1':
                    for currentIndex in descriptorMap[currentPair.disease.id]:
                        for tempIndex in descriptorMap[tempPair.disease.id]:
                            if currentIndex in tempIndex and currentIndex != tempIndex:
                                eliminatedIDs.append(currentPair.disease.id)
                                eliminated = True
                                break
                        if eliminated:
                            break
                    if eliminated:
                        break

            # Store new pair if it is not eliminated
            if not eliminated:
                newPairsInDoc.append(currentPair)
        
        # Add to the new list of pairs
        newPairs.append(newPairsInDoc)

    return newPairs

# Function to loads a dictionary mapping DescriptorUI to their respective TreeNumbers from an XML tree
def loadDescriptorMap(meshFile):
    tree = ET.parse(meshFile)
    root = tree.getroot()

    descriptorMap = {}
    for descriptor in root.findall('.//DescriptorRecord'):
        ui = descriptor.find('DescriptorUI').text
        treeNumbers = [tn.text for tn in descriptor.findall('.//TreeNumber')]
        descriptorMap[ui] = treeNumbers
    
    return descriptorMap

if __name__ == '__main__':
    # Load the pairs that were produced by the first stage
    with open('pairs1.pkl', 'rb') as input_file:
        pairs = pickle.load(input_file)
    
    # Load the annotation hierarchy map from the MeSH dataset
    descriptorMap = loadDescriptorMap('./MeSH/desc2024.xml')

    # Create pairs from the loaded documents
    pairs = hypernymFiltering(pairs, descriptorMap)

    # Open a file to write the output
    with open('pairs2.pkl', 'wb') as outputFile:
        pickle.dump(pairs, outputFile)

    # Close the file after writing
    outputFile.close()