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

            # Split the IDs that are represented as "XXXXX|XXXXX"
            currentChemicalIDs = currentPair.chemical.id.split('|')
            currentDiseaseIDs = currentPair.disease.id.split('|')
            for currentChemicalID in currentChemicalIDs:
                for currentDiseaseID in currentDiseaseIDs:
                    # Remove the pair if the chemical or disease is previously eliminated
                    if currentChemicalID in eliminatedIDs or currentDiseaseID in eliminatedIDs:
                        continue
                    
                    eliminated = False
                    # Compare with the chemical and diseases found later in the document and remove currentPair if more general 
                    # (Replaced the name TreeNumber with Index)
                    for tempPair in pairsInDoc[i + 1:]:
                        
                        # Split the IDs that are represented as "XXXXX|XXXXX"
                        tempChemicalIDs = tempPair.chemical.id.split('|')
                        tempDiseaseIDs = tempPair.disease.id.split('|')
                        for tempChemicalID in tempChemicalIDs:
                            for tempDiseaseID in tempDiseaseIDs:
                                # Check for chemical
                                if currentChemicalID in descriptorMap and tempChemicalID in descriptorMap:
                                    for currentIndex in descriptorMap[currentChemicalID]:
                                        for tempIndex in descriptorMap[tempChemicalID]:
                                            if currentIndex in tempIndex and currentIndex != tempIndex:
                                                eliminatedIDs.add(currentChemicalID)
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
                                if currentDiseaseID in descriptorMap and tempDiseaseID in descriptorMap:
                                    for currentIndex in descriptorMap[currentDiseaseID]:
                                        for tempIndex in descriptorMap[tempDiseaseID]:
                                            if currentIndex in tempIndex and currentIndex != tempIndex:
                                                eliminatedIDs.add(currentDiseaseID)
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
    with open('./Preprocessed/CDRTraining/pairs1.pkl', 'rb') as input_file:
        pairs = pickle.load(input_file)
    
    # Load the annotation hierarchy map from the MeSH dataset
    descriptorMap = loadDescriptorMap('./MeSH/desc2024.xml')

    # Create pairs from the loaded documents
    pairs = hypernymFiltering(pairs, descriptorMap)

    # Open a file to write the output
    with open('./Preprocessed/CDRTraining/pairs2.pkl', 'wb') as outputFile:
        pickle.dump(pairs, outputFile)

    # Close the file after writing
    outputFile.close()