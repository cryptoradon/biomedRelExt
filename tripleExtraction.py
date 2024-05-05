from bioc import pubtator
import xml.etree.ElementTree as ET
import spacy
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

# Function to extract entities, relations, and triplets from the CTD XML file
def getTripletsFromCTD(xml_path):
    entities = set()                        # Set to store unique entities (chemicals and diseases)
    relations = set()                       # Set to store unique relations
    triplets = {}                           # Dictionary to store triplets in the format (chemical, disease, pmid): [relation]
    
    # Parse the CTD XML file and extract information
    for _, elem in ET.iterparse(xml_path, events=('end',)):
        if elem.tag == 'Row':
            chemical = elem.find('ChemicalID').text
            disease = elem.find('DiseaseID').text.split(':')[1] if elem.find('DiseaseID') is not None else 'NIL'
            pmids = elem.find('PubMedIDs').text if elem.find('PubMedIDs') is not None else 'NIL'
            
            relation = 'NIL'
            directEvidence = elem.find('DirectEvidence')
            inferenceScore = elem.find('InferenceScore')
            # Determine the relation based on direct evidence or inference score
            if directEvidence is not None:
                if directEvidence.text == "therapeutic":
                    relation = 'THR'
                elif directEvidence.text == "marker/mechanism":
                    relation = 'CID'
            if inferenceScore is not None and inferenceScore.text:
                relation = 'INF'

            entities.add(chemical)
            entities.add(disease)
            relations.add(relation)

            # Adding to relations array since we may have more than relation
            if pmids != 'NIL':
                pmidsSplitted = pmids.split('|')
                for pmid in pmidsSplitted:
                    key = (chemical, disease, pmid)
                    triplets.setdefault(key, []).append(relation)
                    print("Extracted CTD " + str(key))
            else:
                key = (chemical, disease, 'NIL')
                triplets.setdefault(key, []).append(relation)
                print("Extracted CTD " + str(key))
            elem.clear()

    return entities, relations, triplets

# Function to extract entities, relations, and triplets from the CDR dataset
def getTripletsFromCDR(docs, tripletsCTD):
    entities = set()                        # Set to store unique entities (chemicals and diseases)
    relations = set()                       # Set to store unique relations
    triplets = {}                           # Dictionary to store triplets in the format (id1, id2, pmid): [relation]
    
    # Parse the documents to extract triplet information
    for doc in docs:
        for triplet in doc.relations:
            key = (triplet.id1, triplet.id2, triplet.pmid)
            entities.add(triplet.id1)
            entities.add(triplet.id2)
            # Adding to relations array since we may have more than relation
            if key in tripletsCTD:
                triplets.setdefault(key, []).append(triplet.type)
                relations.add(triplet.type)
            # Store the relation as 'NIL' if it is not in tripletsCTD
            else:
                triplets[key] = ['NIL']
                relations.add('NIL')
            print("Extracted CDR " + str(key))
    return entities, relations, triplets

# Function to combine and process all triplets from the CDR and CTD datasets
def getAllTriplets(tripletsCDR, tripletsCTD, entityToID, relationToID):
    result = set()                          # Set to store unique triplets in the format (entity1, relation, entity2)

    # Process the triplets from the CDR dataset
    for key in tripletsCDR:
        for rel in tripletsCDR[key]:
            # The IDs are changed from strings to numerical IDs here
            result.add((entityToID[key[0]], relationToID[rel], entityToID[key[1]]))

    # Process the triplets from the CTD dataset
    for key in tripletsCTD:
        for rel in tripletsCTD[key]:
            # The IDs are changed from strings to numerical IDs here
            result.add((entityToID[key[0]], relationToID[rel], entityToID[key[1]]))
    
    resultList = list(result)
    resultArray = np.array(resultList)

    print(f"Total Triplets: {len(result)}")
    return resultArray

# Function to extract and process the final set of triplets for separate CDR and CTD sets
def getFinaltriplets(triplets, entityToID, relationToID):
    result = set()                          # Set to store unique triplets in the format (entity1, relation, entity2)

    # Process the input triplets
    for key in triplets:
        for rel in triplets[key]:
            # The IDs are changed from strings to numerical IDs here
            result.add((entityToID[key[0]], relationToID[rel], entityToID[key[1]]))
    
    resultList = list(result)
    resultArray = np.array(resultList)

    print(f"Total Triplets: {len(result)}")
    return resultArray

# Function to split the triplets into training, validation, and test sets
def getAllSplitTriplets(finalTripletsAll, trainRatio=0.995, validationRatio=0.005):
    totalLines = len(finalTripletsAll)
    trainEnd = int(totalLines * trainRatio)
    validationEnd = trainEnd + int(totalLines * validationRatio)
    
    # Split the data
    finalTripletsAllTrain = finalTripletsAll[:trainEnd]
    finalTripletsAllValidation = finalTripletsAll[trainEnd:validationEnd]
    finalTripletsAllTest = finalTripletsAll[validationEnd:]

    return finalTripletsAllTrain, finalTripletsAllValidation, finalTripletsAllTest

if __name__ == '__main__':
    # Extract entities, relations, and triplets from the CTD dataset
    entitiesCTD, relationsCTD, tripletsCTD = getTripletsFromCTD('./CTD_Data/CTD_chemicals_diseases.xml')

    # Load documents from the CDR dataset
    with open('./CDR_Data/CDR.Corpus.v010516/CDR_Training+TestSet.PubTator.txt', 'r') as fp:
        docs = pubtator.load(fp)

    # Extract entities, relations, and triplets from the CDR dataset
    entitiesCDR, relationsCDR, tripletsCDR = getTripletsFromCDR(docs, tripletsCTD)

    # Combine and map entities and relations to unique numerical IDs
    entitiesAll = entitiesCTD.union(entitiesCDR)
    relationsAll = relationsCTD.union(relationsCDR)
    entityToID = {entity: idx for idx, entity in enumerate(entitiesAll)}
    relationToID = {rel: idx for idx, rel in enumerate(sorted(relationsAll))}

    # Save entity and relation mappings to files
    with open('./Knowledge_Representation/entityToID.pkl', 'wb') as f:
        pickle.dump(entityToID, f)
    with open('./Knowledge_Representation/relationToID.pkl', 'wb') as f:
        pickle.dump(relationToID, f)

    # Get all triplets from the CDR and CTD datasets combined
    finalTripletsAll = getAllTriplets(tripletsCDR, tripletsCTD, entityToID, relationToID)
    np.savetxt('./Knowledge_Representation/tripletsAll.txt', finalTripletsAll, fmt='%d %d %d')

    # Split the triplets into training, validation, and test sets
    finalTripletsAllTrain, finalTripletsAllValidation, finalTripletsAllTest  = getAllSplitTriplets(list(finalTripletsAll), trainRatio=0.995, validationRatio=0.005)
    np.savetxt('./Knowledge_Representation/tripletsAllTrain.txt', finalTripletsAllTrain, fmt='%d %d %d')
    np.savetxt('./Knowledge_Representation/tripletsAllValidation.txt', finalTripletsAllValidation, fmt='%d %d %d')
    np.savetxt('./Knowledge_Representation/tripletsAllTest.txt', finalTripletsAllTest, fmt='%d %d %d')

    # Get and save final triplets from the CTD dataset
    finaltripletsCTD = getFinaltriplets(tripletsCTD, entityToID, relationToID)
    np.savetxt('./Knowledge_Representation/tripletsCTD.txt', finaltripletsCTD, fmt='%d %d %d')

    # Get and save final triplets from the CDR dataset
    finaltripletsCDR = getFinaltriplets(tripletsCDR, entityToID, relationToID)
    np.savetxt('./Knowledge_Representation/tripletsCDR.txt', finaltripletsCDR, fmt='%d %d %d')