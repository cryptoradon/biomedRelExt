from bioc import pubtator
import xml.etree.ElementTree as ET
import spacy
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

def getTripletsFromCTD(xml_path):
    triplets = {}
    for _, elem in ET.iterparse(xml_path, events=('end',)):
        if elem.tag == 'Row':
            chemical = elem.find('ChemicalID').text
            disease = elem.find('DiseaseID').text.split(':')[1] if elem.find('DiseaseID') is not None else 'NIL'
            pmids = elem.find('PubMedIDs').text if elem.find('PubMedIDs') is not None else 'NIL'
            
            relation = 'NIL'
            direct_evidence = elem.find('DirectEvidence')
            inference_score = elem.find('InferenceScore')
            if direct_evidence is not None:
                if direct_evidence.text == "therapeutic":
                    relation = 'THR'
                elif direct_evidence.text == "marker/mechanism":
                    relation = 'CID'
            if inference_score is not None and inference_score.text:
                relation = 'INF'

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

    return triplets

def getTripletsFromCDR(docs, tripletsCTD):
    triplets = {}
    for doc in docs:
        for triplet in doc.relations:
            key = (triplet.id1, triplet.id2, triplet.pmid)
            if key in tripletsCTD:
                triplets.setdefault(key, []).append(triplet.type)
            else:
                triplets[key] = ['NIL']
            print("Extracted CDR " + str(key))
    return triplets

def getFinalTriples(triplets):
    result = set()

    for key in triplets:
        for rel in triplets[key]:
            result.add((key[0], rel, key[1]))
    
    result_list = list(result)
    result_array = np.array(result_list)

    print(f"Total Triplets: {len(result)}")
    return result_array

if __name__ == '__main__':
    tripletsCTD = getTripletsFromCTD('./CTD_Data/CTD_chemicals_diseases.xml')

    with open('./CDR_Data/CDR.Corpus.v010516/CDR_Training+TestSet.PubTator.txt', 'r') as fp:
        docs = pubtator.load(fp)

    tripletsCDR = getTripletsFromCDR(docs, tripletsCTD)

    finalTriplesCTD = getFinalTriples(tripletsCTD)

    # Save triples in a format compatible with OpenKE
    np.savetxt('./Knowledge_Representation/triplesCTD.txt', finalTriplesCTD, fmt='%s %s %s')

    finalTriplesCDR = getFinalTriples(tripletsCDR)

    # Save triples in a format compatible with OpenKE
    np.savetxt('./Knowledge_Representation/triplesCDR.txt', finalTriplesCDR, fmt='%s %s %s')

    ctd_train, ctd_test = train_test_split(finalTriplesCTD, test_size=0.20, random_state=42)  # 20% for testing, 80% for training

    # Save CTD training and testing triples
    np.savetxt('./Knowledge_Representation/triplesCTD_train.txt', ctd_train, fmt='%s %s %s')
    np.savetxt('./Knowledge_Representation/triplesCTD_test.txt', ctd_test, fmt='%s %s %s')

    sampledTriplesCTD, _ = train_test_split(finalTriplesCTD, test_size=0.75, random_state=42)
    sampledCTDTrain, sampledCTDTest = train_test_split(sampledTriplesCTD, test_size=0.20, random_state=42)  # 20% of the sampled data for testing

    # Save CTD training and testing triples
    np.savetxt('./Knowledge_Representation/triplesCTD25_train.txt', sampledCTDTrain, fmt='%s %s %s')
    np.savetxt('./Knowledge_Representation/triplesCTD25_test.txt', sampledCTDTest, fmt='%s %s %s')
