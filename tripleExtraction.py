from bioc import pubtator
import xml.etree.ElementTree as ET
import spacy
import pickle

# A class for chemical-disease pairs with context and metadata
'''class Triplet:
    def __init__(self, chemical, disease, relation, pmid):
        self.chemical = chemical                                # Chemical
        self.disease = disease                                  # Disease
        self.relation = relation
        self.pmid = pmid '''                                       # PubMed ID of the article

def getTripletsFromCTR(root):   
    triplets = {}
    for row in root.findall('Row'):
        chemical = row.find('ChemicalID').text
        disease = row.find('DiseaseID').text.split(':')[1]
        pmid = row.find('PubMedIDs').text if row.find('PubMedIDs') is not None else 'NIL'
        relation = 'THR' if row.find('DirectEvidence') is 'therapeutic' else 'NIL'
        relation = 'INF' if row.find('InferenceScore') is not None else 'NIL'
        relation = 'CID' if row.find('DirectEvidence') is 'marker/mechanism' else 'NIL'
        key = (chemical, disease, pmid)
        if key in triplets:
            triplets[key] = triplets[key].append(relation)
        else:
            triplets[key] = [relation]
    return triplets

def getTripletsFromCDR(docs, tripletsCTD):
    triplets = {}
    for doc in docs:                   # Pairs in specific doc
        for triplet in doc.relations:
            key = (triplet.pmid, triplet.chemical.id, triplet.disease.id)
            if key in tripletsCTD:
                if key in triplets:
                    triplets[key] = triplets[key].append(triplet.relation.type)
                else:
                    triplets[key] = [triplet.relation.type]
            else:
                triplets[key] = 'NIL'
    return triplets

if __name__ == '__main__':
    
    tree = ET.parse('./CTD_Data/CTD_chemicals_diseases.xml')
    root = tree.getroot()
    
    # Create pairs from the loaded documents
    tripletsCTD = getTripletsFromCTR(root)
    # Load the document data from a file
    with open('./CDR_Data/CDR.Corpus.v010516/CDR_Training+TestSet.PubTator.txt', 'r') as fp:
        docs = pubtator.load(fp)
    tripletsCDR = getTripletsFromCDR(docs, tripletsCTD)

    print(len(tripletsCTD) + len(tripletsCDR))