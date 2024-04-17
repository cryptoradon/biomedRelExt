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

def getTripletsFromCTD(xml_path):
    triplets = {}
    for _, elem in ET.iterparse(xml_path, events=('end',)):
        if elem.tag == 'Row':
            chemical = elem.find('ChemicalID').text
            disease = elem.find('DiseaseID').text.split(':')[1] if elem.find('DiseaseID') is not None else 'NIL'
            pmid = elem.find('PubMedIDs').text if elem.find('PubMedIDs') is not None else 'NIL'
            relation = 'THR' if elem.find('DirectEvidence') is 'therapeutic' else 'NIL'
            relation = 'INF' if elem.find('InferenceScore') is not None else 'NIL'
            relation = 'CID' if elem.find('DirectEvidence') is 'marker/mechanism' else 'NIL'
            key = (chemical, disease, pmid)
            triplets.setdefault(key, []).append(relation)
            elem.clear()

    return triplets

def getTripletsFromCDR(docs, tripletsCTD):
    triplets = {}
    for doc in docs:
        for triplet in doc.relations:
            key = (triplet.pmid, triplet.chemical.id, triplet.disease.id)
            if key in tripletsCTD:
                triplets.setdefault(key, []).append(triplet.relation.type)
            else:
                triplets[key] = ['NIL']
    return triplets

if __name__ == '__main__':
    tripletsCTD = getTripletsFromCTD('./CTD_Data/CTD_chemicals_diseases.xml')

    with open('./CDR_Data/CDR.Corpus.v010516/CDR_Training+TestSet.PubTator.txt', 'r') as fp:
        docs = pubtator.load(fp)

    tripletsCDR = getTripletsFromCDR(docs, tripletsCTD)

    print(f"Total CTD Triplets: {len(tripletsCTD)}, Total CDR Triplets: {len(tripletsCDR)}")
