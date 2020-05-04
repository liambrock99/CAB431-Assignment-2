import re
import string
import xml.etree.ElementTree as ET
from nltk.corpus import stopwords
from stemming.porter2 import stem
from pathlib import Path
from BowDoc import *

class Processor:
    
    def preprocess(text):
        """ Preprocesses text """
        text = text.lower() 
        text = re.sub(r'\d+', '', text) 
        text = text.translate(str.maketrans('',  '', string.punctuation))
        text = text.split()
        return [stem(term) for term in text if term not in stopwords.words('english') and len(term) > 3]

    def bowdocify(xml_file):
        """ Converts an XML file to a BowDoc """
        root = ET.parse(xml_file).getroot()
        doc_id = root.get('itemid')
        text_el = root.find('text')
        terms = Processor.preprocess(' '.join([child.text for child in text_el]))
        bowdoc = BowDoc(doc_id)
        for term in terms:
            bowdoc.add_term(term)
        return bowdoc

if __name__ == "__main__":
    dataset = Path('dataset101-150')
    topicset = Path('TopicStatements101-150.txt')
    pra = 1.5 # pseudo relevance assumption

    bowdocs = BowDocCollection(101)
    for xml_file in (Path('dataset101-150') / 'Training101').iterdir():
        bowdocs.add_bowdoc(Processor.bowdocify(xml_file))

    results = bowdocs.query([stem('economic'), stem('espionage')])
    with open('test.txt', 'w') as file:
        for k, v in results.items():
            file.write(f'{k}:{v}\n')


    # topics = {}
    # with open(topicset) as file:
    #     topic_nums = []
    #     titles = []
    #     for line in file:
    #         if line.startswith('<num>'):
    #             topic_nums.append((line.strip()[-3:]))
    #         if line.startswith('<title>'):
    #             titles.append((line[7:].strip()))
    #     topics = dict(zip(topic_nums, titles))
    
    # bowdoc_collections = {} # collection of BowDocCollection
    # for topic in topics.keys():
    #     bowdoc_col = BowDocCollection(topic)
    #     dir = dataset / ('Training' + topic)
    #     for xml_file in dir.iterdir():
    #         bowdoc_col.add_bowdoc(p.bowdocify(xml_file))
    #     bowdoc_collections[topic] = bowdoc_col
   