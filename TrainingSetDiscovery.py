import xml.etree.ElementTree as ET
from stemming.porter2 import stem
from pathlib import Path
from BowDoc import *
import re
import string

class Processor:
    
    def __init__(self, stop_words, stemmer):
        self.stop_words = stop_words
        self.stemmer = stemmer

    def preprocess(self, text):
        """ 
        The text is stripped of numbers and punctuation.
        Then the text is tokenized, stop words are removed and remaining terms stemmed with the Porter2 stemming algorithm.
        Returns a list of processed terms.    
        """
        text = text.lower() 
        text = re.sub(r'\d+', '', text) 
        text = text.translate(str.maketrans('',  '', string.punctuation))
        text = text.split()
        return [self.stemmer(term) for term in text if term not in self.stop_words and len(term) > 3]

    def bowdocify(self, xml_file):
        """ Converts an XML file to a BowDoc """
        root = ET.parse(xml_file).getroot()
        doc_id = root.get('itemid')
        text_el = root.find('text')
        terms = self.preprocess(' '.join([child.text for child in text_el]))
        bowdoc = BowDoc(doc_id, terms)
        return bowdoc

if __name__ == "__main__":
    stop_words = open(Path('stopwords.txt')).read().split()
    dataset = Path('dataset101-150')
    topicset = Path('TopicStatements101-150.txt')
    p = Processor(stop_words, stem)

    topics = {}
    with open(topicset) as file:
        topic_nums = []
        titles = []
        for line in file:
            if line.startswith('<num>'):
                topic_nums.append((line.strip()[-3:]))
            if line.startswith('<title>'):
                titles.append((line[7:].strip()))
        topics = dict(zip(topic_nums, titles))
    
    bowdoc_collections = {} # collection of BowDocCollection
    for topic in topics.keys():
        bowdoc_col = BowDocCollection(topic)
        dir = dataset / ('Training' + topic)
        for xml_file in dir.iterdir():
            bowdoc_col.add_bowdoc(p.bowdocify(xml_file))
        bowdoc_collections[topic] = bowdoc_col
   