import re
import string
import xml.etree.ElementTree as ET
from nltk.corpus import stopwords
from stemming.porter2 import stem
from BowDoc import BowDoc

""" Static methods for processing XML documents """
class Processor:
    
    def preprocess(text):
        """Preprocesses given block of text.

        Returns:
            A list of stemmed terms excluding stopwords.

        """
        text = text.lower() 
        text = re.sub(r'\d+', '', text) 
        text = text.translate(str.maketrans('',  '', string.punctuation))
        text = text.split()
        return [stem(term) for term in text if term not in stopwords.words('english') and len(term) > 3]

    def bowdocify(xml_doc):
        """Converts an XML document to a BowDoc representation. 

        Only extracts text from the XML documents <text> element and its child elements.

        Returns:
            A Bag-of-words representation of the XML document.

        """
        root = ET.parse(xml_doc).getroot()
        doc_id = root.get('itemid')
        text_el = root.find('text')
        terms = Processor.preprocess(' '.join([child.text for child in text_el]))
        bowdoc = BowDoc(doc_id)
        for term in terms:
            bowdoc.add_term(term)
        return bowdoc