from math import log

class BowDoc:
    """A Bag-of-words representation of an XML Document.

    Attributes:
        doc_id: Document identifier.
        terms: Dictionary of term:frequency pairs.
    
    """

    def __init__(self, doc_id):
        self.doc_id = doc_id
        self.terms = {}

    def calc_tf(self, term):
        """Calculates the term frequency adjust for document length for the given term."""
        n = len(self.terms)
        tf = self.terms[term] if term in self.terms else 0
        return tf/n

    def get_doc_id(self):
        return self.doc_id

    def get_terms(self):
        return self.terms

    def __iter__(self):
        """Returns an iterator over the sorted term dictionary."""
        return iter(sorted(self.terms.items(), key=lambda x: x[1], reverse=True))

    def add_term(self, term):
        """Adds a term to the term dictionary"""
        try:
            self.terms[term] += 1
        except KeyError:
            self.terms[term] = 1

class BowDocColl:
    """ A collection of BowDoc objects.
    
    Attributes:
        coll_id: Collection identifier.
        coll: Dictionary of doc_id:bowdoc pairs.

    """

    def __init__(self, coll_id):
        self.coll = {}
        self.coll_id = coll_id

    def get_df(self, term):
        """Returns the document frequency of the given term for the collection."""
        df = 0
        for bowdoc in self.coll.values():
            if term in bowdoc.get_terms():
                df += 1
        return df

    def calc_idf(self, term):
        """Returns the inverse document frequency of the given term for the collection."""
        n = len(self.coll)
        df = self.get_df(term)
        return log(n/1+df) 

    def add_bowdoc(self, bowdoc):
        """Adds a BowDoc object to the collection."""
        self.coll[bowdoc.get_doc_id()] = bowdoc

    def __iter__(self):
        """Returns an iterator over the collection."""
        return iter(self.coll.items())

    def calc_tfidf(self, query):
        """ Calculates the TF*IDF value for the given query for each BowDoc in the collection.

            Returns:
                A dictionary of doc_id:tf*idf pairs.

        """
        results = {} # doc_id:tf*idf
        for docid, bowdoc in self:
            tfidf = 0.0
            for term in query:
                idf = self.calc_idf(term)
                tf = bowdoc.calc_tf(term)
                tfidf += (tf * idf)
            results[docid] = tfidf
        return results

