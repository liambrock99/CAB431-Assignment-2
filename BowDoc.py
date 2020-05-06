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

    def __iter__(self):
        return iter(sorted(self.terms.items(), key=lambda x: x[1],reverse=True))

    def get_tf(self):
        """Returns a dictionary of term:freq pairs adjusted for document length."""
        return {k: v/self.calc_dl() for k, v in self.terms.items()}

    def calc_dl(self):
        """Returns the total number of terms in the BowDoc."""
        return sum(self.terms.values())

    def get_doc_id(self):
        return self.doc_id

    def get_terms(self):
        return self.terms

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
    
    def __iter__(self):
        """Returns an iterator over the collection."""
        return iter(self.coll.items())

    def get_df(self):
        """Returns a dictionary of term:df pairs for the collection."""
        df = {}
        for bowdoc in self.coll.values():
            for term in bowdoc.get_terms():
                try:
                    df[term] += 1
                except KeyError:
                    df[term] = 1
        return df
        

    def calc_idf(self, df):
        """Calculates the idf for the given df."""
        return log(len(self.coll)/1+df) 

    def add_bowdoc(self, bowdoc):
        """Adds a BowDoc object to the collection."""
        self.coll[bowdoc.get_doc_id()] = bowdoc

    def calc_tfidf(self, query):
        """Calculates the TF*IDF value for the given query for each BowDoc in the collection.

            Expected that the query has been preprocessed.

            Returns:
                A dictionary of doc_id:tf*idf pairs in descending order.

        """
        results = {} # doc_id:tf
        dfs = self.get_df() # term:df

        for docid, bowdoc in self:
            tfs = bowdoc.get_tf() # term:freq 
            tfidf = 0.0
            for term in query:
                tf = tfs[term] if term in tfs else 0
                df = dfs[term] if term in dfs else 0
                idf = self.calc_idf(df)
                tfidf += tf*idf
            results[docid] = tfidf
        return {k: v for k, v in sorted(results.items(), key=lambda item: -item[1])}