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
        n = self.calc_dl()
        tf = self.terms[term] if term in self.terms else 0
        return tf/n

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

    def calc_df(self, term):
        """Returns the document frequency of the given term for the collection."""
        df = 0
        for bowdoc in self.coll.values():
            if term in bowdoc.get_terms():
                df += 1
        return df
        

    def calc_idf(self, term):
        """Returns the inverse document frequency of the given term for the collection."""
        n = len(self.coll)
        df = self.calc_df(term)
        return log(n/1+df) 

    def add_bowdoc(self, bowdoc):
        """Adds a BowDoc object to the collection."""
        self.coll[bowdoc.get_doc_id()] = bowdoc

    def __iter__(self):
        """Returns an iterator over the collection."""
        return iter(self.coll.items())

    def calc_tfidf(self, query):
        """Calculates the TF*IDF value for the given query for each BowDoc in the collection.

            Expected that the query has been preprocessed.

            Returns:
                A dictionary of doc_id:tf*idf pairs in descending order.

        """
        results = {} # doc_id:tf*idf
        for docid, bowdoc in self:
            tfidf = 0.0
            for term in query:
                idf = self.calc_idf(term)
                tf = bowdoc.calc_tf(term)
                tfidf += (tf * idf)
            results[docid] = tfidf
        return {k: v for k, v in sorted(results.items(), key=lambda item: -item[1])}

    def calc_avgdl(self):
        avgdl = 0.0
        for bowdoc in self.coll.values():
            avgdl += bowdoc.calc_dl()
        return avgdl

    def calc_bm25(self, query):
        """Calculates the BM25 value for the given query for each BowDoc in the collection.

            Expects that the query has been preprocessed.

            Returns:
                A dictionary of doc_id:bm25 pairs in descending order.

        """
        results = {}
        qfs = {} # term:weight 
        for term in query:
            try:
                qfs[term] += 1
            except KeyError:
                qfs[term] = 1
        avgdl = self.calc_avgdl()
        N = len(self.coll)
        print(N)
        for docid, bowdoc in self:
            k = 1.2 * ((1 - 0.75)) + 0.75 * (bowdoc.calc_dl() / avgdl)
            bm25 = 0.0
            for term in qfs.keys():
                n = self.calc_df(term)
                f = bowdoc.calc_tf(term)
                qf = qfs[term]
                bm25 += log(1.0 / ((n + 0.5) / (N - n + 0.5)), 2) * (((1.2 + 1) * f) / (k + f)) * ( ((100 + 1) * qf) / (100 + qf))
            results[docid] = bm25
        return {k: v for k, v in sorted(results.items(), key=lambda item: -item[1])}

            

