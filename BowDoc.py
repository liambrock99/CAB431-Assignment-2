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

    def get_tf(self, term):
        return  self.terms[term] if term in self.terms else 0

    def get_tfs(self):
        """Returns a dictionary of term:freq pairs adjusted for document length."""
        dl = self.calc_dl()
        return {k: v/dl for k, v in self.terms.items()}

    def calc_dl(self):
        """Returns the total number of terms in the BowDoc."""
        return sum(self.terms.values())

    def get_docid(self):
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

    def get_coll_len(self):
        """Returns the length of the collection."""
        return len(self.coll)

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
        self.coll[bowdoc.get_docid()] = bowdoc

    def calc_tfidf(self, query):
        """Calculates the TF*IDF value for the given query for each BowDoc in the collection.

            Expected that the query has been preprocessed.

            Returns:
                A dictionary of doc_id:tf*idf pairs in descending order.

        """
        results = {} # doc_id:tf
        dfs = self.get_df() # term:df

        for docid, bowdoc in self:
            tfs = bowdoc.get_tfs() # term:freq 
            tfidf = 0.0
            for term in query:
                tf = tfs[term] if term in tfs else 0
                df = dfs[term] if term in dfs else 0
                idf = self.calc_idf(df)
                tfidf += tf*idf
            results[docid] = tfidf
        return {k: v for k, v in sorted(results.items(), key=lambda item: -item[1])}
    
    def calc_avgdl(self):
        n = len(self.coll)
        avgdl = 0.0
        for bowdoc in self.coll.values():
            avgdl += bowdoc.calc_dl()
        return avgdl/n

    def calc_bm25(self, query):
        results = {} # docid:bm25
        avgdl = self.calc_avgdl()
        N = len(self.coll)
        dfs = self.get_df()
        qfs = {}
        for qt in query:
            try:
                qfs[qt] += 1
            except KeyError:
                qfs[qt] = 1

        # constants
        k1 = 1.2
        k2 = 100
        b = 0.75
        r = 0 
        R = 0

        for docid, bowdoc in self:
            bm25 = 0.0
            dl = bowdoc.calc_dl()
            K = k1 * ((1-b) + b * dl/avgdl)
            for qt, qf in qfs.items():
                if qt in dfs:
                    n = dfs[qt]
                    f = bowdoc.get_tf(qt)
                    bm25 += log(((r+0.5)/(R-r+0.5)) / ((n-r+0.5)/(N-n-R+r+0.5)), 2) * (((k1+1) * f) / (K+f)) * (((k2+1) * qf) / (k2+qf))
            results[docid] = bm25
        return {k: v for k, v in sorted(results.items(), key=lambda item: -item[1])}