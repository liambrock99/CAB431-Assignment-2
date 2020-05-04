from math import log

class BowDoc:

    def __init__(self, doc_id):
        self.doc_id = doc_id
        self.terms = {}

    def calc_tf(self, term):
        f = self.terms[term] if term in self.terms else 0
        return log(1 + f)

    def get_doc_id(self):
        return self.doc_id

    def get_terms(self):
        return self.terms

    def add_term(self, term):
        try:
            self.terms[term] += 1
        except KeyError:
            self.terms[term] = 1

class BowDocCollection:
    
    def __init__(self, col_id):
        self.bowdocs = {}
        self.col_id = col_id

    def get_bowdocs(self):
        return self.bowdocs

    def calc_idf(self, term):
        n = len(self.bowdocs)
        df = 1 # avoid division-by-zero
        for bowdoc in self.bowdocs.values():
            if term in bowdoc.get_terms():
                df += 1
        return log(n/df) 

    def add_bowdoc(self, bowdoc):
        self.bowdocs[bowdoc.get_doc_id()] = bowdoc

    # Expects a preprocessed tokenized query
    def query(self, query):
        results = {}
        for docid, bowdoc in self.bowdocs.items():
            tfidf = 0.0
            for term in query:
                idf = self.calc_idf(term)
                tf = bowdoc.calc_tf(term)
                tfidf += (tf * idf)
            results[docid] = tfidf
        return results
