class BowDoc:

    def __init__(self, doc_id, terms):
        self.doc_id = doc_id
        self.bow = {}
        for term in terms:
            self.add_term(term)

    def get_doc_id(self):
        return self.doc_id

    def get_bow(self):
        return self.bow

    def add_term(self, term):
        try:
            self.bow[term] += 1
        except KeyError:
            self.bow[term] = 1

class BowDocCollection:
    
    def __init__(self, id):
        self.bow_col = {}
        self.id = id

    def get_bowdoc_collection(self):
        return self.bow_col

    def add_bowdoc(self, bowdoc):
        doc_id = bowdoc.get_doc_id()
        self.bow_col[doc_id] = bowdoc

    def query(self, query):
        pass