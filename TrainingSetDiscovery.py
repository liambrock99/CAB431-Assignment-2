from pathlib import Path
from Processor import *
from BowDoc import *

if __name__ == "__main__":
    dataset = Path('dataset101-150')
    topicset = 'TopicStatements101-150.txt'
    pra = 1.5 # pseudo relevance assumption

    coll = BowDocColl(101)
    for xml_file in (Path('dataset101-150') / 'Training101').iterdir():
        coll.add_bowdoc(Processor.bowdocify(xml_file))

    results = coll.calc_tfidf([stem("economic"), stem("espionage")])
    with open('test_tfidf.txt', 'w') as file:
        for docid, tfidf in results.items():
            file.write(f'{docid}: {tfidf:.5f}\n')
    
    results = coll.calc_bm25([stem("economic"), stem("espionage")])
    with open('test_bm25.txt', 'w') as file:
        for docid, tfidf in results.items():
            file.write(f'{docid}: {tfidf:.5f}\n')

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
   