if __name__ == "__main__":

    from pathlib import Path
    from Processor import *
    from BowDoc import *

    dataset = Path('dataset101-150')
    training = Path('topicassignment101-150')
    baseline = Path('baselineresults')
    topicset = 'TopicStatements101-150.txt'
    pra = 0.02 # pseudo relevance assumption

    topics = {} # topic:query
    with open(topicset) as file:
        topic_nums = []
        titles = []
        for line in file:
            if line.startswith('<num>'):
                topic_nums.append((line.strip()[-3:]))
            if line.startswith('<title>'):
                titles.append((line[7:].strip()))
        topics = dict(zip(topic_nums, titles))
   
    for topic, query in topics.items():
        folder = dataset / f'Training{topic}'
        query = Processor.preprocess(query)
        coll = BowDocColl(topic)
        for xml_doc in folder.iterdir():
            coll.add_bowdoc(Processor.bowdocify(xml_doc))

        tfidfs = coll.calc_tfidf(query)
        bm25s = coll.calc_bm25(query)

        with open(training / f'Training{topic}.txt', 'w') as file:
            for doc_id, tfidf in tfidfs.items():
                d = 1 if tfidf > pra else 0
                file.write(f'R{topic} {doc_id} {d}\n')

        with open(baseline / f'BaselineResult{topic}.dat', 'w') as file:
            for doc_id, bm25 in bm25s.items():
                file.write(f'{doc_id} {bm25:.5f}\n')

        print(f'Completed {topic}...')
        


