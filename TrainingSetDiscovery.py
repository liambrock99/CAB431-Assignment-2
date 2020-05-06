from pathlib import Path
from Processor import *
from BowDoc import *

if __name__ == "__main__":
    dataset = Path('dataset101-150')
    topicset = Path('TopicStatements101-150.txt')
    pra = 1.5 # pseudo relevance assumption

    bowdocs = BowDocColl(101)
    for xml_file in (Path('dataset101-150') / 'Training101').iterdir():
        bowdocs.add_bowdoc(Processor.bowdocify(xml_file))

    results = bowdocs.calc_tfidf([stem('economic'), stem('espionage')])
    with open('test.txt', 'w') as file:
        for k, v in results.items():
            file.write(f'{k}:{v}\n')

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
   