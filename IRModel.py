def training(coll, D, theta):
    T = {}
    ntk = coll.get_df()
    R = 0
    n = coll.get_coll_len()

    dpos = BowDocColl(0)
    for docid, bowdoc in coll:
        if D[docid] == 1:
            dpos.add_bowdoc(bowdoc)
            R += 1 
    T = dpos.get_df()
    
    for term, rtk in T.items():
        T[term] = ((rtk+0.5) / (R-rtk+0.5)) / ((ntk[term]-rtk+0.5) / (n-ntk[term]-R+rtk+0.5))

    mean = 0
    for term, rtk in T.items():
        mean += rtk
    mean = mean/len(T)

    return {t:r for t, r in T.items() if r > mean + theta}

def testing(coll, features):
    ranks = {}
    for docid, bowdoc in coll:
        terms = bowdoc.get_terms()
        for term in features:
            if term in terms:
                try:
                    ranks[docid] += features[term]
                except KeyError:
                    ranks[docid] = features[term]
    return {k: v for k, v in sorted(ranks.items(), key=lambda item: -item[1])}

if __name__ == "__main__":

    from pathlib import Path
    from BowDoc import *
    from Processor import *
    
    dataset = Path('dataset101-150')
    d = Path('topicassignment101-150')
    result = Path('result')

    for folder in dataset.iterdir():
        topic = folder.name[-3:]
        dpath = d / f'Training{topic}.txt'
        D = {}
        with open(dpath) as file:
            for line in file:
                split = line.strip().split()
                D[split[1]] = int(split[2])

        coll = BowDocColl(0)
        for xml_doc in folder.iterdir():
            coll.add_bowdoc(Processor.bowdocify(xml_doc))
        features = training(coll, D, 3.5)
        ranking = testing(coll, features)
        with open(result / f'result{topic}.dat', 'w') as file:
            for k,v in ranking.items():
                file.write(f'{k} {v}\n')


  
