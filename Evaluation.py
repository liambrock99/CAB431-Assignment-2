
def precision_recall(training, B):
    A = {k:v for k,v in training.items() if v == 1}
    i = 0
    for docid in A:
        if docid in B:
            i += 1
    p = i/len(B)
    r = i/len(A)
    F1 = 2*r*p/r+p
    return(p, r, F1)

def evaluate(training_set, result_set):
    evaluation = {} # topic : (precision, recall, F1)
    for (k,v), (_, v2) in zip(training_set.items(), result_set.items()):
        evaluation[k] = precision_recall(v, v2)
    return evaluation

if __name__ == "__main__":
    
    from pathlib import Path

    baseline_results = Path('baselineresults')
    ir_results = Path('result')
    training_set = Path('topicassignment101-150')
    baseline = {} 
    irmodel = {} 
    training = {}
    for ir_result in ir_results.iterdir():
        r = {} # docid : score
        topic = ir_result.name[-7:-4]
        with open(ir_result) as file:
            for line in file:
                split = line.strip().split()
                r[split[0]] = float(split[1])
        irmodel[topic] = r
    for baseline_result in baseline_results.iterdir():
        r = {}
        topic = baseline_result.name[-7:-4]
        with open(baseline_result) as file:
            for line in file:
                split = line.strip().split()
                r[split[0]] = float(split[1])
        baseline[topic] = r
    for training_f in training_set.iterdir():
        r = {}
        topic = training_f.name[-7:-4]
        with open(training_f) as file:
            for line in file:
                split = line.strip().split()
                r[split[1]] = int(split[2])
        training[topic] = r
    
    ir_evaluation = evaluate(training, irmodel)
    with open('EvaluationResult.dat', 'w') as file:
        file.write('topic,precision,recall,F1\n')
        for topic, evaluation in ir_evaluation.items():
            p, r, F1 = evaluation
            file.write(f'{topic},{p:.6f},{r:.6f},{F1:.6f}\n')

    baseline_evaluation = evaluate(training, baseline)
    with open('EvaluationResultBaseline.dat', 'w') as file:
        file.write('topic,precision,recall,F1\n')
        for topic, evaluation in baseline_evaluation.items():
            p, r, F1 = evaluation
            file.write(f'{topic},{p:.6f},{r:.6f},{F1:.6f}\n')
