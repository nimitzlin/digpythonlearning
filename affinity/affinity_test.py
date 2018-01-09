import numpy as np
dataset_filename = "affinity_dataset.txt"
x = np.loadtxt(dataset_filename)
from collections import defaultdict
valid_rules = defaultdict(int)
invalid_rules = defaultdict(int)
num_occurances = defaultdict(int)
for sample in x:
    for premise in range(5):
        if sample[premise] == 0: continue
        num_occurances[premise] += 1
        for conclusion in range(5):
            if premise == conclusion: continue
            if sample[conclusion] == 1:
                valid_rules[(premise, conclusion)] += 1
            else:
                invalid_rules[(premise, conclusion)] += 1

support = valid_rules

confidence = defaultdict(float)
for premise, conclusion in valid_rules.keys():
    rule = (premise, conclusion)
    confidence[rule] = valid_rules[rule] / float(num_occurances[premise])
    print rule, confidence[rule]

from operator import itemgetter
sorted_support = sorted(support.items(), key=itemgetter(1), reverse=True)

print sorted_support
