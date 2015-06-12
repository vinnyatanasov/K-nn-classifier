"""
K-NN classifier

- load in train data as vectors with label
- load in test data and compute distance measure with all train instances
- select the top k and get the most common label, which becomes our label
"""

import numpy


def read_features(file_path):
    feats = set()
    with open(file_path) as file:
        for line in file:
            for word in line.strip().split():
                feats.add(word)
    
    return feats


def make_vect(dimensions, feats_index, line):
    x = numpy.zeros(dimensions)
    for word in line.strip().split():
        x[feats_index[word]] = 1
    
    return x


def compute_similarity(x, y):
    # compute cosine similarity
    dot = numpy.dot(x,y)
    norms = numpy.linalg.norm(x) * numpy.linalg.norm(y)
    cosim = (dot / norms) if norms != 0 else 0
    
    return cosim


def test(k, dimensions, feats_index, train_data, file_path):
    label = 1 if "pos" in file_path else -1
    count = 0
    correct = 0
    # load test data in and compute similarity with all train instances
    with open(file_path) as file:
        for line in file:
            # create instance vector
            vect = make_vect(dimensions, feats_index, line)
            
            # compute similarity with each train instance
            similarities = []
            for item in train_data:
                similarities.append(compute_similarity(vect, item[0]))
            
            # get the k highest indexes and get their corresponding labels
            # choose the most frequent label as prediction
            n = []
            for i in xrange(k):
                maxx = similarities.index(max(similarities))
                n.append(train_data[maxx][1])
                del similarities[maxx]
            
            # if sum of all labels is positive, we have more +1s than -1s
            pred = numpy.sign(sum(n))
            print n, pred
            if (pred == label):
                correct += 1
            count += 1
    
    return 100*(correct/float(count))


def main():
    # set k to be an odd number
    k = 3
    
    # setup feature space
    feats = read_features("data/train.positive")
    feats = feats.union(read_features("data/train.negative"))
    feats = feats.union(read_features("data/test.positive"))
    feats = feats.union(read_features("data/test.negative"))
    feats_index = {}
    for (fid, fval) in enumerate(feats):
        feats_index[fval] = fid
    dimensions = len(feats_index)
    
    # load train data into vectors with labels
    train_data = []
    with open("data/train.positive") as file:
        for line in file:
            # create instance vector
            vect = make_vect(dimensions, feats_index, line)
            train_data.append((vect, +1))
    with open("data/train.negative") as file:
        for line in file:
            # create instance vector
            vect = make_vect(dimensions, feats_index, line)
            train_data.append((vect, -1))
    
    # test knn and print results
    pos_result = test(k, dimensions, feats_index, train_data, "data/test.positive")
    neg_result = test(k, dimensions, feats_index, train_data, "data/test.negative")
    print "Accuracy(%)"
    print "Positive:", pos_result
    print "Negative:", neg_result
    print "Combined:", (pos_result+neg_result)/float(2)


if __name__ == "__main__":
    print "-- K-nearest neighbour algorithm --\n"
    
    main()
    