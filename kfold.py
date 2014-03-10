__author__ = 'James Castiglione'
import sys
import perceptron as p
import winnow as w
from math import ceil


def chunk_data(lines):
    chunks = []
    for i in xrange(10):
        start = int(ceil(float(len(lines)) / 10)) * i
        end = start + len(lines) / 10   # if (start + len(lines) / 10) < len(lines) else len(lines)
        chunks.append(lines[start:end])
    return chunks


def combine_chunks(chunks, start, end):
    data = []
    i = start
    while i != end:
        for line in chunks[i]:
            data.append(line)
        i = (i + 1) % 10
    return data


def paired_t_test(a, b):
    k = len(a)
    diff = 0
    for i in xrange(k):
        diff += a[i] - b[i]
    avg_diff = diff / k
    samp_var = 0
    for i in xrange(k):
        samp_var += (a[i] - b[i] - avg_diff)**2
    samp_var /= k-1
    std_dev = samp_var**.5
    t = avg_diff / (std_dev / k**0.5)
    print "t statistic = %f" % t
    if t > 2.262:                               # t statistic for 9 dof two tailed
        print "Difference is significant"
    else:
        print "Difference is not significant"
    
if __name__ == '__main__':
    data_file = sys.argv[1]
    with open(data_file) as f:
        lines = f.readlines()
    chunks = chunk_data(lines)
    #train_lines = combine_chunks(chunks, i, (i + 9) % 10)
    #print train_lines
    #print len(chunks)
    #print chunks[0], chunks[9]
    configfile = data_file.replace('.data', '.config')
    # 10-fold validate
    percep_results = []
    winnow_results = []
    for i in xrange(10):
        pconf = p.parse_config_perceptron(configfile)
        wconf = w.parse_config_winnow(configfile)
        train_lines = combine_chunks(chunks, i, (i + 8) % 10)
        test_lines = chunks[(i + 9) % 10]
        pc = p.Perceptron(p.parse_data_perceptron(train_lines, pconf), p.parse_data_perceptron(test_lines, pconf))
        wn = w.Winnow(w.parse_data_winnow(train_lines, wconf), w.parse_data_winnow(test_lines, wconf))
        pc.randomize_data()
        wn.randomize_data()
        pc.train_model()
        wn.train_model()
        pr = pc.test_model()
        wr = wn.test_model()
        percep_results.append(float(pr[0])/pr[1])
        winnow_results.append(float(wr[0])/wr[1])

    for i in xrange(len(percep_results)):
        print percep_results[i], winnow_results[i]

    paired_t_test(percep_results, winnow_results)