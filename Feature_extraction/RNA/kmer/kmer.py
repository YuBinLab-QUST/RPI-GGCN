import sys,re
from collections import Counter
import pandas as pd
import itertools

ALPHABET='ACGU'

def readRNAFasta(file):
	with open(file) as f:
		records = f.read()

	if re.search('>', records) == None:
		print('The input RNA sequence must be fasta format.')
		sys.exit(1)
	records = records.split('>')[1:]
	myFasta = []
	for fasta in records:
		array = fasta.split('\n')
		name, sequence = array[0].split()[0], re.sub('[^ACGU-]', '-', ''.join(array[1:]).upper())
		myFasta.append([name, sequence])
	return myFasta

def kmerArray(sequence, k):
    kmer = []
    for i in range(len(sequence) - k + 1):
        kmer.append(sequence[i:i + k])
    return kmer

#########################################################################
def kmer(input_data, k=2, normalize=True):
    
    fastas=readRNAFasta(input_data)
    vector = []
    header = ['#']
    if k < 1:
        print('error, the k must be positive integer.')
        return 0
    for kmer in itertools.product(ALPHABET, repeat=k):
        header.append(''.join(kmer))
    vector.append(header)
    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        kmers = kmerArray(sequence, k)
        count = Counter()
        count.update(kmers)
        if normalize == True:
           for key in count:
               count[key] = count[key] / len(kmers)
        code = [name]
        for j in range(1, len(header)):
            if header[j] in count:
               code.append(count[header[j]])
            else:
                code.append(0)
        vector.append(code)
    return vector

kmer_vector=kmer('Saccharomyces_RNA.txt',k=4)
csv_data=pd.DataFrame(data=kmer_vector)
csv_data.to_csv('Saccharomyces_RNA.csv')

