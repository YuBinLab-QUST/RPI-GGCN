# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 16:33:07 2023

@author: dell
"""
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



def DNC(input_data):
    fastas=readRNAFasta(input_data)
    encodings = []
    dinucleotides = [n1 + n2 for n1 in ALPHABET for n2 in ALPHABET]
    header = ['#'] + dinucleotides
    encodings.append(header)
    AADict = {}
    for i in range(len(ALPHABET)):
        AADict[ALPHABET[i]] = i
    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = [name]
        tmpCode = [0] * 16
        for j in range(len(sequence) - 2 + 1):
            tmpCode[AADict[sequence[j]] * 4 + AADict[sequence[j+1]]] = tmpCode[AADict[sequence[j]] * 4 + AADict[sequence[j+1]]] +1
        if sum(tmpCode) != 0:
            tmpCode = [i/sum(tmpCode) for i in tmpCode]
        code = code + tmpCode
        encodings.append(code)
    return encodings

vector=DNC('RPI369_RNA_N_biaohao.txt')
csv_data=pd.DataFrame(data=vector)
csv_data.to_csv('DNC_out.csv',header=False,index=False)