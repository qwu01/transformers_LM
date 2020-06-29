import sys, re, csv, codecs
from Bio import SeqIO

input_file_name = sys.argv[1] # data/input.fasta
output_file_name = sys.argv[2] # data/output.txt


with codecs.open(input_file_name, 'r', encoding='utf-8', errors='ignore') as f:
    output_list = []
    for record in SeqIO.parse(f, 'fasta'):
        output_list.append(str(record.seq))

with open(output_file_name, "w", encoding='utf-8') as file:
    for s in output_list:
        file.write("%s\n" % s)