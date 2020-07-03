import requests


feature_columns = [
    'id', 'entry name', 'protein names', 'length', 'organism', 'proteome', 'families',
    'feature(DNA BINDING)', 'feature(METAL BINDING)', 'comment(PATHWAY)',
    'comment(PH DEPENDENCE)', 'comment(TEMPERATURE DEPENDENCE)', 'features',
    'comment(SUBCELLULAR LOCATION)', 'feature(INTRAMEMBRANE)', 
    'feature(TOPOLOGICAL DOMAIN)', 'feature(TRANSMEMBRANE)',
    'go(cellular component)', '	go(biological process)', 'go(molecular function)',
]

query_arg = {
    'query': 'organism:"Saccharomyces cerevisiae" AND strain:"ATCC 204508 / S288c" ',
    'format': 'tab',
    'columns': ','.join(feature_columns),
}

result = requests.get('http://www.uniprot.org/uniprot/', params=query_arg)

with open('data/uniprot_data/yeast/saved_2.tsv', 'wt') as out_f:
    print(result.text, file=out_f)