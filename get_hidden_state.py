import torch
from transformers import ReformerTokenizer, ReformerModel


MODEL_MAX_LENGTH = 4608

tokenizer_config_path = "protein_reformer/spiece.model"
tokenizer = ReformerTokenizer(vocab_file=tokenizer_config_path, do_lower_case=True, model_max_length=MODEL_MAX_LENGTH)

model_checkpoint = 'output/checkpoint-6500/'
model = ReformerModel.from_pretrained(model_checkpoint)

sequence_file_path = "data/yeast/yeast.txt"
f = open(sequence_file_path, "r")
sequence_txt = f.readlines()
f.close()

input_sequence_list = [tokenizer(sequence.strip(), truncation=True, return_tensors='pt')['input_ids'].cuda() for sequence in sequence_txt]
model.cuda()
protein_vectors_list = [torch.mean(model(inp)[1][-1], dim=1) for inp in input_sequence_list]
protein_vectors = torch.cat(protein_vectors_list, dim = 0)

from sklearn.manifold import TSNE
protein_vectors_tsne = TSNE(n_components=2).fit_transform(protein_vectors.to('cpu').numpy())
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(protein_vectors_tsne[:,0], protein_vectors_tsne[:,1])
plt.show()