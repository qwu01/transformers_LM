from transformers import ReformerConfig, ReformerTokenizer, ReformerModel
import sentencepiece as spm
import os


assert os.path.exists('protein_reformer/training_vocab.txt') == 1\
    , f'build a lower case amino acid txt file to train tokenizer. content should be: {"ARNDCQEGHILKMFPSTWYVOUBZX".lower()}'
MODEL_MAX_LENGTH = 4608
spm.SentencePieceTrainer.Train("--input=protein_reformer/training_vocab.txt --model_prefix=spiece --vocab_size=30 --pad_id=29 --character_coverage=1.0")
os.system("mv spiece.model spiece.vocab protein_reformer")
tokenizer = ReformerTokenizer(vocab_file="protein_reformer/spiece.model", do_lower_case=True, model_max_length=MODEL_MAX_LENGTH)
tokenizer.save_pretrained("protein_reformer")

configuration = ReformerConfig.from_pretrained("google/reformer-crime-and-punishment")
configuration.axial_pos_shape = (64, 72)
configuration.max_position_embeddings=MODEL_MAX_LENGTH
configuration.vocab_size=tokenizer.vocab_size
configuration.pad_token_id=tokenizer.pad_token_id
# configuration.attn_layers = ["local","lsh","local","lsh"]
configuration.output_hidden_states=True
configuration.save_pretrained('protein_reformer/')
