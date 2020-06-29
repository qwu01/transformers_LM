from Project_A4.data.get_data import get_data
from Project_A4.data.get_batch import get_batch
import torch
import pandas as pd
from argparse import Namespace
import torch.nn.functional as F

args = Namespace(
    writer_dir=None,
    train_batch_size=128,
    eval_batch_size=32,
    n_tokens=None,
    embedding_dim=200,
    fc_hidden_size=200,
    n_layers=4,
    n_heads=4,
    dropout_p=0.2,
    learning_rate=6.,
    gamma_=0.95,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    epochs=42,
    save_model_path="../Project_A4/model/saved_model.pth",
)


input_dataset, TEXT = get_data(args.train_batch_size,
                               args.eval_batch_size,
                               args.device)

args.n_tokens = len(TEXT.vocab.stoi)
print(f"args.n_tokens = {args.n_tokens}")

args.writer_dir = "experiments/run_50"

loaded_model = torch.load("model/saved_model_v2.pth")

for param in loaded_model.parameters():
    param.requires_grad = False

loaded_model.eval()

with torch.no_grad():

    for batch, i in enumerate(range(0, input_dataset["train_data"].size(0) - 1, 35)):

        if batch == 0:
            input_data, targets = get_batch(input_dataset["train_data"], i, 35)

            _, transformer_encoder_output, _, _ = loaded_model(input_data)
            inputs_to_save = input_data
            features_to_save = transformer_encoder_output.mean(0)

            # features_to_save = F.conv1d(transformer_encoder_output.permute(1,0,2),
            #                             torch.ones(1, 35, 1).to(args.device)).squeeze_(1)
        else:
            input_data, _ = get_batch(input_dataset["train_data"], i, 35)
            _, transformer_encoder_output, _, _ = loaded_model(input_data)
            while True:
                try:
                    feature_temp = transformer_encoder_output.mean(0)
                    inputs_to_save = torch.cat((inputs_to_save, input_data), 1)
                    features_to_save = torch.cat((features_to_save, feature_temp), 0)
                    break
                except RuntimeError:
                    break

print(inputs_to_save.shape, features_to_save.shape)

# #
inputsDF = pd.DataFrame(inputs_to_save.to(device="cpu").numpy())
featuresDF = pd.DataFrame(features_to_save.to(device="cpu").numpy())
# #
#
inputsDF.to_csv("res/inputsDF_4.csv")
inputsDF.transpose().to_csv("res/inputsDF_4_transposed.csv")
featuresDF.to_csv("res/featuresDF_4.csv")
#


# import torchtext
# from torchtext.data.utils import get_tokenizer
#
# TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
#                             init_token='<sos>',
#                             eos_token='<eos>',
#                             lower=True)
# train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
# TEXT.build_vocab(train_txt)
# # print(len(train_txt.examples[0].text))
# # print(len(train_txt.examples[0].text[:59520*35]))
# # print(type(train_txt.examples[0].text[0]))
# # def chunks(lst, n):
# input_text_temp = [train_txt.examples[0].text[:59520*35][i:i+35] for i in range(0, 59520*35, 35)]
# input_text = []
#
# print("!")
#
# import csv
#
# with open("res/input_text.csv","w", encoding="utf-8") as f:
#     wr = csv.writer(f)
#     wr.writerows(input_text_temp)
#
# with open('res/input_text.txt', 'w', encoding='utf-8') as f:
#     for sub_list in input_text_temp:
#         for item in sub_list:
#             f.write(item + " ")
#         f.write('\n')