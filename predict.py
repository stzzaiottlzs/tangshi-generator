import torch
import torch.nn as nn
from Dataset_Dataloader import *
from TangShiModel import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict():
    model = TangShiRNN(word2idx_count)
    model.load_state_dict(torch.load("./modules/tangshi_module_100.bin", map_location=torch.device('cpu')))

    model.eval()

    hidden = torch.zeros(1, 1, 128)

    start_word = input("输入第一个字:")

    flag = None

    tangshi_strs = []

    while True:
        if not flag:
            outputs, hidden = model(torch.tensor([[word2idx["S"]]], dtype=torch.long), hidden)
            tangshi_strs.append("S")
            flag = True
        else:
            tangshi_strs.append(start_word)
            outputs, hidden = model(torch.tensor([[word2idx[start_word]]], dtype=torch.long), hidden)
            top_i = torch.argmax(outputs, dim=-1)

            if top_i.item() == word2idx["E"]:
                break

            print(top_i)

            start_word = idx2word[top_i.item()]
        print(tangshi_strs)


if __name__ == '__main__':
    predict()
