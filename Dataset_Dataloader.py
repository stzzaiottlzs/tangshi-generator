import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def deal_tangshi():
    with open("tangshis.txt", "r", encoding="utf-8") as fr:
        lines = fr.read().strip().split("\n")

    tangshis = []
    for line in lines:
        splits = line.split(":")
        if len(splits) != 2:
            continue
        tangshis.append("S" + splits[1] + "E")

    word2idx = {"S": 0, "E": 1}
    word2idx_count = 2

    tangshi_ids = []

    for tangshi in tangshis:
        for word in tangshi:
            if word not in word2idx:
                word2idx[word] = word2idx_count
                word2idx_count += 1

    idx2word = {idx: w for w, idx in word2idx.items()}

    for tangshi in tangshis:
        tangshi_ids.extend([word2idx[w] for w in tangshi])

    return word2idx, idx2word, tangshis, word2idx_count, tangshi_ids


word2idx, idx2word, tangshis, word2idx_count, tangshi_ids = deal_tangshi()


class TangShiDataset(Dataset):
    def __init__(self, tangshi_ids, num_chars):
        # 语料数据
        self.tangshi_ids = tangshi_ids
        # 语料长度
        self.num_chars = num_chars
        # 词的数量
        self.word_count = len(self.tangshi_ids)
        # 句子数量
        self.number = self.word_count // self.num_chars

    def __len__(self):
        return self.number

    def __getitem__(self, idx):
        # 修正索引值到: [0, self.word_count - 1]
        start = min(max(idx, 0), self.word_count - self.num_chars - 2)

        x = self.tangshi_ids[start: start + self.num_chars]
        y = self.tangshi_ids[start + 1: start + 1 + self.num_chars]

        return torch.tensor(x), torch.tensor(y)


def __test_Dataset():
    dataset = TangShiDataset(tangshi_ids, 8)
    x, y = dataset[0]

    print(x, y)


if __name__ == '__main__':
    # deal_tangshi()
    __test_Dataset()
