import time

import torch

from Dataset_Dataloader import *
from TangShiModel import *
import torch.optim as optim
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    dataset = TangShiDataset(tangshi_ids, 128)
    epochs = 100
    model = TangShiRNN(word2idx_count).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for idx in range(epochs):
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
        start_time = time.time()
        total_loss = 0
        total_num = 0
        total_correct = 0
        total_correct_num = 0
        hidden = model.init_hidden()

        for x, y in tqdm(dataloader):
            x = x.to(device)
            y = y.to(device)
            # 隐藏状态
            hidden = model.init_hidden()
            hidden = hidden.to(device)
            # 模型计算
            output, hidden = model(x, hidden)
            # print(output.shape)
            # print(y.shape)
            # 计算损失
            loss = criterion(output.permute(1, 2, 0), y)
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()

            total_loss += loss.sum().item()
            total_num += len(y)
            total_correct_num += y.shape[0] * y.shape[1]
            # print(output.shape)
            total_correct += (torch.argmax(output.permute(1, 0, 2), dim=-1) == y).sum().item()

        print("epoch : %d average_loss : %.3f average_correct : %.3f use_time : %ds" %
              (idx + 1, total_loss / total_num, total_correct / total_correct_num, time.time() - start_time))

        torch.save(model.state_dict(), f"./modules/tangshi_module_{idx + 1}.bin")


if __name__ == '__main__':
    train()
