import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
# 读取文件中的数据
from tqdm import tqdm


def read_data(file_name, num=None):
    with open(os.path.join("data", file_name + ".txt"), encoding="utf-8") as f:
        all_data = f.read().split("\n")

    texts = []
    labels = []
    for data in all_data:
        if data:
            t, l = data.split("\t")
            texts.append(t)
            labels.append(l)
    if num is None:
        return texts, labels
    else:
        return texts[:num], labels[:num]


# 制作对应的word_2_index,index_2_onehot
def bulit_curpus(train_texts):
    word_2_index = {"<PAD>": 0, "<UNK>": 1}
    for text in train_texts:
        for word in text:
            word_2_index[word] = word_2_index.get(word, len(word_2_index))  # 给对应的字赋index
    index_2_onehot = np.eye(len(word_2_index), dtype=np.float32)  # 制作成单位矩阵
    return word_2_index, index_2_onehot


# 制作数据集，方便进行批处理
class ohDataSet(Dataset):
    def __init__(self, texts, labels, word_2_index, index_2_onehot, maxLen):
        self.texts = texts
        self.labels = labels
        self.word_2_index = word_2_index
        self.index_2_onehot = index_2_onehot
        self.maxLen = maxLen

    def __getitem__(self, item):
        # 返回每句话的向量表示，以及其标签
        # 1.根据item获取对应的文本和标签
        text = self.texts[item]
        label = int(self.labels[item])

        # 2.剪裁数据长度置maxLen
        text = text[:self.maxLen]  # 但这时文本长度只能保证不大于maxLen,还可能有小于maxLen

        # 2.将中文文本 -》 index -> onehot
        text_index = [self.word_2_index.get(i, 1)for i in text] # 没有的词，就取第1个index
        text_index = text_index + [0] * (self.maxLen - len(text_index))  # 填充

        text_onehot = self.index_2_onehot[text_index]  # 选择text_index列表对应的下标的onehot叠加，成为二维矩阵

        return text_onehot, label

    def __len__(self):
        return len(self.texts)


# 定义模型类
class ohModel(nn.Module):
    def __init__(self, curpus_len, hidden_num, num_class, maxLen):
        super(ohModel, self).__init__()

        self.fc1 = nn.Linear(in_features=curpus_len, out_features=hidden_num)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc2 = nn.Linear(in_features=hidden_num * maxLen, out_features=num_class)
        self.cross_loss = nn.CrossEntropyLoss()

    def forward(self, txets_onehot, labels=None):
        """
        Args:
            x: [2,6,onehot的长度]
        """
        o1 = self.fc1(txets_onehot)
        o2 = self.relu(o1)
        o3 = self.flatten(o2)
        p = self.fc2(o3)

        self.pred = torch.argmax(p, dim=-1).detach().cpu().numpy().tolist() # 将预测结构存储

        if labels is not None:
            loss = self.cross_loss(p, labels)
            return loss


if __name__ == "__main__":
    train_texts, train_labels = read_data("train")
    dev_texts, dev_labels = read_data("dev")

    assert len(train_texts) == len(train_labels)
    assert len(dev_texts) == len(dev_labels)

    epoch = 5
    batch_size = 60
    maxLen = 25
    hidden_num = 30
    lr = 0.0006

    class_num = len(set(train_labels))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    word_2_index, index_2_onehot = bulit_curpus(train_texts)

    train_dataset = ohDataSet(train_texts, train_labels, word_2_index, index_2_onehot, maxLen)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=False)

    dev_dataset = ohDataSet(dev_texts, dev_labels, word_2_index, index_2_onehot, maxLen)
    print(dev_dataset)
    dev_dataloader = DataLoader(dev_dataset, batch_size, shuffle=False)

    model = ohModel(len(index_2_onehot), hidden_num, class_num, maxLen)
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # 开始训练
    for e in range(epoch):
        for texts, labels in tqdm(train_dataloader):
            texts = texts.to(device)
            labels = labels.to(device)

            loss = model(texts, labels)
            loss.backward()

            optim.step()
            optim.zero_grad()

        right_num = 0
        for texts, labels in tqdm(dev_dataloader):
            # print(texts)
            texts = texts.to(device)
            model(texts)
            right_num += int(sum([i == j for i, j in zip(model.pred, labels)]))  # 计算预测正确的数量
        print(f"dev acc :  {right_num / len(dev_labels) * 100 : .2f}%")