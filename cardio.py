import pandas as pd
import numpy as np

from torch import nn, Tensor
import torch

def trasform_dataset(dataset):
    dataset["age"] = dataset["age"] / max(dataset["age"])
    dataset["height"] = dataset["height"] / max(dataset["height"])
    dataset["weight"] = dataset["weight"] / max(dataset["weight"])
    dataset["ap_hi"] = dataset["ap_hi"] / max(dataset["ap_hi"])
    dataset["ap_lo"] = dataset["ap_lo"] / max(dataset["ap_lo"])
    dataset["gender"]  = dataset["gender"] - 1
    dataset = pd.get_dummies(dataset, columns=['cholesterol', 'gluc'])
    return dataset

def load_dataset():
    train = pd.read_csv("dataset/train.csv")
    test = pd.read_csv("dataset/test.csv")

    train = trasform_dataset(train)
    test = trasform_dataset(test)

    return train.drop(columns="cardio").values, train["cardio"].values.reshape(-1, 1)

class CardioNet():
    def __init__(self):
        self.seq = nn.Sequential(
            nn.Linear(15, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.seq(x)

    def parameters(self):
        return self.seq.parameters()


if __name__ == "__main__":
    X_train, Y_train = load_dataset()

    model = CardioNet()

    # train
    BS = 32
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.BCELoss()

    losses = []
    acc = []

    for epoch in range(400):
        samp = np.random.randint(0, len(X_train), BS)
        X = Tensor(X_train[samp]).float()
        Y = Tensor(Y_train[samp]).float()

        optim.zero_grad()
        out = model(X)
        loss = loss_function(out, Y)
        loss.backward()
        optim.step()

        losses.append(loss.item())
        acc.append((out.round() == Y).sum().item() / BS)

        print(f"Epoch: {epoch} - Loss: {loss.item():.2f} - Acc: {acc[-1]:.2f}")

    import matplotlib.pyplot as plt
    #plt.plot(losses)
    #plt.plot(acc)
    #plt.show()


    # test
    from tqdm import trange

    acc_sum = 0
    for i in (t := trange(X_train.shape[0])):
        samp = np.random.randint(0, len(X_train))
        X = Tensor(X_train[samp]).float()
        Y = Tensor(Y_train[samp]).float()

        out = model(X)
        acc_sum += (out.round() == Y).sum().item()
    print(f"Acc: {acc_sum / X_train.shape[0]:.2f}")