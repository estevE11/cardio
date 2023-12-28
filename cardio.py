import pandas as pd
import numpy as np
import os
from tqdm import trange

from torch import nn, Tensor
import torch

def transform_dataset(dataset, test=False):
    if test:
        dataset = dataset[["age", "height", "weight", "ap_hi", "ap_lo", "gender", "smoke", "alco", "active", "cholesterol", "gluc"]]
    else:
        dataset = dataset[["age", "height", "weight", "ap_hi", "ap_lo", "gender", "smoke", "alco", "active", "cholesterol", "gluc", "cardio"]]
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

    train = transform_dataset(train)
    Y_train = train["cardio"].values.reshape(-1, 1)

    return train.drop(columns="cardio").values, Y_train, transform_dataset(test, test=True).values, test.values

def save_result(result, model):
    submission = pd.read_csv('dataset/sample.csv')
    submission['cardio'] = result

    i = 0
    while os.path.exists(f"submission_{i}.csv"):
        i += 1
    submission.to_csv(f'submission_{i}.csv', index = False)
    torch.save(model.state_dict(), f"cardio_{i}.pth")

class BinaryBranch(nn.Module):
    def __init__(self):
        super(BinaryBranch, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU()
        )

    def __call__(self, x: Tensor) -> Tensor:
        return self.seq(x)

class DecimalBranch(nn.Module):
    def __init__(self):
        super(DecimalBranch, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )

    def __call__(self, x: Tensor) -> Tensor:
        return self.seq(x)


class CardioNet(nn.Module):
    def __init__(self):
        super(CardioNet, self).__init__()
        self.binary_branch = BinaryBranch()
        self.decimal_branch = DecimalBranch()
        self.final = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def __call__(self, x: Tensor) -> Tensor:
        x = x.reshape(-1, 15)
        x_dec = x[:, :5]
        x_bin = x[:, 5:]

        x_bin = self.binary_branch(x_bin)
        x_dec = self.decimal_branch(x_dec)

        x = torch.cat((x_bin, x_dec), dim=1)
        return self.final(x)



if __name__ == "__main__":
    X_train, Y_train, X_test, test = load_dataset()

    model = CardioNet().to("cuda:0")
    model.train()

    # train
    BS = 64
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_schedule = torch.optim.lr_scheduler.StepLR(optim, step_size=400, gamma=0.002)
    loss_function = nn.BCELoss()

    losses = []
    acc = []
    acc_sum = 0

    for epoch in range(3):
        acc_sum = 0
        for i in (t := trange(0, X_train.shape[0], BS)):
            samp = np.random.randint(0, len(X_train), BS)

            X = torch.tensor(X_train[samp], device="cuda:0").float()
            Y = torch.tensor(Y_train[samp], device="cuda:0").float()

            optim.zero_grad()
            out = model(X).to("cuda:0")
            loss = loss_function(out, Y)
            loss.backward()
            optim.step()

            lr_schedule.step()

            losses.append(loss.item())
            acc.append((out.round() == Y).sum().item() / BS)
            acc_sum += acc[-1]

            t.set_description(f"Epoch {epoch+1} => Loss: {loss.item():.2f} - Acc: {acc[-1]:.2f}")
        print(f"Avg acc: {acc_sum / (X_train.shape[0]/BS):.2f}")

    



    acc_sum = 0
    model.eval()
    for i in (t := trange(X_train.shape[0])):
        X = torch.tensor(X_train[i], device="cuda:0").float()
        Y = torch.tensor(Y_train[i], device="cuda:0").float()

        out = model(X).to("cuda:0")
        acc_sum += (out.round() == Y).sum().item()
    print(f"Acc: {acc_sum / X_train.shape[0]:.2f}")

    q = input("Save result? [Y/n]: ")
    if q == "n": exit()

    preds = []
    for i in (t := trange(X_test.shape[0])):
        X = torch.tensor(X_test[i], device="cuda:0").float()

        out = model(X)
        preds.append(int(out.round().item()))

    preds = np.array(preds)
    save_result(preds, model)
    

    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.plot(acc)
    plt.show()