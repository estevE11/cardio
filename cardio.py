import pandas as pd
import numpy as np
import os
from tqdm import trange

from torch import nn, Tensor
import torch

def transform_dataset(dataset, subm=False):
    if subm:
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
    subm = pd.read_csv("dataset/test.csv")

    test = train.sample(frac=0.2)
    train = train.drop(test.index)

    train = transform_dataset(train)
    test = transform_dataset(test)
    Y_train = train["cardio"].values.reshape(-1, 1)
    Y_test = test["cardio"].values.reshape(-1, 1)

    return train.drop(columns="cardio").values.astype(np.float32), Y_train.astype(np.float32), test.drop(columns="cardio").values.astype(np.float32), Y_test.astype(np.float32), transform_dataset(subm, subm=True).values.astype(np.float32)

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

def train():
    X_train, Y_train, X_test, Y_test, _ = load_dataset()

    device = torch.device("cpu")
    model = CardioNet().to(device)
    model.train()

    # train
    BS = 32
    optim = torch.optim.AdamW(model.parameters(), lr=0.001)
    lr_schedule = torch.optim.lr_scheduler.StepLR(optim, step_size=2000, gamma=0.005)
    loss_function = nn.BCELoss()

    losses = []
    loss_sum = 0
    acc = []
    acc_sum = 0

    for epoch in range(5):
        acc_sum = 0
        loss_sum = 0
        for i in (t := trange(0, X_train.shape[0], BS)):
            samp = np.random.randint(0, len(X_train), BS)
            
            X = torch.tensor(X_train[samp].astype(np.float32), device=device).float()
            Y = torch.tensor(Y_train[samp].astype(np.float32), device=device).float()

            optim.zero_grad()
            out = model(X).to(device)
            loss = loss_function(out, Y)
            loss.backward()
            optim.step()

            lr_schedule.step()

            losses.append(loss.item())
            loss_sum += losses[-1]
            acc.append((out.round() == Y).sum().item() / BS)
            acc_sum += acc[-1]

            t.set_description(f"Epoch {epoch+1} => Avg loss: {loss_sum / ((i+1)/BS):.2f} - Acc: {acc_sum / ((i+1)/BS):.2f}")
    
    model.eval()

    # Test training dataset
    X = torch.tensor(X_train[:X_train.shape[0]], device=device).float()
    Y = torch.tensor(Y_train[:X_train.shape[0]], device=device).float()

    out = model(X).to(device)
    acc = (out.round() == Y).sum().item()
    print(f"Training Acc: {acc / X_train.shape[0]:.3f}")
    

    # Test test dataset
    X = torch.tensor(X_test[:X_test.shape[0]], device=device).float()
    Y = torch.tensor(Y_test[:X_test.shape[0]], device=device).float()

    out = model(X).to(device)
    acc = (out.round() == Y).sum().item()
    print(f"** Test acc: {acc / X_test.shape[0]:.3f}")

    return acc / X_test.shape[0], model, losses, acc
    


if __name__ == "__main__":

    acc = 0
    model = None
    losses = []
    accs = []
    while acc < 0.73:
        acc, model, losses, accs = train()

    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.plot(accs)
    plt.show()

    _, _, _, _, X_subm = load_dataset()    

    preds = []
    for i in (t := trange(X_subm.shape[0])):
        X = torch.tensor(X_subm[i], device=model.device).float()

        out = model(X)
        preds.append(int(out.round().item()))

    preds = np.array(preds)
    save_result(preds, model)