import pandas as pd
import numpy as np
import os
from tqdm import trange

from tinygrad import nn, Tensor
from tinygrad.nn import optim
from tinygrad.jit import TinyJit
from tinygrad.nn.state import get_parameters

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

    return train.drop(columns="cardio").values.astype(np.float32), Y_train.astype(np.float32), transform_dataset(test, test=True).values.astype(np.float32), test.values.astype(np.float32)

def save_result(result, model):
    submission = pd.read_csv('dataset/sample.csv')
    submission['cardio'] = result

    i = 0
    while os.path.exists(f"submission_{i}.csv"):
        i += 1
    submission.to_csv(f'submission_{i}.csv', index = False)
    #torch.save(model.state_dict(), f"cardio_{i}.pth")

class BinaryBranch():
    def __init__(self):
        super(BinaryBranch, self).__init__()
        self.l1 = nn.Linear(10, 32)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.l1(x)
        x = x.relu()
        return x

class DecimalBranch():
    def __init__(self):
        super(DecimalBranch, self).__init__()
        self.l1 = nn.Linear(5, 32)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.l1(x)
        x = x.relu()
        #x = x.batchnorm()
        x = x.relu()
        return x


class CardioNet():
    def __init__(self):
        super(CardioNet, self).__init__()
        self.binary_branch = BinaryBranch()
        self.decimal_branch = DecimalBranch()
        self.last = nn.Linear(64, 1)
    
    def __call__(self, x: Tensor) -> Tensor:
        x = x.reshape(-1, 15)
        x_dec = x[:, :5]
        x_bin = x[:, 5:]

        x = self.binary_branch(x_bin)
        dec = self.decimal_branch(x_dec)
        x = x.cat(dec, dim=1)

        x = self.last(x)
        x = x.sigmoid()
        return x



if __name__ == "__main__":
    X_train, Y_train, X_test, test = load_dataset()

    model = CardioNet()

    # train
    BS = 64
    optim = optim.Adam(get_parameters(model), lr=0.001)
    #lr_schedule = torch.optim.lr_scheduler.StepLR(optim, step_size=400, gamma=0.002)

    losses = []
    accs = []
    acc_sum = 0

    @TinyJit
    def train_step(samp):
        with Tensor.train():
            X = Tensor(X_train[samp]).float()
            Y = Tensor(Y_train[samp]).float()

            optim.zero_grad()
            out = model(X)
            loss = out.binary_crossentropy(Y)
            loss.backward()
            optim.step()

            #lr_schedule.step()

        return loss.realize(), (out.numpy().round() == Y.numpy()).sum().item() / BS

    for epoch in range(3):
        acc_sum = 0
        for i in (t := trange(0, X_train.shape[0], BS)):
            samp = np.random.randint(0, len(X_train), BS)
            loss, acc = train_step(samp) 
            
            losses.append(loss.item())
            accs.append(acc)
            acc_sum += accs[-1]

            t.set_description(f"Epoch {epoch+1} => Loss: {loss.item():.2f} - Acc: {accs[-1]:.2f}")
        print(f"Avg acc: {acc_sum / (X_train.shape[0]/BS):.2f}")

    


    X = Tensor(X_train[:X_train.shape[0]]).float()
    Y = Tensor(Y_train[:Y_train.shape[0]]).float()

    out = model(X)
    acc = (out.numpy().round() == Y.numpy()).sum().item()
    print(f"{acc/X_train.shape[0]:.2f}")



    q = input("Save result? [Y/n]: ")
    if q == "n": exit()

    preds = []
    for i in (t := trange(X_test.shape[0])):
        X = Tensor(X_test[i]).float()

        out = model(X)
        preds.append(int(out.round().item()))

    preds = np.array(preds)
    save_result(preds, model)
    

    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.plot(accs)
    plt.show()