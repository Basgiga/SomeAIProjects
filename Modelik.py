import cv2 as cv
import torch
from torch import nn
import pathlib
import sklearn
import pandas as pd
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

n = 1000

X, y = make_circles(n, noise = 0.03, random_state = 42)

circles = pd.DataFrame({"X1": X[:,0],
                        "X2": X[:,1],
                        "label": y})
print(circles.head(10))

def wizualizacja():
    plt.scatter(x=X[:,0],y=X[:,1],c=y)
    plt.show()
#wizualizacja()



"""
zmieniamy dane w tensory zeby pracowac w pytorchy
i rozdzielamy dane
"""

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

"""
jezdiemy model
"""
device = "cuda" if torch.cuda.is_available() else "cpu"

#print(device, X_train,y_train)

class CircleModel_Naiwna(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features= 8) # in tyle ile mamy atrybutow do zbadania, out to zazwyczaj wielokrotnosc 8mki do tylu zwiekszamy ilosc atrybutow, to bedzie przesyalc do nastepnej warstywy
        self.layer_2 = nn.Linear(in_features=8, out_features=1) #

        """
        #mozna to zrobic tez za pomoca Sequenital:
        model= nn.Sequential(
            nn.Linear(in_features=2, out_features= 8),
            nn.Linear(in_features=8, out_features=1)
        )
        
        ||
        ||
        \/
        """
        self.two_layers = nn.Sequential(
            nn.Linear(in_features=2, out_features= 8),
            nn.Linear(in_features=8, out_features=1))


    def forward(self, x): #metoda forward passu
            return self.layer_2(self.layer_1(x)) # x -> layer_1 -> layer_2 -> output

model = CircleModel_Naiwna().to(device)
#print(model)


"""
uzyjemy
Loss : cross entropy - nn.BCElossWithLogitLoss()
Optimizer : Adam - torch.optim.Adam()
accuracy
"""
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr=0.1)

def accuracy_fn(y_true,y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc


"""
training loop
"""

X_train,y_train = X_train.to(device), y_train.to(device)
X_test,y_test = X_test.to(device), y_test.to(device)

def Training(ile_epok, loss_fn, optimizer):
    for epoch in range(ile_epok):
        model.train()

        y_pred_logits = model1(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_pred_logits)) # zmienianie logitów na wartości prawdopodobieństwa, bo inaczej to sa losowe liczby typu 5.3, squeeze bo sie nie zgadzaly wymiary w loss

        loss = loss_fn(y_pred_logits,y_train) #!!
        optimizer.zero_grad()
        acc = accuracy_fn(y_train,y_pred)

        loss.backward()

        optimizer.step()

        model1.eval()

        with torch.inference_mode():
            test_logits = model1(X_test).squeeze()
            test_y_pred = torch.round(torch.sigmoid(test_logits))

            test_loss = loss_fn(test_logits,y_test)
            test_acc = accuracy_fn(y_test,test_y_pred)

        if epoch % 10 == 0:
            print(f"epoch: {epoch} | loss: {loss} , accuracy: {acc}| Test loss: {test_loss} , Test acc: {test_acc}")

#Training(100,loss_fn,optimizer)

"""
teraz wiemy czemy jest naiwana,
uzywa jednej warsty liniowej zeby rodzielic kolko
"""


class CircleModel_nobas(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2,
                                 out_features=8)  # in tyle ile mamy atrybutow do zbadania, out to zazwyczaj wielokrotnosc 8mki do tylu zwiekszamy ilosc atrybutow, to bedzie przesyalc do nastepnej warstywy
        self.layer_2 = nn.Linear(in_features=8, out_features=8)
        self.layer_3 = nn.Linear(in_features=8, out_features=8)
        self.layer_4 = nn.Linear(in_features=8, out_features=8)
        self.layer_5 = nn.Linear(in_features=8, out_features=8)
        self.layer_6 = nn.Linear(in_features=8, out_features=1)

    def forward(self, x):  # metoda forward passu
        return self.layer6(self.layer5(self.layer4(self.layer_3(self.layer_2(self.layer_1(x))))))

model1 = CircleModel_nobas().to(device)

optimizer1 = torch.optim.Adam(params = model1.parameters(), lr=0.1)

def Training1(ile_epok, loss_fn, optimizer):
    for epoch in range(ile_epok):
        model.train()

        y_pred_logits = model1(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_pred_logits)) # zmienianie logitów na wartości prawdopodobieństwa, bo inaczej to sa losowe liczby typu 5.3, squeeze bo sie nie zgadzaly wymiary w loss

        loss = loss_fn(y_pred_logits,y_train) #!!
        optimizer.zero_grad()
        acc = accuracy_fn(y_train,y_pred)

        loss.backward()

        optimizer.step()

        model.eval()

        with torch.inference_mode():
            test_logits = model1(X_test).squeeze()
            test_y_pred = torch.round(torch.sigmoid(test_logits))

            test_loss = loss_fn(test_logits,y_test)
            test_acc = accuracy_fn(y_test,test_y_pred)

        if epoch % 10 == 0:
            print(f"epoch: {epoch} | loss: {loss} , accuracy: {acc}| Test loss: {test_loss} , Test acc: {test_acc}")

#Training1(100,loss_fn,optimizer1)

"""
ten tez nie dziala, czemu?
bo dodawanie wiecej warstw liniowych nie sprawi ze rozwiazemy "problem kolkowy"

"""

class CircleModel_Sigma(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2,
                                 out_features=10)  # in tyle ile mamy atrybutow do zbadania, out to zazwyczaj wielokrotnosc 8mki do tylu zwiekszamy ilosc atrybutow, to bedzie przesyalc do nastepnej warstywy
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        #self.layer_3 = nn.Linear(in_features=8, out_features=8)
        self.relu = nn.ReLU()
        self.layer_5 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):  # metoda forward passu
        return self.layer_5(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

model2 = CircleModel_Sigma().to(device)

optimizer2 = torch.optim.Adam(params = model2.parameters(), lr=0.1)

def Training2(ile_epok, loss_fn, optimizer):
    for epoch in range(ile_epok):
        model2.train()

        y_pred_logits = model2(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_pred_logits)) # zmienianie logitów na wartości prawdopodobieństwa, bo inaczej to sa losowe liczby typu 5.3, squeeze bo sie nie zgadzaly wymiary w loss

        loss = loss_fn(y_pred_logits,y_train) #!!
        acc = accuracy_fn(y_train,y_pred)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        model2.eval()

        with torch.inference_mode():
            test_logits = model2(X_test).squeeze()
            test_y_pred = torch.round(torch.sigmoid(test_logits))

            test_loss = loss_fn(test_logits,y_test)
            test_acc = accuracy_fn(y_test,test_y_pred)

        if epoch % 100 == 0:
            print(f"epoch: {epoch} | loss: {loss} , accuracy: {acc}| Test loss: {test_loss} , Test acc: {test_acc}")

Training2(1000,loss_fn,optimizer2)

