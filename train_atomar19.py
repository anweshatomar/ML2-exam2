import os
import numpy as np
import cv2
import torch
import scipy
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
# %% --------------------------------------- Set-Up --------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# %% -------------------------------------- Data Laoding ---------------------------------------------------------------
if "train" not in os.listdir():
    os.system("wget https://storage.googleapis.com/exam-deep-learning/train-Exam2.zip")
    os.system("unzip train-Exam2.zip")
# %% -------------------------------------- Functions ------------------------------------------------------------------
def encod(targets):
    #ecoding the target variables
    test=["red blood cell", "difficult", "gametocyte", "trophozoite","ring", "schizont", "leukocyte"]
    enc=np.zeros(7)
    for i in range(len(test)):
        for j in range(len(targets)):
            if (test[i]==targets[j]):
                enc[i]=1
    return enc
# %% -------------------------------------- Data Prep ------------------------------------------------------------------
DATA_DIR = os.getcwd() + "/train/"
RESIZE_TO = 400
x, y, labels = [], [],[]
for path in [f for f in os.listdir(DATA_DIR) if f[-4:] == ".png"]:
    img_file=(cv2.resize(cv2.imread(DATA_DIR + path), (RESIZE_TO, RESIZE_TO)))
    img_file = img_file.transpose((2, 0, 1))
    img = img_file.astype("float") / 255.0
    x.append(torch.from_numpy(np.asarray(img)))
    with open(DATA_DIR + path[:-4] + ".txt", "r") as s:
         label = [line.strip() for line in s]
         vals_enc=encod(label)
    y.append(torch.from_numpy(np.asarray(vals_enc)))


X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.20)
x_train, y_train=torch.stack(X_train).to(device),torch.stack(y_train).to(device)
x_test,  y_test =torch.stack(X_test).to(device) , torch.stack(y_test).to(device)
print(len(x_train))
# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 0.001
N_EPOCHS = 20
BATCH_SIZE = 30
DROPOUT = 0.05
# %% -------------------------------------- CNN Class ------------------------------------------------------------------
#https://towardsdatascience.com/understanding-and-calculating-the-number-of-parameters-in-convolution-neural-networks-cnns-fc88790d530d
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 18, (3, 3))
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(18, 36, (3, 3))
        self.pool2 = nn.MaxPool2d((2, 2),
                                  ceil_mode=False)
        self.conv3 = nn.Conv2d(36, 72, (3, 3))
        self.pool3 = nn.MaxPool2d((2, 2))
        self.conv4 = nn.Conv2d(72, 72, (3, 3))
        self.pool4 = nn.MaxPool2d((2, 2))
        self.linear1 = nn.Linear(72 * 23 * 23, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 7)
        self.drop1 = nn.Dropout(DROPOUT)
        self.drop2 = nn.Dropout(DROPOUT)
        self.drop3 = nn.Dropout(DROPOUT)
        self.sigmoid = nn.Sigmoid()
        self.act = torch.relu

    def forward(self, x):
        x = self.drop1(self.pool1(self.act(self.conv1(x))))
        x = self.drop1(self.pool2(self.act(self.conv2(x))))
        x = self.drop1(self.pool3(self.act(self.conv3(x))))
        x = self.drop1(self.pool4(self.act(self.conv4(x))))
        x = self.linear3(self.drop2(self.act(self.linear2(self.drop2(self.act(self.linear1(x.view(len(x), -1))))))))
        return self.sigmoid(x)

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = CNN().type('torch.DoubleTensor').to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCELoss()
# %% -------------------------------------- Training Loop ----------------------------------------------------------
print("Starting training loop...")
for epoch in range(N_EPOCHS):
    model.train()
    loss_train = 0
    for batch in range(len(x_train) // BATCH_SIZE):
        inds = slice(batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE)
        optimizer.zero_grad()
        logits = model(x_train[inds])
        loss = criterion(logits, y_train[inds])
        loss.backward()
        loss_train += loss.item()
        optimizer.step()
    with torch.no_grad():
        y_test_pred = model(x_test)
        loss = criterion(y_test_pred, y_test)
        loss_test = loss.item()
        print("Epoch {} | Train Loss {:.5f}, - Test Loss {:.5f}".format(
            epoch, loss_train / BATCH_SIZE, loss_test))
torch.save(model.state_dict(), "model_atomar19.pt")
model.eval()
print("Perfect Result: Loss ---> " + str(loss_test))
print(model)

