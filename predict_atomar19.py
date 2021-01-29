import cv2
import numpy as np
import torch
import torch.nn as nn
DROPOUT = 0.05
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

def predict(x):
    RESIZE_TO=400
    X=[]
    for img_path in [f for f in x if f[-4:] == ".png"]:
        img_file = (cv2.resize(cv2.imread(img_path), (RESIZE_TO, RESIZE_TO)))
        img_file = img_file.transpose((2, 0, 1))
        img = img_file.astype("float") / 255.0
        X.append(torch.FloatTensor(np.array(img)))
    X=torch.stack(X)
     # Here you would load your model (.pt) and use it on x to get y_pred, and then return y_pred
    model=CNN()
    model.load_state_dict(torch.load("model_atomar19.pt"))
    model.eval()
    y_pred = model(X)
    y_pred = y_pred.detach().clone()
    y_pred=torch.round(y_pred)
    y_pred=y_pred.float()
    return y_pred

# %% -------------------------------------------------------------------------------------------------------------------