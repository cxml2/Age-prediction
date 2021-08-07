from models.imports import *


class CNN(Module):
    def __init__(self, biasconv=True, biaslinear=True, starter=256):
        super().__init__()
        self.convactivation = ReLU(inplace=True)
        self.convdropout = Dropout()
        self.max_pool2d = MaxPool2d((2, 2), (2, 2))
        self.conv1 = Conv2d(3, 12, (5, 5), (1, 1), (1, 1), bias=biasconv)
        self.conv1batchnorm = BatchNorm2d(12)
        self.conv2 = Conv2d(12, 24, (5, 5), (1, 1), (1, 1), bias=biasconv)
        self.conv2batchnorm = BatchNorm2d(24)
        self.conv3 = Conv2d(24, 12, (5, 5), (1, 1), (1, 1), bias=biasconv)
        self.conv3batchnorm = BatchNorm2d(12)
        self.lineardropout = Dropout()
        self.linearactivation = ReLU()
        self.max_pool1d = MaxPool1d((2, 2), (2, 2))
        self.linear1 = Linear(24 * 19 * 19, starter, bias=biaslinear)
        self.linear1batchnorm = BatchNorm1d(starter)
        self.linear2 = Linear(starter, starter * 2, bias=biaslinear)
        self.linear2batchnorm = BatchNorm1d(starter * 2)
        self.linear3 = Linear(starter * 2, starter, bias=biaslinear)
        self.linear3batchnorm = BatchNorm1d(starter)
        self.output = Linear(starter, 4)

    def forward(self, X):
        preds = self.convactivation(self.max_pool2d(self.conv1batchnorm(self.conv1(X))))
        preds = self.convactivation(
            self.max_pool2d(self.conv2batchnorm(self.conv2(preds)))
        )
        preds = self.convdropout(preds)
        preds = preds.view(-1, 24 * 19 * 19)
        preds = self.linearactivation(self.linear1batchnorm(self.linear1(preds)))
        preds = self.linearactivation(self.linear2batchnorm(self.linear2(preds)))
        preds = self.linearactivation(self.linear3batchnorm(self.linear3(preds)))
        preds = self.lineardropout(preds)
        preds = self.output(preds)
        # preds = Softmax()(preds)
        return preds
