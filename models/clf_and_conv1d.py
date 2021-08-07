from models.imports import *


class Clf_and_Conv1d(Module):
    def __init__(self, biasconv=True, biaslinear=True, starter=128):
        super().__init__()
        self.convactivation = LeakyReLU()
        self.convdropout = Dropout()
        self.max_pool2d = MaxPool2d((2, 2), (2, 2))
        self.conv1 = Conv1d(3, 12, (5, 5), bias=biasconv)
        self.conv1batchnorm = BatchNorm2d(12)
        self.conv2 = Conv1d(12, 24, (5, 5), bias=biasconv)
        self.conv2batchnorm = BatchNorm2d(24)
        self.lineardropout = Dropout()
        self.linearactivation = ReLU()
        self.linear1 = Linear(24 * 18 * 18, starter, bias=biaslinear)
        self.linear1batchnorm = BatchNorm1d(starter)
        self.linear2 = Linear(starter, starter * 2, bias=biaslinear)
        self.linear2batchnorm = BatchNorm1d(starter * 2)
        self.linear3 = Linear(starter * 2, starter, bias=biaslinear)
        self.linear3batchnorm = BatchNorm1d(starter)
        self.output = Linear(starter,4)

    def forward(self, X):
        preds = self.convactivation(
            self.max_pool2d(self.convdropout(self.conv1batchnorm(self.conv1(X))))
        )
        preds = self.convactivation(
            self.max_pool2d(self.convdropout(self.conv2batchnorm(self.conv2(preds))))
        )
        preds = preds.view(-1, 24 * 18 * 18)
        preds = self.linearactivation(
            self.lineardropout(self.linear1batchnorm(self.linear1(preds)))
        )
        preds = self.convactivation(
            self.lineardropout(self.linear2batchnorm(self.linear2(preds)))
        )
        preds = self.convactivation(
            self.lineardropout(self.linear3batchnorm(self.linear3(preds)))
        )
        preds = self.output(preds)
        return preds

