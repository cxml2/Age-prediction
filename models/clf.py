from models.imports import *


class Clf(Module):
    def __init__(self, biaslinear=True, starter=256):
        super().__init__()
        self.lineardropout = Dropout()
        self.linearactivation = LeakyReLU()
        self.max_pool1d = MaxPool1d((2, 2), (2, 2))
        self.linear1 = Linear(3*84*84, starter, bias=biaslinear)
        self.linear1batchnorm = BatchNorm1d(starter)
        self.linear2 = Linear(starter, starter * 2, bias=biaslinear)
        self.linear2batchnorm = BatchNorm1d(starter * 2)
        self.linear3 = Linear(starter * 2, starter * 4, bias=biaslinear)
        self.linear3batchnorm = BatchNorm1d(starter * 4)
        self.linear4 = Linear(starter * 4, starter * 8, bias=biaslinear)
        self.linear4batchnorm = BatchNorm1d(starter * 8)
        self.linear5 = Linear(starter * 8, starter * 4, bias=biaslinear)
        self.linear5batchnorm = BatchNorm1d(starter * 4)
        self.output = Linear(starter * 4, 4) 

    def forward(self, X):
        X = X.view(-1,3*84*84)
        preds = self.linearactivation(

                self.lineardropout(self.linear1batchnorm(self.linear1(X)))
            
        )
        preds = self.linearactivation(

                self.lineardropout(self.linear2batchnorm(self.linear2(preds)))
 
        )
        preds = self.linearactivation(

                self.lineardropout(self.linear3batchnorm(self.linear3(preds)))
            
        )
        preds = self.linearactivation(
      
                self.lineardropout(self.linear4batchnorm(self.linear4(preds)))
            
        )
        preds = self.linearactivation(

                self.lineardropout(self.linear5batchnorm(self.linear5(preds)))
            
        )
        preds = self.output(preds)
        return preds
        
