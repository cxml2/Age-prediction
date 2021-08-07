from torchvision.models import *
from models.clf import Clf
from models.clf import CNN
from models.clf_and_conv1d import Clf_and_Conv1d
from helper_funtions import *

hp = Help_Funcs()
# X_train, y_train, X_test, y_test, X_val, y_val, labels, X, y, data = hp.load_data()
# print(labels)

# torch.save(X_train, "./data/cleaned/X_train-1.pt")
# torch.save(X_test, "./data/cleaned/X_test-1.pt")
# torch.save(y_test, "./data/cleaned/y_test-1.pt")
# torch.save(y_train, "./data/cleaned/y_train-1.pt")
# torch.save(X_val, "./data/cleaned/X_val-1.pt")
# torch.save(y_val, "./data/cleaned/y_val-1.pt")

X_train = torch.load("./data/cleaned/X_train.pt")
X_test = torch.load("./data/cleaned/X_test.pt")
y_test = torch.load("./data/cleaned/y_test.pt")
y_train = torch.load("./data/cleaned/y_train.pt")
X_val = torch.load("./data/cleaned/X_val.pt")
y_val = torch.load("./data/cleaned/y_val.pt")
torch.cuda.empty_cache()
batch_size = 30
optimizer = Adamax
lr = 0.001


class TL_Model(Module):
    def __init__(self, model, num_of_classes=4):
        super().__init__()
        self.model = model
        self.output = Linear(1000, num_of_classes)

    def forward(self, X):
        preds = self.model(X)
        preds = self.output(preds)
        preds = Sigmoid()(preds)
        return preds


model = TL_Model(resnet34(pretrained=True))
model = hp.train(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    criterion,
    optimizer(model.parameters(), lr=lr),
    lr=lr,
    PROJECT_NAME=PROJECT_NAME,
    name=f"Final",
    batch_size=batch_size,
    epochs=100,
)
torch.save(model, "./trained_models/model-age.pt")
torch.save(model, "./trained_models/model-age.pth")
torch.save(model.state_dict(), "./trained_models/model-age-sd.pt")
torch.save(model.state_dict(), "./trained_models/model-age-sd.pth")
model = torch.load("./trained_models/model.pt")
paths = os.listdir("./data/test/")
new_paths = []
for path in paths:
    new_paths.append(f"./data/test/{path}")
hp.get_multiple_preds(paths=new_paths, model=model, IMG_SIZE=84)
