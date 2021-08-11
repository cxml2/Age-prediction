from torchvision.models import *
from models.clf import Clf
from models.clf import CNN
from models.clf_and_conv1d import Clf_and_Conv1d
from helper_funtions import *

# hp = Help_Funcs()


X_train = torch.load()
X_test = torch.load(
y_test = torch.load()
y_train = torch.load()
X_val = torch.load()
y_val = torch.load()
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
