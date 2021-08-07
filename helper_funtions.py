from imports import *


class Help_Funcs:
    def load_data(self, IMG_SIZE=84, dir="./data/raw/"):
        data = []
        labels = {}
        idx = -1
        for label in os.listdir(dir):
            idx += 1
            labels[f"{dir}{label}/"] = [idx, -1]
        for label in tqdm(labels):
            for file in os.listdir(label):
                try:
                    file = label + file
                    img = cv2.imread(file)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    labels[label][1] += 1
                    data.append(
                        [np.array(transformations(np.array(img))), labels[label][0]]
                    )
                except Exception as e:
                    print(file)
                    print(e)
        np.random.shuffle(data)
        VAL_SPLIT = 938
        TEST_SPLIT = 1250
        X = []
        y = []
        for d in data:
            X.append(d[0])
            y.append(d[1])
        print(len(data))
        X_train = torch.from_numpy(np.array(X[:-TEST_SPLIT]))
        y_train = torch.from_numpy(np.array(y[:-TEST_SPLIT]))
        X_test = torch.from_numpy(np.array(X[-TEST_SPLIT:]))
        y_test = torch.from_numpy(np.array(y[-TEST_SPLIT:]))
        X_val = torch.from_numpy(np.array(X_test[VAL_SPLIT:]))
        y_val = torch.from_numpy(np.array(y_test[VAL_SPLIT:]))
        X_test = torch.from_numpy(np.array(X_test[:VAL_SPLIT]))
        y_test = torch.from_numpy(np.array(y_test[:VAL_SPLIT]))
        print(f"X_train len : {len(X_train)}")
        print(f"X_test len : {len(X_test)}")
        print(f"X_val len : {len(X_val)}")
        print(f"y_train len : {len(y_train)}")
        print(f"y_test len : {len(y_test)}")
        print(f"y_val len : {len(y_val)}")
        for label in labels:
            print(label)
        return X_train, y_train, X_test, y_test, X_val, y_val, labels, X, y, data

    def __accuracy(self, model, X, y):
        correct = -1
        total = -1
        preds = model(X)
        if len(preds) != len(y):
            raise ValueError("Preds is not Equal to the len of y")
        for pred, y_batch in zip(preds, y):
            pred = int(torch.argmax(pred))
            if pred == int(y_batch):
                correct += 1
            total += 1
        acc = round(correct / total, 3)
        return acc

    def accuracy(self, model, X, y):
        accs = []
        model.eval()
        acc = self.__accuracy(model, X, y)
        accs.append(acc)
        model.train()
        acc = self.__accuracy(model, X, y)
        accs.append(acc)
        acc = np.mean(accs)
        return acc

    def __loss(self, model, X, y, criterion):
        preds = model(X)
        loss = criterion(preds, y)
        return loss.item()

    def loss(self, model, X, y, criterion):
        losses = []
        loss = self.__loss(model, X, y, criterion)
        losses.append(loss)
        loss = np.mean(losses)
        return loss

    def accuracy_preds(self, preds, y):
        correct = -1
        total = -1
        for pred, y_batch in zip(preds, y):
            pred = int(torch.argmax(pred))
            if pred == int(y_batch):
                correct += 1
            total += 1
        acc = round(correct / total, 3)
        return acc

    def train(
        self,
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        criterion,
        optimizer,
        lr,
        PROJECT_NAME,
        name,
        batch_size,
        epochs,
    ):
        # try:
        hp = Help_Funcs()
        wandb.init(
            project=PROJECT_NAME,
            name=name,
            config={"criterion": criterion, "lr": lr, "batch_size": batch_size},
        )
        torch.cuda.empty_cache()
        for _ in tqdm(range(epochs)):
            torch.cuda.empty_cache()
            for idx in range(0, len(X_train), batch_size):
                torch.cuda.empty_cache()
                X_batch = X_train[idx : idx + batch_size].to(device).float()
                y_batch = y_train[idx : idx + batch_size].to(device)
                model.to(device)
                preds = model(X_batch.float())
                preds.to(device)
                loss = criterion(preds, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()
            model.eval()
            model.to("cpu")
            torch.cuda.empty_cache()
            wandb.log({"loss": loss.item()})
            torch.cuda.empty_cache()
            wandb.log({"val_loss": self.loss(model, X_test, y_test, criterion)})
            torch.cuda.empty_cache()
            wandb.log(
                {"accuracy": self.accuracy(model, X_batch.to("cpu"), y_batch.to("cpu"))}
            )
            wandb.log({"accuracy": self.accuracy_preds(preds, y_batch.to("cpu"))})
            torch.cuda.empty_cache()
            wandb.log({"val_accuracy": self.accuracy(model, X_test, y_test)})
            torch.cuda.empty_cache()
            model.train()
            model.to(device)
        paths = os.listdir("./data/test/")
        new_paths = []
        for path in paths:
            new_paths.append(f"./data/test/{path}")
        hp.get_multiple_preds(paths=new_paths, model=model, IMG_SIZE=84)
        paths = os.listdir("./output/")
        for path in paths:
            wandb.log({f"img/{path}": wandb.Image(cv2.imread(f"./output/{path}"))})
        wandb.finish()
        return model

    def get_faces(self, paths) -> dict or bool:
        idx = 0
        imgs_dict = {}
        for path in paths:
            image = face_recognition.load_image_file(path)
            face_locations = face_recognition.face_locations(image)
            if len(face_locations) > 0:
                for face_location in tqdm(face_locations):
                    idx += 1
                    im = Image.open(fr"{path}")
                    print(face_locations)
                    left = face_location[3] - 125
                    top = face_location[0] + 15
                    right = face_location[1] + 62 + 31
                    bottom = face_location[2] + 62 + 31
                    im1 = im.crop((left, top, right, bottom))  # Croping into the Image
                    im1.save(f"./output/{idx}.png")
                    # The below is the proccess of adding the idx and idx img to the dir and adding it to 'imgs_dict'
                    if path in list(imgs_dict.keys()):
                        imgs_dict[path][0].append(idx)
                        imgs_dict[path][1].append(f"./output/{idx}.png")
                    else:
                        imgs_dict[path] = [
                            [idx],
                            [f"./output/{idx}.png"],
                        ]
        return imgs_dict

    def get_multiple_preds(
        self,
        paths,
        model,
        labels={"0": "6-20", "1": "25-30", "2": "60-98", "3": "42-48"},
        num_of_times=50,
        IMG_SIZE=84,
    ) -> dict:
        with torch.no_grad():
            preds = {}
            hp = Help_Funcs()
            faces_results = hp.get_faces(paths)
            for _ in range(num_of_times):
                imgs = []
                for key, val in zip(faces_results.keys(), faces_results.values()):
                    img = cv2.imread(val[1][0])
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    imgs.append(img)
                if imgs == []:
                    break
                preds_model = model(
                    torch.tensor(np.array(imgs)).view(-1, 3, 84, 84).to(device).float()
                )
                for key, val, pred in zip(
                    faces_results.keys(), faces_results.values(), preds_model
                ):
                    pred = torch.argmax(pred)
                    pred = int(pred)
                    if val[0][0] in preds.keys():
                        preds[val[0][0]][0][str(pred)] += 1
                    else:
                        preds[val[0][0]] = [
                            {"0": 0, "1": 0, "2": 0, "3": 0},
                            [key, val[1][0]],
                        ]
                        preds[val[0][0]][0][str(pred)] += 1
            results = {}
            for idx, log in zip(preds.keys(), preds.values()):
                files = log[1]
                log = log[0]
                best_class = -1
                new_log = {}
                for key, val in zip(log.keys(), log.values()):
                    new_log[val] = key
                best_class = new_log[max(new_log.keys())]
                img = cv2.imread(files[1])
                results[idx] = [
                    [best_class, files[0], files[1], img.tolist()],
                    files[0],
                    files[1],
                ]
                plt.figure(figsize=(10, 7))
                plt.imshow(img)
                plt.title(f"{labels[str(best_class)]}")
                plt.savefig(f"./output/{idx}.png")
            return results
