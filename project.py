import os
from pyexpat import model
import random
from tqdm import tqdm
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset 
import pandas as pd
from numpy import vstack
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch.optim import SGD
from torch.nn import BCELoss
import json

class PeopleDataset(T.utils.data.Dataset):
    def __init__(self, src_file, num_rows=None):
        df = pd.read_csv(src_file)
        df.drop(df.columns[[0]], axis=1, inplace=True)
        print(df.columns)
        df.Class = df.Class.astype('float64')
        y_tmp = df['Class'].values
        x_tmp = df.drop('Class', axis=1).values
                

        self.x_data = T.tensor(x_tmp,dtype=T.float64).to(device)
        self.y_data = T.tensor(y_tmp,dtype=T.float64).to(device)

        print(type(self.x_data))
        print(len(self.x_data))

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if T.is_tensor(idx):
            idx = idx.tolist()
        preds = self.x_data[idx].type(T.FloatTensor)
        pol = self.y_data[idx].type(T.LongTensor)
        sample = [preds, pol]
        return sample


class MLPUpdated(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(30, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)
    
    fed_acc, fed_pre, fed_recall, fed_f1 = list(), list(), list(), list()

for cln in {3,6,10}:
    mp = {}

    # Can be changed to 12,16,24,32
    num_clients = 16
    # Change it to 3, 6, 10, 16
    num_selected = cln
    num_rounds = 50
    epochs = 5
    batch_size = 1024
    device = "cpu"
    device = T.device(device)
    fed_acc, fed_pre, fed_recall, fed_f1 = list(), list(), list(), list()

    # Dividing the training data into num_clients, with each client having equal number of data
    traindata = PeopleDataset('C:/Users/DAMAN/Desktop/Pre-Project/creditcard_train_SMOTE_1.csv')
    print(len(traindata))
    traindata_split = T.utils.data.random_split(traindata, [int(len(traindata) / num_clients) for _ in range(num_clients)])
    train_loader = [T.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in traindata_split]

    test_file = 'C:/Users/DAMAN/Desktop/Pre-Project/creditcard_test.csv'
    test_ds = PeopleDataset(test_file)
    test_loader = T.utils.data.DataLoader(test_ds,batch_size=batch_size, shuffle=True)

    def client_update(client_model, optimizer, train_loader, epoch=5):
        """
        This function updates/trains client model on client data
        """
        model.train()
        for e in range(epoch):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = client_model(data)
                binary_loss = T.nn.BCEWithLogitsLoss()
                target = target.unsqueeze(1)
                target = target.float()
                loss = binary_loss(output, target)
                loss.backward()
                optimizer.step()
        return loss.item()

    def server_aggregate(global_model, client_models):
        """
        This function has aggregation method 'mean'
        """
        ### This will take simple mean of the weights of models ###
        global_dict = global_model.state_dict()
        for k in global_dict.keys():
            global_dict[k] = T.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
        global_model.load_state_dict(global_dict)
        for model in client_models:
            model.load_state_dict(global_model.state_dict())

    def test(global_model, test_loader):
        """This function test the global model on test data and returns test loss and test accuracy """
        model.eval()
        test_loss = 0
        correct = 0
        actuals, predictions = list(), list()
        with T.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = global_model(data)
                binary_loss = T.nn.BCEWithLogitsLoss()
                target = target.unsqueeze(1)
                target = target.float()
                test_loss += binary_loss(output, target)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                actual = target.numpy()
                pr = output.detach().numpy()
                pr = pr.round()
                predictions.append(pr)
                actuals.append(actual)

        test_loss /= len(test_loader.dataset)
        predictions, actuals = vstack(predictions), vstack(actuals)
        # calculate accuracy
        acc = accuracy_score(actuals, predictions)
        # calculate precision
        prescision = precision_score(actuals, predictions)
        # calculate recall
        recall = recall_score(actuals, predictions)
        # calculate f1
        f1 = f1_score(actuals, predictions)
        fed_acc.append(acc)
        fed_pre.append(prescision)
        fed_recall.append(recall)
        fed_f1.append(f1)
        print()
        print(confusion_matrix(actuals, predictions))
        return test_loss, acc, prescision, recall, f1

    ###########################################
    #### Initializing models and optimizer  ####
    ############################################

    #### global model ##########
    # global_model =  MLPUpdated().cuda()
    global_model = MLPUpdated().to(device)

    ############## client models ##############
    # client_models = [ MLPUpdated().cuda() for _ in range(num_selected)]
    client_models = [ MLPUpdated().to(device) for _ in range(num_selected)]
    for model in client_models:
        model.load_state_dict(global_model.state_dict()) ### initial synchronizing with global model 

    ############### optimizers ################
    opt = [optim.SGD(model.parameters(), lr=0.01) for model in client_models]

    print(len(fed_acc))
    print(len(fed_pre))
    print(len(fed_recall))
    print(len(fed_f1))

    ###### List containing info about learning #########
    losses_train = []
    losses_test = []
    acc_train = []
    acc_test = []
    # Runnining FL

    import time
    start_time = time.time()
    for r in range(num_rounds):
        # select random clients
        client_idx = np.random.permutation(num_clients)[:num_selected]
        # client update
        loss = 0
        for i in tqdm(range(num_selected)):
            loss += client_update(client_models[i], opt[i], train_loader[client_idx[i]], epoch=epochs)
        
        losses_train.append(loss)
        # server aggregate
        server_aggregate(global_model, client_models)
        
        test_loss, acc, prescision, recall, f1= test(global_model, test_loader)
        losses_test.append(test_loss)
        acc_test.append(acc)
        print('%d-th round' % r)
        print('average train loss %0.3g | test loss %0.3g | test acc: %0.3f | test prescision: %0.3f | test recall: %0.3f | test f1: %0.3f' % (loss / num_selected, test_loss, acc, prescision, recall, f1))

    print("--- %s seconds ---" % (time.time() - start_time))
    # time[24]['fed'] = (time.time() - start_time)

    print(len(fed_acc))
    print(len(fed_pre))
    print(len(fed_recall))
    print(len(fed_f1))

    mp[str(num_selected)] = {'fed_acc': fed_acc, 'fed_pre': fed_pre, 'fed_recall': fed_recall, 'fed_f1': fed_f1}
    print(mp)
    print(len(mp))

    # To write the results in json file
    data = mp
    a_file = open(str(num_selected) + "_results.json", "w")
    json.dump(data, a_file)
    a_file.close()

    # To read the results in json file
    a_file = open(str(cln) + "_results.json", "r")
    a_dictionary = json.load(a_file)

    mp = a_dictionary
    print(a_dictionary.keys())
    print(len(a_dictionary[str(num_selected)]['fed_acc']))

    i = cln
    a_file = open(str(i) + "_results.json", "r")
    a_dictionary = json.load(a_file)

    mp = a_dictionary

    import matplotlib.pyplot as plt
    x = [i for i in range(1, num_rounds+1)]

    print("Number of Rounds: ", num_rounds)

    fig = plt.figure()
    fig.set_size_inches(25.5, 10.5)

    plt.subplot(2, 2, 1)
    plt.plot(x, mp[str(num_selected)]['fed_acc'])
    plt.legend([str(num_selected)])
    plt.ylabel("Accuracy")
    plt.xlabel("Rounds")

    plt.subplot(2, 2, 2)
    plt.plot(x, mp[str(num_selected)]['fed_pre'])
    plt.legend([str(num_selected)])
    plt.ylabel("Precision")
    plt.xlabel("Rounds")

    plt.subplot(2, 2, 3)
    plt.plot(x, mp[str(num_selected)]['fed_recall'])
    plt.legend([str(num_selected)])
    plt.ylabel("Recall")
    plt.xlabel("Rounds")

    plt.subplot(2, 2, 4)
    plt.plot(x, mp[str(num_selected)]['fed_f1'])
    plt.legend([str(num_selected)])
    plt.ylabel("F1 score")
    plt.xlabel("Rounds")
    plt.show()