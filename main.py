from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from utils.config_utils import read_args, load_config, Dict2Object
from threading import Thread
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    """
    tain the model and return the training accuracy
    :param args: input arguments
    :param model: neural network model
    :param device: the device where model stored
    :param train_loader: data loader
    :param optimizer: optimizer
    :param epoch: current epoch
    :return:
    """
    model.train()
    correct = 0
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # calculate training accuracy and loss
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        train_loss += loss.item()
    training_acc = 100. * correct / len(train_loader.dataset)
    training_loss = train_loss / len(train_loader.dataset)
    return training_acc, training_loss


def test(model, device, test_loader):
    """
    test the model and return the tesing accuracy
    :param model: neural network model
    :param device: the device where model stored
    :param test_loader: data loader
    :return:
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            '''Fill your code'''
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  
            pred = output.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()
            

    testing_acc = 100. * correct / len(test_loader.dataset)
    testing_loss = test_loss / len(test_loader.dataset)
    return testing_acc, testing_loss


def plot(epoches, performance,title):
    """
    plot the model peformance, 
    generate line charts basd on the recods training loss, 
    testing loss and testing accuracy for each run 
    :param epoches: recorded epoches
    :param performance: recorded performance
    :return:
    """
    """Fill your code"""
    plt.figure()
    plt.plot(epoches, performance)
    plt.xlabel('Epoch')
    plt.ylabel('Performance')
    plt.savefig(title+'.png')
    
def plot1(epoches, performance,title):
    """
    plot the model peformance, 
    generate line charts basd on the recods training loss, 
    testing loss and testing accuracy for each run 
    :param epoches: recorded epoches
    :param performance: recorded performance
    :return:
    """
    """Fill your code"""
    plt.figure()
    plt.plot(epoches, performance)
    plt.xlabel('Epoch')
    plt.ylabel('Performance')
    plt.savefig(title+'.png')
    
    


def run(config,r_seed,time):
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    use_mps = not config.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(config.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': config.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': config.test_batch_size, 'shuffle': True}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # download data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)

    """add random seed to the DataLoader, pls modify this function"""
    torch.manual_seed(r_seed)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=config.lr)

    """record the performance"""
    epoches = [0]*(config.epochs+1)
    training_accuracies = [0]*(config.epochs+1)
    training_loss = [0]*(config.epochs+1)
    testing_accuracies = [0]*(config.epochs+1)
    testing_loss = [0]*(config.epochs+1)

    scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)
    
    for epoch in range(1, config.epochs + 1):
        training_accuracies[epoch], training_loss[epoch] = train(config, model, device, train_loader, optimizer, epoch)    
        """record training info, Fill your code"""
        testing_accuracies[epoch], testing_loss[epoch] = test(model, device, test_loader)
        epoches[epoch] = epoch
        """record testing info, Fill your code"""
        scheduler.step()
        """update the records, Fill your code"""
        train_acc = training_accuracies[epoch]
        train_loss = training_loss[epoch]
        test_acc= testing_accuracies[epoch]
        test_loss = testing_loss[epoch]
        f=open('results_run_{time}.txt'.format(time=time), 'a')
        f.write('training_accuracies_{epoch}:{train_acc}\n'.format(epoch=epoch,train_acc=train_acc))
        f.write('training_loss_{epoch}:{train_loss}\n'.format(epoch=epoch,train_loss=train_loss))
        f.write('testing_accuracies_{epoch}:{test_acc}\n'.format(epoch=epoch,test_acc=test_acc))
        f.write('testing_loss_{epoch}:{test_loss}\n'.format(epoch=epoch,test_loss=test_loss))
        
    """plotting training performance with the records"""
    plot(epoches, training_loss,f'Training_loss_Record_{time}'.format(time=time))
    plot(epoches, training_accuracies,'Training_accuracies_Record_{time}'.format(time=time))
    

    """plotting testing performance with the records"""
    plot(epoches, testing_loss,'Testing_loss_record_{time}'.format(time=time))
    plot(epoches, testing_accuracies,'Testing_accuracies_record_{time}'.format(time=time))

    if config.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
    
    f.close()


def plot_mean(epoches):
    """
    Read the recorded results.
    Plot the mean results after three runs.
    :return:
    """
    """fill your code"""
    
    mean_train_acc = [0]*(epoches+1)
    mean_train_loss = [0]*(epoches+1)
    mean_test_acc = [0]*(epoches+1)
    mean_test_loss = [0]*(epoches+1)
    mean_epochs=[0]*(epoches+1)
    for i in range(1,4):
        for j in range (1,epoches+1):
            mean_train_acc[j] += float(open('results_run_{i}.txt'.format(i=i)).readlines()[4*j-4].split(':')[1])
            mean_train_loss[j] += float(open('results_run_{i}.txt'.format(i=i)).readlines()[4*j-3].split(':')[1])
            mean_test_acc[j] += float(open('results_run_{i}.txt'.format(i=i)).readlines()[4*j-2].split(':')[1])
            mean_test_loss[j] += float(open('results_run_{i}.txt'.format(i=i)).readlines()[4*j-1].split(':')[1])
    for i in range(1,epoches+1):
        mean_epochs[i] = i
        mean_train_acc[i] /= 3
        mean_train_loss[i] /= 3
        mean_test_acc[i] /= 3
        mean_test_loss[i] /= 3
    plot1(mean_epochs, mean_train_acc,'Mean_Training_accuracies_Record')
    plot1(mean_epochs, mean_train_loss,'Mean_Training_loss_Record')
    plot1(mean_epochs, mean_test_acc,'Mean_Testing_accuracies_Record')
    plot1(mean_epochs, mean_test_loss,'Mean_Testing_loss_Record')

if __name__ == '__main__':
    
    arg = read_args()

    """toad training settings"""
    config = load_config(arg)
    
    

    """train model and record results"""
    t1=Thread(target=run,args=(config,123,1))
    t2=Thread(target=run,args=(config,321,2))
    t3=Thread(target=run,args=(config,666,3))
    t1.start()
    t2.start()
    t3.start()
    t1.join()
    t2.join()
    t3.join()
    """plot the mean results"""
    plot_mean(config.epochs)