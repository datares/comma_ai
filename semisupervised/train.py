import torch
import torch.nn as nn
import torch.optim as optim
from semisupervised.models import resnet18
from tqdm import tqdm

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr *= (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] *= (0.1 ** (epoch // 30))
def train_one_epoch(model, train_loader, optimizer, criterion, epoch, batch_size=100):
    model.train()
    step = 0
    train_loss = 0
    train_acc = 0
    for data, target in tqdm(train_loader, desc="epoch " + str(epoch), mininterval=1):
        adjust_learning_rate(optimizer, epoch)
        # data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.data
        y_pred = output.data.max(1)[1]

        acc = float(y_pred.eq(target.data).sum()) / len(data) * 100.
        train_acc += acc
        step += 1
        if step % 100 == 0:
            print("[Epoch {0:4d}] Loss: {1:2.3f} Acc: {2:.3f}%".format(epoch, loss.data, acc), end='')
            for param_group in optimizer.param_groups:
                print(",  Current learning rate is: {}".format(param_group['lr']))

    length = len(train_loader.dataset) // batch_size
    return train_loss / length, train_acc / length
def test(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="eval", mininterval=1):
            output = model(x)
            pred = output.data.max(1)[1]
            correct += pred.eq(y.data).sum()

    acc = 100.0 * float(correct) / len(test_loader.dataset)
    return acc
def train(train_loader, initial_learning_rate, test_loader=[], initial_momentum=0, epoch_num=5):
    model = resnet18()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate)#, momentum=initial_momentum)

    for epoch in range(1, epoch_num+1):
        train_one_epoch(model, train_loader, optimizer, criterion, epoch)
        #acc = test(model, test_loader)
        #print("Test accuracy: {.3f}%".format(acc))
