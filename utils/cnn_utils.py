import torch
import torch.nn as nn


def trec_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    correct = []
    #round predictions to the closest integer
    rounded_preds = torch.round(torch.softmax(preds, dim=1))
    for x, yy in zip(rounded_preds, y): 
        correct.append((torch.argmax(x) == yy).float()) #convert into float for division
    acc = np.sum(correct) / len(correct)
    return acc



def train_model(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label.type(torch.LongTensor).to(device))
        acc = trec_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)



def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label.type(torch.LongTensor).to(device))
            acc = trec_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
