import numpy as np
import torch


def trec_accuracy(preds, y):
    """
    Calculate multiclass average accuracy per batch
    Inputs:
        predictions - torch.tensor of shape [batch_size, n_classes]
        y - torch.tensor of shape [batch_size, 1]
    Output:
        acc - average accuracy (float)
    """
    correct = []
    # round predictions to the closest integer
    rounded_preds = torch.round(torch.softmax(preds, dim=1))
    for x, yy in zip(rounded_preds, y):
        correct.append((torch.argmax(x) == yy).float())  # convert into float for division
    acc = np.sum(correct) / len(correct)
    return acc


def train_model(model, iterator, optimizer, criterion):
    '''
    train language model (1 epoch)
    Inputs:
        model - torch model
        iterator - data iterator
        optimizer - torch optimizer
        criterion - torch loss function
    Outputs:
        average epoch loss
        average epoch accuracy
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    '''
    Evaluate model on dataset.
    Inputs:
        model - torch model
        iterator - dataset iterator
        criterion - torch loss function
    Outputs:
        average epoch loss [float]
        average epoch accuracy [float]
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def epoch_time(start_time, end_time):
    '''
    Calculate time elapsed by each epoch.
    '''
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

