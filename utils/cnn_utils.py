import torch
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def trec_accuracy(preds, y):
	"""
	Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
	"""
	max_preds = preds.argmax(dim=1, keepdim=True) # get the index of the max probability
	correct = max_preds.squeeze(1).eq(y)
	correct = correct.detach().to('cpu')
	return correct.sum() / torch.FloatTensor([y.shape[0]])

def train(model, iterator, optimizer, criterion, tokenizer):
	'''
	train model for one epoch
	'''
	global device
	epoch_loss = 0
	epoch_acc = 0
	model.train()
	half = len(iterator) // 2
	for i, batch in enumerate(iterator):
		if i <= half:
			batch_ = torch.stack(batch['input_ids'], dim=0).permute(1, 0)
			batch_ = batch_.to(device)
			optimizer.zero_grad()
			predictions = model(batch_)
			loss = criterion(predictions, batch['label-coarse'].long().to(device))
			acc = trec_accuracy(predictions, batch['label-coarse'].to(device))
			loss.backward()
			optimizer.step()
			epoch_loss += loss.item()
			epoch_acc += acc.item()
		else:
			break
	return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, tokenizer):
	'''
	evaluate model for one epoch
	'''
	global device
	epoch_loss = 0
	epoch_acc = 0
	model.eval()
	with torch.no_grad():
		for batch in iterator:
			batch_ = torch.stack(batch['input_ids'], dim=0).permute(1, 0)
			batch_ = batch_.to(device)
			predictions = model(batch_)
			loss = criterion(predictions, batch['label-coarse'].long().to(device))
			acc = trec_accuracy(predictions, batch['label-coarse'].to(device))
			epoch_loss += loss.item()
			epoch_acc += acc.item()
	return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
	'''
	compute elapsed time for training
	'''
	elapsed_time = end_time - start_time
	elapsed_mins = int(elapsed_time / 60)
	elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
	return elapsed_mins, elapsed_secs