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

def get_text_length(batch, tokenizer):
	'''
	get texts length for batches
	'''
	result = []
	for i in range(batch.shape[1]):
		result.append((sum(batch[:, i] != tokenizer.pad_token_id).item()))
	return torch.tensor(result, dtype=int, device='cpu')

def train(model, iterator, optimizer, criterion, tokenizer):
	'''
	train model for one epoch
	'''
	global device
	epoch_loss = 0
	epoch_acc = 0
	model.train()
	for batch in iterator:
		batch_ = torch.stack(batch['input_ids'], dim=0)
		batch_ = batch_.to(device)
		optimizer.zero_grad()
		text_lengths = get_text_length(batch_, tokenizer) 
		predictions = model(batch_, text_lengths)
		loss = criterion(predictions, batch['label-coarse'].long().to(device))
		acc = trec_accuracy(predictions, batch['label-coarse'].to(device))
		loss.backward()
		optimizer.step()
		epoch_loss += loss.item()
		epoch_acc += acc.item()
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
			batch_ = torch.stack(batch['input_ids'], dim=0)
			batch_ = batch_.to(device)
			text_lengths = get_text_length(batch_, tokenizer)
			predictions = model(batch_, text_lengths)
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