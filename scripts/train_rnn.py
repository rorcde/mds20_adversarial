import sys
import os
sys.path.append('../')
import time
from models.classifiers import TextGRU
from utils.rnn_utils import train, evaluate, epoch_time
from utils.prepare_dataset import prepare_trec_dataset
import torch
import torch.nn as nn 
import argparse
import logging
logging.basicConfig(level=logging.INFO, filename='train_rnn_logs.log')


def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=64,
		help='batch_size for RNN model training, default = 64')
	parser.add_argument('--hidden_size', type=int, default=128,
		help='hidden_size for RNN model, default = 128')
	parser.add_argument('--emb_dim', type=int, default=100,
		help='embedding dimension for RNN model, default=100')
	parser.add_argument('--out_dim', type=int, default=6,
		help='dimension of the output for RNN model, default=6')
	parser.add_argument('--dropout', type=float, default=0.1,
		help='dropout for RNN model, default=0.1')
	parser.add_argument('--learning_rate', type=float, default=0.001,
		help='learning_rate for RNN model training, default = 0.001')
	parser.add_argument('--path_to_save', type=str, default='../data/',
		help='path to save the trained model, default= "../data/"')
	parser.add_argument('--n_epochs', type=int, default=50,
		help='number of epochs to train rnn model, default=50')
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = parse_arguments()
	PATH = os.path.join(args.path_to_save)
	trainloader, testloader, tokenizer = prepare_trec_dataset(batch_size=args.batch_size)

	model = TextGRU(
		vocab_size=tokenizer.vocab_size,
		emb_dim=args.emb_dim,
		hidden_dim=args.hidden_size,
		out_dim=args.out_dim,
		dropout=args.dropout,
		pad_idx=tokenizer.pad_token_id)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
	criterion = nn.CrossEntropyLoss()
	model = model.to(device)
	criterion = criterion.to(device)


	N_EPOCHS = args.n_epochs
	best_valid_loss = float('inf')
	losses = []
	for epoch in range(N_EPOCHS):
		start_time = time.time()
		train_loss, train_acc = train(model, trainloader, optimizer, criterion, tokenizer)
		valid_loss, valid_acc = evaluate(model, testloader, criterion, tokenizer)
		losses.append(train_loss)
		end_time = time.time()
		epoch_mins, epoch_secs = epoch_time(start_time, end_time)
		if valid_loss < best_valid_loss:
			best_valid_loss = valid_loss
			torch.save(model.state_dict(), PATH+'textrnn_trec.pt')
		print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
		print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
		print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

		if epoch >= 3:
			if train_loss >= losses[-1] and train_loss >= losses[-2] and train_loss >= losses[-3]:
				print('Early stopping')
				break


