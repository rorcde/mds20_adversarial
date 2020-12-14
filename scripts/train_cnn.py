import sys
import os 
sys.path.append('../')
import time
from models.classifiers import TextCNN
from utils.cnn_utils import train, evaluate, epoch_time
from utils.prepare_dataset import prepare_trec_dataset
import torch
import torch.nn as nn 
import argparse
import logging
logging.basicConfig(level=logging.INFO, filename='train_cnn_logs.log')


def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=64,
		help='batch_size for CNN model training, default = 64')
	parser.add_argument('--emb_dim', type=int, default=100,
		help='embedding dimension for CNN model, default=100')
	parser.add_argument('--n_filters', type=int, default=8,
		help='number of filters for CNN model, default = 8')
	parser.add_argument('--filter_sizes_1', type=int, default=3,
		help='first filter size for CNN model, default=3')
	parser.add_argument('--filter_sizes_2', type=int, default=4,
		help='second filter size for CNN model, default=4')
	parser.add_argument('--filter_sizes_3', type=int, default=5,
		help='third filter size for CNN model, default=5')
	parser.add_argument('--out_dim', type=int, default=6,
		help='dimension of the output for CNN model, default=6')
	parser.add_argument('--dropout', type=float, default=0.1,
		help='dropout for CNN model, default=0.1')
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
	PATH = os.path.join('../data/')
	trainloader, testloader, tokenizer = prepare_trec_dataset(batch_size=args.batch_size)

	model = TextCNN(
		vocab_size=tokenizer.vocab_size,
		emb_dim=args.emb_dim,
		n_filters=args.n_filters,
		filter_sizes=[args.filter_sizes_1, args.filter_sizes_2, args.filter_sizes_3],
		dropout=args.dropout,
		output_dim=args.out_dim,
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
			torch.save(model.state_dict(),PATH+'textcnn_trec.pt')
		print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
		print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
		print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

		if epoch >= 3:
			if train_loss >= losses[-1] and train_loss >= losses[-2] and train_loss >= losses[-3]:
				print('Early stopping')
				break