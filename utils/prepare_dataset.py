from transformers import AutoTokenizer
from datasets import load_dataset
import torch

def prepare_trec_dataset():
	tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
	train_dataset = load_dataset('trec', split='train')
	train_dataset = train_dataset.map(
		lambda e: tokenizer(e['text'], truncation=True, padding='max_length'),
		batched=True)
	test_dataset = load_dataset('trec', split='test')
	test_dataset = test_dataset.map(
		lambda e: tokenizer(e['text'], truncation=True, padding='max_length'),
		batched=True)
	trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
	testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
	return trainloader, testloader, tokenizer	