from sentence_transformers import SentenceTransformer, util
from transformers import AutoModel, AutoTokenizer
from nltk.tokenize import NLTKWordTokenizer
from nltk.corpus import stopwords
import torch


model = SentenceTransformer('bert-base-nli-mean-tokens')
word_tokenizer = NLTKWordTokenizer()
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
tweet_model = AutoModel.from_pretrained("vinai/bertweet-base")

stop_words = set(stopwords.words('english'))


def tweet_similarity(sentences):
	cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
	sen1 = normalize_text(sentences)
	tk1 =tokenizer.encode(sen1, return_tensors='pt')
	tk2 = tokenizer.encode(sen2, return_tensors='pt')
	with torch.no_grad():
		embd1 = tweet_model(tk1, convert_to_tensor=True)[0]
		embd2 = tweet_model(tk2)[0]
	cosine = cos(embd1, embd2)

	if cosine >0.5:
		return 1
	else:
		return 0


def normalize_text(sentences):
	normalized_sentences =[]
	for txt in sentences:
		tkns = word_tokenizer.tokenize(txt)
		tkns = [''.join(t.split('-')).lower() for t in tkns if t not in stop_words and t not in '@.,!#$%*:;"']
		normalized_sentences.append(' '.join(tkns))
	return normalized_sentences
