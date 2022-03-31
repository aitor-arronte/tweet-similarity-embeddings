from sentence_transformers import SentenceTransformer, util
from transformers import AutoModel, AutoTokenizer
from nltk.tokenize import NLTKWordTokenizer
from nltk.corpus import stopwords
import torch


model = SentenceTransformer('bert-base-nli-mean-tokens')
word_tokenizer = NLTKWordTokenizer()
#tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
#tweet_model = AutoModel.from_pretrained("vinai/bertweet-base")

stop_words = set(stopwords.words('english'))


def tweet_similarity(sent, sentences):
	emb1 = model.encode([sent])
	emb2 = model.encode(sentences)
	cosine = util.cos_sim(emb1, emb2)
	indices=[]
	for i in range(0, cosine.size(1)):
		if cosine[0][i] > 0.6:
			indices.append(i)
	return indices


def normalize_text(sentences):
	normalized_sentences =[]
	for txt in sentences:
		tkns = word_tokenizer.tokenize(txt)
		tkns = [''.join(t.split('-')).lower() for t in tkns if t not in stop_words and t not in '@.,!#$%*:;"']
		normalized_sentences.append(' '.join(tkns))
	return normalized_sentences
