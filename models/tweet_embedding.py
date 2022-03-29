from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
from nltk.tokenize import TweetTokenizer

model = SentenceTransformer('bert-base-nli-mean-tokens')
tweet_tokenize = TweetTokenizer()
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
tweet_model = AutoModel.from_pretrained("vinai/bertweet-base")


def tweet_similarity(sen1, sen2):
	tweet_tkns1 = tokenizer(tweet_tokenize(sen1))
	tweet_tkns2 = tokenizer(tweet_tokenize(sen2))
	embd1 = tweet_model(sen1)
	embd2 = tweet_model(sen2)
	cosine = cosine_similarity(embd1, embd2)

	if cosine >0.5:
		return 1
	else:
		return 0