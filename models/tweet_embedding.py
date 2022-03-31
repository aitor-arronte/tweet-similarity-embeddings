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


#Using Bertweet to get embeddings with tokenization
def tweet_embedding(sent, sentences, tweet_model, tokenizer):
	sentences = normalize_text(sentences)
	max_len = max([len(s) for s in sentences])
	tokens = {'input_ids': [], 'attention_mask': []}
	tkn = tokenizer.encode_plus(sent, max_length=130,
                                       truncation=True, padding='max_length',
                                       return_tensors='pt')
	tokens['input_ids'].append(tkn['input_ids'][0])
	tokens['attention_mask'].append(tkn['attention_mask'][0])

	for sen in sentences:
		tkn = tokenizer.encode_plus(sen, max_length=130,
		                            truncation=True, padding='max_length',
		                            return_tensors='pt')
		tokens['input_ids'].append(tkn['input_ids'][0])
		tokens['attention_mask'].append(tkn['attention_mask'][0])

	tokens['input_ids'] = torch.stack(tokens['input_ids'])
	tokens['attention_mask'] = torch.stack(tokens['attention_mask'])

	with torch.no_grad():
		output = tweet_model(**tokens)

	tweet_embeddings = mean_pooling(output, tokens['attention_mask'])

	emb1 = tweet_embeddings[0]
	emb2 = tweet_embeddings[1:]
	cosine = util.cos_sim(emb1, emb2)
	indices = []
	for i in range(0, cosine.size(1)):
		if cosine[0][i] > 0.6:
			indices.append(i)
	return indices


#Mean Pooling using the attention mask of the tokens
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)



def normalize_text(sentences):
	normalized_sentences =[]
	for txt in sentences:
		tkns = word_tokenizer.tokenize(txt)
		tkns = [''.join(t.split('-')).lower() for t in tkns if t not in stop_words and t not in '@.,!#$%*:;"']
		normalized_sentences.append(' '.join(tkns))
	return normalized_sentences
