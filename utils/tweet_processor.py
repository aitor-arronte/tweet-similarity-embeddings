import tweepy
from distracker import settings
from models.tweet_embedding import tweet_similarity

#sys.path.append('$HOME/.cargo/bin')

tkn= getattr(settings, "BEARER_TOKEN", None)


#GET tweets based on query and return amplification score
def get_tweets_score(q):
	tweets=[]
	sentences = []
	client = tweepy.Client(bearer_token=tkn)
	query = q+' -is:retweet lang:en'
	tweets_original = client.search_recent_tweets(query=query, tweet_fields=['author_id','conversation_id', 'in_reply_to_user_id', 'public_metrics',
	                                                                'created_at', 'referenced_tweets', 'geo'],  max_results=100)
	query2 = q + ' -is:retweet is:reply lang:en'
	tweets_res = client.search_recent_tweets(query=query2,
	                              tweet_fields=['author_id', 'conversation_id', 'in_reply_to_user_id', 'public_metrics',
	                                            'created_at', 'referenced_tweets', 'geo'], max_results=100)
	conversations =[{'conversation':t['conversation_id'], 'reference':t.referenced_tweets[0]['id']} for t in tweets_res.data if t['conversation_id']]
	conv_responses = get_message_tree(conversations)
	if tweets_original.data is not None:
		#Loop over original tweets
		for i in range(0, len(tweets_original)-1):
			if tweets_original.data[i].public_metrics['retweet_count'] > 10:
				sentences.append(tweets_res[i].data.text)
				tweets.append(tweets_res[i].data)

	if tweets_res.data is not None:
		#Loop over replies and responses
		for i in range(0, len(conv_responses)-1):
			if tweets_res.data[i].public_metrics['retweet_count'] > 10:
				sentences.append(tweets_res[i].data.text)
				tweets.append(tweets_res[i].data)
			if conv_responses[i][0].data is not None and conv_responses[i][0].data.public_metrics['retweet_count']>10:
				sentences.append(conv_responses[i][0].data.text)
				tweets.append(conv_responses[i][0].data)
			if conv_responses[i][1] is not None:
				if conv_responses[i][1].data.public_metrics['retweet_count']>10:
					sentences.append(conv_responses[i][1].data.text)
					tweets.append(conv_responses[i][1].data)
	score = 0
	if len(sentences)>1:
		#Tweet similarity given a query returns indices
		ind_similar = tweet_similarity(q, sentences)
		for i in ind_similar:
			rt = tweets[i].public_metrics['retweet_count']
			lk = tweets[i].public_metrics['like_count']
			score += (0.7*rt)+(0.3*lk)
	return score


#Based on a list of conversation ids, return the original message and response if exists
def get_message_tree(convers):
	conversations =[]
	client = tweepy.Client(bearer_token=tkn)
	for con in convers:
		if con['conversation'] == con['reference']:
			tweet = client.get_tweet(con['conversation'], tweet_fields=['author_id','conversation_id', 'in_reply_to_user_id',
			                                                            'public_metrics', 'created_at', 'referenced_tweets', 'geo'])
			conversations.append((tweet,None))
		elif con['conversation'] != con['reference']:
			tweet_conv = client.get_tweet(con['conversation'], tweet_fields=['author_id','conversation_id', 'in_reply_to_user_id',
			                                                                 'public_metrics', 'created_at', 'referenced_tweets', 'geo'])
			tweet_ref = client.get_tweet(con['reference'], tweet_fields=['author_id','conversation_id', 'in_reply_to_user_id',
			                                                             'public_metrics', 'created_at', 'referenced_tweets', 'geo'])
			conversations.append((tweet_conv, tweet_ref))
	return conversations