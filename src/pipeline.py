import re
import os
import operator
import csv
import sys
import trie_search as ts
import pandas as pd
from TextPreprocessor import TextPreprocessor
from FeatureExtractor import FeatureExtractor
from main.parser import Parser
from greek_stemmer import GreekStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier




def final_prediction(row):
	if row['OrgPrediction'] == 'Org':
		return 'Org'
	elif row['RespAPrediction'] == 'RespA':
		return 'RespA'
	else:
		return 'Irrelevant'

stemmer = GreekStemmer()
v = CountVectorizer(ngram_range=(1,2),lowercase=False)
tp = TextPreprocessor("greek_stopwords.txt")
fe = FeatureExtractor("foreis.csv",tp,0.04)
df_train_org = fe.train_organisations("org_training_plus.csv")



ratio = 0.04

# CREATE ORG TIRE INDEX BELOW

# stemmed_foreis = []
# freq_foreis = {}
# with open("foreis.csv") as fp:
# 	pat = re.compile("[^\w\.]+")
# 	for cnt,line in enumerate(fp):
# 		l = []
# 		cleanLine = ' '.join(pat.split(line))
# 		for w in cleanLine.split():
# 			stem = stemmer.stem(w)
# 			l.append(stem.upper()) # create upper case stemmed organisations
# 		foreas = tp.getCleanText(" ".join(l)) # create one string
# 		wordgrams = v.fit(list([foreas])).vocabulary_.keys()
# 		for wgram in wordgrams:
# 			if wgram in freq_foreis:
# 				freq_foreis[wgram] = freq_foreis[wgram] + 1
# 			else:
# 				freq_foreis[wgram] = 1
# 		stemmed_foreis.append(foreas) # insert it to a list with all organisations

# temp_df = pd.DataFrame(list(freq_foreis.items()),columns=['stems','freq'])

# selected_df = temp_df[temp_df['freq']/len(stemmed_foreis)>ratio]

# maxvalue = selected_df['freq'].max()
# meanvalue = selected_df['freq'].mean()

# most_freq = selected_df[selected_df.freq > int(meanvalue)]

# freq_stems = selected_df['stems'].values.tolist()
# freqstemscopy = []
# for s in freq_stems:
# 	if not tp.hasNumbers(s) and len(s) > 3:
# 		freqstemscopy.append(s)

# org_trie = ts.TrieSearch(freqstemscopy)

#print(freqstemscopy)

# CREATE ORG TIRE INDEX ABOVE


# respas = tp.getTerms("2/RespAs/")
# most_frequent_respas_stems_ordered = respas[0]
# respas_paragraphs = respas[2]
# weights = respas[1]

# non_respas = tp.getTerms("2/Non-RespAs/")
# most_frequent_non_respas_stems_ordered = non_respas[0]
# non_respas_paragraphs = non_respas[2]

#print(list(respas_paragraphs.StemmedParagraph))

respas_p_df = tp.getParagraphsFromFolder("2/RespAs/",-1)
respas = tp.getTermFrequency(list(respas_p_df['StemmedParagraph']))
most_frequent_respas_stems_ordered = respas[0]
weights = respas[1]

non_respas_p_df = tp.getParagraphsFromFolder("2/Non-RespAs/",-1)
non_respas = tp.getTermFrequency(list(non_respas_p_df['StemmedParagraph']))
most_frequent_non_respas_stems_ordered = non_respas[0]

# ORIGINAL TEST BELOW

# train_respas = respas_paragraphs[0:(int(len(respas_paragraphs)*0.8))]
# test_respas = respas_paragraphs[(int(len(respas_paragraphs)*0.8))+1:len(respas_paragraphs)]

# train_non_respas = non_respas_paragraphs[0:(int(len(non_respas_paragraphs)*0.8))]
# test_non_respas = non_respas_paragraphs[(int(len(non_respas_paragraphs)*0.8))+1:len(non_respas_paragraphs)]

# ORIGINAL TEST ABOVE


# CREATE TRIE INDEX FOR RESPAS BELOW

num_non_respa_docs = len(non_respas_p_df.index)

selected_df = non_respas[0][non_respas[0]['frequency']/num_non_respa_docs>ratio]

sublist = list(selected_df['stems'])

#most_frequent_non_respas_stems = list(most_frequent_non_respas_stems_ordered['stems'])

#sublist = most_frequent_non_respas_stems[0:(int(len(most_frequent_non_respas_stems[0:len(most_frequent_non_respas_stems)])/2))] # get top 50% of non-respas

# sorted_df = pd.DataFrame(respas[3])
# sorted_df.columns = ['stems','freq']

subtraction = [x for x in list(most_frequent_respas_stems_ordered['stems']) if x not in sublist] # respas - sublist

subtraction_df = pd.DataFrame({'stems':subtraction})

new_df = pd.merge(subtraction_df,most_frequent_respas_stems_ordered,on='stems') # merge subtracted and sorted on stems column

maxvalue = new_df['frequency'].max() # get max frequency
meanvalue = new_df['frequency'].max() # get mean frequency
#most_freq = new_df[new_df.frequency > int(maxvalue*0.7)] # get items for which frequency is greater than the 70% of the max frequency

most_freq = new_df[new_df.frequency > int(meanvalue)]
freqstems = most_freq['stems'].values.tolist()
freqstemscopy=[]

for s in freqstems:
    if not tp.hasNumbers(s): 
        freqstemscopy.append(s)        
trie = ts.TrieSearch(freqstemscopy) # create trie index from sublist terms that do not contain numbers

# CREATE TRIE INDEX FOR RESPAS ABOVE

df_train_respa = pd.DataFrame()
df_test_respa = pd.DataFrame()
#df_train = df_train.append({'TMC':0,'LMP':0,'M1':0,'M2':0}, ignore_index=True)
#df_train = df_train.append({'TMC':1,'LMP':1,'M1':1,'M2':1}, ignore_index=True)

#print(df_train)

regexAlphaDot = re.compile('^([a-z]|[α-ω]){1,2}(\.)')
regexAlphaPar = re.compile('^([a-z]|[α-ω]){1,2}(\))')
regexAlphaCapDot = re.compile('^([Α-Ω]|[A-Z]){1,2}(\.)')
regexAlphaCapPar = re.compile('^([Α-Ω]|[A-Z]){1,2}(\))')
regexNumDot = re.compile('^([0-9]){1,2}(\.)')
regexNumPar = re.compile('^([0-9]){1,2}(\))')

#used_features = ['OrgTotalMatchingCharacters','OrgLongestMatchingCharacters','OrgMatchingUnigrams','OrgMatchingBigrams','TotalMatchingCharacters','LongestMatchingPattern','SumMatchingEntries','SumMatchingEntriesLength','MatchingUnigrams','MatchingBigrams','AlphaBulletDot','AlphaBulletPar','AlphaBulletCapDot','AlphaBulletCapPar','DigitBulletDot','DigitBulletPar','TotalMatchingRatio','FirstPatternOffset','WordsInCapital','FirstWordInCapitalOffset','UnqMatchingUnigrams','UnqMatchingBigrams']


org_used_features = ['OrgTotalMatchingCharacters','OrgMatchingUnigrams','OrgMatchingBigrams','AlphaBulletDot','AlphaBulletPar','AlphaBulletCapDot','AlphaBulletCapPar','DigitBulletDot','DigitBulletPar']

respa_used_features = ['MatchedPatternsCount','UnqMatchedPatternsCount','TotalMatchingCharacters','SumMatchingEntries','SumMatchingEntriesLength','MatchingUnigrams','MatchingBigrams','TotalMatchingRatio','UnqMatchingUnigrams','UnqMatchingBigrams']


for index,row in respas_p_df.iterrows():
	raw_paragraph = row['RawParagraph']
	stemmed_paragraph = row['StemmedParagraph']
	totalMatchingCharacters = 0
	longestMatchingPattern = 0
	# the variables below refer to wordgrams
	matching1grams = 0
	matching2grams = 0
	unq_matching1grams = 0
	unq_matching2grams = 0
	words_in_capital = 0
	first_capital_word_offset = 0
	first_pattern_offset = 0
	isAlphaBulletDot = 0
	isAlphaBulletPar = 0
	isAlphaBulletCapDot = 0
	isAlphaBulletCapPar = 0
	isDigitBulletDot = 0
	isDigitBulletPar = 0
	org_totalMatchingCharacters = 0
	org_longestMatchingPattern = 0
	org_matching1grams = 0
	org_matching2grams = 0
	container = set()
	sum_matching_entries = 0
	sum_matching_entries_len = 0
	# === Code block below removes unigrams that are contained in bigrams ===
	allpatterns = list(trie.search_all_patterns(stemmed_paragraph))
	unigrams = []
	subpatterns = [] # contains bigrams and unigrams that are not contained in bigrams
	first_pattern = True
	for pat,start_idx in allpatterns:
		if first_pattern:
			first_pattern_offset = start_idx
			first_pattern = False
		#print(pat)
		parts = pat.split()
		if (len(parts) == 2):
			first_word = parts[0]
			second_word = parts[1]
			unigrams.append(first_word)
			unigrams.append(second_word)
	for pat,start_idx in allpatterns:
		if pat not in unigrams:
			subpatterns.append(pat)
	# === Code block above removes unigrams that are contained in bigrams ===

	matchedPatterns = ''
	patterns_so_far = []
	for pattern in subpatterns:
		totalMatchingCharacters += len(pattern)
		matchedPatterns += pattern
		matchedPatterns += '|'
		if (len(pattern) > longestMatchingPattern): longestMatchingPattern = len(pattern)
		if (len(pattern.split()) == 1):
			if pattern not in patterns_so_far:
				unq_matching1grams += 1
				patterns_so_far.append(pattern)
			matching1grams += 1
			if pattern not in container:
				sum_matching_entries += weights.get(pattern,0.0)
				sum_matching_entries_len += (weights.get(pattern,0.0)*len(pattern))
				container.add(pattern)
		if (len(pattern.split()) == 2):
			if pattern not in patterns_so_far:
				unq_matching2grams += 1
				patterns_so_far.append(pattern)
			matching2grams += 1
			if pattern not in container:
				sum_matching_entries += weights.get(pattern,0.0)
				sum_matching_entries_len += (weights.get(pattern,0.0)*len(pattern))
				container.add(pattern)

		if regexAlphaDot.search(raw_paragraph): isAlphaBulletDot = 1
		if regexAlphaPar.search(raw_paragraph): isAlphaBulletPar = 1
		if regexAlphaCapDot.search(raw_paragraph): isAlphaBulletCapDot = 1
		if regexAlphaCapPar.search(raw_paragraph): isAlphaBulletCapPar = 1
		if regexNumDot.search(raw_paragraph): isDigitBulletDot = 1
		if regexNumPar.search(raw_paragraph): isDigitBulletPar = 1
	orgMatchedPatterns = ''
	for pattern,start_idx in fe.org_trie.search_all_patterns(stemmed_paragraph):
		orgMatchedPatterns += pattern
		orgMatchedPatterns += '|'
		org_totalMatchingCharacters += len(pattern)
		if (len(pattern) > org_longestMatchingPattern): org_longestMatchingPattern = len(pattern)
		if (len(pattern.split()) == 1): org_matching1grams += 1
		if (len(pattern.split()) == 2): org_matching2grams += 1
	words_in_capital = tp.get_words_in_capital(raw_paragraph)
	first_capital_word_offset = tp.get_first_word_in_capital_offset(raw_paragraph)
	df_train_respa = df_train_respa.append({'Class':'RespA','UnqMatchedPatternsCount':len(set(allpatterns)),'MatchedPatternsCount':len(allpatterns),'OrgTotalMatchingCharacters':org_totalMatchingCharacters,'OrgLongestMatchingCharacters':org_longestMatchingPattern,'OrgMatchingUnigrams':org_matching1grams,'OrgMatchingBigrams':org_matching2grams,'TotalMatchingCharacters':totalMatchingCharacters,'LongestMatchingPattern':longestMatchingPattern, 'SumMatchingEntries':sum_matching_entries,'SumMatchingEntriesLength':sum_matching_entries_len,'MatchingUnigrams':matching1grams,'MatchingBigrams':matching2grams,'AlphaBulletDot':isAlphaBulletDot,'AlphaBulletPar':isAlphaBulletPar,'AlphaBulletCapDot':isAlphaBulletCapDot,'AlphaBulletCapPar':isAlphaBulletCapPar,'DigitBulletDot':isDigitBulletDot,'DigitBulletPar':isDigitBulletPar,'RawParagraph':raw_paragraph,'StemmedParagraph':stemmed_paragraph,'StemmedParagraphLength':len(stemmed_paragraph),'RawParagraphLength':len(raw_paragraph),'MatchedPatterns':matchedPatterns,'OrgMatchedPatterns':orgMatchedPatterns,'FirstPatternOffset':first_pattern_offset,'WordsInCapital':words_in_capital,'FirstWordInCapitalOffset':first_capital_word_offset,'UnqMatchingUnigrams':unq_matching1grams,'UnqMatchingBigrams':unq_matching2grams}, ignore_index=True)


for index,row in non_respas_p_df.iterrows():
	raw_paragraph = row['RawParagraph']
	stemmed_paragraph = row['StemmedParagraph']
	totalMatchingCharacters = 0
	longestMatchingPattern = 0
	# the variables below refer to wordgrams
	matching1grams = 0
	matching2grams = 0
	unq_matching1grams = 0
	unq_matching2grams = 0
	words_in_capital = 0
	first_capital_word_offset = 0
	first_pattern_offset = 0
	isAlphaBulletDot = 0
	isAlphaBulletPar = 0
	isAlphaBulletCapDot = 0
	isAlphaBulletCapPar = 0
	isDigitBulletDot = 0
	isDigitBulletPar = 0
	org_totalMatchingCharacters = 0
	org_longestMatchingPattern = 0
	org_matching1grams = 0
	org_matching2grams = 0
	container = set()
	sum_matching_entries = 0
	sum_matching_entries_len = 0
	# === Code block below removes unigrams that are contained in bigrams ===
	allpatterns = list(trie.search_all_patterns(stemmed_paragraph))
	unigrams = []
	subpatterns = [] # contains bigrams and unigrams that are not contained in bigrams
	first_pattern = True
	for pat,start_idx in allpatterns:
		if first_pattern:
			first_pattern_offset = start_idx
			first_pattern = False
		#print(pat)
		parts = pat.split()
		if (len(parts) == 2):
			first_word = parts[0]
			second_word = parts[1]
			unigrams.append(first_word)
			unigrams.append(second_word)
	for pat,start_idx in allpatterns:
		if pat not in unigrams:
			subpatterns.append(pat)
	# === Code block above removes unigrams that are contained in bigrams ===

	matchedPatterns = ''
	patterns_so_far = []
	for pattern in subpatterns:
		totalMatchingCharacters += len(pattern)
		matchedPatterns += pattern
		matchedPatterns += '|'
		if (len(pattern) > longestMatchingPattern): longestMatchingPattern = len(pattern)
		if (len(pattern.split()) == 1):
			if pattern not in patterns_so_far:
				unq_matching1grams += 1
				patterns_so_far.append(pattern)
			matching1grams += 1
			if pattern not in container:
				sum_matching_entries += weights.get(pattern,0.0)
				sum_matching_entries_len += (weights.get(pattern,0.0)*len(pattern))
				container.add(pattern)
		if (len(pattern.split()) == 2):
			if pattern not in patterns_so_far:
				unq_matching2grams += 1
				patterns_so_far.append(pattern)
			matching2grams += 1
			if pattern not in container:
				sum_matching_entries += weights.get(pattern,0.0)
				sum_matching_entries_len += (weights.get(pattern,0.0)*len(pattern))
				container.add(pattern)

		if regexAlphaDot.search(raw_paragraph): isAlphaBulletDot = 1
		if regexAlphaPar.search(raw_paragraph): isAlphaBulletPar = 1
		if regexAlphaCapDot.search(raw_paragraph): isAlphaBulletCapDot = 1
		if regexAlphaCapPar.search(raw_paragraph): isAlphaBulletCapPar = 1
		if regexNumDot.search(raw_paragraph): isDigitBulletDot = 1
		if regexNumPar.search(raw_paragraph): isDigitBulletPar = 1
	orgMatchedPatterns = ''
	for pattern,start_idx in fe.org_trie.search_all_patterns(stemmed_paragraph):
		orgMatchedPatterns += pattern
		orgMatchedPatterns += '|'
		org_totalMatchingCharacters += len(pattern)
		if (len(pattern) > org_longestMatchingPattern): org_longestMatchingPattern = len(pattern)
		if (len(pattern.split()) == 1): org_matching1grams += 1
		if (len(pattern.split()) == 2): org_matching2grams += 1
	words_in_capital = tp.get_words_in_capital(raw_paragraph)
	first_capital_word_offset = tp.get_first_word_in_capital_offset(raw_paragraph)
	df_train_respa = df_train_respa.append({'Class':'Non-RespA','UnqMatchedPatternsCount':len(set(allpatterns)),'MatchedPatternsCount':len(allpatterns),'OrgTotalMatchingCharacters':org_totalMatchingCharacters,'OrgLongestMatchingCharacters':org_longestMatchingPattern,'OrgMatchingUnigrams':org_matching1grams,'OrgMatchingBigrams':org_matching2grams,'TotalMatchingCharacters':totalMatchingCharacters,'LongestMatchingPattern':longestMatchingPattern,'SumMatchingEntries':sum_matching_entries,'SumMatchingEntriesLength':sum_matching_entries_len,'MatchingUnigrams':matching1grams,'MatchingBigrams':matching2grams,'AlphaBulletDot':isAlphaBulletDot,'AlphaBulletPar':isAlphaBulletPar,'AlphaBulletCapDot':isAlphaBulletCapDot,'AlphaBulletCapPar':isAlphaBulletCapPar,'DigitBulletDot':isDigitBulletDot,'DigitBulletPar':isDigitBulletPar,'RawParagraph':raw_paragraph,'StemmedParagraph':stemmed_paragraph,'StemmedParagraphLength':len(stemmed_paragraph),'RawParagraphLength':len(raw_paragraph),'MatchedPatterns':matchedPatterns,'OrgMatchedPatterns':orgMatchedPatterns,'FirstPatternOffset':first_pattern_offset,'WordsInCapital':words_in_capital,'FirstWordInCapitalOffset':first_capital_word_offset,'UnqMatchingUnigrams':unq_matching1grams,'UnqMatchingBigrams':unq_matching2grams}, ignore_index=True)

df_train_respa['TotalMatchingRatio'] = df_train_respa.apply(lambda row: row.TotalMatchingCharacters/len(row.StemmedParagraph),axis=1)

df_train_respa.to_csv("training.csv",sep='\t')


































parser = Parser()

paragraphs_per_article = pd.DataFrame()

for root,dirs,files in os.walk("feks"):
	for filename in files:
		txt = parser.get_txt(filename.replace(".pdf",""), '/home/latex/Downloads/gsoc2018-GG-extraction-master/src/feks/', '/home/latex/Desktop/')
		articles = parser.get_articles(txt)
		for num,article in articles.items():
			article_paragraphs = parser.get_paragraphs(article)
			isPrevAlphaBulletDot = 0
			isPrevAlphaBulletPar = 0
			isPrevAlphaBulletCapDot = 0
			isPrevAlphaBulletCapPar = 0
			isPrevDigitBulletDot = 0
			isPrevDigitBulletPar = 0
			for raw_paragraph in article_paragraphs:
				stemmed_paragraph = tp.getCleanText(raw_paragraph)
				totalMatchingCharacters = 0
				longestMatchingPattern = 0
				# the variables below refer to wordgrams
				matching1grams = 0
				matching2grams = 0
				unq_matching1grams = 0
				unq_matching2grams = 0
				words_in_capital = 0
				first_capital_word_offset = 0
				first_pattern_offset = 0
				isAlphaBulletDot = 0
				isAlphaBulletPar = 0
				isAlphaBulletCapDot = 0
				isAlphaBulletCapPar = 0
				isDigitBulletDot = 0
				isDigitBulletPar = 0
				org_totalMatchingCharacters = 0
				org_longestMatchingPattern = 0
				org_matching1grams = 0
				org_matching2grams = 0
				container = set()
				sum_matching_entries = 0
				sum_matching_entries_len = 0
				# === Code block below removes unigrams that are contained in bigrams ===
				allpatterns = list(trie.search_all_patterns(stemmed_paragraph))
				unigrams = []
				subpatterns = [] # contains bigrams and unigrams that are not contained in bigrams
				first_pattern = True
				for pat,start_idx in allpatterns:
					if first_pattern:
						first_pattern_offset = start_idx
						first_pattern = False
					#print(pat)
					parts = pat.split()
					if (len(parts) == 2):
						first_word = parts[0]
						second_word = parts[1]
						unigrams.append(first_word)
						unigrams.append(second_word)
				for pat,start_idx in allpatterns:
					if pat not in unigrams:
						subpatterns.append(pat)
				# === Code block above removes unigrams that are contained in bigrams ===

				matchedPatterns = ''
				patterns_so_far = []
				for pattern in subpatterns:
					matchedPatterns += pattern
					matchedPatterns += '|'
					totalMatchingCharacters += len(pattern)
					if (len(pattern) > longestMatchingPattern): longestMatchingPattern = len(pattern)
					if (len(pattern.split()) == 1):
						if pattern not in patterns_so_far:
							unq_matching1grams += 1
							patterns_so_far.append(pattern)
						matching1grams += 1
						if pattern not in container:
							sum_matching_entries += weights.get(pattern,0.0)
							sum_matching_entries_len += (weights.get(pattern,0.0)*len(pattern))
							container.add(pattern)
					if (len(pattern.split()) == 2):
						if pattern not in patterns_so_far:
							unq_matching1grams += 1
							patterns_so_far.append(pattern)
						matching2grams += 1
						if pattern not in container:
							sum_matching_entries += weights.get(pattern,0.0)
							sum_matching_entries_len += (weights.get(pattern,0.0)*len(pattern))
							container.add(pattern)
					if regexAlphaDot.search(raw_paragraph): isAlphaBulletDot = 1
					if regexAlphaPar.search(raw_paragraph): isAlphaBulletPar = 1
					if regexAlphaCapDot.search(raw_paragraph): isAlphaBulletCapDot = 1
					if regexAlphaCapPar.search(raw_paragraph): isAlphaBulletCapPar = 1
					if regexNumDot.search(raw_paragraph): isDigitBulletDot = 1
					if regexNumPar.search(raw_paragraph): isDigitBulletPar = 1
				orgMatchedPatterns = ''
				for pattern,start_idx in fe.org_trie.search_all_patterns(stemmed_paragraph):
					orgMatchedPatterns += pattern
					orgMatchedPatterns += '|'
					org_totalMatchingCharacters += len(pattern)
					if (len(pattern) > org_longestMatchingPattern): org_longestMatchingPattern = len(pattern)
					if (len(pattern.split()) == 1): org_matching1grams += 1
					if (len(pattern.split()) == 2): org_matching2grams += 1
				words_in_capital = tp.get_words_in_capital(raw_paragraph)
				first_capital_word_offset = tp.get_first_word_in_capital_offset(raw_paragraph)
				paragraphs_per_article = paragraphs_per_article.append({'Filename':filename,'Class':'NoIdea','UnqMatchedPatternsCount':len(set(allpatterns)),'MatchedPatternsCount':len(allpatterns),'ArticleNo':num,'OrgTotalMatchingCharacters':org_totalMatchingCharacters,'OrgLongestMatchingCharacters':org_longestMatchingPattern,'OrgMatchingUnigrams':org_matching1grams,'OrgMatchingBigrams':org_matching2grams,'TotalMatchingCharacters':totalMatchingCharacters,'LongestMatchingPattern':longestMatchingPattern, 'SumMatchingEntries':sum_matching_entries,'SumMatchingEntriesLength':sum_matching_entries_len,'MatchingUnigrams':matching1grams,'MatchingBigrams':matching2grams,'AlphaBulletDot':isAlphaBulletDot,'AlphaBulletPar':isAlphaBulletPar,'AlphaBulletCapDot':isAlphaBulletCapDot,'AlphaBulletCapPar':isAlphaBulletCapPar,'DigitBulletDot':isDigitBulletDot,'DigitBulletPar':isDigitBulletPar,'PrevAlphaBulletDot':isPrevAlphaBulletDot,'PrevAlphaBulletPar':isPrevAlphaBulletPar,'PrevAlphaBulletCapDot':isPrevAlphaBulletCapDot,'PrevAlphaBulletCapPar':isPrevAlphaBulletCapPar,'PrevDigitBulletDot':isPrevDigitBulletDot,'PrevDigitBulletPar':isPrevDigitBulletPar,'RawParagraph':raw_paragraph,'StemmedParagraph':stemmed_paragraph,'StemmedParagraphLength':len(stemmed_paragraph),'RawParagraphLength':len(raw_paragraph),'MatchedPatterns':matchedPatterns,'OrgMatchedPatterns':orgMatchedPatterns,'FirstPatternOffset':first_pattern_offset,'WordsInCapital':words_in_capital,'FirstWordInCapitalOffset':first_capital_word_offset,'UnqMatchingUnigrams':unq_matching1grams,'UnqMatchingBigrams':unq_matching2grams}, ignore_index=True)
				isPrevAlphaBulletDot = isAlphaBulletDot
				isPrevAlphaBulletPar = isAlphaBulletPar
				isPrevAlphaBulletCapDot = isAlphaBulletCapDot
				isPrevAlphaBulletCapPar = isAlphaBulletCapPar
				isPrevDigitBulletDot = isDigitBulletDot
				isPrevDigitBulletPar = isDigitBulletPar

paragraphs_per_article['TotalMatchingRatio'] = paragraphs_per_article.apply(lambda row: (row.TotalMatchingCharacters/len(row.StemmedParagraph) if len(row.StemmedParagraph) != 0 else 0),axis=1)



respa_classifier = svm.SVC(C=1,gamma='auto')
respa_classifier.fit(df_train_respa[respa_used_features].values,df_train_respa['Class'])
respa_prediction = respa_classifier.predict(paragraphs_per_article[respa_used_features])

org_classifier = svm.SVC(C=1,gamma='auto')
org_classifier.fit(df_train_org[org_used_features].values,df_train_org['Class'])
org_prediction = org_classifier.predict(paragraphs_per_article[org_used_features])

paragraphs_per_article['RespAPrediction'] = pd.Series(respa_prediction)

paragraphs_per_article['OrgPrediction'] = pd.Series(org_prediction)

paragraphs_per_article['TotalPrediction'] = paragraphs_per_article.apply(lambda row: final_prediction(row),axis=1)

paragraphs_per_article.to_csv("article_predictions.csv",sep='\t')

#df_train_respa.to_csv("entire_training.csv",sep='\t')

# for p in train_respas:
# 	totalMatchingCharacters = 0
# 	longestMatchingPattern = 0
# 	# the variables below refer to wordgrams
# 	matching1grams = 0
# 	matching2grams = 0
# 	isAlphaBulletWithPar = 0
# 	isAlphaBulletWithDot = 0
# 	isDigitBullet = 0
# 	container = set()
# 	sum_matching_entries = 0
# 	sum_matching_entries_len = 0
# 	# === Code block below removes unigrams that are contained in bigrams ===
# 	allpatterns = [i[0] for i in trie.search_all_patterns(p)]
# 	unigrams = []
# 	subpatterns = [] # contains bigrams and unigrams that are not contained in bigrams
# 	for pat in allpatterns:
# 		#print(pat)
# 		parts = pat.split()
# 		if (len(parts) == 2):
# 			first_word = parts[0]
# 			second_word = parts[1]
# 			unigrams.append(first_word)
# 			unigrams.append(second_word)
# 	for pat in allpatterns:
# 		if pat not in unigrams:
# 			subpatterns.append(pat)
# 	# === Code block above removes unigrams that are contained in bigrams ===

# 	for pattern in subpatterns:
# 		totalMatchingCharacters += len(pattern)
# 		if (len(pattern) > longestMatchingPattern): longestMatchingPattern = len(pattern)
# 		if (len(pattern.split()) == 1):
# 			matching1grams += 1
# 			if pattern not in container:
# 				sum_matching_entries += weights.get(pattern,0.0)
# 				sum_matching_entries_len += (weights.get(pattern,0.0)*len(pattern))
# 				container.add(pattern)
# 		if (len(pattern.split()) == 2):
# 			matching2grams += 1
# 			if pattern not in container:
# 				sum_matching_entries += weights.get(pattern,0.0)
# 				sum_matching_entries_len += (weights.get(pattern,0.0)*len(pattern))
# 				container.add(pattern)
# 		if (p[:1].isdigit()): isDigitBullet = 1
# 		elif ((p[:1].isalpha() and p[1:2] == ")") or p[:1].isalpha() and p[1:2] == " "): isAlphaBulletWithPar = 1
# 		elif (p[:1].isalpha() and p[1:2] == ")"): isAlphaBulletWithDot = 1
# 	df_train = df_train.append({'Class':'RespA','TMC':totalMatchingCharacters,'LMP':longestMatchingPattern, 'SME':sum_matching_entries,'SMEL':sum_matching_entries_len,'M1':matching1grams,'M2':matching2grams,'AB':isAlphaBulletWithPar,'DB':isDigitBullet,'ABD':isAlphaBulletWithDot,'Paragraph':p}, ignore_index=True)


# for p in train_non_respas:
# 	totalMatchingCharacters = 0
# 	longestMatchingPattern = 0
# 	# the variables below refer to wordgrams
# 	matching1grams = 0
# 	matching2grams = 0
# 	isAlphaBulletWithPar = 0
# 	isAlphaBulletWithDot = 0
# 	isDigitBullet = 0
# 	container = set()
# 	sum_matching_entries = 0
# 	sum_matching_entries_len = 0
# 	# === Code block below removes unigrams that are contained in bigrams ===
# 	allpatterns = [i[0] for i in trie.search_all_patterns(p)]
# 	unigrams = []
# 	subpatterns = [] # contains bigrams and unigrams that are not contained in bigrams
# 	for pat in allpatterns:
# 		#print(pat)
# 		parts = pat.split()
# 		if (len(parts) == 2):
# 			first_word = parts[0]
# 			second_word = parts[1]
# 			unigrams.append(first_word)
# 			unigrams.append(second_word)
# 	for pat in allpatterns:
# 		if pat not in unigrams:
# 			subpatterns.append(pat)
# 	# === Code block above removes unigrams that are contained in bigrams ===

# 	for pattern in subpatterns:
# 		totalMatchingCharacters += len(pattern)
# 		if (len(pattern) > longestMatchingPattern): longestMatchingPattern = len(pattern)
# 		if (len(pattern.split()) == 1):
# 			matching1grams += 1
# 			if pattern not in container:
# 				sum_matching_entries += weights.get(pattern,0.0)
# 				sum_matching_entries_len += (weights.get(pattern,0.0)*len(pattern))
# 				container.add(pattern)
# 		if (len(pattern.split()) == 2):
# 			matching2grams += 1
# 			if pattern not in container:
# 				sum_matching_entries += weights.get(pattern,0.0)
# 				sum_matching_entries_len += (weights.get(pattern,0.0)*len(pattern))
# 				container.add(pattern)
# 		if (p[:1].isdigit()): isDigitBullet = 1
# 		elif ((p[:1].isalpha() and p[1:2] == ")") or p[:1].isalpha() and p[1:2] == " "): isAlphaBulletWithPar = 1
# 		elif (p[:1].isalpha() and p[1:2] == ")"): isAlphaBulletWithDot = 1
# 	df_train = df_train.append({'Class':'Non-RespA','TMC':totalMatchingCharacters,'LMP':longestMatchingPattern, 'SME':sum_matching_entries,'SMEL':sum_matching_entries_len,'M1':matching1grams,'M2':matching2grams,'AB':isAlphaBulletWithPar,'DB':isDigitBullet,'ABD':isAlphaBulletWithDot,'Paragraph':p}, ignore_index=True)


# for p in test_respas:
# 	totalMatchingCharacters = 0
# 	longestMatchingPattern = 0
# 	# the variables below refer to wordgrams
# 	matching1grams = 0
# 	matching2grams = 0
# 	isAlphaBulletWithPar = 0
# 	isAlphaBulletWithDot = 0
# 	isDigitBullet = 0
# 	container = set()
# 	sum_matching_entries = 0
# 	sum_matching_entries_len = 0
# 	# === Code block below removes unigrams that are contained in bigrams ===
# 	allpatterns = [i[0] for i in trie.search_all_patterns(p)]
# 	unigrams = []
# 	subpatterns = [] # contains bigrams and unigrams that are not contained in bigrams
# 	for pat in allpatterns:
# 		#print(pat)
# 		parts = pat.split()
# 		if (len(parts) == 2):
# 			first_word = parts[0]
# 			second_word = parts[1]
# 			unigrams.append(first_word)
# 			unigrams.append(second_word)
# 	for pat in allpatterns:
# 		if pat not in unigrams:
# 			subpatterns.append(pat)
# 	# === Code block above removes unigrams that are contained in bigrams ===

# 	for pattern in subpatterns:
# 		totalMatchingCharacters += len(pattern)
# 		if (len(pattern) > longestMatchingPattern): longestMatchingPattern = len(pattern)
# 		if (len(pattern.split()) == 1):
# 			matching1grams += 1
# 			if pattern not in container:
# 				sum_matching_entries += weights.get(pattern,0.0)
# 				sum_matching_entries_len += (weights.get(pattern,0.0)*len(pattern))
# 				container.add(pattern)
# 		if (len(pattern.split()) == 2):
# 			matching2grams += 1
# 			if pattern not in container:
# 				sum_matching_entries += weights.get(pattern,0.0)
# 				sum_matching_entries_len += (weights.get(pattern,0.0)*len(pattern))
# 				container.add(pattern)
# 		if (p[:1].isdigit()): isDigitBullet = 1
# 		elif ((p[:1].isalpha() and p[1:2] == ")") or p[:1].isalpha() and p[1:2] == " "): isAlphaBulletWithPar = 1
# 		elif (p[:1].isalpha() and p[1:2] == ")"): isAlphaBulletWithDot = 1
# 	df_test = df_test.append({'Class':'RespA','TMC':totalMatchingCharacters,'LMP':longestMatchingPattern, 'SME':sum_matching_entries,'SMEL':sum_matching_entries_len,'M1':matching1grams,'M2':matching2grams,'AB':isAlphaBulletWithPar,'DB':isDigitBullet,'ABD':isAlphaBulletWithDot,'Paragraph':p}, ignore_index=True)


# for p in test_non_respas:
# 	totalMatchingCharacters = 0
# 	longestMatchingPattern = 0
# 	# the variables below refer to wordgrams
# 	matching1grams = 0
# 	matching2grams = 0
# 	isAlphaBulletWithPar = 0
# 	isAlphaBulletWithDot = 0
# 	isDigitBullet = 0
# 	container = set()
# 	sum_matching_entries = 0
# 	sum_matching_entries_len = 0
# 	# === Code block below removes unigrams that are contained in bigrams ===
# 	allpatterns = [i[0] for i in trie.search_all_patterns(p)]
# 	unigrams = []
# 	subpatterns = [] # contains bigrams and unigrams that are not contained in bigrams
# 	for pat in allpatterns:
# 		#print(pat)
# 		parts = pat.split()
# 		if (len(parts) == 2):
# 			first_word = parts[0]
# 			second_word = parts[1]
# 			unigrams.append(first_word)
# 			unigrams.append(second_word)
# 	for pat in allpatterns:
# 		if pat not in unigrams:
# 			subpatterns.append(pat)
# 	# === Code block above removes unigrams that are contained in bigrams ===

# 	for pattern in subpatterns:
# 		totalMatchingCharacters += len(pattern)
# 		if (len(pattern) > longestMatchingPattern): longestMatchingPattern = len(pattern)
# 		if (len(pattern.split()) == 1):
# 			matching1grams += 1
# 			if pattern not in container:
# 				sum_matching_entries += weights.get(pattern,0.0)
# 				container.add(pattern)
# 		if (len(pattern.split()) == 2):
# 			matching2grams += 1
# 			if pattern not in container:
# 				sum_matching_entries += weights.get(pattern,0.0)
# 				container.add(pattern)
# 		if (p[:1].isdigit()): isDigitBullet = 1
# 		elif ((p[:1].isalpha() and p[1:2] == ")") or p[:1].isalpha() and p[1:2] == " "): isAlphaBulletWithPar = 1
# 		elif (p[:1].isalpha() and p[1:2] == ")"): isAlphaBulletWithDot = 1
# 	df_test = df_test.append({'Class':'Non-RespA','TMC':totalMatchingCharacters,'LMP':longestMatchingPattern, 'SME':sum_matching_entries,'SMEL':sum_matching_entries_len,'M1':matching1grams,'M2':matching2grams,'AB':isAlphaBulletWithPar,'DB':isDigitBullet,'ABD':isAlphaBulletWithDot,'Paragraph':p}, ignore_index=True)

# df_test['TMR'] = df_test.apply(lambda row: row.TMC/len(row.Paragraph),axis=1)


#print(df_train)

#df_train.to_csv("train.csv",sep='\t')

# gnb = GaussianNB()
# used_features = ['TMC','LMP','M1','M2','AB','DB','ABD','TMR']

# gnb.fit(df_train[used_features].values,df_train['Class'])
# y_pred = gnb.predict(df_test[used_features])

# print("Number of mislabeled points out of a total {} points : {}, Naive Bayes performance {:05.2f}%"
#       .format(
#           df_test.shape[0],
#           (df_test['Class'] != y_pred).sum(),
#           100*(1-(df_test['Class'] != y_pred).sum()/df_test.shape[0])
# ))

# clf = svm.SVC(gamma=0.001,C=100.)
# clf.fit(df_train[used_features].values,df_train['Class'])
# pred = clf.predict(df_test[used_features])

# print("Number of mislabeled points out of a total {} points : {}, SVM performance {:05.2f}%"
#       .format(
#           df_test.shape[0],
#           (df_test['Class'] != pred).sum(),
#           100*(1-(df_test['Class'] != pred).sum()/df_test.shape[0])
# ))


# ACTUAL TESTING BELOW IN FEK
# parser = Parser()
# txt = parser.get_txt('test_education', '/home/latex/Downloads/gsoc2018-GG-extraction-master/src/', '/home/latex/Desktop/')
# articles = parser.get_articles(txt)
# # for num,article in articles.items():
# # 	print('===================')
# # 	print(article)
# # 	print('===================')

# sum_art = pd.DataFrame()
# avg_art = pd.DataFrame()
# paragraphs_per_article = pd.DataFrame()
# for num,article in articles.items():
# 	article_paragraphs = parser.get_paragraphs(article)
# 	for raw_paragraph in article_paragraphs:
# 		stemmed_paragraph = tp.getCleanText(raw_paragraph)
# 		totalMatchingCharacters = 0
# 		longestMatchingPattern = 0
# 		# the variables below refer to wordgrams
# 		matching1grams = 0
# 		matching2grams = 0
# 		unq_matching1grams = 0
# 		unq_matching2grams = 0
# 		words_in_capital = 0
# 		first_capital_word_offset = 0
# 		first_pattern_offset = 0
# 		isAlphaBulletDot = 0
# 		isAlphaBulletPar = 0
# 		isAlphaBulletCapDot = 0
# 		isAlphaBulletCapPar = 0
# 		isDigitBulletDot = 0
# 		isDigitBulletPar = 0
# 		org_totalMatchingCharacters = 0
# 		org_longestMatchingPattern = 0
# 		org_matching1grams = 0
# 		org_matching2grams = 0
# 		container = set()
# 		sum_matching_entries = 0
# 		sum_matching_entries_len = 0
# 		# === Code block below removes unigrams that are contained in bigrams ===
# 		allpatterns = list(trie.search_all_patterns(stemmed_paragraph))
# 		unigrams = []
# 		subpatterns = [] # contains bigrams and unigrams that are not contained in bigrams
# 		first_pattern = True
# 		for pat,start_idx in allpatterns:
# 			if first_pattern:
# 				first_pattern_offset = start_idx
# 				first_pattern = False
# 			#print(pat)
# 			parts = pat.split()
# 			if (len(parts) == 2):
# 				first_word = parts[0]
# 				second_word = parts[1]
# 				unigrams.append(first_word)
# 				unigrams.append(second_word)
# 		for pat,start_idx in allpatterns:
# 			if pat not in unigrams:
# 				subpatterns.append(pat)
# 		# === Code block above removes unigrams that are contained in bigrams ===

# 		matchedPatterns = ''
# 		patterns_so_far = []
# 		for pattern in subpatterns:
# 			matchedPatterns += pattern
# 			matchedPatterns += '|'
# 			totalMatchingCharacters += len(pattern)
# 			if (len(pattern) > longestMatchingPattern): longestMatchingPattern = len(pattern)
# 			if (len(pattern.split()) == 1):
# 				if pattern not in patterns_so_far:
# 					unq_matching1grams += 1
# 					patterns_so_far.append(pattern)
# 				matching1grams += 1
# 				if pattern not in container:
# 					sum_matching_entries += weights.get(pattern,0.0)
# 					sum_matching_entries_len += (weights.get(pattern,0.0)*len(pattern))
# 					container.add(pattern)
# 			if (len(pattern.split()) == 2):
# 				if pattern not in patterns_so_far:
# 					unq_matching1grams += 1
# 					patterns_so_far.append(pattern)
# 				matching2grams += 1
# 				if pattern not in container:
# 					sum_matching_entries += weights.get(pattern,0.0)
# 					sum_matching_entries_len += (weights.get(pattern,0.0)*len(pattern))
# 					container.add(pattern)
# 			if regexAlphaDot.search(raw_paragraph): isAlphaBulletDot = 1
# 			if regexAlphaPar.search(raw_paragraph): isAlphaBulletPar = 1
# 			if regexAlphaCapDot.search(raw_paragraph): isAlphaBulletCapDot = 1
# 			if regexAlphaCapPar.search(raw_paragraph): isAlphaBulletCapPar = 1
# 			if regexNumDot.search(raw_paragraph): isDigitBulletDot = 1
# 			if regexNumPar.search(raw_paragraph): isDigitBulletPar = 1
# 		orgMatchedPatterns = ''
# 		for pattern,start_idx in org_trie.search_all_patterns(stemmed_paragraph):
# 			orgMatchedPatterns += pattern
# 			orgMatchedPatterns += '|'
# 			org_totalMatchingCharacters += len(pattern)
# 			if (len(pattern) > org_longestMatchingPattern): org_longestMatchingPattern = len(pattern)
# 			if (len(pattern.split()) == 1): org_matching1grams += 1
# 			if (len(pattern.split()) == 2): org_matching2grams += 1
# 		words_in_capital = tp.get_words_in_capital(raw_paragraph)
# 		first_capital_word_offset = tp.get_first_word_in_capital_offset(raw_paragraph)
# 		paragraphs_per_article = paragraphs_per_article.append({'Class':'NoIdea','UnqMatchedPatternsCount':len(set(allpatterns)),'MatchedPatternsCount':len(allpatterns),'ArticleNo':num,'OrgTotalMatchingCharacters':org_totalMatchingCharacters,'OrgLongestMatchingCharacters':org_longestMatchingPattern,'OrgMatchingUnigrams':org_matching1grams,'OrgMatchingBigrams':org_matching2grams,'TotalMatchingCharacters':totalMatchingCharacters,'LongestMatchingPattern':longestMatchingPattern, 'SumMatchingEntries':sum_matching_entries,'SumMatchingEntriesLength':sum_matching_entries_len,'MatchingUnigrams':matching1grams,'MatchingBigrams':matching2grams,'AlphaBulletDot':isAlphaBulletDot,'AlphaBulletPar':isAlphaBulletPar,'AlphaBulletCapDot':isAlphaBulletCapDot,'AlphaBulletCapPar':isAlphaBulletCapPar,'DigitBulletDot':isDigitBulletDot,'DigitBulletPar':isDigitBulletPar,'RawParagraph':raw_paragraph,'StemmedParagraph':stemmed_paragraph,'StemmedParagraphLength':len(stemmed_paragraph),'RawParagraphLength':len(raw_paragraph),'MatchedPatterns':matchedPatterns,'OrgMatchedPatterns':orgMatchedPatterns,'FirstPatternOffset':first_pattern_offset,'WordsInCapital':words_in_capital,'FirstWordInCapitalOffset':first_capital_word_offset,'UnqMatchingUnigrams':unq_matching1grams,'UnqMatchingBigrams':unq_matching2grams}, ignore_index=True)

# paragraphs_per_article['TotalMatchingRatio'] = paragraphs_per_article.apply(lambda row: (row.TotalMatchingCharacters/len(row.StemmedParagraph) if len(row.StemmedParagraph) != 0 else 0),axis=1)
	
# #paragraphs_per_article.to_csv("fek_features.csv",sep='\t')


# respa_classifier = svm.SVC(C=1,gamma='auto')
# respa_classifier.fit(df_train_respa[respa_used_features].values,df_train_respa['Class'])
# respa_prediction = respa_classifier.predict(paragraphs_per_article[respa_used_features])

# org_classifier = svm.SVC(C=1,gamma='auto')
# org_classifier.fit(df_train_respa[org_used_features].values,df_train_respa['Class'])
# org_prediction = org_classifier.predict(paragraphs_per_article[org_used_features])

# paragraphs_per_article['RespAPrediction'] = pd.Series(respa_prediction)

# paragraphs_per_article['OrgPrediction'] = pd.Series(org_prediction)

# def final_prediction(row):
# 	if row['OrgPrediction'] == 'Org':
# 		return 'Org'
# 	elif row['RespAPrediction'] == 'RespA':
# 		return 'RespA'
# 	else:
# 		return 'Irrelevant'

# paragraphs_per_article['TotalPrediction'] = paragraphs_per_article.apply(lambda row: final_prediction(row),axis=1)

#paragraphs_per_article.to_csv("fek_predictions.csv",sep='\t')

	#paragraphs_per_article = paragraphs_per_article.append(paragraphs_per_article.agg(['sum','mean']))
	#values = paragraphs_per_article.tail(2)
	#sum_row = values.head(1)
	#sum_row.drop('Class',axis=1,inplace=True)
	#sum_row.drop('Paragraph',axis=1,inplace=True)
	#sum_row.loc['sum','Class'] = 'RespA'
	#sum_row.loc['sum','Paragraph'] = num
	#avg_row = values.tail(1)
	#avg_row.loc['sum','Class'] = 'RespA'
	#avg_row.loc['sum','Paragraph'] = num

	#print(sum_row)
	#print(avg_row)
	#sum_art = sum_art.append(sum_row.loc['sum'], ignore_index=True)

# clf2 = svm.SVC(gamma=0.001,C=100.)
# clf2.fit(df_train[used_features].values,df_train['Class'])
# pred2 = clf.predict(df_test[used_features])

#paragraphs_per_article.drop('Class',axis=1,inplace=True)
#paragraphs_per_article['Prediction'] = pd.Series(pred2)
#paragraphs_per_article.to_csv("paideia.csv",sep='\t')

#print(paragraphs_per_article)

# rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
# rf.fit(df_train[used_features].values,df_train['Class'])
# rf_pred = rf.predict(df_test[used_features])

# print("Number of mislabeled points out of a total {} points : {}, RF performance {:05.2f}%"
#       .format(
#           df_test.shape[0],
#           (df_test['Class'] != rf_pred).sum(),
#           100*(1-(df_test['Class'] != rf_pred).sum()/df_test.shape[0])
# ))

#df_test['Prediction'] = pd.Series(pred)

#df_test.to_csv("predictions.csv",sep='\t')



# print term frequency to csv file
#with open("list.csv",'w') as myfile:
#	wr = csv.writer(myfile)
#	for i in sorted_list:
#		myfile.write(i[0] + "," + str(i[1]) + "\n")