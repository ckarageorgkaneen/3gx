import re
import os
import operator
import csv
import sys
import trie_search as ts
import pandas as pd
from main.parser import Parser
from greek_stemmer import GreekStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

class FeatureExtractor:

	def __init__(self,organisations_file,text_preprocessor,ratio):
		stemmer = GreekStemmer()
		v = CountVectorizer(ngram_range=(1,2),lowercase=False)
		stemmed_organisations = []
		freq_organisations = {}
		with open(organisations_file) as fp:
			pat = re.compile("[^\w\.]+")
			for cnt,line in enumerate(fp):
				l = []
				cleanLine = ' '.join(pat.split(line))
				for w in cleanLine.split():
					stem = stemmer.stem(w)
					l.append(stem.upper()) # create upper case stemmed organisations
				organisation = text_preprocessor.getCleanText(" ".join(l)) # create one string
			wordgrams = v.fit(list([organisation])).vocabulary_.keys()
			for wgram in wordgrams:
				if wgram in freq_organisations:
					freq_organisations[wgram] = freq_organisations[wgram] + 1
				else:
					freq_organisations[wgram] = 1
			stemmed_organisations.append(organisation) # insert it to a list with all organisations

		temp_df = pd.DataFrame(list(freq_organisations.items()),columns=['stems','freq'])

		selected_df = temp_df[temp_df['freq']/len(stemmed_organisations)>ratio]

		maxvalue = selected_df['freq'].max()
		meanvalue = selected_df['freq'].mean()

		most_freq = selected_df[selected_df.freq > int(meanvalue)]

		freq_stems = selected_df['stems'].values.tolist()
		freqstemscopy = []
		for s in freq_stems:
			if not text_preprocessor.hasNumbers(s) and len(s) > 3:
				freqstemscopy.append(s)

		print(freqstemscopy)
		self.org_trie = ts.TrieSearch(freqstemscopy)
		self.text_preprocessor = text_preprocessor


	def get_words_in_capital(self,text):
		return self.text_preprocessor.get_words_in_capital(text)

	def regex_applies(self,regex,text):
		if regex.search(text):
			return 1
		else:
			return 0

	def get_organisational_features(self,text):
		org_total_matching_characters = 0
		org_longest_matching_pattern = 0
		org_matching_unigrams = 0
		org_matching_bigrams = 0
		for pattern,start_idx in self.org_trie.search_all_patterns(text):
			org_total_matching_characters += len(pattern)
			if (len(pattern) > org_longest_matching_pattern): org_longest_matching_pattern = len(pattern)
			if (len(pattern.split()) == 1): org_matching_unigrams += 1
			if (len(pattern.split()) == 2): org_matching_bigrams += 1
		organisational_features = {'OrgTotalMatchingCharacters':org_total_matching_characters,'OrgLongestMatchingPattern':org_longest_matching_pattern,'OrgMatchingUnigrams':org_matching_unigrams,'OrgMatchingBigrams':org_matching_bigrams}
		return organisational_features

	def train_organisations(self,filename):
		regexAlphaDot = re.compile('^([a-z]|[α-ω]){1,2}(\.)')
		regexAlphaPar = re.compile('^([a-z]|[α-ω]){1,2}(\))')
		regexAlphaCapDot = re.compile('^([Α-Ω]|[A-Z]){1,2}(\.)')
		regexAlphaCapPar = re.compile('^([Α-Ω]|[A-Z]){1,2}(\))')
		regexNumDot = re.compile('^([0-9]){1,2}(\.)')
		regexNumPar = re.compile('^([0-9]){1,2}(\))')
		train_org_df = pd.read_csv(filename,sep='\t')
		cols = [0,3,4,5,6,7]
		train_org_df.drop(train_org_df.columns[cols],axis=1,inplace=True)
		train_org_df.columns = ['Class','RawParagraph']
		train_org_df['StemmedParagraph'] = train_org_df.apply(lambda row: self.text_preprocessor.getCleanText(row['RawParagraph']),axis=1)
		train_org_df['AlphaBulletDot'] = train_org_df.apply(lambda row: self.regex_applies(regexAlphaDot,row['RawParagraph']),axis=1)
		train_org_df['AlphaBulletPar'] = train_org_df.apply(lambda row: self.regex_applies(regexAlphaPar,row['RawParagraph']),axis=1)
		train_org_df['AlphaBulletCapDot'] = train_org_df.apply(lambda row: self.regex_applies(regexAlphaCapDot,row['RawParagraph']),axis=1)
		train_org_df['AlphaBulletCapPar'] = train_org_df.apply(lambda row: self.regex_applies(regexAlphaCapPar,row['RawParagraph']),axis=1)
		train_org_df['DigitBulletDot'] = train_org_df.apply(lambda row: self.regex_applies(regexNumDot,row['RawParagraph']),axis=1)
		train_org_df['DigitBulletPar'] = train_org_df.apply(lambda row: self.regex_applies(regexNumPar,row['RawParagraph']),axis=1)
		features_df = pd.DataFrame()
		#features_df.columns = ['OrgTotalMatchingCharacters','OrgLongestMatchingPattern','OrgMatchingUnigrams','OrgMatchingBigrams']
		for index,row in train_org_df.iterrows():
			#print(row['StemmedParagraph'])
			features_df = features_df.append(self.get_organisational_features(row['StemmedParagraph']),ignore_index=True)
		new_df = pd.concat([train_org_df,features_df],axis=1)
		return new_df