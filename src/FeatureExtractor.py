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
				print(line)
				l = []
				clean_line = ' '.join(pat.split(line.replace('"','')))
				if clean_line != "":
					for w in clean_line.split():
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

		self.org_trie = ts.TrieSearch(freqstemscopy)
		self.text_preprocessor = text_preprocessor


	def get_words_in_capital(self,text):
		return self.text_preprocessor.get_words_in_capital(text)

	def regex_applies(self,regex,text):
		if regex.search(text):
			return 1
		else:
			return 0

	def extract_features_from_trie_patterns(self,patterns,weights):
		total_matching_characters = 0
		longest_matching_pattern = 0
		# the variables below refer to wordgrams
		matching_unigrams = 0
		matching_bigrams = 0
		unq_matching_unigrams = 0
		unq_matching_bigrams = 0
		first_pattern_offset = 0
		sum_matching_entries = 0
		sum_matching_entries_len = 0
		#matchedPatterns = ''
		patterns_so_far = []
		container = set()
		for pattern in patterns:
			#matchedPatterns += pattern
			#matchedPatterns += '|'
			total_matching_characters += len(pattern)
			if (len(pattern) > longest_matching_pattern): longest_matching_pattern = len(pattern)
			if (len(pattern.split()) == 1):
				if pattern not in patterns_so_far:
					unq_matching_unigrams += 1
					patterns_so_far.append(pattern)
				matching_unigrams += 1
			if (len(pattern.split()) == 2):
				if pattern not in patterns_so_far:
					unq_matching_bigrams += 1
					patterns_so_far.append(pattern)
				matching_bigrams += 1
			if pattern not in container:
				sum_matching_entries += weights.get(pattern,0.0)
				sum_matching_entries_len += (weights.get(pattern,0.0)*len(pattern))
				container.add(pattern)
		features = {'UnqMatchedPatternsCount':len(container),'MatchedPatternsCount':len(patterns),'TotalMatchingCharacters':total_matching_characters,'LongestMatchingPattern':longest_matching_pattern,'SumMatchingEntries':sum_matching_entries,'SumMatchingEntriesLength':sum_matching_entries_len,'MatchingUnigrams':matching_unigrams,'MatchingBigrams':matching_bigrams,'UnqMatchingUnigrams':unq_matching_unigrams,'UnqMatchingBigrams':unq_matching_bigrams,'FirstPatternOffset':first_pattern_offset}
		return features

	def extract_organisational_features(self,text):
		#orgMatchedPatterns = ''
		org_total_matching_characters = 0
		org_longest_matching_pattern = 0
		org_matching_unigrams = 0
		org_matching_bigrams = 0
		for pattern,start_idx in self.org_trie.search_all_patterns(text):
			# orgMatchedPatterns += pattern
			# orgMatchedPatterns += '|'
			org_total_matching_characters += len(pattern)
			if (len(pattern) > org_longest_matching_pattern): org_longest_matching_pattern = len(pattern)
			if (len(pattern.split()) == 1): org_matching_unigrams += 1
			if (len(pattern.split()) == 2): org_matching_bigrams += 1
		organisational_features = {'OrgTotalMatchingCharacters':org_total_matching_characters,'OrgLongestMatchingPattern':org_longest_matching_pattern,'OrgMatchingUnigrams':org_matching_unigrams,'OrgMatchingBigrams':org_matching_bigrams}
		return organisational_features

	def extract_organisational_features_from_file(self,filename):
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
			features_df = features_df.append(self.extract_organisational_features(row['StemmedParagraph']),ignore_index=True)
		new_df = pd.concat([train_org_df,features_df],axis=1)
		return new_df