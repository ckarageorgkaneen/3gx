import re
import os
import operator
import csv
import sys
import trie_search as ts
import pandas as pd
import pickle
import utils
from parser import Parser
from greek_stemmer import GreekStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

class FeatureExtractor:

	def __init__(self,organisations_file,text_preprocessor,ratio,headersize):
		stemmer = GreekStemmer()
		v = CountVectorizer(ngram_range=(1,2),lowercase=False)
		stemmed_organisations = []
		freq_organisations = {}
		with open(organisations_file) as fp:
			pat = re.compile("[^\w\.]+")
			for cnt,line in enumerate(fp):
				#print(line)
				l = []
				clean_line = ' '.join(pat.split(line.replace('"','')))
				if clean_line != "":
					for w in clean_line.split():
						stem = stemmer.stem(w)
						l.append(stem.upper()) # create upper case stemmed organisations
					organisation = text_preprocessor.getCleanText(" ".join(l),headersize) # create one string
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
		self.headersize = headersize


	def read_org_trie_from_file(self,text):
		self.org_trie = pickle.load(open(text,"rb"))

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

	def read_training_file(self,filename):
		columns = ['UID','AlphaBulletCapDot','AlphaBulletCapPar','AlphaBulletDot','AlphaBulletPar','Class','DigitBulletDot','DigitBulletPar','FirstPatternOffset','FirstWordInCapitalOffset','LongestMatchingPattern','MatchedPatternsCount','MatchingBigrams','MatchingUnigrams','OrgMatchingBigrams','OrgMatchingUnigrams','OrgTotalMatchingCharacters','RawParagraph','RawParagraphLength','StemmedParagraph','StemmedParagraphLength','SumMatchingEntries','SumMatchingEntriesLength','TotalMatchingCharacters','UnqMatchedPatternsCount','UnqMatchingBigrams','UnqMatchingUnigrams','WordsInCapital','TotalMatchingRatio']
		train_data = pd.read_csv(filename,sep='\t')
		train_data.columns = columns
		return train_data

	def extract_features(self,stemmed_paragraph,all_patterns, weights):
		#all_patterns = list(trie.search_all_patterns(stemmed_paragraph))
		# === Code block below removes unigrams that are contained in bigrams ===
		subpatterns = utils.remove_unigrams_contained_in_bigrams(all_patterns)
		# === Code block above removes unigrams that are contained in bigrams ===
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
		for pattern in subpatterns:
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
		features = {'UnqMatchedPatternsCount':len(container),'MatchedPatternsCount':len(subpatterns),'TotalMatchingCharacters':total_matching_characters,'LongestMatchingPattern':longest_matching_pattern,'SumMatchingEntries':sum_matching_entries,'SumMatchingEntriesLength':sum_matching_entries_len,'MatchingUnigrams':matching_unigrams,'MatchingBigrams':matching_bigrams,'UnqMatchingUnigrams':unq_matching_unigrams,'UnqMatchingBigrams':unq_matching_bigrams,'FirstPatternOffset':first_pattern_offset}
		return features


	def extract_features_from_file(self,filename,weights,trie,tp,headersize):
		train_data = pd.read_csv(filename,sep='|')
		regexDep = re.compile('^([a-z]|[A-Z]|[Α-Ω]|[α-ω]|[0-9]){1,4}(\.|\))(\sΤμήμα)')
		regexAlphaDot = re.compile('^([a-z]|[α-ω]){1,4}(\.)')
		regexAlphaPar = re.compile('^([a-z]|[α-ω]){1,4}(\))')
		regexAlphaCapDot = re.compile('^([Α-Ω]|[A-Z]){1,4}(\.)')
		regexAlphaCapPar = re.compile('^([Α-Ω]|[A-Z]){1,4}(\))')
		regexNumDot = re.compile('^([0-9]){1,4}(\.)')
		regexNumPar = re.compile('^([0-9]){1,4}(\))')
		

		train_data['AlphaBulletDot'] = train_data.apply(lambda row: self.regex_applies(regexAlphaDot,row.RawParagraph),axis=1)
		train_data['AlphaBulletPar'] = train_data.apply(lambda row: self.regex_applies(regexAlphaPar,row.RawParagraph),axis=1)
		train_data['AlphaBulletCapDot'] = train_data.apply(lambda row: self.regex_applies(regexAlphaCapDot,row.RawParagraph),axis=1)
		train_data['AlphaBulletCapPar'] = train_data.apply(lambda row: self.regex_applies(regexAlphaCapPar,row.RawParagraph),axis=1)
		train_data['DigitBulletDot'] = train_data.apply(lambda row: self.regex_applies(regexNumDot,row.RawParagraph),axis=1)
		train_data['DigitBulletPar'] = train_data.apply(lambda row: self.regex_applies(regexNumPar,row.RawParagraph),axis=1)
		train_data['DigitBulletPar'] = train_data.apply(lambda row: self.regex_applies(regexNumPar,row.RawParagraph),axis=1)
		train_data['StemmedParagraph'] = train_data.apply(lambda row: tp.getStemmedParagraph(row.RawParagraph,headersize),axis=1)
		
		train_data['ExtraFeatures'] = train_data.apply(lambda row: self.extract_features(row.StemmedParagraph,trie.search_all_patterns(row.StemmedParagraph),weights),axis=1)
		
		train_data['UnqMatchedPatternsCount'] = train_data.apply(lambda row: row.ExtraFeatures['UnqMatchedPatternsCount'],axis=1)
		train_data['MatchedPatternsCount'] = train_data.apply(lambda row: row.ExtraFeatures['MatchedPatternsCount'],axis=1)
		train_data['TotalMatchingCharacters'] = train_data.apply(lambda row: row.ExtraFeatures['TotalMatchingCharacters'],axis=1)
		train_data['LongestMatchingPattern'] = train_data.apply(lambda row: row.ExtraFeatures['LongestMatchingPattern'],axis=1)
		train_data['SumMatchingEntries'] = train_data.apply(lambda row: row.ExtraFeatures['SumMatchingEntries'],axis=1)
		train_data['SumMatchingEntriesLength'] = train_data.apply(lambda row: row.ExtraFeatures['SumMatchingEntriesLength'],axis=1)
		train_data['MatchingUnigrams'] = train_data.apply(lambda row: row.ExtraFeatures['MatchingUnigrams'],axis=1)
		train_data['MatchingBigrams'] = train_data.apply(lambda row: row.ExtraFeatures['MatchingBigrams'],axis=1)
		train_data['UnqMatchingUnigrams'] = train_data.apply(lambda row: row.ExtraFeatures['UnqMatchingUnigrams'],axis=1)
		train_data['UnqMatchingBigrams'] = train_data.apply(lambda row: row.ExtraFeatures['UnqMatchingBigrams'],axis=1)
		train_data['FirstPatternOffset'] = train_data.apply(lambda row: row.ExtraFeatures['FirstPatternOffset'],axis=1)
		train_data['Class'] = train_data.apply(lambda row: row.Prediction,axis=1)
		
		train_data.drop(columns=['ExtraFeatures','Prediction'],axis=1,inplace=True)
		
		return train_data
		
	def update_organisational_features_from_file(self,filename1,filename2,weights,trie,tp,headersize):
		regexDep = re.compile('^([a-z]|[A-Z]|[Α-Ω]|[α-ω]|[0-9]){1,4}(\.|\))(\sΤμήμα)')
		regexAlphaDot = re.compile('^([a-z]|[α-ω]){1,4}(\.)')
		regexAlphaPar = re.compile('^([a-z]|[α-ω]){1,4}(\))')
		regexAlphaCapDot = re.compile('^([Α-Ω]|[A-Z]){1,4}(\.)')
		regexAlphaCapPar = re.compile('^([Α-Ω]|[A-Z]){1,4}(\))')
		regexNumDot = re.compile('^([0-9]){1,4}(\.)')
		regexNumPar = re.compile('^([0-9]){1,4}(\))')
		train_org_df = pd.read_csv(filename1,sep='\t')

		train_org_df2 = pd.read_csv(filename2,sep='|')
		train_org = train_org_df2[['RawParagraph','Prediction']]
		train_org['Class'] = train_org.apply(lambda row: row.Prediction,axis=1)
		train_org.drop(columns=['Prediction'],axis=1,inplace=True)

		isOrg = train_org['Class'] == 'Org'

		train_org = train_org[isOrg]
	

		cols = [0,3,4,5,6,7]
		train_org_df.drop(train_org_df.columns[cols],axis=1,inplace=True)
		train_org_df.columns = ['Class','RawParagraph']

		train_org_df = train_org_df.append(train_org,sort=True)

		train_org_df = train_org_df.reset_index(drop=True)

		#train_org_df = train_org_df.loc[~train_org_df.index.duplicated(keep='first')]

		train_org_df['StemmedParagraph'] = train_org_df.apply(lambda row: self.text_preprocessor.getCleanText(row['RawParagraph'],self.headersize),axis=1)
		train_org_df['BulletDepartment'] = train_org_df.apply(lambda row: self.regex_applies(regexDep,row['RawParagraph']),axis=1)
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
		regexDep = re.compile('^([a-z]|[A-Z]|[Α-Ω]|[α-ω]|[0-9]){1,4}(\.|\))(\sΤμήμα)')
		regexAlphaDot = re.compile('^([a-z]|[α-ω]){1,4}(\.)')
		regexAlphaPar = re.compile('^([a-z]|[α-ω]){1,4}(\))')
		regexAlphaCapDot = re.compile('^([Α-Ω]|[A-Z]){1,4}(\.)')
		regexAlphaCapPar = re.compile('^([Α-Ω]|[A-Z]){1,4}(\))')
		regexNumDot = re.compile('^([0-9]){1,4}(\.)')
		regexNumPar = re.compile('^([0-9]){1,4}(\))')
		train_org_df = pd.read_csv(filename,sep='\t')
		cols = [0,3,4,5,6,7]
		train_org_df.drop(train_org_df.columns[cols],axis=1,inplace=True)
		train_org_df.columns = ['Class','RawParagraph']
		train_org_df['StemmedParagraph'] = train_org_df.apply(lambda row: self.text_preprocessor.getCleanText(row['RawParagraph'],self.headersize),axis=1)
		train_org_df['BulletDepartment'] = train_org_df.apply(lambda row: self.regex_applies(regexDep,row['RawParagraph']),axis=1)
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
