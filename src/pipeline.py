import re
import os
import operator
import csv
import sys
import pandas as pd
from TextPreprocessor import TextPreprocessor
from FeatureExtractor import FeatureExtractor
from main.parser import Parser
import util.utils as utils
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier




tp = TextPreprocessor("greek_stopwords.txt")
fe = FeatureExtractor("foreis3.csv",tp,0.015)
df_train_org = fe.extract_organisational_features_from_file("org_training_plus.csv")



ratio = 0.04



respas_p_df = tp.getParagraphsFromFolder("2/RespAs/",-1)
respas = tp.getTermFrequency(list(respas_p_df['StemmedParagraph']))
most_frequent_respas_stems_ordered = respas[0]
weights = respas[1]

non_respas_p_df = tp.getParagraphsFromFolder("2/Non-RespAs/",-1)
non_respas = tp.getTermFrequency(list(non_respas_p_df['StemmedParagraph']))
most_frequent_non_respas_stems_ordered = non_respas[0]


# CREATE TRIE INDEX FOR RESPAS BELOW

num_non_respa_docs = len(non_respas_p_df.index)

trie = utils.create_trie_index(most_frequent_non_respas_stems_ordered,most_frequent_respas_stems_ordered,num_non_respa_docs,ratio)

# CREATE TRIE INDEX FOR RESPAS ABOVE

df_train_respa = pd.DataFrame()
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

respa_used_features = ['MatchedPatternsCount','UnqMatchedPatternsCount','TotalMatchingCharacters','SumMatchingEntries','SumMatchingEntriesLength','MatchingUnigrams','MatchingBigrams','TotalMatchingRatio','UnqMatchingUnigrams','UnqMatchingBigrams','AlphaBulletDot','AlphaBulletPar','AlphaBulletCapDot','AlphaBulletCapPar','DigitBulletDot','DigitBulletPar']


for index,row in respas_p_df.iterrows():
	raw_paragraph = row['RawParagraph']
	stemmed_paragraph = row['StemmedParagraph']
	words_in_capital = tp.get_words_in_capital(raw_paragraph)
	first_capital_word_offset = tp.get_first_word_in_capital_offset(raw_paragraph)
	isAlphaBulletDot = fe.regex_applies(regexAlphaDot,raw_paragraph)
	isAlphaBulletPar = fe.regex_applies(regexAlphaPar,raw_paragraph)
	isAlphaBulletCapDot = fe.regex_applies(regexAlphaCapDot,raw_paragraph)
	isAlphaBulletCapPar = fe.regex_applies(regexAlphaCapPar,raw_paragraph)
	isDigitBulletDot = fe.regex_applies(regexNumDot,raw_paragraph)
	isDigitBulletPar = fe.regex_applies(regexNumPar,raw_paragraph)
	all_patterns = list(trie.search_all_patterns(stemmed_paragraph))
	# === Code block below removes unigrams that are contained in bigrams ===
	subpatterns = utils.remove_unigrams_contained_in_bigrams(all_patterns)
	# === Code block above removes unigrams that are contained in bigrams ===

	pattern_features = fe.extract_features_from_trie_patterns(subpatterns,weights)
	organisational_features = fe.extract_organisational_features(stemmed_paragraph)
	df_train_respa = df_train_respa.append({'Class':'RespA','UnqMatchedPatternsCount':pattern_features['UnqMatchedPatternsCount'],'MatchedPatternsCount':pattern_features['MatchedPatternsCount'],'OrgTotalMatchingCharacters':organisational_features['OrgTotalMatchingCharacters'],'OrgMatchingUnigrams':organisational_features['OrgMatchingUnigrams'],'OrgMatchingBigrams':organisational_features['OrgMatchingBigrams'],'TotalMatchingCharacters':pattern_features['TotalMatchingCharacters'],'LongestMatchingPattern':pattern_features['LongestMatchingPattern'], 'SumMatchingEntries':pattern_features['SumMatchingEntries'],'SumMatchingEntriesLength':pattern_features['SumMatchingEntriesLength'],'MatchingUnigrams':pattern_features['MatchingUnigrams'],'MatchingBigrams':pattern_features['MatchingBigrams'],'AlphaBulletDot':isAlphaBulletDot,'AlphaBulletPar':isAlphaBulletPar,'AlphaBulletCapDot':isAlphaBulletCapDot,'AlphaBulletCapPar':isAlphaBulletCapPar,'DigitBulletDot':isDigitBulletDot,'DigitBulletPar':isDigitBulletPar,'RawParagraph':raw_paragraph,'StemmedParagraph':stemmed_paragraph,'StemmedParagraphLength':len(stemmed_paragraph),'RawParagraphLength':len(raw_paragraph),'FirstPatternOffset':pattern_features['FirstPatternOffset'],'WordsInCapital':words_in_capital,'FirstWordInCapitalOffset':first_capital_word_offset,'UnqMatchingUnigrams':pattern_features['UnqMatchingUnigrams'],'UnqMatchingBigrams':pattern_features['UnqMatchingBigrams']}, ignore_index=True)


for index,row in non_respas_p_df.iterrows():
	raw_paragraph = row['RawParagraph']
	stemmed_paragraph = row['StemmedParagraph']
	words_in_capital = tp.get_words_in_capital(raw_paragraph)
	first_capital_word_offset = tp.get_first_word_in_capital_offset(raw_paragraph)
	isAlphaBulletDot = fe.regex_applies(regexAlphaDot,raw_paragraph)
	isAlphaBulletPar = fe.regex_applies(regexAlphaPar,raw_paragraph)
	isAlphaBulletCapDot = fe.regex_applies(regexAlphaCapDot,raw_paragraph)
	isAlphaBulletCapPar = fe.regex_applies(regexAlphaCapPar,raw_paragraph)
	isDigitBulletDot = fe.regex_applies(regexNumDot,raw_paragraph)
	isDigitBulletPar = fe.regex_applies(regexNumPar,raw_paragraph)

	all_patterns = list(trie.search_all_patterns(stemmed_paragraph))
	# === Code block below removes unigrams that are contained in bigrams ===
	subpatterns = utils.remove_unigrams_contained_in_bigrams(all_patterns)
	# === Code block above removes unigrams that are contained in bigrams ===

	pattern_features = fe.extract_features_from_trie_patterns(subpatterns,weights)
	organisational_features = fe.extract_organisational_features(stemmed_paragraph)
	df_train_respa = df_train_respa.append({'Class':'Non-RespA','UnqMatchedPatternsCount':pattern_features['UnqMatchedPatternsCount'],'MatchedPatternsCount':pattern_features['MatchedPatternsCount'],'OrgTotalMatchingCharacters':organisational_features['OrgTotalMatchingCharacters'],'OrgMatchingUnigrams':organisational_features['OrgMatchingUnigrams'],'OrgMatchingBigrams':organisational_features['OrgMatchingBigrams'],'TotalMatchingCharacters':pattern_features['TotalMatchingCharacters'],'LongestMatchingPattern':pattern_features['LongestMatchingPattern'], 'SumMatchingEntries':pattern_features['SumMatchingEntries'],'SumMatchingEntriesLength':pattern_features['SumMatchingEntriesLength'],'MatchingUnigrams':pattern_features['MatchingUnigrams'],'MatchingBigrams':pattern_features['MatchingBigrams'],'AlphaBulletDot':isAlphaBulletDot,'AlphaBulletPar':isAlphaBulletPar,'AlphaBulletCapDot':isAlphaBulletCapDot,'AlphaBulletCapPar':isAlphaBulletCapPar,'DigitBulletDot':isDigitBulletDot,'DigitBulletPar':isDigitBulletPar,'RawParagraph':raw_paragraph,'StemmedParagraph':stemmed_paragraph,'StemmedParagraphLength':len(stemmed_paragraph),'RawParagraphLength':len(raw_paragraph),'FirstPatternOffset':pattern_features['FirstPatternOffset'],'WordsInCapital':words_in_capital,'FirstWordInCapitalOffset':first_capital_word_offset,'UnqMatchingUnigrams':pattern_features['UnqMatchingUnigrams'],'UnqMatchingBigrams':pattern_features['UnqMatchingBigrams']}, ignore_index=True)

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
				words_in_capital = tp.get_words_in_capital(raw_paragraph)
				first_capital_word_offset = tp.get_first_word_in_capital_offset(raw_paragraph)
				isAlphaBulletDot = fe.regex_applies(regexAlphaDot,raw_paragraph)
				isAlphaBulletPar = fe.regex_applies(regexAlphaPar,raw_paragraph)
				isAlphaBulletCapDot = fe.regex_applies(regexAlphaCapDot,raw_paragraph)
				isAlphaBulletCapPar = fe.regex_applies(regexAlphaCapPar,raw_paragraph)
				isDigitBulletDot = fe.regex_applies(regexNumDot,raw_paragraph)
				isDigitBulletPar = fe.regex_applies(regexNumPar,raw_paragraph)
				all_patterns = list(trie.search_all_patterns(stemmed_paragraph))
				# === Code block below removes unigrams that are contained in bigrams ===
				subpatterns = utils.remove_unigrams_contained_in_bigrams(all_patterns)
				# === Code block above removes unigrams that are contained in bigrams ===
				pattern_features = fe.extract_features_from_trie_patterns(subpatterns,weights)
				organisational_features = fe.extract_organisational_features(stemmed_paragraph)

				paragraphs_per_article = paragraphs_per_article.append({'Filename':filename,'Class':'NoIdea','UnqMatchedPatternsCount':pattern_features['UnqMatchedPatternsCount'],'MatchedPatternsCount':pattern_features['MatchedPatternsCount'],'ArticleNo':num,'OrgTotalMatchingCharacters':organisational_features['OrgTotalMatchingCharacters'],'OrgMatchingUnigrams':organisational_features['OrgMatchingUnigrams'],'OrgMatchingBigrams':organisational_features['OrgMatchingBigrams'],'TotalMatchingCharacters':pattern_features['TotalMatchingCharacters'],'LongestMatchingPattern':pattern_features['LongestMatchingPattern'], 'SumMatchingEntries':pattern_features['SumMatchingEntries'],'SumMatchingEntriesLength':pattern_features['SumMatchingEntriesLength'],'MatchingUnigrams':pattern_features['MatchingUnigrams'],'MatchingBigrams':pattern_features['MatchingBigrams'],'AlphaBulletDot':isAlphaBulletDot,'AlphaBulletPar':isAlphaBulletPar,'AlphaBulletCapDot':isAlphaBulletCapDot,'AlphaBulletCapPar':isAlphaBulletCapPar,'DigitBulletDot':isDigitBulletDot,'DigitBulletPar':isDigitBulletPar,'PrevAlphaBulletDot':isPrevAlphaBulletDot,'PrevAlphaBulletPar':isPrevAlphaBulletPar,'PrevAlphaBulletCapDot':isPrevAlphaBulletCapDot,'PrevAlphaBulletCapPar':isPrevAlphaBulletCapPar,'PrevDigitBulletDot':isPrevDigitBulletDot,'PrevDigitBulletPar':isPrevDigitBulletPar,'RawParagraph':raw_paragraph,'StemmedParagraph':stemmed_paragraph,'StemmedParagraphLength':len(stemmed_paragraph),'RawParagraphLength':len(raw_paragraph),'FirstPatternOffset':pattern_features['FirstPatternOffset'],'WordsInCapital':words_in_capital,'FirstWordInCapitalOffset':first_capital_word_offset,'UnqMatchingUnigrams':pattern_features['UnqMatchingUnigrams'],'UnqMatchingBigrams':pattern_features['UnqMatchingBigrams']}, ignore_index=True)
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

paragraphs_per_article['TotalPrediction'] = paragraphs_per_article.apply(lambda row: utils.final_prediction(row),axis=1)

paragraphs_per_article.to_csv("article_predictions.csv",sep='\t')

