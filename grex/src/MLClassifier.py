import re
import pandas as pd
import pickle
import utils
from parser import Parser
from TextPreprocessor import TextPreprocessor
from FeatureExtractor import FeatureExtractor
from sklearn import svm
from functools import reduce

class MLClassifier:

	def __init__(self,stopwords_file,organisations_file,org_ratio,org_headersize):
		self.tp = TextPreprocessor(stopwords_file)
		self.fe = FeatureExtractor(organisations_file,self.tp,org_ratio,org_headersize)
		self.trie = ""
		self.weights = ""
		self.org_classifier = ""
		self.respa_classifier = ""
		self.respa_used_features = ['MatchedPatternsCount','UnqMatchedPatternsCount','TotalMatchingCharacters','SumMatchingEntries','SumMatchingEntriesLength','MatchingUnigrams','MatchingBigrams','TotalMatchingRatio','UnqMatchingUnigrams','UnqMatchingBigrams','AlphaBulletDot','AlphaBulletPar','AlphaBulletCapDot','AlphaBulletCapPar','DigitBulletDot','DigitBulletPar']
		self.org_used_features = ['BulletDepartment','OrgTotalMatchingCharacters','OrgMatchingUnigrams','OrgMatchingBigrams','AlphaBulletDot','AlphaBulletPar','AlphaBulletCapDot','AlphaBulletCapPar','DigitBulletDot','DigitBulletPar']

	def read_org_trie_from_file(self,org_pickle_file):
		self.fe.read_org_trie_from_file(org_pickle_file)

	def read_trie_index_from_file(self,trie):
		self.trie = pickle.load(open(trie,"rb"))#pickle/1trie.pkl

	def read_weights_from_file(self,weightfile):
		self.weights = pickle.load(open(weightfile,"rb"))#pickle/1weights.pkl

	def org_classifier_from_file(self,filename):
		org_train_data = self.fe.extract_organisational_features_from_file(filename)

		org_classifier = svm.SVC(C=1,gamma='auto')
		org_classifier.fit(org_train_data[self.org_used_features].values,org_train_data['Class'])

		self.org_classifier = org_classifier

	def update_org_classifier(self,oldfile,newfile,headersize):
		org_train_data = self.fe.update_organisational_features_from_file(oldfile,newfile,self.weights,self.trie,self.tp,headersize)

		org_classifier = svm.SVC(C=1,gamma='auto')
		org_classifier.fit(org_train_data[self.org_used_features].values,org_train_data['Class'])

		self.org_classifier = org_classifier

	def respa_classifier_from_file(self,filename,headersize):
		train_data = self.fe.extract_features_from_file(filename,self.weights,self.trie,self.tp,headersize)#ExampleFile.csv
		train_data['TotalMatchingRatio'] = train_data.apply(lambda row: (row.TotalMatchingCharacters/len(row.StemmedParagraph) if len(row.StemmedParagraph) != 0 else 0),axis=1)

		respa_classifier = svm.SVC(C=1,gamma='auto')
		respa_classifier.fit(train_data[self.respa_used_features].values,train_data['Class'])
		
		self.respa_classifier = respa_classifier

	def predict_pdf_file(self,filename,respa_headersize,org_headersize,pdf_directory,out_txt_directory):
		paragraphs_per_article = pd.DataFrame()

		regexAlphaDot = re.compile('^([a-z]|[α-ω]){1,4}(\.)')
		regexAlphaPar = re.compile('^([a-z]|[α-ω]){1,4}(\))')
		regexAlphaCapDot = re.compile('^([Α-Ω]|[A-Z]){1,4}(\.)')
		regexAlphaCapPar = re.compile('^([Α-Ω]|[A-Z]){1,4}(\))')
		regexNumDot = re.compile('^([0-9]){1,4}(\.)')
		regexNumPar = re.compile('^([0-9]){1,4}(\))')
		regexDep = re.compile('^([a-z]|[A-Z]|[Α-Ω]|[α-ω]|[0-9]){1,4}(\.|\))(\sΤμήμα)')


		parser = Parser()

		txt = parser.get_txt(filename.replace(".pdf",""), pdf_directory, out_txt_directory)#/home/latex/Downloads/gsoc2018-GG-extraction-master/src/respa_feks/ -- /home/latex/Desktop/respa_feks_txt/
		txt = re.sub('([0-9]){1,4}(\s\n)ΕΦΗΜΕΡΙΣ ΤΗΣ ΚΥΒΕΡΝΗΣΕΩΣ \(ΤΕΥΧΟΣ ΠΡΩΤΟ\)(\s\n)','',txt)
		articles = parser.get_articles(txt)
		for num,article in articles.items():
			articleNo = num
			article_paragraphs = parser.get_paragraphs(article)
			isPrevAlphaBulletDot = 0
			isPrevAlphaBulletPar = 0
			isPrevAlphaBulletCapDot = 0
			isPrevAlphaBulletCapPar = 0
			isPrevDigitBulletDot = 0
			isPrevDigitBulletPar = 0
			# =========== code block below splits paragraphs based on bullets ===========
			trimmed_paragraphs = []
			for p in article_paragraphs:
				sublist = list(filter(lambda x: len(x)>1, re.split('((\n)([α-ω])([α-ω])(\)))',p)))
				if (len(sublist) > 1):
					if len(sublist[0]) <= 3:
						for x in range(0,len(sublist)-1,2):
							trimmed_paragraphs.append(sublist[x] + sublist[x+1])
						if len(sublist) % 2 != 0: trimmed_paragraphs.append(sublist[len(sublist)-1])
					else:
						trimmed_paragraphs.append(sublist[0])
						for x in range(1,len(sublist)-1,2):
							trimmed_paragraphs.append(sublist[x] + sublist[x+1])
						if len(sublist) % 2 == 0: trimmed_paragraphs.append(sublist[len(sublist)-1])
				else:
					trimmed_paragraphs.append(p)
			# =========== code block above splits paragraphs based on bullets ===========
			for raw_paragraph in trimmed_paragraphs:
				stemmed_paragraph = self.tp.getCleanText(raw_paragraph,respa_headersize)
				words_in_capital = self.tp.get_words_in_capital(raw_paragraph)
				first_capital_word_offset = self.tp.get_first_word_in_capital_offset(raw_paragraph)
				isAlphaBulletDot = self.fe.regex_applies(regexAlphaDot,raw_paragraph)
				isAlphaBulletPar = self.fe.regex_applies(regexAlphaPar,raw_paragraph)
				isAlphaBulletCapDot = self.fe.regex_applies(regexAlphaCapDot,raw_paragraph)
				isAlphaBulletCapPar = self.fe.regex_applies(regexAlphaCapPar,raw_paragraph)
				isDigitBulletDot = self.fe.regex_applies(regexNumDot,raw_paragraph)
				isDigitBulletPar = self.fe.regex_applies(regexNumPar,raw_paragraph)
				isDep = self.fe.regex_applies(regexDep,raw_paragraph)
				all_patterns = list(self.trie.search_all_patterns(stemmed_paragraph))
				# === Code block below removes unigrams that are contained in bigrams ===
				subpatterns = utils.remove_unigrams_contained_in_bigrams(all_patterns)
				# === Code block above removes unigrams that are contained in bigrams ===
				pattern_features = self.fe.extract_features_from_trie_patterns(subpatterns,self.weights)
				organisational_features = self.fe.extract_organisational_features(self.tp.getCleanText(raw_paragraph,org_headersize))

				paragraphs_per_article = paragraphs_per_article.append({'Filename':filename,'Class':'NoIdea','UnqMatchedPatternsCount':pattern_features['UnqMatchedPatternsCount'],'BulletDepartment':isDep,'MatchedPatternsCount':pattern_features['MatchedPatternsCount'],'ArticleNo':articleNo,'OrgTotalMatchingCharacters':organisational_features['OrgTotalMatchingCharacters'],'OrgMatchingUnigrams':organisational_features['OrgMatchingUnigrams'],'OrgMatchingBigrams':organisational_features['OrgMatchingBigrams'],'TotalMatchingCharacters':pattern_features['TotalMatchingCharacters'],'LongestMatchingPattern':pattern_features['LongestMatchingPattern'], 'SumMatchingEntries':pattern_features['SumMatchingEntries'],'SumMatchingEntriesLength':pattern_features['SumMatchingEntriesLength'],'MatchingUnigrams':pattern_features['MatchingUnigrams'],'MatchingBigrams':pattern_features['MatchingBigrams'],'AlphaBulletDot':isAlphaBulletDot,'AlphaBulletPar':isAlphaBulletPar,'AlphaBulletCapDot':isAlphaBulletCapDot,'AlphaBulletCapPar':isAlphaBulletCapPar,'DigitBulletDot':isDigitBulletDot,'DigitBulletPar':isDigitBulletPar,'PrevAlphaBulletDot':isPrevAlphaBulletDot,'PrevAlphaBulletPar':isPrevAlphaBulletPar,'PrevAlphaBulletCapDot':isPrevAlphaBulletCapDot,'PrevAlphaBulletCapPar':isPrevAlphaBulletCapPar,'PrevDigitBulletDot':isPrevDigitBulletDot,'PrevDigitBulletPar':isPrevDigitBulletPar,'RawParagraph':raw_paragraph,'StemmedParagraph':stemmed_paragraph,'StemmedParagraphLength':len(stemmed_paragraph),'RawParagraphLength':len(raw_paragraph),'FirstPatternOffset':pattern_features['FirstPatternOffset'],'WordsInCapital':words_in_capital,'FirstWordInCapitalOffset':first_capital_word_offset,'UnqMatchingUnigrams':pattern_features['UnqMatchingUnigrams'],'UnqMatchingBigrams':pattern_features['UnqMatchingBigrams']}, ignore_index=True)
				isPrevAlphaBulletDot = isAlphaBulletDot
				isPrevAlphaBulletPar = isAlphaBulletPar
				isPrevAlphaBulletCapDot = isAlphaBulletCapDot
				isPrevAlphaBulletCapPar = isAlphaBulletCapPar
				isPrevDigitBulletDot = isDigitBulletDot
				isPrevDigitBulletPar = isDigitBulletPar

		paragraphs_per_article['TotalMatchingRatio'] = paragraphs_per_article.apply(lambda row: (row.TotalMatchingCharacters/len(row.StemmedParagraph) if len(row.StemmedParagraph) != 0 else 0),axis=1)
		respa_prediction = self.respa_classifier.predict(paragraphs_per_article[self.respa_used_features])
		org_prediction = self.org_classifier.predict(paragraphs_per_article[self.org_used_features])
		paragraphs_per_article['RespAPrediction'] = pd.Series(respa_prediction)

		paragraphs_per_article['OrgPrediction'] = pd.Series(org_prediction)

		paragraphs_per_article['Prediction'] = paragraphs_per_article.apply(lambda row: utils.total_prediction(row),axis=1)

		adj_predictions = pd.DataFrame()

		previous_prediction = ''

		for index,row in paragraphs_per_article.iterrows():
			if re.search(r'\b(ΠΕ|ΤΕ|ΔΕ|ΥΕ)\b',row['RawParagraph']) and re.search(r'\b(Κλάδος|θέση|θέσεις|θέσεων|Υπάλληλος)\b',row['RawParagraph'],re.IGNORECASE) or re.search(r'\b(θέση|θέσεις|θέσεων)\b',row['RawParagraph'],re.IGNORECASE) and re.search(r'\bΚλάδος\b',row['RawParagraph'],re.IGNORECASE):
				row['Prediction'] = 'Positions'
			elif row['AlphaBulletDot'] == row['PrevAlphaBulletDot'] and row['AlphaBulletPar'] == row['PrevAlphaBulletPar'] and row['AlphaBulletCapDot'] == row['PrevAlphaBulletCapDot'] and row['AlphaBulletCapPar'] == row['PrevAlphaBulletCapPar'] and row['DigitBulletDot'] == row['PrevDigitBulletDot'] and row['DigitBulletPar'] == row['PrevDigitBulletPar'] and previous_prediction != '':
				row['Prediction'] = previous_prediction
			adj_predictions = adj_predictions.append(row, ignore_index=True)
			lines = row['RawParagraph'].splitlines()
			total_line_len = reduce((lambda x, y: x + y), list(map(lambda l: len(l), lines)))
			line_ratio = total_line_len/len(lines)
			if (line_ratio <= 4): row['Prediction'] = 'Irrelevant'
			previous_prediction = row['Prediction']

		adj_predictions[['ArticleNo','RawParagraph','Prediction']].to_csv(filename + '.csv',sep='\t')

		return adj_predictions

	def respa_classifier_from_pdf_files(self,respa_directory,headersize1,non_respa_directory,headersize2,ratio,create_trie):
		respas_p_df = self.tp.getParagraphsFromFolder(respa_directory,headersize1)
		respas = self.tp.getTermFrequency(list(respas_p_df['StemmedParagraph']))
		most_frequent_respas_stems_ordered = respas[0]
		weights = respas[1]

		non_respas_p_df = self.tp.getParagraphsFromFolder(non_respa_directory,headersize2)
		non_respas = self.tp.getTermFrequency(list(non_respas_p_df['StemmedParagraph']))
		most_frequent_non_respas_stems_ordered = non_respas[0]

		num_non_respa_docs = len(non_respas_p_df.index)

		if (create_trie): self.trie = utils.create_trie_index(most_frequent_non_respas_stems_ordered,most_frequent_respas_stems_ordered,num_non_respa_docs,ratio,self.tp)

		df_train_respa = pd.DataFrame()

		regexAlphaDot = re.compile('^([a-z]|[α-ω]){1,4}(\.)')
		regexAlphaPar = re.compile('^([a-z]|[α-ω]){1,4}(\))')
		regexAlphaCapDot = re.compile('^([Α-Ω]|[A-Z]){1,4}(\.)')
		regexAlphaCapPar = re.compile('^([Α-Ω]|[A-Z]){1,4}(\))')
		regexNumDot = re.compile('^([0-9]){1,4}(\.)')
		regexNumPar = re.compile('^([0-9]){1,4}(\))')
		regexDep = re.compile('^([a-z]|[A-Z]|[Α-Ω]|[α-ω]|[0-9]){1,4}(\.|\))(\sΤμήμα)')

		for index,row in respas_p_df.iterrows():
			raw_paragraph = row['RawParagraph']
			stemmed_paragraph = row['StemmedParagraph']
			words_in_capital = self.tp.get_words_in_capital(raw_paragraph)
			first_capital_word_offset = self.tp.get_first_word_in_capital_offset(raw_paragraph)
			isAlphaBulletDot = self.fe.regex_applies(regexAlphaDot,raw_paragraph)
			isAlphaBulletPar = self.fe.regex_applies(regexAlphaPar,raw_paragraph)
			isAlphaBulletCapDot = self.fe.regex_applies(regexAlphaCapDot,raw_paragraph)
			isAlphaBulletCapPar = self.fe.regex_applies(regexAlphaCapPar,raw_paragraph)
			isDigitBulletDot = self.fe.regex_applies(regexNumDot,raw_paragraph)
			isDigitBulletPar = self.fe.regex_applies(regexNumPar,raw_paragraph)
			isDep = self.fe.regex_applies(regexDep,raw_paragraph)
			all_patterns = list(self.trie.search_all_patterns(stemmed_paragraph))
			# === Code block below removes unigrams that are contained in bigrams ===
			subpatterns = utils.remove_unigrams_contained_in_bigrams(all_patterns)
			# === Code block above removes unigrams that are contained in bigrams ===

			pattern_features = self.fe.extract_features_from_trie_patterns(subpatterns,weights)
			organisational_features = self.fe.extract_organisational_features(stemmed_paragraph)
			df_train_respa = df_train_respa.append({'Class':'RespA','UnqMatchedPatternsCount':pattern_features['UnqMatchedPatternsCount'],'MatchedPatternsCount':pattern_features['MatchedPatternsCount'],'BulletDepartment':isDep,'OrgTotalMatchingCharacters':organisational_features['OrgTotalMatchingCharacters'],'OrgMatchingUnigrams':organisational_features['OrgMatchingUnigrams'],'OrgMatchingBigrams':organisational_features['OrgMatchingBigrams'],'TotalMatchingCharacters':pattern_features['TotalMatchingCharacters'],'LongestMatchingPattern':pattern_features['LongestMatchingPattern'], 'SumMatchingEntries':pattern_features['SumMatchingEntries'],'SumMatchingEntriesLength':pattern_features['SumMatchingEntriesLength'],'MatchingUnigrams':pattern_features['MatchingUnigrams'],'MatchingBigrams':pattern_features['MatchingBigrams'],'AlphaBulletDot':isAlphaBulletDot,'AlphaBulletPar':isAlphaBulletPar,'AlphaBulletCapDot':isAlphaBulletCapDot,'AlphaBulletCapPar':isAlphaBulletCapPar,'DigitBulletDot':isDigitBulletDot,'DigitBulletPar':isDigitBulletPar,'RawParagraph':raw_paragraph,'StemmedParagraph':stemmed_paragraph,'StemmedParagraphLength':len(stemmed_paragraph),'RawParagraphLength':len(raw_paragraph),'FirstPatternOffset':pattern_features['FirstPatternOffset'],'WordsInCapital':words_in_capital,'FirstWordInCapitalOffset':first_capital_word_offset,'UnqMatchingUnigrams':pattern_features['UnqMatchingUnigrams'],'UnqMatchingBigrams':pattern_features['UnqMatchingBigrams']}, ignore_index=True)


		for index,row in non_respas_p_df.iterrows():
			raw_paragraph = row['RawParagraph']
			stemmed_paragraph = row['StemmedParagraph']
			words_in_capital = self.tp.get_words_in_capital(raw_paragraph)
			first_capital_word_offset = self.tp.get_first_word_in_capital_offset(raw_paragraph)
			isAlphaBulletDot = self.fe.regex_applies(regexAlphaDot,raw_paragraph)
			isAlphaBulletPar = self.fe.regex_applies(regexAlphaPar,raw_paragraph)
			isAlphaBulletCapDot = self.fe.regex_applies(regexAlphaCapDot,raw_paragraph)
			isAlphaBulletCapPar = self.fe.regex_applies(regexAlphaCapPar,raw_paragraph)
			isDigitBulletDot = self.fe.regex_applies(regexNumDot,raw_paragraph)
			isDigitBulletPar = self.fe.regex_applies(regexNumPar,raw_paragraph)
			isDep = self.fe.regex_applies(regexDep,raw_paragraph)
			all_patterns = list(self.trie.search_all_patterns(stemmed_paragraph))
			# === Code block below removes unigrams that are contained in bigrams ===
			subpatterns = utils.remove_unigrams_contained_in_bigrams(all_patterns)
			# === Code block above removes unigrams that are contained in bigrams ===

			pattern_features = self.fe.extract_features_from_trie_patterns(subpatterns,weights)
			organisational_features = self.fe.extract_organisational_features(stemmed_paragraph)
			df_train_respa = df_train_respa.append({'Class':'Non-RespA','UnqMatchedPatternsCount':pattern_features['UnqMatchedPatternsCount'],'MatchedPatternsCount':pattern_features['MatchedPatternsCount'],'BulletDepartment':isDep,'OrgTotalMatchingCharacters':organisational_features['OrgTotalMatchingCharacters'],'OrgMatchingUnigrams':organisational_features['OrgMatchingUnigrams'],'OrgMatchingBigrams':organisational_features['OrgMatchingBigrams'],'TotalMatchingCharacters':pattern_features['TotalMatchingCharacters'],'LongestMatchingPattern':pattern_features['LongestMatchingPattern'], 'SumMatchingEntries':pattern_features['SumMatchingEntries'],'SumMatchingEntriesLength':pattern_features['SumMatchingEntriesLength'],'MatchingUnigrams':pattern_features['MatchingUnigrams'],'MatchingBigrams':pattern_features['MatchingBigrams'],'AlphaBulletDot':isAlphaBulletDot,'AlphaBulletPar':isAlphaBulletPar,'AlphaBulletCapDot':isAlphaBulletCapDot,'AlphaBulletCapPar':isAlphaBulletCapPar,'DigitBulletDot':isDigitBulletDot,'DigitBulletPar':isDigitBulletPar,'RawParagraph':raw_paragraph,'StemmedParagraph':stemmed_paragraph,'StemmedParagraphLength':len(stemmed_paragraph),'RawParagraphLength':len(raw_paragraph),'FirstPatternOffset':pattern_features['FirstPatternOffset'],'WordsInCapital':words_in_capital,'FirstWordInCapitalOffset':first_capital_word_offset,'UnqMatchingUnigrams':pattern_features['UnqMatchingUnigrams'],'UnqMatchingBigrams':pattern_features['UnqMatchingBigrams']}, ignore_index=True)

		df_train_respa['TotalMatchingRatio'] = df_train_respa.apply(lambda row: row.TotalMatchingCharacters/len(row.StemmedParagraph),axis=1)

		self.respa_classifier = svm.SVC(C=1,gamma='auto')
		self.respa_classifier.fit(df_train_respa[self.respa_used_features].values,df_train_respa['Class'])

		df_train_respa.to_csv("training_file.csv",sep="\t")

		return df_train_respa

	def classifier_from_enriched_train_samples(self,oldfile,newfile,headersize1,headersize2,ratio):

		old_train_data = self.fe.read_training_file(oldfile)[['Class','StemmedParagraph','RawParagraph']]
		new_train_data_respa = self.fe.extract_features_from_file(newfile,self.weights,self.trie,self.tp,headersize1)[['Class','StemmedParagraph','RawParagraph']]
		new_train_data_non_respa = self.fe.extract_features_from_file(newfile,self.weights,self.trie,self.tp,headersize2)[['Class','StemmedParagraph','RawParagraph']]

		merged_df = old_train_data.append(new_train_data_respa).append(new_train_data_non_respa)

		merged_df = merged_df.reset_index(drop=True)

		isRespA = merged_df['Class'] == 'RespA'
		isNonRespA = merged_df['Class'] == 'Non-RespA'

		respas_p_df = merged_df[isRespA][['StemmedParagraph','RawParagraph']]
		non_respas_p_df = merged_df[isNonRespA][['StemmedParagraph','RawParagraph']]

		#respas_p_df = self.tp.getParagraphsFromFolder(respa_directory,headersize1) #stemmed, raw
		respas = self.tp.getTermFrequency(list(respas_p_df['StemmedParagraph']))
		most_frequent_respas_stems_ordered = respas[0]
		weights = respas[1]

		#non_respas_p_df = self.tp.getParagraphsFromFolder(non_respa_directory,headersize2)
		non_respas = self.tp.getTermFrequency(list(non_respas_p_df['StemmedParagraph']))
		most_frequent_non_respas_stems_ordered = non_respas[0]

		num_non_respa_docs = len(non_respas_p_df.index)

		self.trie = utils.create_trie_index(most_frequent_non_respas_stems_ordered,most_frequent_respas_stems_ordered,num_non_respa_docs,ratio,self.tp)

		df_train_respa = pd.DataFrame()

		regexAlphaDot = re.compile('^([a-z]|[α-ω]){1,4}(\.)')
		regexAlphaPar = re.compile('^([a-z]|[α-ω]){1,4}(\))')
		regexAlphaCapDot = re.compile('^([Α-Ω]|[A-Z]){1,4}(\.)')
		regexAlphaCapPar = re.compile('^([Α-Ω]|[A-Z]){1,4}(\))')
		regexNumDot = re.compile('^([0-9]){1,4}(\.)')
		regexNumPar = re.compile('^([0-9]){1,4}(\))')
		regexDep = re.compile('^([a-z]|[A-Z]|[Α-Ω]|[α-ω]|[0-9]){1,4}(\.|\))(\sΤμήμα)')

		for index,row in respas_p_df.iterrows():
			raw_paragraph = row['RawParagraph']
			stemmed_paragraph = row['StemmedParagraph']
			words_in_capital = self.tp.get_words_in_capital(raw_paragraph)
			first_capital_word_offset = self.tp.get_first_word_in_capital_offset(raw_paragraph)
			isAlphaBulletDot = self.fe.regex_applies(regexAlphaDot,raw_paragraph)
			isAlphaBulletPar = self.fe.regex_applies(regexAlphaPar,raw_paragraph)
			isAlphaBulletCapDot = self.fe.regex_applies(regexAlphaCapDot,raw_paragraph)
			isAlphaBulletCapPar = self.fe.regex_applies(regexAlphaCapPar,raw_paragraph)
			isDigitBulletDot = self.fe.regex_applies(regexNumDot,raw_paragraph)
			isDigitBulletPar = self.fe.regex_applies(regexNumPar,raw_paragraph)
			isDep = self.fe.regex_applies(regexDep,raw_paragraph)
			all_patterns = list(self.trie.search_all_patterns(stemmed_paragraph))
			# === Code block below removes unigrams that are contained in bigrams ===
			subpatterns = utils.remove_unigrams_contained_in_bigrams(all_patterns)
			# === Code block above removes unigrams that are contained in bigrams ===

			pattern_features = self.fe.extract_features_from_trie_patterns(subpatterns,weights)
			organisational_features = self.fe.extract_organisational_features(stemmed_paragraph)
			df_train_respa = df_train_respa.append({'Class':'RespA','UnqMatchedPatternsCount':pattern_features['UnqMatchedPatternsCount'],'MatchedPatternsCount':pattern_features['MatchedPatternsCount'],'BulletDepartment':isDep,'OrgTotalMatchingCharacters':organisational_features['OrgTotalMatchingCharacters'],'OrgMatchingUnigrams':organisational_features['OrgMatchingUnigrams'],'OrgMatchingBigrams':organisational_features['OrgMatchingBigrams'],'TotalMatchingCharacters':pattern_features['TotalMatchingCharacters'],'LongestMatchingPattern':pattern_features['LongestMatchingPattern'], 'SumMatchingEntries':pattern_features['SumMatchingEntries'],'SumMatchingEntriesLength':pattern_features['SumMatchingEntriesLength'],'MatchingUnigrams':pattern_features['MatchingUnigrams'],'MatchingBigrams':pattern_features['MatchingBigrams'],'AlphaBulletDot':isAlphaBulletDot,'AlphaBulletPar':isAlphaBulletPar,'AlphaBulletCapDot':isAlphaBulletCapDot,'AlphaBulletCapPar':isAlphaBulletCapPar,'DigitBulletDot':isDigitBulletDot,'DigitBulletPar':isDigitBulletPar,'RawParagraph':raw_paragraph,'StemmedParagraph':stemmed_paragraph,'StemmedParagraphLength':len(stemmed_paragraph),'RawParagraphLength':len(raw_paragraph),'FirstPatternOffset':pattern_features['FirstPatternOffset'],'WordsInCapital':words_in_capital,'FirstWordInCapitalOffset':first_capital_word_offset,'UnqMatchingUnigrams':pattern_features['UnqMatchingUnigrams'],'UnqMatchingBigrams':pattern_features['UnqMatchingBigrams']}, ignore_index=True)


		for index,row in non_respas_p_df.iterrows():
			raw_paragraph = row['RawParagraph']
			stemmed_paragraph = row['StemmedParagraph']
			words_in_capital = self.tp.get_words_in_capital(raw_paragraph)
			first_capital_word_offset = self.tp.get_first_word_in_capital_offset(raw_paragraph)
			isAlphaBulletDot = self.fe.regex_applies(regexAlphaDot,raw_paragraph)
			isAlphaBulletPar = self.fe.regex_applies(regexAlphaPar,raw_paragraph)
			isAlphaBulletCapDot = self.fe.regex_applies(regexAlphaCapDot,raw_paragraph)
			isAlphaBulletCapPar = self.fe.regex_applies(regexAlphaCapPar,raw_paragraph)
			isDigitBulletDot = self.fe.regex_applies(regexNumDot,raw_paragraph)
			isDigitBulletPar = self.fe.regex_applies(regexNumPar,raw_paragraph)
			isDep = self.fe.regex_applies(regexDep,raw_paragraph)
			all_patterns = list(self.trie.search_all_patterns(stemmed_paragraph))
			# === Code block below removes unigrams that are contained in bigrams ===
			subpatterns = utils.remove_unigrams_contained_in_bigrams(all_patterns)
			# === Code block above removes unigrams that are contained in bigrams ===

			pattern_features = self.fe.extract_features_from_trie_patterns(subpatterns,weights)
			organisational_features = self.fe.extract_organisational_features(stemmed_paragraph)
			df_train_respa = df_train_respa.append({'Class':'Non-RespA','UnqMatchedPatternsCount':pattern_features['UnqMatchedPatternsCount'],'MatchedPatternsCount':pattern_features['MatchedPatternsCount'],'BulletDepartment':isDep,'OrgTotalMatchingCharacters':organisational_features['OrgTotalMatchingCharacters'],'OrgMatchingUnigrams':organisational_features['OrgMatchingUnigrams'],'OrgMatchingBigrams':organisational_features['OrgMatchingBigrams'],'TotalMatchingCharacters':pattern_features['TotalMatchingCharacters'],'LongestMatchingPattern':pattern_features['LongestMatchingPattern'], 'SumMatchingEntries':pattern_features['SumMatchingEntries'],'SumMatchingEntriesLength':pattern_features['SumMatchingEntriesLength'],'MatchingUnigrams':pattern_features['MatchingUnigrams'],'MatchingBigrams':pattern_features['MatchingBigrams'],'AlphaBulletDot':isAlphaBulletDot,'AlphaBulletPar':isAlphaBulletPar,'AlphaBulletCapDot':isAlphaBulletCapDot,'AlphaBulletCapPar':isAlphaBulletCapPar,'DigitBulletDot':isDigitBulletDot,'DigitBulletPar':isDigitBulletPar,'RawParagraph':raw_paragraph,'StemmedParagraph':stemmed_paragraph,'StemmedParagraphLength':len(stemmed_paragraph),'RawParagraphLength':len(raw_paragraph),'FirstPatternOffset':pattern_features['FirstPatternOffset'],'WordsInCapital':words_in_capital,'FirstWordInCapitalOffset':first_capital_word_offset,'UnqMatchingUnigrams':pattern_features['UnqMatchingUnigrams'],'UnqMatchingBigrams':pattern_features['UnqMatchingBigrams']}, ignore_index=True)

		df_train_respa['TotalMatchingRatio'] = df_train_respa.apply(lambda row: row.TotalMatchingCharacters/len(row.StemmedParagraph),axis=1)

		self.respa_classifier = svm.SVC(C=1,gamma='auto')
		self.respa_classifier.fit(df_train_respa[self.respa_used_features].values,df_train_respa['Class'])

		return df_train_respa