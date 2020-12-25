import re
import os
import operator
import csv
import sys
import trie_search as ts
import pandas as pd
from parser import Parser
from greek_stemmer import GreekStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

class TextPreprocessor:

	def __init__(self,stopwordsFile):
		# read stopwords from a file
		stopwords = []
		with open(stopwordsFile) as fp:
			for cnt,line in enumerate(fp):
				stopwords.append(" " + line + " ")
		# read stopwords from a file
		self.stopwords = stopwords
		self.regex = re.compile('^([0-9]|[Α-Ω]|[A-Z]|[a-z]|[α-ω]){1,4}(\.|\))')


	def removeStopWords(self,text):
		for sw in self.stopwords:
			text = text.replace(sw.replace("\n",""),' ')
		return text

	def preprocessText(self,string):
		string = re.sub(u"ά",u"α",string)
		string = re.sub(u"ό",u"ο",string)
		string = re.sub(u"έ",u"ε",string)
		string = re.sub(u"ί",u"ι",string)
		string = re.sub(u"ή",u"η",string)
		string = re.sub(u"ώ",u"ω",string)
		string = re.sub(u"ύ",u"υ",string)
		string = re.sub(u"Ά",u"α",string)
		string = re.sub(u"Ό",u"ο",string)
		string = re.sub(u"Έ",u"ε",string)
		string = re.sub(u"Ί",u"ι",string)
		string = re.sub(u"Ή",u"η",string)
		string = re.sub(u"Ώ",u"ω",string)
		string = re.sub(u"Ύ",u"υ",string)
		string = re.sub(u"ϋ",u"υ",string)
		string = re.sub(u"ϊ",u"ι",string)
		return string.upper()

	def getStemmedParagraph(self,text,headersize):
		stemmer = GreekStemmer()
		words = []
		for word in self.removeStopWords(self.preprocessText(text))[:headersize].split():
			stem = stemmer.stem(word)
			words.append(stem)
		return " ".join(words)

	def getCleanText(self,text,headersize):
		text = text.replace('\n',' ')
		#pat = re.compile("\W+")
		pat = re.compile("[^\w\.]+")
		text = ' '.join(pat.split(text))
		text = self.getStemmedParagraph(text,headersize)
		return text

	def getParagraphsFromFolder(self,folder,headersize):
		stemmed_paragraphs = [] # contains all texts from RespA folder, stemmed
		raw_paragraphs = []
		for root,dirs,files in os.walk(folder):
			for filename in sorted(files): # use better sorting for files
				parlines = []
				with open(folder + filename, errors="ignore") as fp:
					for cnt,line in enumerate(fp):
						if line.replace("\n","")[-1:] == "-" or line.replace("\n","")[-1:] == "−":
							parlines.append(line.replace("\n","").replace("-","").replace("−","")) # remove hyphens that split words from paragraphs
						else:
							parlines.append(line.replace("\n",""))
					#pat = re.compile("\W+")
					pat = re.compile("[^\w\.]+") #remove punctuation except dot from acronyms
					paragraph = "".join(parlines)
					raw_paragraphs.append(paragraph)
					#paragraph = clean_and_caps(paragraph,startsWithNumber(paragraph),startsWithAlpha(paragraph),headersize, stemming)
					paragraph = self.getStemmedParagraph(' '.join(pat.split(paragraph)),headersize) # remove punctuation from paragraph and stem paragraph
					stemmed_paragraphs.append(paragraph)
		return pd.DataFrame({'StemmedParagraph':stemmed_paragraphs,'RawParagraph':raw_paragraphs})

	def getTermFrequency(self,paragraphs):
		v2 = CountVectorizer(ngram_range=(1,2),lowercase=False)
		stem_freq = {}
		for p in paragraphs:
			wgrams = v2.fit(list([p])).vocabulary_.keys() # create wordgrams that range from 1 to 2
			for wgram in wgrams:
				if wgram in stem_freq:
					stem_freq[wgram] = stem_freq[wgram] + 1
				else:
					stem_freq[wgram] = 1
		sorted_tuple_list = list(reversed(sorted(stem_freq.items(),key=operator.itemgetter(1))))
		df = pd.DataFrame(sorted_tuple_list,columns=['stems','frequency'])
		return (df,stem_freq)


	def getTerms(self,folder):
		v2 = CountVectorizer(ngram_range=(1,2),lowercase=False)
		freq = {}
		paragraphs = [] # contains all texts from RespA folder
		raw_paragraphs = []
		for root,dirs,files in os.walk(folder):
		#filenames = [convertFilenames(f) for f in files]
			for filename in sorted(files): # use better sorting for files
				parlines = []
				with open(folder + filename) as fp:
					for cnt,line in enumerate(fp):
						if line.replace("\n","")[-1:] == "-" or line.replace("\n","")[-1:] == "−":
							parlines.append(line.replace("\n","").replace("-","").replace("−","")) # remove hyphens that split words from paragraphs
						else:
							parlines.append(line.replace("\n",""))
					#pat = re.compile("\W+")
					pat = re.compile("[^\w\.]+") #remove punctuation except dot from acronyms
					paragraph = "".join(parlines)
					raw_paragraphs.append(paragraph)
					paragraph = self.getStemmedParagraph(' '.join(pat.split(paragraph))) # remove punctuation from paragraph and stem paragraph
					#print(paragraph)
					paragraphs.append(list([paragraph]))
					wgrams = v2.fit(paragraphs).vocabulary_.keys() # create wordgrams that range from 1 to 2
					for wgram in wgrams:
						if wgram in freq:
							counter = freq.get(wgram,0) + 1
							freq[wgram] = counter
						else:
							freq[wgram] = 1

		sorted_list = list(reversed(sorted(freq.items(),key=operator.itemgetter(1))))
		#print(sorted_list)
		df_paragraphs = pd.DataFrame({'StemmedParagraph':paragraphs,'RawParagraph':raw_paragraphs})
		# [i[0] for i in sorted_list]
		return ([i[0] for i in sorted_list],freq,df_paragraphs,sorted_list) # returns (most frequent stems, stemmed paragraphs, most frequent stems with frequency, first column not reversed both arguments, raw paragraphs)

	def get_words_in_capital(self,txt, keeponly=-1,shift=' '):
		if keeponly > 0:
			txt = txt[:keeponly]
		txt = re.sub(self.regex, "", txt)
		text = txt.replace('\n',' ')
		pat = re.compile("[^\w\.]")
		dot = re.compile("\.")
		#pat = re.compile("\W+")
		toks = []
		for a in pat.split(text):
			if len(dot.findall(a))==1:
				for t in dot.split(a):
					if len(t)>1:
						toks.append(t.strip())
			else:
				if len(a.strip())>0:
					toks.append(a.strip())
		text = ' '.join(toks)
		count = 0
		for word in (shift+text).split()[1:]:
			if word[0].isupper():
				count += 1
		return count

	def get_first_pattern_offset(cltxt,trie):
		val = len(cltxt)*100
		for pattern, start_idx in trie.search_all_patterns(cltxt):
			val = start_idx
			break
		return val

	def get_first_word_in_capital_offset(self,txt, keeponly=-1,shift=' '):
		if keeponly > 0:
			txt = txt[:keeponly]
		txt = re.sub(self.regex, "", txt)
		text = txt.replace('\n',' ')
		pat = re.compile("[^\w\.]")
		dot = re.compile("\.")
		#pat = re.compile("\W+")
		toks = []
		for a in pat.split(text):
			if len(dot.findall(a))==1:
				for t in dot.split(a):
					if len(t)>1:
						toks.append(t.strip())
			else:
				if len(a.strip())>0:
					toks.append(a.strip())
		text = ' '.join(toks)
		wordoffset = 0
		count = 1
		for word in (shift+text).split()[1:]:
			count += 1
			if word[0].isupper():
				wordoffset = count
				break
		return wordoffset

	def startsWithAlpha(self,txt):
		if self.regexAlpha.search(txt):
			if txt[:5].count('.') > 1:
				return 0
			else: 
				return 1
		else:
			return 0

	def startsWithNumber(self,txt):
		if self.regexNum.search(txt):
			if txt[:5].count('.') > 1:
				return 0
			else:
				return 1
		else:
			return 0

	def divide(self,TotalMatchingCharacters,stemmedlen):
		if stemmedlen == 0: 
			return 0 
		else: 
			return TotalMatchingCharacters/stemmedlen

	def hasNumbers(self,inputString):
		return any(char.isdigit() for char in inputString)
