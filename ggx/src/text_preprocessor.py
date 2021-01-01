import operator
import os
import pandas as pd
import re

from greek_stemmer import GreekStemmer
from sklearn.feature_extraction.text import CountVectorizer


_GREEK_VOWELS = {
    'ά': 'α',
    'ό': 'ο',
    'έ': 'ε',
    'ί': 'ι',
    'ή': 'η',
    'ώ': 'ω',
    'ύ': 'υ',
    'Ά': 'α',
    'Ό': 'ο',
    'Έ': 'ε',
    'Ί': 'ι',
    'Ή': 'η',
    'Ώ': 'ω',
    'Ύ': 'υ',
    'ϋ': 'υ',
    'ϊ': 'ι'
}

class TextPreprocessor:

    # Regex
    _REGEX_GREEK_BULLETS = re.compile(
        r'^([0-9]|[Α-Ω]|[A-Z]|[a-z]|[α-ω]){1,4}(\.|\))')

    def __init__(self, stopwordsFile):
        # Read stopwords from a file
        stopwords = []
        with open(stopwordsFile) as fp:
            for cnt, line in enumerate(fp):
                stopwords.append(f' {line} ')
        # read stopwords from a file
        self.stopwords = stopwords

    def removeStopWords(self, text):
        for sw in self.stopwords:
            text = text.replace(sw.replace('\n', ''), ' ')
        return text

    def preprocessText(self, string):
        for accented_letter, unaccented_letter in _GREEK_VOWELS.items():
            string = re.sub(accented_letter, unaccented_letter, string)
        return string.upper()

    def getStemmedParagraph(self, text, headersize):
        stemmer = GreekStemmer()
        words = []
        preprocessed_text = self.preprocessText(text)
        for word in self.removeStopWords(
                preprocessed_text)[:headersize].split():
            stem = stemmer.stem(word)
            words.append(stem)
        return ' '.join(words)

    def getCleanText(self, text, headersize):
        text = text.replace('\n', ' ')
        pat = re.compile(r'[^\w\.]+')
        text = ' '.join(pat.split(text))
        text = self.getStemmedParagraph(text, headersize)
        return text

    def getParagraphsFromFolder(self, folder, headersize):
        # Contains all texts from RespA folder, stemmed
        stemmed_paragraphs = []
        raw_paragraphs = []
        for root, dirs, files in os.walk(folder):
            for filename in sorted(files):  # use better sorting for files
                parlines = []
                with open(folder + filename, errors='ignore') as fp:
                    for cnt, line in enumerate(fp):
                        if line.replace('\n', '')[-1:] == '-' \
                                or line.replace('\n', '')[-1:] == '−':
                            # remove hyphens that split words from paragraphs
                            parlines.append(line.replace('\n', '').replace(
                                '-', '').replace('−', ''))
                        else:
                            parlines.append(line.replace('\n', ''))
                    # Remove punctuation except dot from acronyms
                    pat = re.compile(r'[^\w\.]+')
                    paragraph = ''.join(parlines)
                    raw_paragraphs.append(paragraph)
                    # Remove punctuation from paragraph and stem paragraph
                    paragraph = self.getStemmedParagraph(
                        ' '.join(pat.split(paragraph)), headersize)
                    stemmed_paragraphs.append(paragraph)
        return pd.DataFrame({
            'StemmedParagraph': stemmed_paragraphs,
            'RawParagraph': raw_paragraphs
        })

    def getTermFrequency(self, paragraphs):
        v2 = CountVectorizer(ngram_range=(1, 2), lowercase=False)
        stem_freq = {}
        for p in paragraphs:
            # create wordgrams that range from 1 to 2
            wgrams = v2.fit(list([p])).vocabulary_.keys()
            for wgram in wgrams:
                if wgram in stem_freq:
                    stem_freq[wgram] = stem_freq[wgram] + 1
                else:
                    stem_freq[wgram] = 1
        sorted_tuple_list = list(
            reversed(sorted(stem_freq.items(), key=operator.itemgetter(1))))
        df = pd.DataFrame(sorted_tuple_list, columns=['stems', 'frequency'])
        return (df, stem_freq)

    def getTerms(self, folder):
        v2 = CountVectorizer(ngram_range=(1, 2), lowercase=False)
        freq = {}
        # Contains all texts from RespA folder
        paragraphs = []
        raw_paragraphs = []
        for root, dirs, files in os.walk(folder):
            # Use better sorting for files
            for filename in sorted(files):
                parlines = []
                with open(folder + filename) as fp:
                    for cnt, line in enumerate(fp):
                        if line.replace('\n', '')[-1:] == '-' \
                                or line.replace('\n', '')[-1:] == '−':
                            # Remove hyphens that split words from paragraphs
                            parlines.append(line.replace('\n', '').replace(
                                '-', '').replace('−', ''))
                        else:
                            parlines.append(line.replace('\n', ''))
                    # Remove punctuation except dot from acronyms
                    pat = re.compile(r'[^\w\.]+')
                    paragraph = ''.join(parlines)
                    raw_paragraphs.append(paragraph)
                    # Remove punctuation from paragraph and stem paragraph
                    paragraph = self.getStemmedParagraph(
                        ' '.join(pat.split(paragraph)))
                    paragraphs.append(list([paragraph]))
                    # Create wordgrams that range from 1 to 2
                    wgrams = v2.fit(paragraphs).vocabulary_.keys()
                    for wgram in wgrams:
                        if wgram in freq:
                            counter = freq.get(wgram, 0) + 1
                            freq[wgram] = counter
                        else:
                            freq[wgram] = 1
        sorted_list = list(
            reversed(sorted(freq.items(), key=operator.itemgetter(1))))
        df_paragraphs = pd.DataFrame({
            'StemmedParagraph': paragraphs,
            'RawParagraph': raw_paragraphs
        })
        # Return most frequent stems, stemmed paragraphs,
        # most frequent stems with frequency, first column not reversed both
        # arguments, raw paragraphs
        return ([i[0] for i in sorted_list], freq, df_paragraphs, sorted_list)

    def get_words_in_capital(self, txt, keeponly=-1, shift=' '):
        if keeponly > 0:
            txt = txt[:keeponly]
        txt = re.sub(self._REGEX_GREEK_BULLETS, '', txt)
        text = txt.replace('\n', ' ')
        pat = re.compile(r'[^\w\.]')
        dot = re.compile(r'\.')
        toks = []
        for a in pat.split(text):
            if len(dot.findall(a)) == 1:
                for t in dot.split(a):
                    if len(t) > 1:
                        toks.append(t.strip())
            else:
                if len(a.strip()) > 0:
                    toks.append(a.strip())
        text = ' '.join(toks)
        count = 0
        for word in (shift + text).split()[1:]:
            if word[0].isupper():
                count += 1
        return count

    def get_first_pattern_offset(cltxt, trie):
        val = len(cltxt) * 100
        for pattern, start_idx in trie.search_all_patterns(cltxt):
            val = start_idx
            break
        return val

    def get_first_word_in_capital_offset(self, txt, keeponly=-1, shift=' '):
        if keeponly > 0:
            txt = txt[:keeponly]
        txt = re.sub(self._REGEX_GREEK_BULLETS, '', txt)
        text = txt.replace('\n', ' ')
        pat = re.compile(r'[^\w\.]')
        dot = re.compile(r'\.')
        toks = []
        for a in pat.split(text):
            if len(dot.findall(a)) == 1:
                for t in dot.split(a):
                    if len(t) > 1:
                        toks.append(t.strip())
            else:
                if len(a.strip()) > 0:
                    toks.append(a.strip())
        text = ' '.join(toks)
        wordoffset = 0
        count = 1
        for word in (shift + text).split()[1:]:
            count += 1
            if word[0].isupper():
                wordoffset = count
                break
        return wordoffset

    def startsWithAlpha(self, txt):
        if self._REGEX_GREEK_BULLETSAlpha.search(txt):
            return int(not (txt[:5].count('.') > 1))
        else:
            return 0

    def startsWithNumber(self, txt):
        if self._REGEX_GREEK_BULLETSNum.search(txt):
            return int(not (txt[:5].count('.') > 1))
        else:
            return 0

    def divide(self, TotalMatchingCharacters, stemmedlen):
        if stemmedlen == 0:
            return 0
        else:
            return TotalMatchingCharacters / stemmedlen

    def hasNumbers(self, inputString):
        return any(char.isdigit() for char in inputString)
