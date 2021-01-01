import functools
import pandas as pd
import pickle
import re

from sklearn import svm

from . import utils
from .parser import Parser
from .feature_extractor import FeatureExtractor
from .text_preprocessor import TextPreprocessor

import trie_search as ts

class Classifier:

    # Parameters
    _SVC_GAMMA_PARAM = 'auto'

    # Keys
    _FREQUENCY = 'frequency'
    _STEMS = 'stems'
    _CLASS = 'Class'
    _FILENAME = 'Filename'
    _ARTICLE_NUMBER = 'ArticleNo'
    _PREDICTION = 'Prediction'
    _RESPA_PREDICTION = 'RespAPrediction'
    _ORG_PREDICTION = 'OrgPrediction'
    _LONGEST_MATCHING_PATTERN = 'LongestMatchingPattern'
    _PREV_ALPHA_BULLET_DOT = 'PrevAlphaBulletDot'
    _PREV_ALPHA_BULLET_PAR = 'PrevAlphaBulletPar'
    _PREV_ALPHA_BULLET_CAP_DOT = 'PrevAlphaBulletCapDot'
    _PREV_ALPHA_BULLET_CAP_PAR = 'PrevAlphaBulletCapPar'
    _PREV_DIGIT_BULLET_DOT = 'PrevDigitBulletDot'
    _PREV_DIGIT_BULLET_PAR = 'PrevDigitBulletPar'
    _RAW_PARAGRAPH = 'RawParagraph'
    _STEMMED_PARAGRAPH = 'StemmedParagraph'
    _STEMMED_PARAGRAPH_LENGTH = 'StemmedParagraphLength'
    _RAW_PARAGRAPH_LENGTH = 'RawParagraphLength'
    _FIRST_PATTERN_OFFSET = 'FirstPatternOffset'
    _WORDS_IN_CAPITAL = 'WordsInCapital'
    _FIRST_WORDS_IN_CAPITAL_OFFSET = 'FirstWordInCapitalOffset'
    _MATCHED_PATTERNS_COUNT = 'MatchedPatternsCount'
    _UNQ_MATCHED_PATTERNS_COUNT = 'UnqMatchedPatternsCount'
    _TOTAL_MATCHING_CHARACTERS = 'TotalMatchingCharacters'
    _SUM_MATCHING_ENTRIES = 'SumMatchingEntries'
    _SUM_MATCHING_ENTRIES_LENGTH = 'SumMatchingEntriesLength'
    _MATCHING_UNIGRAMS = 'MatchingUnigrams'
    _MATCHING_BIGRAMS = 'MatchingBigrams'
    _TOTAL_MATCHING_RATIO = 'TotalMatchingRatio'
    _UNQ_MATCHING_UNIGRAMS = 'UnqMatchingUnigrams'
    _UNQ_MATCHING_BIGRAMS = 'UnqMatchingBigrams'
    _ALPHA_BULLET_DOT = 'AlphaBulletDot'
    _ALPHA_BULLET_PAR = 'AlphaBulletPar'
    _ALPHA_BULLET_CAP_DOT = 'AlphaBulletCapDot'
    _ALPHA_BULLET_CAP_PAR = 'AlphaBulletCapPar'
    _DIGIT_BULLET_DOT = 'DigitBulletDot'
    _DIGIT_BULLET_PAR = 'DigitBulletPar'
    _BULLET_DEPARTMENT = 'BulletDepartment'
    _ORG_TOTAL_MATCHING_CHARACTERS = 'OrgTotalMatchingCharacters'
    _ORG_MATCHING_UNIGRAMS = 'OrgMatchingUnigrams'
    _ORG_MATCHING_BIGRAMS = 'OrgMatchingBigrams'
    _RESPA_USED_FEATURES = [
        _MATCHED_PATTERNS_COUNT,
        _UNQ_MATCHED_PATTERNS_COUNT,
        _TOTAL_MATCHING_CHARACTERS,
        _SUM_MATCHING_ENTRIES,
        _SUM_MATCHING_ENTRIES_LENGTH,
        _MATCHING_UNIGRAMS,
        _MATCHING_BIGRAMS,
        _TOTAL_MATCHING_RATIO,
        _UNQ_MATCHING_UNIGRAMS,
        _UNQ_MATCHING_BIGRAMS,
        _ALPHA_BULLET_DOT,
        _ALPHA_BULLET_PAR,
        _ALPHA_BULLET_CAP_DOT,
        _ALPHA_BULLET_CAP_PAR,
        _DIGIT_BULLET_DOT,
        _DIGIT_BULLET_PAR
    ]
    _ORG_USED_FEATURES = [
        _BULLET_DEPARTMENT,
        _ORG_TOTAL_MATCHING_CHARACTERS,
        _ORG_MATCHING_UNIGRAMS,
        _ORG_MATCHING_BIGRAMS,
        _ALPHA_BULLET_DOT,
        _ALPHA_BULLET_PAR,
        _ALPHA_BULLET_CAP_DOT,
        _ALPHA_BULLET_CAP_PAR,
        _DIGIT_BULLET_DOT,
        _DIGIT_BULLET_PAR
    ]

    # Values
    _ORG_VALUE = 'Org'
    _POSITIONS_VALUE = 'Positions'
    _NO_IDEA_VALUE = 'NoIdea'
    _IRRELEVANT_VALUE = 'Irrelevant'
    _NON_RESPA_VALUE = 'Non-RespA'
    _RESPA_VALUE = 'RespA'

    # Regex
    _REGEX_ALPHA_DOT = re.compile(r'^([a-z]|[α-ω]){1,4}(\.)')
    _REGEX_ALPHA_PAR = re.compile(r'^([a-z]|[α-ω]){1,4}(\))')
    _REGEX_ALPHA_CAP_DOT = re.compile(r'^([Α-Ω]|[A-Z]){1,4}(\.)')
    _REGEX_ALPHA_CAP_PAR = re.compile(r'^([Α-Ω]|[A-Z]){1,4}(\))')
    _REGEX_NUM_DOT = re.compile(r'^([0-9]){1,4}(\.)')
    _REGEX_NUM_PAR = re.compile(r'^([0-9]){1,4}(\))')
    _REGEX_DEP = re.compile(
        r'^([a-z]|[A-Z]|[Α-Ω]|[α-ω]|[0-9]){1,4}(\.|\))(\sΤμήμα)')
    _REGEX_PATTERN_GREEK_BULLETS = r'((\n)([α-ω])([α-ω])(\)))'
    _REGEX_PATTERN_GAZZETTE_TITLE = \
        r'([0-9]){1,4}(\s\n)ΕΦΗΜΕΡΙΣ ΤΗΣ ΚΥΒΕΡΝΗΣΕΩΣ\(ΤΕΥΧΟΣ ΠΡΩΤΟ\)(\s\n)'
    _REGEX_PATTERN_BRANCH_ATOM = r'\bΚλάδος\b'
    _REGEX_PATTERN_EDUCATION_ATOMS = r'\b(ΠΕ|ΤΕ|ΔΕ|ΥΕ)\b'
    _REGEX_PATTERN_POSITION_ATOMS = r'\b(θέση|θέσεις|θέσεων)\b'
    _REGEX_PATTERN_MIXED_POSITION_ATOMS = \
        r'\b(Κλάδος|θέση|θέσεις|θέσεων|Υπάλληλος)\b'

    # Files
    _TRAINING_FILE = 'training_file.csv'

    def __init__(self, stopwords_file, organizations_file, org_ratio,
                 org_headersize):
        self.tp = TextPreprocessor(stopwords_file)
        self.fe = FeatureExtractor(organizations_file, self.tp, org_ratio,
                                   org_headersize)
        self.trie = ''
        self.weights = ''
        self.org_classifier = ''
        self.respa_classifier = ''

    def total_prediction(self, row):
        if row[self._ORG_PREDICTION] == self._ORG_VALUE:
            return self._ORG_VALUE
        elif row[self._RESPA_PREDICTION] == self._RESPA_VALUE:
            return self._RESPA_VALUE
        else:
            return self._IRRELEVANT_VALUE

    def create_trie_index(self, most_frequent_non_respas_stems_ordered,
                          most_frequent_respas_stems_ordered,
                          num_non_respa_docs, ratio, tp):
        selected_df = most_frequent_non_respas_stems_ordered[
            most_frequent_non_respas_stems_ordered[
                self._FREQUENCY] / num_non_respa_docs > ratio]
        sublist = list(selected_df[self._STEMS])
        subtraction = [x for x in list(
            most_frequent_respas_stems_ordered[self._STEMS])
            if x not in sublist]  # respas - sublist
        subtraction_df = pd.DataFrame({self._STEMS: subtraction})
        # Merge subtracted and sorted on stems column
        new_df = pd.merge(
            subtraction_df,
            most_frequent_respas_stems_ordered, on=self._STEMS)
        meanvalue = new_df[self._FREQUENCY].mean()  # get mean frequency
        # Get items for which frequency is greater than the mean frequency
        most_freq = new_df[new_df.frequency > int(meanvalue)]
        freqstems = most_freq[self._STEMS].values.tolist()
        freqstemscopy = []
        for s in freqstems:
            if not tp.hasNumbers(s) and len(s) > 3:
                freqstemscopy.append(s)
        # Create trie index from sublist terms that do not contain numbers
        trie = ts.TrieSearch(freqstemscopy)
        return trie

    def read_org_trie_from_file(self, org_pickle_file):
        self.fe.read_org_trie_from_file(org_pickle_file)

    def read_trie_index_from_file(self, trie):
        with open(trie, 'rb') as f:
            self.trie = pickle.load(f)  # pickle/1trie.pkl

    def read_weights_from_file(self, weightfile):
        with open(weightfile, 'rb') as f:
            self.weights = pickle.load(f)  # pickle/1weights.pkl

    def org_classifier_from_file(self, filename):
        org_train_data = self.fe.extract_organisational_features_from_file(
            filename)
        org_classifier = svm.SVC(C=1, gamma=self._SVC_GAMMA_PARAM)
        org_classifier.fit(org_train_data[
            self._ORG_USED_FEATURES].values, org_train_data[self._CLASS])
        self.org_classifier = org_classifier

    def update_org_classifier(self, oldfile, newfile, headersize):
        org_train_data = self.fe.update_organisational_features_from_file(
            oldfile, newfile, self.weights, self.trie, self.tp, headersize)
        org_classifier = svm.SVC(C=1, gamma=self._SVC_GAMMA_PARAM)
        org_classifier.fit(org_train_data[
            self._ORG_USED_FEATURES].values, org_train_data[self._CLASS])
        self.org_classifier = org_classifier

    def respa_classifier_from_file(self, filename, headersize):
        train_data = self.fe.extract_features_from_file(
            filename, self.weights,
            self.trie, self.tp, headersize)  # ExampleFile.csv
        train_data[self._TOTAL_MATCHING_RATIO] = train_data.apply(lambda row: (
            row.TotalMatchingCharacters / len(row.StemmedParagraph) if len(
                row.StemmedParagraph) != 0 else 0), axis=1)
        respa_classifier = svm.SVC(C=1, gamma=self._SVC_GAMMA_PARAM)
        respa_classifier.fit(
            train_data[self._RESPA_USED_FEATURES].values,
            train_data[self._CLASS])
        self.respa_classifier = respa_classifier

    def predict_pdf_file(self, filename, respa_headersize, org_headersize,
                         pdf_directory, out_txt_directory):
        paragraphs_per_article = pd.DataFrame()
        parser = Parser()
        txt = parser.get_pdf_txt(filename.replace('.pdf', ''),
                                 pdf_directory, out_txt_directory)
        txt = re.sub(self._REGEX_PATTERN_GAZZETTE_TITLE, '', txt)
        articles = parser.get_articles(txt)
        for num, article in articles.items():
            articleNo = num
            article_paragraphs = parser.get_paragraphs(article)
            isPrevAlphaBulletDot = 0
            isPrevAlphaBulletPar = 0
            isPrevAlphaBulletCapDot = 0
            isPrevAlphaBulletCapPar = 0
            isPrevDigitBulletDot = 0
            isPrevDigitBulletPar = 0
            # Split paragraphs based on bullets ===
            trimmed_paragraphs = []
            for p in article_paragraphs:
                sublist = list(filter(
                    lambda x: len(x) > 1,
                    re.split(self._REGEX_PATTERN_GREEK_BULLETS, p)))
                if (len(sublist) > 1):
                    if len(sublist[0]) <= 3:
                        for x in range(0, len(sublist) - 1, 2):
                            trimmed_paragraphs.append(
                                sublist[x] + sublist[x + 1])
                        if len(sublist) % 2 != 0:
                            trimmed_paragraphs.append(
                                sublist[len(sublist) - 1])
                    else:
                        trimmed_paragraphs.append(sublist[0])
                        for x in range(1, len(sublist) - 1, 2):
                            trimmed_paragraphs.append(
                                sublist[x] + sublist[x + 1])
                        if len(sublist) % 2 == 0:
                            trimmed_paragraphs.append(
                                sublist[len(sublist) - 1])
                else:
                    trimmed_paragraphs.append(p)
            # ===
            for raw_paragraph in trimmed_paragraphs:
                stemmed_paragraph = self.tp.getCleanText(
                    raw_paragraph, respa_headersize)
                words_in_capital = self.tp.get_words_in_capital(raw_paragraph)
                first_capital_word_offset = \
                    self.tp.get_first_word_in_capital_offset(raw_paragraph)
                isAlphaBulletDot = self.fe.regex_applies(
                    self._REGEX_ALPHA_DOT, raw_paragraph)
                isAlphaBulletPar = self.fe.regex_applies(
                    self._REGEX_ALPHA_PAR, raw_paragraph)
                isAlphaBulletCapDot = self.fe.regex_applies(
                    self._REGEX_ALPHA_CAP_DOT, raw_paragraph)
                isAlphaBulletCapPar = self.fe.regex_applies(
                    self._REGEX_ALPHA_CAP_PAR, raw_paragraph)
                isDigitBulletDot = self.fe.regex_applies(
                    self._REGEX_NUM_DOT, raw_paragraph)
                isDigitBulletPar = self.fe.regex_applies(
                    self._REGEX_NUM_PAR, raw_paragraph)
                isDep = self.fe.regex_applies(self._REGEX_DEP, raw_paragraph)
                all_patterns = list(
                    self.trie.search_all_patterns(stemmed_paragraph))
                # Remove unigrams that are contained in bigrams ===
                subpatterns = utils.remove_unigrams_contained_in_bigrams(
                    all_patterns)
                # ===
                pattern_features = self.fe.extract_features_from_trie_patterns(
                    subpatterns, self.weights)
                organisational_features = \
                    self.fe.extract_organisational_features(
                        self.tp.getCleanText(raw_paragraph, org_headersize))
                paragraphs_per_article = paragraphs_per_article.append({
                    self._FILENAME: filename,
                    self._CLASS: self._NO_IDEA_VALUE,
                    self._UNQ_MATCHED_PATTERNS_COUNT: pattern_features[
                        self._UNQ_MATCHED_PATTERNS_COUNT],
                    self._BULLET_DEPARTMENT: isDep,
                    self._MATCHED_PATTERNS_COUNT: pattern_features[
                        self._MATCHED_PATTERNS_COUNT],
                    self._ARTICLE_NUMBER: articleNo,
                    self._ORG_TOTAL_MATCHING_CHARACTERS:
                        organisational_features[
                            self._ORG_TOTAL_MATCHING_CHARACTERS],
                    self._ORG_MATCHING_UNIGRAMS: organisational_features[
                        self._ORG_MATCHING_UNIGRAMS],
                    self._ORG_MATCHING_BIGRAMS: organisational_features[
                        self._ORG_MATCHING_BIGRAMS],
                    self._TOTAL_MATCHING_CHARACTERS: pattern_features[
                        self._TOTAL_MATCHING_CHARACTERS],
                    self._LONGEST_MATCHING_PATTERN: pattern_features[
                        self._LONGEST_MATCHING_PATTERN],
                    self._SUM_MATCHING_ENTRIES: pattern_features[
                        self._SUM_MATCHING_ENTRIES],
                    self._SUM_MATCHING_ENTRIES_LENGTH: pattern_features[
                        self._SUM_MATCHING_ENTRIES_LENGTH],
                    self._MATCHING_UNIGRAMS: pattern_features[
                        self._MATCHING_UNIGRAMS],
                    self._MATCHING_BIGRAMS: pattern_features[
                        self._MATCHING_BIGRAMS],
                    self._ALPHA_BULLET_DOT: isAlphaBulletDot,
                    self._ALPHA_BULLET_PAR: isAlphaBulletPar,
                    self._ALPHA_BULLET_CAP_DOT: isAlphaBulletCapDot,
                    self._ALPHA_BULLET_CAP_PAR: isAlphaBulletCapPar,
                    self._DIGIT_BULLET_DOT: isDigitBulletDot,
                    self._DIGIT_BULLET_PAR: isDigitBulletPar,
                    self._PREV_ALPHA_BULLET_DOT: isPrevAlphaBulletDot,
                    self._PREV_ALPHA_BULLET_PAR: isPrevAlphaBulletPar,
                    self._PREV_ALPHA_BULLET_CAP_DOT:
                        isPrevAlphaBulletCapDot,
                    self._PREV_ALPHA_BULLET_CAP_PAR:
                        isPrevAlphaBulletCapPar,
                    self._PREV_DIGIT_BULLET_DOT: isPrevDigitBulletDot,
                    self._PREV_DIGIT_BULLET_PAR: isPrevDigitBulletPar,
                    self._RAW_PARAGRAPH: raw_paragraph,
                    self._STEMMED_PARAGRAPH: stemmed_paragraph,
                    self._STEMMED_PARAGRAPH_LENGTH: len(stemmed_paragraph),
                    self._RAW_PARAGRAPH_LENGTH: len(raw_paragraph),
                    self._FIRST_PATTERN_OFFSET: pattern_features[
                        self._FIRST_PATTERN_OFFSET],
                    self._WORDS_IN_CAPITAL: words_in_capital,
                    self._FIRST_WORDS_IN_CAPITAL_OFFSET:
                        first_capital_word_offset,
                    self._UNQ_MATCHING_UNIGRAMS: pattern_features[
                        self._UNQ_MATCHING_UNIGRAMS],
                    self._UNQ_MATCHING_BIGRAMS: pattern_features[
                        self._UNQ_MATCHING_BIGRAMS]
                }, ignore_index=True)
                isPrevAlphaBulletDot = isAlphaBulletDot
                isPrevAlphaBulletPar = isAlphaBulletPar
                isPrevAlphaBulletCapDot = isAlphaBulletCapDot
                isPrevAlphaBulletCapPar = isAlphaBulletCapPar
                isPrevDigitBulletDot = isDigitBulletDot
                isPrevDigitBulletPar = isDigitBulletPar
        paragraphs_per_article[self._TOTAL_MATCHING_RATIO] = \
            paragraphs_per_article.apply(
            lambda row: (
                row.TotalMatchingCharacters / len(row.StemmedParagraph)
                if len(row.StemmedParagraph) != 0 else 0), axis=1)
        respa_prediction = self.respa_classifier.predict(
            paragraphs_per_article[self._RESPA_USED_FEATURES])
        org_prediction = self.org_classifier.predict(
            paragraphs_per_article[self._ORG_USED_FEATURES])
        paragraphs_per_article[
            self._RESPA_PREDICTION] = pd.Series(respa_prediction)
        paragraphs_per_article[
            self._ORG_PREDICTION] = pd.Series(org_prediction)
        paragraphs_per_article[self._PREDICTION] = \
            paragraphs_per_article.apply(
            lambda row: self.total_prediction(row), axis=1)
        adj_predictions = pd.DataFrame()
        previous_prediction = ''
        for index, row in paragraphs_per_article.iterrows():
            if re.search(self._REGEX_PATTERN_EDUCATION_ATOMS,
                         row[self._RAW_PARAGRAPH]) \
                and re.search(self._REGEX_PATTERN_MIXED_POSITION_ATOMS,
                              row[self._RAW_PARAGRAPH], re.IGNORECASE) \
                or re.search(self._REGEX_PATTERN_POSITION_ATOMS,
                             row[self._RAW_PARAGRAPH],
                             re.IGNORECASE) \
                and re.search(self._REGEX_PATTERN_BRANCH_ATOM,
                              row[self._RAW_PARAGRAPH],
                              re.IGNORECASE):
                row[self._PREDICTION] = self._POSITIONS_VALUE
            elif row[self._ALPHA_BULLET_DOT] == row[
                self._PREV_ALPHA_BULLET_DOT] \
                    and row[self._ALPHA_BULLET_PAR] == row[
                        self._PREV_ALPHA_BULLET_PAR] \
                    and row[self._ALPHA_BULLET_CAP_DOT] == row[
                        self._PREV_ALPHA_BULLET_CAP_DOT] \
                    and row[self._ALPHA_BULLET_CAP_PAR] == row[
                        self._PREV_ALPHA_BULLET_CAP_PAR] \
                    and row[self._DIGIT_BULLET_DOT] == row[
                        self._PREV_DIGIT_BULLET_DOT] \
                    and row[self._DIGIT_BULLET_PAR] == row[
                        self._PREV_DIGIT_BULLET_PAR] \
                    and previous_prediction != '':
                row[self._PREDICTION] = previous_prediction
            adj_predictions = adj_predictions.append(row, ignore_index=True)
            lines = row[self._RAW_PARAGRAPH].splitlines()
            total_line_len = functools.reduce(
                (lambda x, y: x + y), list(map(lambda l: len(l), lines)))
            line_ratio = total_line_len / len(lines)
            if (line_ratio <= 4):
                row[self._PREDICTION] = self._IRRELEVANT_VALUE
            previous_prediction = row[self._PREDICTION]
        adj_predictions[[
            self._ARTICLE_NUMBER, self._RAW_PARAGRAPH,
            self._PREDICTION]].to_csv(filename + '.csv', sep='\t')
        return adj_predictions

    def respa_classifier_from_pdf_files(self, respa_directory, headersize1,
                                        non_respa_directory, headersize2,
                                        ratio, create_trie):
        respas_pdf = self.tp.getParagraphsFromFolder(
            respa_directory, headersize1)
        respas = self.tp.getTermFrequency(
            list(respas_pdf[self._STEMMED_PARAGRAPH]))
        most_frequent_respas_stems_ordered = respas[0]
        weights = respas[1]
        non_respas_pdf = self.tp.getParagraphsFromFolder(
            non_respa_directory, headersize2)
        non_respas = self.tp.getTermFrequency(
            list(non_respas_pdf[self._STEMMED_PARAGRAPH]))
        most_frequent_non_respas_stems_ordered = non_respas[0]
        num_non_respa_docs = len(non_respas_pdf.index)
        if (create_trie):
            self.trie = self.create_trie_index(
                most_frequent_non_respas_stems_ordered,
                most_frequent_respas_stems_ordered,
                num_non_respa_docs, ratio, self.tp)
        df_train_respa = pd.DataFrame()
        for index, row in respas_pdf.iterrows():
            raw_paragraph = row[self._RAW_PARAGRAPH]
            stemmed_paragraph = row[self._STEMMED_PARAGRAPH]
            words_in_capital = self.tp.get_words_in_capital(raw_paragraph)
            first_capital_word_offset = \
                self.tp.get_first_word_in_capital_offset(raw_paragraph)
            isAlphaBulletDot = self.fe.regex_applies(
                self._REGEX_ALPHA_DOT, raw_paragraph)
            isAlphaBulletPar = self.fe.regex_applies(
                self._REGEX_ALPHA_PAR, raw_paragraph)
            isAlphaBulletCapDot = self.fe.regex_applies(
                self._REGEX_ALPHA_CAP_DOT, raw_paragraph)
            isAlphaBulletCapPar = self.fe.regex_applies(
                self._REGEX_ALPHA_CAP_PAR, raw_paragraph)
            isDigitBulletDot = self.fe.regex_applies(
                self._REGEX_NUM_DOT, raw_paragraph)
            isDigitBulletPar = self.fe.regex_applies(
                self._REGEX_NUM_PAR, raw_paragraph)
            isDep = self.fe.regex_applies(self._REGEX_DEP, raw_paragraph)
            all_patterns = list(
                self.trie.search_all_patterns(stemmed_paragraph))
            # Remove unigrams that are contained in bigrams ===
            subpatterns = utils.remove_unigrams_contained_in_bigrams(
                all_patterns)
            # ===
            pattern_features = self.fe.extract_features_from_trie_patterns(
                subpatterns, weights)
            organisational_features = self.fe.extract_organisational_features(
                stemmed_paragraph)
            df_train_respa = df_train_respa.append({
                self._CLASS: self._RESPA_VALUE,
                self._UNQ_MATCHED_PATTERNS_COUNT: pattern_features[
                    self._UNQ_MATCHED_PATTERNS_COUNT],
                self._MATCHED_PATTERNS_COUNT: pattern_features[
                    self._MATCHED_PATTERNS_COUNT],
                self._BULLET_DEPARTMENT: isDep,
                self._ORG_TOTAL_MATCHING_CHARACTERS: organisational_features[
                    self._ORG_TOTAL_MATCHING_CHARACTERS],
                self._ORG_MATCHING_UNIGRAMS: organisational_features[
                    self._ORG_MATCHING_UNIGRAMS],
                self._ORG_MATCHING_BIGRAMS: organisational_features[
                    self._ORG_MATCHING_BIGRAMS],
                self._TOTAL_MATCHING_CHARACTERS: pattern_features[
                    self._TOTAL_MATCHING_CHARACTERS],
                self._LONGEST_MATCHING_PATTERN: pattern_features[
                    self._LONGEST_MATCHING_PATTERN],
                self._SUM_MATCHING_ENTRIES: pattern_features[
                    self._SUM_MATCHING_ENTRIES],
                self._SUM_MATCHING_ENTRIES_LENGTH: pattern_features[
                    self._SUM_MATCHING_ENTRIES_LENGTH],
                self._MATCHING_UNIGRAMS: pattern_features[
                    self._MATCHING_UNIGRAMS],
                self._MATCHING_BIGRAMS: pattern_features[
                    self._MATCHING_BIGRAMS],
                self._ALPHA_BULLET_DOT: isAlphaBulletDot,
                self._ALPHA_BULLET_PAR: isAlphaBulletPar,
                self._ALPHA_BULLET_CAP_DOT: isAlphaBulletCapDot,
                self._ALPHA_BULLET_CAP_PAR: isAlphaBulletCapPar,
                self._DIGIT_BULLET_DOT: isDigitBulletDot,
                self._DIGIT_BULLET_PAR: isDigitBulletPar,
                self._RAW_PARAGRAPH: raw_paragraph,
                self._STEMMED_PARAGRAPH: stemmed_paragraph,
                self._STEMMED_PARAGRAPH_LENGTH: len(stemmed_paragraph),
                self._RAW_PARAGRAPH_LENGTH: len(raw_paragraph),
                self._FIRST_PATTERN_OFFSET: pattern_features[
                    self._FIRST_PATTERN_OFFSET],
                self._WORDS_IN_CAPITAL: words_in_capital,
                self._FIRST_WORDS_IN_CAPITAL_OFFSET:
                    first_capital_word_offset,
                self._UNQ_MATCHING_UNIGRAMS: pattern_features[
                    self._UNQ_MATCHING_UNIGRAMS],
                self._UNQ_MATCHING_BIGRAMS: pattern_features[
                    self._UNQ_MATCHING_BIGRAMS]}, ignore_index=True)

        for index, row in non_respas_pdf.iterrows():
            raw_paragraph = row[self._RAW_PARAGRAPH]
            stemmed_paragraph = row[self._STEMMED_PARAGRAPH]
            words_in_capital = self.tp.get_words_in_capital(raw_paragraph)
            first_capital_word_offset = \
                self.tp.get_first_word_in_capital_offset(raw_paragraph)
            isAlphaBulletDot = self.fe.regex_applies(
                self._REGEX_ALPHA_DOT, raw_paragraph)
            isAlphaBulletPar = self.fe.regex_applies(
                self._REGEX_ALPHA_PAR, raw_paragraph)
            isAlphaBulletCapDot = self.fe.regex_applies(
                self._REGEX_ALPHA_CAP_DOT, raw_paragraph)
            isAlphaBulletCapPar = self.fe.regex_applies(
                self._REGEX_ALPHA_CAP_PAR, raw_paragraph)
            isDigitBulletDot = self.fe.regex_applies(
                self._REGEX_NUM_DOT, raw_paragraph)
            isDigitBulletPar = self.fe.regex_applies(
                self._REGEX_NUM_PAR, raw_paragraph)
            isDep = self.fe.regex_applies(self._REGEX_DEP, raw_paragraph)
            all_patterns = list(
                self.trie.search_all_patterns(stemmed_paragraph))
            # Remove unigrams that are contained in bigrams ===
            subpatterns = utils.remove_unigrams_contained_in_bigrams(
                all_patterns)
            # ===
            pattern_features = self.fe.extract_features_from_trie_patterns(
                subpatterns, weights)
            organisational_features = self.fe.extract_organisational_features(
                stemmed_paragraph)
            df_train_respa = df_train_respa.append({
                self._CLASS: self._NON_RESPA_VALUE,
                self._UNQ_MATCHED_PATTERNS_COUNT: pattern_features[
                    self._UNQ_MATCHED_PATTERNS_COUNT],
                self._MATCHED_PATTERNS_COUNT: pattern_features[
                    self._MATCHED_PATTERNS_COUNT],
                self._BULLET_DEPARTMENT: isDep,
                self._ORG_TOTAL_MATCHING_CHARACTERS: organisational_features[
                    self._ORG_TOTAL_MATCHING_CHARACTERS],
                self._ORG_MATCHING_UNIGRAMS: organisational_features[
                    self._ORG_MATCHING_UNIGRAMS],
                self._ORG_MATCHING_BIGRAMS: organisational_features[
                    self._ORG_MATCHING_BIGRAMS],
                self._TOTAL_MATCHING_CHARACTERS: pattern_features[
                    self._TOTAL_MATCHING_CHARACTERS],
                self._LONGEST_MATCHING_PATTERN: pattern_features[
                    self._LONGEST_MATCHING_PATTERN],
                self._SUM_MATCHING_ENTRIES: pattern_features[
                    self._SUM_MATCHING_ENTRIES],
                self._SUM_MATCHING_ENTRIES_LENGTH: pattern_features[
                    self._SUM_MATCHING_ENTRIES_LENGTH],
                self._MATCHING_UNIGRAMS: pattern_features[
                    self._MATCHING_UNIGRAMS],
                self._MATCHING_BIGRAMS: pattern_features[
                    self._MATCHING_BIGRAMS],
                self._ALPHA_BULLET_DOT: isAlphaBulletDot,
                self._ALPHA_BULLET_PAR: isAlphaBulletPar,
                self._ALPHA_BULLET_CAP_DOT: isAlphaBulletCapDot,
                self._ALPHA_BULLET_CAP_PAR: isAlphaBulletCapPar,
                self._DIGIT_BULLET_DOT: isDigitBulletDot,
                self._DIGIT_BULLET_PAR: isDigitBulletPar,
                self._RAW_PARAGRAPH: raw_paragraph,
                self._STEMMED_PARAGRAPH: stemmed_paragraph,
                self._STEMMED_PARAGRAPH_LENGTH: len(stemmed_paragraph),
                self._RAW_PARAGRAPH_LENGTH: len(raw_paragraph),
                self._FIRST_PATTERN_OFFSET: pattern_features[
                    self._FIRST_PATTERN_OFFSET],
                self._WORDS_IN_CAPITAL: words_in_capital,
                self._FIRST_WORDS_IN_CAPITAL_OFFSET:
                    first_capital_word_offset,
                self._UNQ_MATCHING_UNIGRAMS: pattern_features[
                    self._UNQ_MATCHING_UNIGRAMS],
                self._UNQ_MATCHING_BIGRAMS: pattern_features[
                    self._UNQ_MATCHING_BIGRAMS]}, ignore_index=True)
        df_train_respa[self._TOTAL_MATCHING_RATIO] = df_train_respa.apply(
            lambda row: row.TotalMatchingCharacters / len(
                row.StemmedParagraph), axis=1)
        self.respa_classifier = svm.SVC(C=1, gamma=self._SVC_GAMMA_PARAM)
        self.respa_classifier.fit(df_train_respa[
            self._RESPA_USED_FEATURES].values, df_train_respa[self._CLASS])
        df_train_respa.to_csv(self._TRAINING_FILE, sep="\t")
        return df_train_respa

    def classifier_from_enriched_train_samples(self, oldfile, newfile,
                                               headersize1, headersize2,
                                               ratio):
        old_train_data = self.fe.read_training_file(
            oldfile)[[self._CLASS, self._STEMMED_PARAGRAPH,
                      self._RAW_PARAGRAPH]]
        new_train_data_respa = self.fe.extract_features_from_file(
            newfile, self.weights, self.trie, self.tp, headersize1)[
            [self._CLASS, self._STEMMED_PARAGRAPH,
             self._RAW_PARAGRAPH]]
        new_train_data_non_respa = self.fe.extract_features_from_file(
            newfile, self.weights, self.trie, self.tp, headersize2)[
            [self._CLASS, self._STEMMED_PARAGRAPH,
             self._RAW_PARAGRAPH]]
        merged_df = old_train_data.append(
            new_train_data_respa).append(new_train_data_non_respa)
        merged_df = merged_df.reset_index(drop=True)
        isRespA = merged_df[self._CLASS] == self._RESPA_VALUE
        isNonRespA = merged_df[self._CLASS] == self._NON_RESPA_VALUE
        respas_pdf = merged_df[isRespA][[
            self._STEMMED_PARAGRAPH, self._RAW_PARAGRAPH]]
        non_respas_pdf = merged_df[isNonRespA][[
            self._STEMMED_PARAGRAPH, self._RAW_PARAGRAPH]]
        respas = self.tp.getTermFrequency(
            list(respas_pdf[self._STEMMED_PARAGRAPH]))
        most_frequent_respas_stems_ordered = respas[0]
        weights = respas[1]
        non_respas = self.tp.getTermFrequency(
            list(non_respas_pdf[self._STEMMED_PARAGRAPH]))
        most_frequent_non_respas_stems_ordered = non_respas[0]
        num_non_respa_docs = len(non_respas_pdf.index)
        self.trie = self.create_trie_index(
            most_frequent_non_respas_stems_ordered,
            most_frequent_respas_stems_ordered,
            num_non_respa_docs, ratio, self.tp)
        df_train_respa = pd.DataFrame()
        for index, row in respas_pdf.iterrows():
            raw_paragraph = row[self._RAW_PARAGRAPH]
            stemmed_paragraph = row[self._STEMMED_PARAGRAPH]
            words_in_capital = self.tp.get_words_in_capital(raw_paragraph)
            first_capital_word_offset = \
                self.tp.get_first_word_in_capital_offset(raw_paragraph)
            isAlphaBulletDot = self.fe.regex_applies(
                self._REGEX_ALPHA_DOT, raw_paragraph)
            isAlphaBulletPar = self.fe.regex_applies(
                self._REGEX_ALPHA_PAR, raw_paragraph)
            isAlphaBulletCapDot = self.fe.regex_applies(
                self._REGEX_ALPHA_CAP_DOT, raw_paragraph)
            isAlphaBulletCapPar = self.fe.regex_applies(
                self._REGEX_ALPHA_CAP_PAR, raw_paragraph)
            isDigitBulletDot = self.fe.regex_applies(
                self._REGEX_NUM_DOT, raw_paragraph)
            isDigitBulletPar = self.fe.regex_applies(
                self._REGEX_NUM_PAR, raw_paragraph)
            isDep = self.fe.regex_applies(self._REGEX_DEP, raw_paragraph)
            all_patterns = list(
                self.trie.search_all_patterns(stemmed_paragraph))
            # Remove unigrams that are contained in bigrams ===
            subpatterns = utils.remove_unigrams_contained_in_bigrams(
                all_patterns)
            # ===
            pattern_features = self.fe.extract_features_from_trie_patterns(
                subpatterns, weights)
            organisational_features = self.fe.extract_organisational_features(
                stemmed_paragraph)
            df_train_respa = df_train_respa.append({
                self._CLASS: self._RESPA_VALUE,
                self._UNQ_MATCHED_PATTERNS_COUNT: pattern_features[
                    self._UNQ_MATCHED_PATTERNS_COUNT],
                self._MATCHED_PATTERNS_COUNT: pattern_features[
                    self._MATCHED_PATTERNS_COUNT],
                self._BULLET_DEPARTMENT: isDep,
                self._ORG_TOTAL_MATCHING_CHARACTERS: organisational_features[
                    self._ORG_TOTAL_MATCHING_CHARACTERS],
                self._ORG_MATCHING_UNIGRAMS: organisational_features[
                    self._ORG_MATCHING_UNIGRAMS],
                self._ORG_MATCHING_BIGRAMS: organisational_features[
                    self._ORG_MATCHING_BIGRAMS],
                self._TOTAL_MATCHING_CHARACTERS: pattern_features[
                    self._TOTAL_MATCHING_CHARACTERS],
                self._LONGEST_MATCHING_PATTERN: pattern_features[
                    self._LONGEST_MATCHING_PATTERN],
                self._SUM_MATCHING_ENTRIES: pattern_features[
                    self._SUM_MATCHING_ENTRIES],
                self._SUM_MATCHING_ENTRIES_LENGTH: pattern_features[
                    self._SUM_MATCHING_ENTRIES_LENGTH],
                self._MATCHING_UNIGRAMS: pattern_features[
                    self._MATCHING_UNIGRAMS],
                self._MATCHING_BIGRAMS: pattern_features[
                    self._MATCHING_BIGRAMS],
                self._ALPHA_BULLET_DOT: isAlphaBulletDot,
                self._ALPHA_BULLET_PAR: isAlphaBulletPar,
                self._ALPHA_BULLET_CAP_DOT: isAlphaBulletCapDot,
                self._ALPHA_BULLET_CAP_PAR: isAlphaBulletCapPar,
                self._DIGIT_BULLET_DOT: isDigitBulletDot,
                self._DIGIT_BULLET_PAR: isDigitBulletPar,
                self._RAW_PARAGRAPH: raw_paragraph,
                self._STEMMED_PARAGRAPH: stemmed_paragraph,
                self._STEMMED_PARAGRAPH_LENGTH: len(stemmed_paragraph),
                self._RAW_PARAGRAPH_LENGTH: len(raw_paragraph),
                self._FIRST_PATTERN_OFFSET: pattern_features[
                    self._FIRST_PATTERN_OFFSET],
                self._WORDS_IN_CAPITAL: words_in_capital,
                self._FIRST_WORDS_IN_CAPITAL_OFFSET:
                    first_capital_word_offset,
                self._UNQ_MATCHING_UNIGRAMS: pattern_features[
                    self._UNQ_MATCHING_UNIGRAMS],
                self._UNQ_MATCHING_BIGRAMS: pattern_features[
                    self._UNQ_MATCHING_BIGRAMS]}, ignore_index=True)
        for index, row in non_respas_pdf.iterrows():
            raw_paragraph = row[self._RAW_PARAGRAPH]
            stemmed_paragraph = row[self._STEMMED_PARAGRAPH]
            words_in_capital = self.tp.get_words_in_capital(raw_paragraph)
            first_capital_word_offset = \
                self.tp.get_first_word_in_capital_offset(raw_paragraph)
            isAlphaBulletDot = self.fe.regex_applies(
                self._REGEX_ALPHA_DOT, raw_paragraph)
            isAlphaBulletPar = self.fe.regex_applies(
                self._REGEX_ALPHA_PAR, raw_paragraph)
            isAlphaBulletCapDot = self.fe.regex_applies(
                self._REGEX_ALPHA_CAP_DOT, raw_paragraph)
            isAlphaBulletCapPar = self.fe.regex_applies(
                self._REGEX_ALPHA_CAP_PAR, raw_paragraph)
            isDigitBulletDot = self.fe.regex_applies(
                self._REGEX_NUM_DOT, raw_paragraph)
            isDigitBulletPar = self.fe.regex_applies(
                self._REGEX_NUM_PAR, raw_paragraph)
            isDep = self.fe.regex_applies(self._REGEX_DEP, raw_paragraph)
            all_patterns = list(
                self.trie.search_all_patterns(stemmed_paragraph))
            # Remove unigrams that are contained in bigrams ===
            subpatterns = utils.remove_unigrams_contained_in_bigrams(
                all_patterns)
            # ===
            pattern_features = self.fe.extract_features_from_trie_patterns(
                subpatterns, weights)
            organisational_features = self.fe.extract_organisational_features(
                stemmed_paragraph)
            df_train_respa = df_train_respa.append({
                self._CLASS: self._NON_RESPA_VALUE,
                self._UNQ_MATCHED_PATTERNS_COUNT: pattern_features[
                    self._UNQ_MATCHED_PATTERNS_COUNT],
                self._MATCHED_PATTERNS_COUNT: pattern_features[
                    self._MATCHED_PATTERNS_COUNT],
                self._BULLET_DEPARTMENT: isDep,
                self._ORG_TOTAL_MATCHING_CHARACTERS: organisational_features[
                    self._ORG_TOTAL_MATCHING_CHARACTERS],
                self._ORG_MATCHING_UNIGRAMS: organisational_features[
                    self._ORG_MATCHING_UNIGRAMS],
                self._ORG_MATCHING_BIGRAMS: organisational_features[
                    self._ORG_MATCHING_BIGRAMS],
                self._TOTAL_MATCHING_CHARACTERS: pattern_features[
                    self._TOTAL_MATCHING_CHARACTERS],
                self._LONGEST_MATCHING_PATTERN: pattern_features[
                    self._LONGEST_MATCHING_PATTERN],
                self._SUM_MATCHING_ENTRIES: pattern_features[
                    self._SUM_MATCHING_ENTRIES],
                self._SUM_MATCHING_ENTRIES_LENGTH: pattern_features[
                    self._SUM_MATCHING_ENTRIES_LENGTH],
                self._MATCHING_UNIGRAMS: pattern_features[
                    self._MATCHING_UNIGRAMS],
                self._MATCHING_BIGRAMS: pattern_features[
                    self._MATCHING_BIGRAMS],
                self._ALPHA_BULLET_DOT: isAlphaBulletDot,
                self._ALPHA_BULLET_PAR: isAlphaBulletPar,
                self._ALPHA_BULLET_CAP_DOT: isAlphaBulletCapDot,
                self._ALPHA_BULLET_CAP_PAR: isAlphaBulletCapPar,
                self._DIGIT_BULLET_DOT: isDigitBulletDot,
                self._DIGIT_BULLET_PAR: isDigitBulletPar,
                self._RAW_PARAGRAPH: raw_paragraph,
                self._STEMMED_PARAGRAPH: stemmed_paragraph,
                self._STEMMED_PARAGRAPH_LENGTH: len(stemmed_paragraph),
                self._RAW_PARAGRAPH_LENGTH: len(raw_paragraph),
                self._FIRST_PATTERN_OFFSET:
                    pattern_features[self._FIRST_PATTERN_OFFSET],
                self._WORDS_IN_CAPITAL: words_in_capital,
                self._FIRST_WORDS_IN_CAPITAL_OFFSET:
                    first_capital_word_offset,
                self._UNQ_MATCHING_UNIGRAMS: pattern_features[
                    self._UNQ_MATCHING_UNIGRAMS],
                self._UNQ_MATCHING_BIGRAMS: pattern_features[
                    self._UNQ_MATCHING_BIGRAMS]}, ignore_index=True)
        df_train_respa[self._TOTAL_MATCHING_RATIO] = df_train_respa.apply(
            lambda row: row.TotalMatchingCharacters / len(
                row.StemmedParagraph), axis=1)
        self.respa_classifier = svm.SVC(C=1, gamma=self._SVC_GAMMA_PARAM)
        self.respa_classifier.fit(df_train_respa[
            self._RESPA_USED_FEATURES].values, df_train_respa[self._CLASS])
        return df_train_respa
