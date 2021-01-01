import pandas as pd
import pickle
import re
import trie_search as ts

from . import utils

from greek_stemmer import GreekStemmer
from sklearn.feature_extraction.text import CountVectorizer

class FeatureExtractor:

    # Keys
    _FREQUENCY_KEY = 'freq'
    _STEMS_KEY = 'stems'
    _PREDICTION = 'Prediction'
    _EXTRA_FEATURES = 'ExtraFeatures'
    _UNQ_MATCHED_PATTERNS_COUNT = 'UnqMatchedPatternsCount'
    _MATCHED_PATTERNS_COUNT = 'MatchedPatternsCount'
    _TOTAL_MATCHING_CHARACTERS = 'TotalMatchingCharacters'
    _LONGEST_MATCHING_PATTERN = 'LongestMatchingPattern'
    _SUM_MATCHING_ENTRIES = 'SumMatchingEntries'
    _SUM_MATCHING_ENTRIES_LENGTH = 'SumMatchingEntriesLength'
    _MATCHING_UNIGRAMS = 'MatchingUnigrams'
    _MATCHING_BIGRAMS = 'MatchingBigrams'
    _UNQ_MATCHING_UNIGRAMS = 'UnqMatchingUnigrams'
    _UNQ_MATCHING_BIGRAMS = 'UnqMatchingBigrams'
    _FIRST_PATTERN_OFFSET = 'FirstPatternOffset'
    _UID = 'UID'
    _ALPHA_BULLET_CAP_DOT = 'AlphaBulletCapDot'
    _ALPHA_BULLET_CAP_PAR = 'AlphaBulletCapPar'
    _ALPHA_BULLET_DOT = 'AlphaBulletDot'
    _ALPHA_BULLET_PAR = 'AlphaBulletPar'
    _BULLET_DEPARTMENT = 'BulletDepartment'
    _CLASS = 'Class'
    _DIGIT_BULLET_DOT = 'DigitBulletDot'
    _DIGIT_BULLET_PAR = 'DigitBulletPar'
    _FIRST_WORDS_IN_CAPITAL_OFFSET = 'FirstWordInCapitalOffset'
    _ORG_MATCHING_BIGRAMS = 'OrgMatchingBigrams'
    _ORG_MATCHING_UNIGRAMS = 'OrgMatchingUnigrams'
    _ORG_TOTAL_MATCHING_CHARACTERS = 'OrgTotalMatchingCharacters'
    _RAW_PARAGRAPH = 'RawParagraph'
    _RAW_PARAGRAPH_LENGTH = 'RawParagraphLength'
    _STEMMED_PARAGRAPH = 'StemmedParagraph'
    _STEMMED_PARAGRAPH_LENGTH = 'StemmedParagraphLength'
    _WORDS_IN_CAPITAL = 'WordsInCapital'
    _TOTAL_MATCHING_RATIO = 'TotalMatchingRatio'
    _ORG_LONGEST_MATCHING_PATTERN = 'OrgLongestMatchingPattern'
    _COLUMNS = [
        _UID,
        _ALPHA_BULLET_CAP_DOT,
        _ALPHA_BULLET_CAP_PAR,
        _ALPHA_BULLET_DOT,
        _ALPHA_BULLET_PAR,
        _BULLET_DEPARTMENT,
        _CLASS,
        _DIGIT_BULLET_DOT,
        _DIGIT_BULLET_PAR,
        _FIRST_PATTERN_OFFSET,
        _FIRST_WORDS_IN_CAPITAL_OFFSET,
        _LONGEST_MATCHING_PATTERN,
        _MATCHED_PATTERNS_COUNT,
        _MATCHING_BIGRAMS,
        _MATCHING_UNIGRAMS,
        _ORG_MATCHING_BIGRAMS,
        _ORG_MATCHING_UNIGRAMS,
        _ORG_TOTAL_MATCHING_CHARACTERS,
        _RAW_PARAGRAPH,
        _RAW_PARAGRAPH_LENGTH,
        _STEMMED_PARAGRAPH,
        _STEMMED_PARAGRAPH_LENGTH,
        _SUM_MATCHING_ENTRIES,
        _SUM_MATCHING_ENTRIES_LENGTH,
        _TOTAL_MATCHING_CHARACTERS,
        _UNQ_MATCHED_PATTERNS_COUNT,
        _UNQ_MATCHING_BIGRAMS,
        _UNQ_MATCHING_UNIGRAMS,
        _WORDS_IN_CAPITAL,
        _TOTAL_MATCHING_RATIO
    ]

    # Regex
    _REGEX_ALPHA_DOT = re.compile(r'^([a-z]|[α-ω]){1,4}(\.)')
    _REGEX_ALPHA_PAR = re.compile(r'^([a-z]|[α-ω]){1,4}(\))')
    _REGEX_ALPHA_CAP_DOT = re.compile(r'^([Α-Ω]|[A-Z]){1,4}(\.)')
    _REGEX_ALPHA_CAP_PAR = re.compile(r'^([Α-Ω]|[A-Z]){1,4}(\))')
    _REGEX_NUM_DOT = re.compile(r'^([0-9]){1,4}(\.)')
    _REGEX_NUM_PAR = re.compile(r'^([0-9]){1,4}(\))')
    _REGEX_DEP = re.compile(
        r'^([a-z]|[A-Z]|[Α-Ω]|[α-ω]|[0-9]){1,4}(\.|\))(\sΤμήμα)')

    def __init__(self, organizations_file, text_preprocessor, ratio,
                 headersize):
        stemmer = GreekStemmer()
        v = CountVectorizer(ngram_range=(1, 2), lowercase=False)
        stemmed_organizations = []
        freq_organizations = {}
        with open(organizations_file) as fp:
            pat = re.compile(r'[^\w\.]+')
            for cnt, line in enumerate(fp):
                list_ = []
                clean_line = ' '.join(pat.split(line.replace('"', '')))
                if clean_line:
                    for w in clean_line.split():
                        stem = stemmer.stem(w)
                        # create upper case stemmed organizations
                        list_.append(stem.upper())
                    organisation = text_preprocessor.getCleanText(
                        " ".join(list_), headersize)  # create one string
                    wordgrams = v.fit(list([organisation])).vocabulary_.keys()
                    for wgram in wordgrams:
                        if wgram in freq_organizations:
                            freq_organizations[wgram] += 1
                        else:
                            freq_organizations[wgram] = 1
                    # insert it to a list with all organizations
                    stemmed_organizations.append(organisation)
        temp_df = pd.DataFrame(
            list(freq_organizations.items()),
            columns=[self._STEMS_KEY, self._FREQUENCY_KEY])
        selected_df = temp_df[
            temp_df[self._FREQUENCY_KEY] / len(stemmed_organizations) > ratio]
        freq_stems = selected_df[self._STEMS_KEY].values.tolist()
        freqstemscopy = []
        for s in freq_stems:
            if not text_preprocessor.hasNumbers(s) and len(s) > 3:
                freqstemscopy.append(s)
        self.org_trie = ts.TrieSearch(freqstemscopy)
        self.text_preprocessor = text_preprocessor
        self.headersize = headersize

    def read_org_trie_from_file(self, text):
        self.org_trie = pickle.load(open(text, 'rb'))

    def get_words_in_capital(self, text):
        return self.text_preprocessor.get_words_in_capital(text)

    def regex_applies(self, regex, text):
        if regex.search(text):
            return 1
        else:
            return 0

    def extract_features_from_trie_patterns(self, patterns, weights):
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
        patterns_so_far = []
        container = set()
        for pattern in patterns:
            total_matching_characters += len(pattern)
            if (len(pattern) > longest_matching_pattern):
                longest_matching_pattern = len(pattern)
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
                sum_matching_entries += weights.get(pattern, 0.0)
                sum_matching_entries_len += weights.get(pattern, 0.0) * len(
                    pattern)
                container.add(pattern)
        features = {
            self._UNQ_MATCHED_PATTERNS_COUNT: len(container),
            self._MATCHED_PATTERNS_COUNT: len(patterns),
            self._TOTAL_MATCHING_CHARACTERS: total_matching_characters,
            self._LONGEST_MATCHING_PATTERN: longest_matching_pattern,
            self._SUM_MATCHING_ENTRIES: sum_matching_entries,
            self._SUM_MATCHING_ENTRIES_LENGTH: sum_matching_entries_len,
            self._MATCHING_UNIGRAMS: matching_unigrams,
            self._MATCHING_BIGRAMS: matching_bigrams,
            self._UNQ_MATCHING_UNIGRAMS: unq_matching_unigrams,
            self._UNQ_MATCHING_BIGRAMS: unq_matching_bigrams,
            self._FIRST_PATTERN_OFFSET: first_pattern_offset
        }
        return features

    def read_training_file(self, filename):
        train_data = pd.read_csv(filename, sep='\t')
        train_data.columns = self._COLUMNS
        return train_data

    def extract_features(self, stemmed_paragraph, all_patterns, weights):
        subpatterns = utils.remove_unigrams_contained_in_bigrams(all_patterns)
        total_matching_characters = 0
        longest_matching_pattern = 0
        # Wordgrams
        matching_unigrams = 0
        matching_bigrams = 0
        unq_matching_unigrams = 0
        unq_matching_bigrams = 0
        first_pattern_offset = 0
        sum_matching_entries = 0
        sum_matching_entries_len = 0
        patterns_so_far = []
        container = set()
        for pattern in subpatterns:
            total_matching_characters += len(pattern)
            if len(pattern) > longest_matching_pattern:
                longest_matching_pattern = len(pattern)
            if len(pattern.split()) == 1:
                if pattern not in patterns_so_far:
                    unq_matching_unigrams += 1
                    patterns_so_far.append(pattern)
                matching_unigrams += 1
            if len(pattern.split()) == 2:
                if pattern not in patterns_so_far:
                    unq_matching_bigrams += 1
                    patterns_so_far.append(pattern)
                matching_bigrams += 1
            if pattern not in container:
                sum_matching_entries += weights.get(pattern, 0.0)
                sum_matching_entries_len += weights.get(pattern, 0.0) * len(
                    pattern)
                container.add(pattern)
        features = {
            self._UNQ_MATCHED_PATTERNS_COUNT: len(container),
            self._MATCHED_PATTERNS_COUNT: len(subpatterns),
            self._TOTAL_MATCHING_CHARACTERS: total_matching_characters,
            self._LONGEST_MATCHING_PATTERN: longest_matching_pattern,
            self._SUM_MATCHING_ENTRIES: sum_matching_entries,
            self._SUM_MATCHING_ENTRIES_LENGTH: sum_matching_entries_len,
            self._MATCHING_UNIGRAMS: matching_unigrams,
            self._MATCHING_BIGRAMS: matching_bigrams,
            self._UNQ_MATCHING_UNIGRAMS: unq_matching_unigrams,
            self._UNQ_MATCHING_BIGRAMS: unq_matching_bigrams,
            self._FIRST_PATTERN_OFFSET: first_pattern_offset
        }
        return features

    def extract_features_from_file(self, filename, weights, trie, tp,
                                   headersize):
        train_data = pd.read_csv(filename, sep='|')
        self._REGEX_ALPHA_DOT = re.compile(r'^([a-z]|[α-ω]){1,4}(\.)')
        self._REGEX_ALPHA_PAR = re.compile(r'^([a-z]|[α-ω]){1,4}(\))')
        self._REGEX_ALPHA_CAP_DOT = re.compile(r'^([Α-Ω]|[A-Z]){1,4}(\.)')
        self._REGEX_ALPHA_CAP_PAR = re.compile(r'^([Α-Ω]|[A-Z]){1,4}(\))')
        self._REGEX_NUM_DOT = re.compile(r'^([0-9]){1,4}(\.)')
        self._REGEX_NUM_PAR = re.compile(r'^([0-9]){1,4}(\))')
        train_data[self._ALPHA_BULLET_DOT] = train_data.apply(
            lambda row: self.regex_applies(
                self._REGEX_ALPHA_DOT, row.RawParagraph),
            axis=1)
        train_data[self._ALPHA_BULLET_PAR] = train_data.apply(
            lambda row: self.regex_applies(
                self._REGEX_ALPHA_PAR, row.RawParagraph),
            axis=1)
        train_data[self._ALPHA_BULLET_CAP_DOT] = train_data.apply(
            lambda row: self.regex_applies(
                self._REGEX_ALPHA_CAP_DOT, row.RawParagraph),
            axis=1)
        train_data[self._ALPHA_BULLET_CAP_PAR] = train_data.apply(
            lambda row: self.regex_applies(
                self._REGEX_ALPHA_CAP_PAR, row.RawParagraph),
            axis=1)
        train_data[self._DIGIT_BULLET_DOT] = train_data.apply(
            lambda row: self.regex_applies(
                self._REGEX_NUM_DOT, row.RawParagraph),
            axis=1)
        train_data[self._DIGIT_BULLET_PAR] = train_data.apply(
            lambda row: self.regex_applies(
                self._REGEX_NUM_PAR, row.RawParagraph),
            axis=1)
        train_data[self._DIGIT_BULLET_PAR] = train_data.apply(
            lambda row: self.regex_applies(
                self._REGEX_NUM_PAR, row.RawParagraph),
            axis=1)
        train_data[self._STEMMED_PARAGRAPH] = train_data.apply(
            lambda row: tp.getStemmedParagraph(row.RawParagraph, headersize),
            axis=1)
        train_data[self._EXTRA_FEATURES] = train_data.apply(
            lambda row: self.extract_features(
                row.StemmedParagraph,
                trie.search_all_patterns(row.StemmedParagraph), weights),
            axis=1)
        train_data[self._UNQ_MATCHED_PATTERNS_COUNT] = train_data.apply(
            lambda row: row.ExtraFeatures[
                self._UNQ_MATCHED_PATTERNS_COUNT], axis=1)
        train_data[self._MATCHED_PATTERNS_COUNT] = train_data.apply(
            lambda row: row.ExtraFeatures[
                self._MATCHED_PATTERNS_COUNT], axis=1)
        train_data[self._TOTAL_MATCHING_CHARACTERS] = train_data.apply(
            lambda row: row.ExtraFeatures[
                self._TOTAL_MATCHING_CHARACTERS], axis=1)
        train_data[self._LONGEST_MATCHING_PATTERN] = train_data.apply(
            lambda row: row.ExtraFeatures[
                self._LONGEST_MATCHING_PATTERN], axis=1)
        train_data[self._SUM_MATCHING_ENTRIES] = train_data.apply(
            lambda row: row.ExtraFeatures[self._SUM_MATCHING_ENTRIES], axis=1)
        train_data[self._SUM_MATCHING_ENTRIES_LENGTH] = train_data.apply(
            lambda row: row.ExtraFeatures[
                self._SUM_MATCHING_ENTRIES_LENGTH], axis=1)
        train_data[self._MATCHING_UNIGRAMS] = train_data.apply(
            lambda row: row.ExtraFeatures[self._MATCHING_UNIGRAMS], axis=1)
        train_data[self._MATCHING_BIGRAMS] = train_data.apply(
            lambda row: row.ExtraFeatures[self._MATCHING_BIGRAMS], axis=1)
        train_data[self._UNQ_MATCHING_UNIGRAMS] = train_data.apply(
            lambda row: row.ExtraFeatures[self._UNQ_MATCHING_UNIGRAMS], axis=1)
        train_data[self._UNQ_MATCHING_BIGRAMS] = train_data.apply(
            lambda row: row.ExtraFeatures[self._UNQ_MATCHING_BIGRAMS], axis=1)
        train_data[self._FIRST_PATTERN_OFFSET] = train_data.apply(
            lambda row: row.ExtraFeatures[self._FIRST_PATTERN_OFFSET], axis=1)
        train_data[self._CLASS] = train_data.apply(
            lambda row: row.Prediction, axis=1)
        train_data.drop(
            columns=[self._EXTRA_FEATURES, self._PREDICTION],
            axis=1, inplace=True)
        return train_data

    def update_organisational_features_from_file(
            self, filename1, filename2, weights, trie, tp, headersize):
        train_org_df = pd.read_csv(filename1, sep='\t')
        train_org_df2 = pd.read_csv(filename2, sep='|')
        train_org = train_org_df2[[self._RAW_PARAGRAPH, self._PREDICTION]]
        train_org[self._CLASS] = train_org.apply(
            lambda row: row.Prediction, axis=1)
        train_org.drop(columns=[self._PREDICTION], axis=1, inplace=True)
        isOrg = train_org[self._CLASS] == 'Org'
        train_org = train_org[isOrg]
        cols = [0, 3, 4, 5, 6, 7]
        train_org_df.drop(train_org_df.columns[cols], axis=1, inplace=True)
        train_org_df.columns = [self._CLASS, self._RAW_PARAGRAPH]
        train_org_df = train_org_df.append(train_org, sort=True)
        train_org_df = train_org_df.reset_index(drop=True)
        train_org_df[self._STEMMED_PARAGRAPH] = train_org_df.apply(
            lambda row: self.text_preprocessor.getCleanText(
                row[self._RAW_PARAGRAPH], self.headersize), axis=1)
        train_org_df[self._BULLET_DEPARTMENT] = train_org_df.apply(
            lambda row: self.regex_applies(
                self._REGEX_DEP, row[self._RAW_PARAGRAPH]),
            axis=1)
        train_org_df[self._ALPHA_BULLET_DOT] = train_org_df.apply(
            lambda row: self.regex_applies(
                self._REGEX_ALPHA_DOT, row[self._RAW_PARAGRAPH]),
            axis=1)
        train_org_df[self._ALPHA_BULLET_PAR] = train_org_df.apply(
            lambda row: self.regex_applies(
                self._REGEX_ALPHA_PAR, row[self._RAW_PARAGRAPH]),
            axis=1)
        train_org_df[self._ALPHA_BULLET_CAP_DOT] = train_org_df.apply(
            lambda row: self.regex_applies(
                self._REGEX_ALPHA_CAP_DOT, row[self._RAW_PARAGRAPH]),
            axis=1)
        train_org_df[self._ALPHA_BULLET_CAP_PAR] = train_org_df.apply(
            lambda row: self.regex_applies(
                self._REGEX_ALPHA_CAP_PAR, row[self._RAW_PARAGRAPH]),
            axis=1)
        train_org_df[self._DIGIT_BULLET_DOT] = train_org_df.apply(
            lambda row: self.regex_applies(
                self._REGEX_NUM_DOT, row[self._RAW_PARAGRAPH]),
            axis=1)
        train_org_df[self._DIGIT_BULLET_PAR] = train_org_df.apply(
            lambda row: self.regex_applies(
                self._REGEX_NUM_PAR, row[self._RAW_PARAGRAPH]),
            axis=1)
        features_df = pd.DataFrame()
        for index, row in train_org_df.iterrows():
            features_df = features_df.append(
                self.extract_organisational_features(
                    row[self._STEMMED_PARAGRAPH]), ignore_index=True)

        new_df = pd.concat([train_org_df, features_df], axis=1)
        return new_df

    def extract_organisational_features(self, text):
        org_total_matching_characters = 0
        org_longest_matching_pattern = 0
        org_matching_unigrams = 0
        org_matching_bigrams = 0
        for pattern, start_idx in self.org_trie.search_all_patterns(text):
            org_total_matching_characters += len(pattern)
            if (len(pattern) > org_longest_matching_pattern):
                org_longest_matching_pattern = len(pattern)
            if (len(pattern.split()) == 1):
                org_matching_unigrams += 1
            if (len(pattern.split()) == 2):
                org_matching_bigrams += 1
        organisational_features = {
            self._ORG_TOTAL_MATCHING_CHARACTERS: org_total_matching_characters,
            self._ORG_LONGEST_MATCHING_PATTERN: org_longest_matching_pattern,
            self._ORG_MATCHING_UNIGRAMS: org_matching_unigrams,
            self._ORG_MATCHING_BIGRAMS: org_matching_bigrams
        }
        return organisational_features

    def extract_organisational_features_from_file(self, filename):
        train_org_df = pd.read_csv(filename, sep='\t')
        cols = [0, 3, 4, 5, 6, 7]
        train_org_df.drop(train_org_df.columns[cols], axis=1, inplace=True)
        train_org_df.columns = [self._CLASS, self._RAW_PARAGRAPH]
        train_org_df[self._STEMMED_PARAGRAPH] = train_org_df.apply(
            lambda row: self.text_preprocessor.getCleanText(
                row[self._RAW_PARAGRAPH], self.headersize), axis=1)
        train_org_df[self._BULLET_DEPARTMENT] = train_org_df.apply(
            lambda row: self.regex_applies(
                self._REGEX_DEP, row[self._RAW_PARAGRAPH]),
            axis=1)
        train_org_df[self._ALPHA_BULLET_DOT] = train_org_df.apply(
            lambda row: self.regex_applies(
                self._REGEX_ALPHA_DOT, row[self._RAW_PARAGRAPH]),
            axis=1)
        train_org_df[self._ALPHA_BULLET_PAR] = train_org_df.apply(
            lambda row: self.regex_applies(
                self._REGEX_ALPHA_PAR, row[self._RAW_PARAGRAPH]),
            axis=1)
        train_org_df[self._ALPHA_BULLET_CAP_DOT] = train_org_df.apply(
            lambda row: self.regex_applies(
                self._REGEX_ALPHA_CAP_DOT, row[self._RAW_PARAGRAPH]), axis=1)
        train_org_df[self._ALPHA_BULLET_CAP_PAR] = train_org_df.apply(
            lambda row: self.regex_applies(
                self._REGEX_ALPHA_CAP_PAR, row[self._RAW_PARAGRAPH]), axis=1)
        train_org_df[self._DIGIT_BULLET_DOT] = train_org_df.apply(
            lambda row: self.regex_applies(
                self._REGEX_NUM_DOT, row[self._RAW_PARAGRAPH]),
            axis=1)
        train_org_df[self._DIGIT_BULLET_PAR] = train_org_df.apply(
            lambda row: self.regex_applies(
                self._REGEX_NUM_PAR, row[self._RAW_PARAGRAPH]),
            axis=1)
        features_df = pd.DataFrame()
        for index, row in train_org_df.iterrows():
            features_df = features_df.append(
                self.extract_organisational_features(
                    row[self._STEMMED_PARAGRAPH]),
                ignore_index=True)
        new_df = pd.concat([train_org_df, features_df], axis=1)
        return new_df
