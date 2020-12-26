from collections import OrderedDict
from helper import Helper


def n_gram_is_respa(n_gram, respa_kw_pair):
    return any([(respa_kw_pair[0] in word) for word in n_gram]) and \
        any([(respa_kw_pair[1] in word) for word in n_gram])


def bigram_is_respa(bi_gram, special_respa_kw_pair):
    return special_respa_kw_pair[0] in bi_gram[0] and \
        special_respa_kw_pair[1][0] == bi_gram[1][0]


class Analyzer:
    """A class for doing n-gram Responsibility Assignments (RespAs) analysis
    of Public Administration Organizations (PAOrg) Presidential Decree Issues
    """
    _COMMON_BIGRAM_PAIRS_KEY = 'common_bigram_pairs'
    _COMMON_QUADGRAM_PAIRS_KEY = 'common_quadgram_pairs'
    _QUADGRAM_ANALYSIS_KEY = 'quadgram_analysis'
    _BIGRAM_ANALYSIS_KEY = 'bigram_analysis'
    _RESPA_KEYWORDS = {
        'primary': ["αρμόδι", "αρμοδι", "αρμοδιότητ", "ευθύνη", "εύθυν"],
        'secondary': ["για", "εξής"],
        _COMMON_BIGRAM_PAIRS_KEY: [("αρμόδι", "για"), ("ευθύνη", "για"),
                                   ("εύθυν", "για"), ("αρμοδιότητ", "ακόλουθ"),
                                   ("αρμοδιότητ", "μεταξύ"),
                                   ("ρμοδιότητες", "τ")],
        _COMMON_QUADGRAM_PAIRS_KEY: [("αρμοδιότητ", "έχει"),
                                     ("αρμοδιότητ", "εξής"),
                                     ("αρμοδιότητ", "είναι")]
    }

    def _do_bi_gram_analysis(self, txt):
        word_bi_grams = Helper.get_word_n_grams(txt, 2)
        analysis_data = OrderedDict()
        for respa_kw_pair in self._RESPA_KEYWORDS[
                self._COMMON_BIGRAM_PAIRS_KEY][:-1]:
            occurences = sum([n_gram_is_respa(bigram, respa_kw_pair)
                              for bigram in word_bi_grams])
            analysis_data[respa_kw_pair] = occurences
        # Manage special ("ρμοδιότητες", "τ") case separately
        special_respa_kw_pair = self._RESPA_KEYWORDS[
            self._COMMON_BIGRAM_PAIRS_KEY][-1]
        special_occurences = sum(
            [bigram_is_respa(bigram, special_respa_kw_pair)
             for bigram in word_bi_grams])
        analysis_data[
            special_respa_kw_pair] = special_occurences
        return analysis_data

    def _do_quad_qram_analysis(self, txt):
        word_quad_grams = Helper.get_word_n_grams(txt, 4)
        quad_gram_analysis_data = OrderedDict()
        for respa_kw_pair in self._RESPA_KEYWORDS[
                self._COMMON_QUADGRAM_PAIRS_KEY]:
            occurences = sum([n_gram_is_respa(quadgram, respa_kw_pair)
                              for quadgram in word_quad_grams])
            quad_gram_analysis_data[respa_kw_pair] = occurences
        return quad_gram_analysis_data

    def get_analysis_vector(self, txt):
        """Return a list of raw bi/quad-gram RespA analysis weights

        @param txt: Article, or Presidential Decree Organization Issue
            e.g. "ΠΡΟΕΔΡΙΚΟ ΔΙΑΤΑΓΜΑ ΥΠ’ ΑΡΙΘΜ. 18
                  Οργανισμός Υπουργείου Παιδείας, Έρευνας και
                  Θρησκευμάτων. ..."

        Returns: list of integers, e.g. [12, 12, 5, 3, 0, 40, 5, 0, 26]
        """
        txt = Helper.clean_up_txt(txt)
        bigram_data_dict = self._do_bi_gram_analysis(txt)
        quadgram_data_dict = self._do_quad_qram_analysis(txt)
        bigram_data_vector = bigram_data_dict.values()
        quadgram_data_vector = quadgram_data_dict.values()
        n_gram_data_vector = bigram_data_vector + quadgram_data_vector
        return n_gram_data_vector
