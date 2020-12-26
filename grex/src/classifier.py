import os
import pathlib
from helper import Helper


class ParagraphRespAClassifier(object):
    """A class that classifies Public Administration Organization (PAOrg)
    Presidential Decree paragrams as RespA-related or not.
    """

    _RESPA_KEY = 'respa'
    _NON_RESPA_KEY = 'non_respa'
    _UNIT_KEYWORDS = ["ΤΜΗΜΑ", "ΓΡΑΦΕΙΟ ", "ΓΡΑΦΕΙΑ ", "ΑΥΤΟΤΕΛΕΣ ",
                      "ΑΥΤΟΤΕΛΗ ", "ΔΙΕΥΘΥΝΣ", "ΥΠΗΡΕΣΙΑ ", "ΣΥΜΒΟΥΛΙ",
                      'ΓΡΑΜΜΑΤΕIA ', "ΥΠΟΥΡΓ", "ΕΙΔΙΚΟΣ ΛΟΓΑΡΙΑΣΜΟΣ",
                      "MONAΔ", "ΠΕΡΙΦΕΡΕΙ"]
    _RESPA_PAIRS = [("ΑΡΜΟΔΙΟΤΗΤ", ":"), ("ΑΡΜΟΔΙΟΤΗΤ", "."),
                    ('ΑΡΜΟΔΙΟΤΗΤΕΣ', 'ΥΠΗΡΕΣΙΩΝ')]
    _RESPA_KEYWORD_TRIOS = [("ΑΡΜΟΔ", "ΓΙΑ", ":"), ("ΩΣ", "ΣΚΟΠΟ", ":"),
                            ("ΑΡΜΟΔΙΟΤ", "ΕΧΕΙ", ":"),
                            ("ΑΡΜΟΔΙΟΤ", "ΕΞΗΣ", ":"),
                            ("ΑΡΜΟΔΙΟΤ", "ΠΟΥ", ":"),
                            ("ΑΡΜΟΔΙΟΤ", "ΕΙΝΑΙ", ":"),
                            ("ΑΡΜΟΔΙΟΤ", "ΤΟΥ", ":"),
                            ("ΑΡΜΟΔΙΟΤ", "ΑΚΟΛΟΥΘ", ":"),
                            ("ΑΡΜΟΔΙΟΤ", "ΜΕΤΑΞΥ", ":"),
                            ("ΑΡΜΟΔΙΟΤ", "ΕΠΙ", ":"),
                            ("ΑΡΜΟΔΙΟΤ", "ΣΕ", ":"),
                            ("ΑΡΜΟΔΙΟΤ", "ΠΕΡΙΛΑΜΒ", ":")]
    package_dir = pathlib.Path(__file__).parents[1]
    rel_data_dir = 'data/respa_clf_models/paragraph_respa_classifier_data/'
    abs_data_dir = os.path.join(package_dir, rel_data_dir)[0]
    _RESPA_TRAINING_DATA_FILE = os.path.join(abs_data_dir,
                                             'respa_paragraphs_dict.pkl')
    _NON_RESPA_TRAINING_DATA_FILE = os.path.join(
        abs_data_dir, 'non_respa_paragraphs_dict.pkl')

    def __init__(self):
        self.training_data = {}
        # Load training data
        self.training_data[self._RESPA_KEY] = Helper.load_pickle_file(
            self._RESPA_TRAINING_DATA_FILE)
        self.training_data[self._NON_RESPA_KEY] = Helper.load_pickle_file(
            self._NON_RESPA_TRAINING_DATA_FILE)

    def has_units(self, paragraph):
        """Return True if paragraph contains units.

        @param paragraph: Any RespA or non-RespA related paragraph
         """
        paragraph = Helper.normalize_txt(paragraph)
        return any(unit_kw in paragraph
                   for unit_kw in self._UNIT_KEYWORDS)

    def has_only_units(self, paragraph):
        """Return True if paragraph contains only units, without
        anything RespA related or containing ':'.

        @param paragraph: Any RespA or non-RespA related paragraph
        """
        paragraph = Helper.normalize_txt(paragraph)
        return any((((
            unit_kw in paragraph) and (
            resp_kw_trio[0] not in paragraph) and (
            resp_kw_trio[1] not in paragraph) and (
            resp_kw_trio[2] not in paragraph)))
            for unit_kw in self._UNIT_KEYWORDS
            for resp_kw_trio in self._RESPA_KEYWORD_TRIOS)

    def has_units_and_respas(self, paragraph):
        """Returns True if paragraph contains units, something RespA-related
        and does not contain ':'.

        @param paragraph: Any RespA or non-RespA related paragraph
        """
        paragraph = Helper.normalize_txt(paragraph)
        return any((((unit_kw in paragraph) and (
            resp_kw_trio[0] in paragraph) and (
            resp_kw_trio[1] in paragraph) and (
            resp_kw_trio[2] not in paragraph)))
            for unit_kw in self._UNIT_KEYWORDS
            for resp_kw_trio in self._RESPA_KEYWORD_TRIOS)

    def has_units_followed_by_respas(self, paragraph):
        """Return True if paragraph contains units, ':' and
        something RespA-related.

        @param paragraph: Any RespA or non-RespA related paragraph
        """
        paragraph = Helper.normalize_txt(paragraph)
        return any((((unit_kw in paragraph) and (
            resp_kw_trio[0] in paragraph) and (
            resp_kw_trio[1] in paragraph) and (
            resp_kw_trio[2] in paragraph)))
            for unit_kw in self._UNIT_KEYWORDS
            for resp_kw_trio in self._RESPA_KEYWORD_TRIOS)

    def has_respas_decl(self, paragraph):
        """Return True if paragraph contains respas_decl (respa-list initiator)

        @param paragraph: Any RespA or non-RespA related paragraph
        """
        paragraph = Helper.normalize_txt(paragraph)
        return (self._RESPA_PAIRS[0][
            0] in paragraph and self._RESPA_PAIRS[0][1] in paragraph) or (
            self._RESPA_PAIRS[1][0] in paragraph and (
                self._RESPA_PAIRS[1][1] == paragraph[-1] or self._RESPA_PAIRS[
                    1][1] == paragraph[-2])) or (self._RESPA_PAIRS[
                        2][0] in paragraph and self._RESPA_PAIRS[
                        2][1] in paragraph)
