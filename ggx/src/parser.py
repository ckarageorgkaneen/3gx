import io
import os
import re
import subprocess


def get_special_regex_disjunction(key_list):
    regex_disj_str = ''
    for key in key_list:
        key = str(key).replace(' ', '\\s+')
        regex_disj_str += str(key) + '|'
    return regex_disj_str[:-1]


class Parser:
    """Parse and extract data from Decision and Presidential Decree Issues"""

    # Keys
    _ARTICLE_KEY = 'Άρθρο'
    _LAST_ARTICLE_KEYS = [
        'Έναρξη Ισχύος',
        'Έναρξη ισχύος',
        'Η ισχύς του παρόντος',
        'EΝΑΡΞΗ ΙΣΧΥΟΣ'
    ]

    # Regex
    _REGEX_PATTERN_ARTICLES = \
        fr'({_ARTICLE_KEY}\s*\d+\s*\n.+?)(?={_ARTICLE_KEY}\s*\d+\s*\n)'
    last_article_regex_disjunction = \
        get_special_regex_disjunction(_LAST_ARTICLE_KEYS)
    _REGEX_PATTERN_LAST_ARTICLES = \
        fr'({_ARTICLE_KEY}\s*\d+\s*\n' \
        fr'(?:{last_article_regex_disjunction}).+?\.\s*\n)'
    _REGEX_PATTERN_PARAGRAPHS = \
        r'\n?\s*([Ά-ΏΑ-Ωα-ωά-ώBullet\d+\(•\-\−]+[\.\)α-ω ][\s\S]+?' \
        r'(?:[\.\:](?=\s*\n)|\,(?=\s*\n(?:[α-ω\d]+[\.\)]|Bullet))))'
    _REGEX_PATTERN_CIDS = r'\(cid:\d+\)'
    _REGEX_PATTERNS_PARAGRAPH_PRELIMS = [
        r'(?:Τεύχος|ΤΕΥΧΟΣ)\s*[Α-Ω].*\nΕΦΗΜΕΡΙ.*\n[0-9]*\n',
        r'[0-9]*\s*\nΕΦΗΜΕΡΙ.*\n(?:Τεύχος|ΤΕΥΧΟΣ)\s*[Α-Ω].*\n',
        r'ΕΦΗΜΕΡΙ.*\s*\n[0-9]*\s*\n',
        r'.ρθρο\s*[0-9]*\s*\n',
    ]

    def _pdf_text(self, in_pdf, out_txt):
        """ Return cleaned-up text of pdf.

        @param in_pdf: The input pdf file
        @param out_txt: The output txt file
        """
        if not os.path.exists(in_pdf):
            print(f'{in_pdf} does not exist!')
        if os.path.exists(out_txt):
            print(f'{out_txt} already exists! Fetching text anyway...')
        else:
            print(f'{in_pdf} -> {out_txt}:', end='')
            escaped_input = re.escape(in_pdf)
            escaped_output = re.escape(out_txt)
            cmd = f'pdf2txt.py {escaped_input} > {escaped_output}'
            subprocess.call(cmd, shell=True)
        with open(out_txt) as out_file:
            text = out_file.read()
        print('DONE.')
        # Clean up text
        cid_occurs = re.findall(self._REGEX_PATTERN_CIDS, text)
        # Ignore cid occurences
        for cid in cid_occurs:
            text = text.replace(cid, '')
        new_text = ''
        # Overwrite .txt
        with io.StringIO(text) as in_file, open(out_txt, 'w') as out_file:
            for line in in_file:
                # Skip empty lines
                if not line.strip():
                    continue
                out_file.write(line)
                new_text += line
        # Read .txt locally again
        # text = ''
        # with open(out_txt) as out_file:
        #     text = out_file.read()
        return new_text

    def get_articles(self, txt):
        """Return a dictionary of articles contained within an Issue.

        @param txt: GG Issue containing articles
            e.g. 'Άρθρο 1\nΑποστολή \nΤο Υπουργείο Ανάπτυξης και
                  Ανταγωνιστικότητας\nέχει ως αποστολή τη διαμόρφωση της
                  αναπτυξιακής \nπολιτικής της χώρας που στοχεύει στην
                  προώθηση ...'
        """
        articles = re.findall(self._REGEX_PATTERN_ARTICLES,
                              txt, flags=re.DOTALL)
        last_articles = re.findall(
            self._REGEX_PATTERN_LAST_ARTICLES, txt, flags=re.DOTALL)
        articles.extend(last_articles)
        return {i + 1: article for i, article in enumerate(articles)}

    def get_paragraphs(self, txt):
        _REGEX_PATTERN_TAB_WHITESPACE = r'[\t ]+'
        _REGEX_PATTERN_HYPHEN_MINUS_SPACE = r'\-[\s]+'
        _REGEX_PATTERN_DASH_SPACE = r'\−[\s]+'
        # Cleanup text
        txt = re.sub(_REGEX_PATTERN_TAB_WHITESPACE, ' ', txt)
        txt = re.sub(_REGEX_PATTERN_HYPHEN_MINUS_SPACE, '', txt)
        txt = re.sub(_REGEX_PATTERN_DASH_SPACE, '', txt)
        txt = txt.replace('\f', '')
        # Remove preliminaries
        for regex in self._REGEX_PATTERNS_PARAGRAPH_PRELIMS:
            pat = re.compile(regex)
            txt = pat.sub('', txt)
        # Get paragraphs
        paragraphs = re.findall(self._REGEX_PATTERN_PARAGRAPHS, txt)
        return paragraphs

    def get_pdf_txt(self, filename, pdf_path, txt_path):
        """Return cleaned-up text of pdf.

        @param filename: The input pdf file name (extensionless)
        @param pdf_path: The input pdf's dir path
        @param txt_path: The output txt's dir path
        """
        return self._pdf_text(f'{pdf_path}{filename}.pdf',
                              f'{txt_path}{filename}.txt')
