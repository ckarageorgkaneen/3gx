from context import main, unittest, call, getcwd, os, errno, shutil, Context
from collections import defaultdict
from pickle import dump, HIGHEST_PROTOCOL

class AnalyzerTest(Context):

	def test_get_respa_kw_analysis_of_paorg_pres_decree_txts_1(self):

			ref_respa_pdf_path = self.test_pdfs_dir + '/Presidential_Decree_Issues/'
			txt_1 = self.get_txt('1_Pres_Decree', pdf_path=ref_respa_pdf_path)
			txt_2 = self.get_txt('2_Pres_Decree', pdf_path=ref_respa_pdf_path)
			txt_3 = self.get_txt('3_Pres_Decree', pdf_path=ref_respa_pdf_path)
			txt_4 = self.get_txt('4_Pres_Decree', pdf_path=ref_respa_pdf_path)
			txt_5 = self.get_txt('5_Pres_Decree', pdf_path=ref_respa_pdf_path)

			## 
			#  Decision Contents
			##
			dec_contents_1 = self.parser.get_dec_contents_from_txt(txt_1);
			dec_contents_2 = self.parser.get_dec_contents_from_txt(txt_2); 
			dec_contents_3 = self.parser.get_dec_contents_from_txt(txt_3); 
			dec_contents_4 = self.parser.get_dec_contents_from_txt(txt_4);
			dec_contents_5 = self.parser.get_dec_contents_from_txt(txt_5);
			
			## 
			#  Decision Summaries
			## 
			dec_summaries_1 = self.parser.get_dec_summaries_from_txt(txt_1, dec_contents_1);
			dec_summaries_2 = self.parser.get_dec_summaries_from_txt(txt_2, dec_contents_2);
			dec_summaries_3 = self.parser.get_dec_summaries_from_txt(txt_3, dec_contents_3);
			dec_summaries_4 = self.parser.get_dec_summaries_from_txt(txt_4, dec_contents_4);
			dec_summaries_5 = self.parser.get_dec_summaries_from_txt(txt_5, dec_contents_5);

			## 
			#  Decisions
			##
			decisions_1 = self.parser.get_decisions_from_txt(txt_1, len(dec_summaries_1))
			decisions_2 = self.parser.get_decisions_from_txt(txt_2, len(dec_summaries_2))
			decisions_3 = self.parser.get_decisions_from_txt(txt_3, len(dec_summaries_3))
			decisions_4 = self.parser.get_decisions_from_txt(txt_4, len(dec_summaries_4))
			decisions_5 = self.parser.get_decisions_from_txt(txt_5, len(dec_summaries_5))

			self.assertTrue(len(decisions_1) == 1);
			self.assertTrue(len(decisions_2) == 1); 
			self.assertTrue(len(decisions_3) == 1); 
			self.assertTrue(len(decisions_4) == 1);
			self.assertTrue(len(decisions_5) == 1);

			# Convert any dict to list
			if isinstance(decisions_1, dict): decisions_1 = list(decisions_1.values())
			if isinstance(decisions_2, dict): decisions_2 = list(decisions_2.values())
			if isinstance(decisions_3, dict): decisions_3 = list(decisions_3.values())
			if isinstance(decisions_4, dict): decisions_4 = list(decisions_4.values())
			if isinstance(decisions_5, dict): decisions_5 = list(decisions_5.values())

			articles_1 = self.parser.get_articles_from_txt(decisions_1[0])
			articles_2 = self.parser.get_articles_from_txt(decisions_2[0])
			articles_3 = self.parser.get_articles_from_txt(decisions_3[0])
			articles_4 = self.parser.get_articles_from_txt(decisions_4[0])
			articles_5 = self.parser.get_articles_from_txt(decisions_5[0])

			# Convert any dict to list
			if isinstance(articles_1, dict): articles_1 = list(articles_1.values())
			if isinstance(articles_2, dict): articles_2 = list(articles_2.values())
			if isinstance(articles_3, dict): articles_3 = list(articles_3.values())
			if isinstance(articles_4, dict): articles_4 = list(articles_4.values())
			if isinstance(articles_5, dict): articles_5 = list(articles_5.values())

			
			analysis_txt_1_data_sums = self.analyzer.analyze_issue_from_articles(articles_1)
			analysis_txt_2_data_sums = self.analyzer.analyze_issue_from_articles(articles_2)
			analysis_txt_3_data_sums = self.analyzer.analyze_issue_from_articles(articles_3)
			analysis_txt_4_data_sums = self.analyzer.analyze_issue_from_articles(articles_4)
			analysis_txt_5_data_sums = self.analyzer.analyze_issue_from_articles(articles_5)

			print(dec_summaries_1, "\n", analysis_txt_1_data_sums, "\n")
			print(dec_summaries_2, "\n", analysis_txt_2_data_sums, "\n")
			print(dec_summaries_3, "\n", analysis_txt_3_data_sums, "\n")
			print(dec_summaries_4, "\n", analysis_txt_4_data_sums, "\n")
			print(dec_summaries_5, "\n", analysis_txt_5_data_sums, "\n")

			print(self.analyzer.get_n_gram_analysis_data_sums_vector(articles_1))
			print(self.analyzer.get_n_gram_analysis_data_sums_vector(articles_2))
			print(self.analyzer.get_n_gram_analysis_data_sums_vector(articles_3))
			print(self.analyzer.get_n_gram_analysis_data_sums_vector(articles_4))
			print(self.analyzer.get_n_gram_analysis_data_sums_vector(articles_5))


	def test_get_respa_kw_analysis_of_paorg_pres_decree_txts_2(self):

			ref_respa_pdf_path = self.test_pdfs_dir + '/Presidential_Decree_Issues/'
			txt_1 = self.get_txt('6_Pres_Decree', pdf_path=ref_respa_pdf_path)
			txt_2 = self.get_txt('7_Pres_Decree', pdf_path=ref_respa_pdf_path)
			txt_3 = self.get_txt('8_Pres_Decree', pdf_path=ref_respa_pdf_path)

			## 
			#  Decision Contents
			##
			dec_contents_1 = self.parser.get_dec_contents_from_txt(txt_1);
			dec_contents_2 = self.parser.get_dec_contents_from_txt(txt_2); 
			dec_contents_3 = self.parser.get_dec_contents_from_txt(txt_3); 
			
			## 
			#  Decision Summaries
			## 
			dec_summaries_1 = self.parser.get_dec_summaries_from_txt(txt_1, dec_contents_1);
			dec_summaries_2 = self.parser.get_dec_summaries_from_txt(txt_2, dec_contents_2);
			dec_summaries_3 = self.parser.get_dec_summaries_from_txt(txt_3, dec_contents_3);

			## 
			#  Decisions
			##
			decisions_1 = self.parser.get_decisions_from_txt(txt_1, len(dec_summaries_1))
			decisions_2 = self.parser.get_decisions_from_txt(txt_2, len(dec_summaries_2))
			decisions_3 = self.parser.get_decisions_from_txt(txt_3, len(dec_summaries_3))

			self.assertTrue(len(decisions_1) == 1);
			self.assertTrue(len(decisions_2) == 1); 
			self.assertTrue(len(decisions_3) == 1); 

			# Convert any dict to list
			if isinstance(decisions_1, dict): decisions_1 = list(decisions_1.values())
			if isinstance(decisions_2, dict): decisions_2 = list(decisions_2.values())
			if isinstance(decisions_3, dict): decisions_3 = list(decisions_3.values())

			articles_1 = self.parser.get_articles_from_txt(decisions_1[0])
			articles_2 = self.parser.get_articles_from_txt(decisions_2[0])
			articles_3 = self.parser.get_articles_from_txt(decisions_3[0])

			# Convert any dict to list
			if isinstance(articles_1, dict): articles_1 = list(articles_1.values())
			if isinstance(articles_2, dict): articles_2 = list(articles_2.values())
			if isinstance(articles_3, dict): articles_3 = list(articles_3.values())
			
			analysis_txt_1_data_sums = self.analyzer.analyze_issue_from_articles(articles_1)
			analysis_txt_2_data_sums = self.analyzer.analyze_issue_from_articles(articles_2)
			analysis_txt_3_data_sums = self.analyzer.analyze_issue_from_articles(articles_3)
			
			print(dec_summaries_1, "\n", analysis_txt_1_data_sums, "\n")
			print(dec_summaries_2, "\n", analysis_txt_2_data_sums, "\n")
			print(dec_summaries_3, "\n", analysis_txt_3_data_sums, "\n")

			print(self.analyzer.get_n_gram_analysis_data_sums_vector(articles_1))
			print(self.analyzer.get_n_gram_analysis_data_sums_vector(articles_2))
			print(self.analyzer.get_n_gram_analysis_data_sums_vector(articles_3))

	def test_respa_kw_analysis_of_paorg_pres_decree_articles_1(self):

		ref_respa_pdf_path = self.test_pdfs_dir + '/Presidential_Decree_Issues/'
		txt_1 = self.get_txt('1_Pres_Decree', pdf_path=ref_respa_pdf_path)
		txt_2 = self.get_txt('2_Pres_Decree', pdf_path=ref_respa_pdf_path)
		txt_3 = self.get_txt('3_Pres_Decree', pdf_path=ref_respa_pdf_path)
		txt_4 = self.get_txt('4_Pres_Decree', pdf_path=ref_respa_pdf_path)
		txt_5 = self.get_txt('5_Pres_Decree', pdf_path=ref_respa_pdf_path)

		## 
		#  Decision Contents
		##
		dec_contents_1 = self.parser.get_dec_contents_from_txt(txt_1);
		dec_contents_2 = self.parser.get_dec_contents_from_txt(txt_2); 
		dec_contents_3 = self.parser.get_dec_contents_from_txt(txt_3); 
		dec_contents_4 = self.parser.get_dec_contents_from_txt(txt_4);
		dec_contents_5 = self.parser.get_dec_contents_from_txt(txt_5);
		
		
		## 
		#  Decision Summaries
		## 
		dec_summaries_1 = self.parser.get_dec_summaries_from_txt(txt_1, dec_contents_1);
		dec_summaries_2 = self.parser.get_dec_summaries_from_txt(txt_2, dec_contents_2);
		dec_summaries_3 = self.parser.get_dec_summaries_from_txt(txt_3, dec_contents_3);
		dec_summaries_4 = self.parser.get_dec_summaries_from_txt(txt_4, dec_contents_4);
		dec_summaries_5 = self.parser.get_dec_summaries_from_txt(txt_5, dec_contents_5);

		## 
		#  Decisions
		##
		decisions_1 = self.parser.get_decisions_from_txt(txt_1, len(dec_summaries_1))
		decisions_2 = self.parser.get_decisions_from_txt(txt_2, len(dec_summaries_2))
		decisions_3 = self.parser.get_decisions_from_txt(txt_3, len(dec_summaries_3))
		decisions_4 = self.parser.get_decisions_from_txt(txt_4, len(dec_summaries_4))
		decisions_5 = self.parser.get_decisions_from_txt(txt_5, len(dec_summaries_5))

		# Convert any dict to list
		if isinstance(decisions_1, dict): decisions_1 = list(decisions_1.values())
		if isinstance(decisions_2, dict): decisions_2 = list(decisions_2.values())
		if isinstance(decisions_3, dict): decisions_3 = list(decisions_3.values())
		if isinstance(decisions_4, dict): decisions_4 = list(decisions_4.values())
		if isinstance(decisions_5, dict): decisions_5 = list(decisions_5.values())

		articles_1 = self.parser.get_articles_from_txt(decisions_1[0])
		articles_2 = self.parser.get_articles_from_txt(decisions_2[0])
		articles_3 = self.parser.get_articles_from_txt(decisions_3[0])
		articles_4 = self.parser.get_articles_from_txt(decisions_4[0])
		articles_5 = self.parser.get_articles_from_txt(decisions_5[0])

		# Convert any dict to list
		if isinstance(articles_1, dict): articles_1 = list(articles_1.values())
		if isinstance(articles_2, dict): articles_2 = list(articles_2.values())
		if isinstance(articles_3, dict): articles_3 = list(articles_3.values())
		if isinstance(articles_4, dict): articles_4 = list(articles_4.values())
		if isinstance(articles_5, dict): articles_5 = list(articles_5.values())

		print(self.analyzer.get_custom_n_gram_analysis_data_vectors(articles_1))
		print(self.analyzer.get_custom_n_gram_analysis_data_vectors(articles_2))
		print(self.analyzer.get_custom_n_gram_analysis_data_vectors(articles_3))
		print(self.analyzer.get_custom_n_gram_analysis_data_vectors(articles_4))
		print(self.analyzer.get_custom_n_gram_analysis_data_vectors(articles_5))

	def test_respa_kw_analysis_of_paorg_pres_decree_articles_2(self):

		ref_respa_pdf_path = self.test_pdfs_dir + '/Presidential_Decree_Issues/'
		txt_1 = self.get_txt('6_Pres_Decree', pdf_path=ref_respa_pdf_path)
		txt_2 = self.get_txt('7_Pres_Decree', pdf_path=ref_respa_pdf_path)
		txt_3 = self.get_txt('8_Pres_Decree', pdf_path=ref_respa_pdf_path)

		## 
		#  Decision Contents
		##
		dec_contents_1 = self.parser.get_dec_contents_from_txt(txt_1);
		dec_contents_2 = self.parser.get_dec_contents_from_txt(txt_2); 
		dec_contents_3 = self.parser.get_dec_contents_from_txt(txt_3); 
		
		
		## 
		#  Decision Summaries
		## 
		dec_summaries_1 = self.parser.get_dec_summaries_from_txt(txt_1, dec_contents_1);
		dec_summaries_2 = self.parser.get_dec_summaries_from_txt(txt_2, dec_contents_2);
		dec_summaries_3 = self.parser.get_dec_summaries_from_txt(txt_3, dec_contents_3);

		## 
		#  Decisions
		##
		decisions_1 = self.parser.get_decisions_from_txt(txt_1, len(dec_summaries_1))
		decisions_2 = self.parser.get_decisions_from_txt(txt_2, len(dec_summaries_2))
		decisions_3 = self.parser.get_decisions_from_txt(txt_3, len(dec_summaries_3))

		# Convert any dict to list
		if isinstance(decisions_1, dict): decisions_1 = list(decisions_1.values())
		if isinstance(decisions_2, dict): decisions_2 = list(decisions_2.values())
		if isinstance(decisions_3, dict): decisions_3 = list(decisions_3.values())

		articles_1 = self.parser.get_articles_from_txt(decisions_1[0])
		articles_2 = self.parser.get_articles_from_txt(decisions_2[0])
		articles_3 = self.parser.get_articles_from_txt(decisions_3[0])

		# Convert any dict to list
		if isinstance(articles_1, dict): articles_1 = list(articles_1.values())
		if isinstance(articles_2, dict): articles_2 = list(articles_2.values())
		if isinstance(articles_3, dict): articles_3 = list(articles_3.values())

		print(self.analyzer.get_custom_n_gram_analysis_data_vectors(articles_1))
		print(self.analyzer.get_custom_n_gram_analysis_data_vectors(articles_2))
		print(self.analyzer.get_custom_n_gram_analysis_data_vectors(articles_3))

	def test_respa_kw_analysis_of_paorg_pres_decree_paragraphs_1(self):
		pdf_path = self.test_pdfs_dir + '/Presidential_Decree_Issues/'
		get_txt = self.get_txt
		txt_1 = get_txt('1_Pres_Decree', pdf_path=pdf_path)
		txt_2 = get_txt('2_Pres_Decree', pdf_path=pdf_path)
		txt_3 = get_txt('3_Pres_Decree', pdf_path=pdf_path)
		txt_4 = get_txt('4_Pres_Decree', pdf_path=pdf_path)
		txt_5 = get_txt('5_Pres_Decree', pdf_path=pdf_path)
		txt_6 = get_txt('6_Pres_Decree', pdf_path=pdf_path)
		txt_7 = get_txt('7_Pres_Decree', pdf_path=pdf_path)
		txt_8 = get_txt('8_Pres_Decree', pdf_path=pdf_path)

		get_paragraphs = self.parser.get_paragraphs
		paragraphs_1 = get_paragraphs(txt_1)
		paragraphs_2 = get_paragraphs(txt_2)
		paragraphs_3 = get_paragraphs(txt_3)
		paragraphs_4 = get_paragraphs(txt_4)
		paragraphs_5 = get_paragraphs(txt_5)
		paragraphs_6 = get_paragraphs(txt_6)
		paragraphs_7 = get_paragraphs(txt_7)
		paragraphs_8 = get_paragraphs(txt_8)

		print(self.analyzer.get_n_gram_analysis_data_vectors(paragraphs_1))
		print(self.analyzer.get_n_gram_analysis_data_vectors(paragraphs_2))
		print(self.analyzer.get_n_gram_analysis_data_vectors(paragraphs_3))
		print(self.analyzer.get_n_gram_analysis_data_vectors(paragraphs_4))
		print(self.analyzer.get_n_gram_analysis_data_vectors(paragraphs_5))
		print(self.analyzer.get_n_gram_analysis_data_vectors(paragraphs_6))
		print(self.analyzer.get_n_gram_analysis_data_vectors(paragraphs_7))
		print(self.analyzer.get_n_gram_analysis_data_vectors(paragraphs_8))
		
	def test_respa_kw_analysis_of_paorg_pres_decree_paragraphs_2(self):
		pdf_path = self.test_pdfs_dir + '/Presidential_Decree_Issues/for_training_data/Non-RespAs/'
		txt_path = self.test_txts_dir + '/for_training_data/Non-RespAs/'
		get_txt = self.get_txt
		txts = [get_txt(str(file), pdf_path=pdf_path, txt_path=txt_path)
				for file in range(1, 23+1)]

		get_paragraphs = self.parser.get_paragraphs

		paragraphs_of_txts = [get_paragraphs(txt)
							  for txt in txts]
	
		for paragraphs in paragraphs_of_txts:
			print(len(self.analyzer.get_n_gram_analysis_data_vectors(paragraphs)))

	def test_respa_kw_analysis_of_paorg_pres_decree_paragraphs_3(self):
		pdf_path = self.test_pdfs_dir + '/Presidential_Decree_Issues/for_training_data/RespAs/'
		txt_path = self.test_txts_dir + '/for_training_data/RespAs/'
		get_txt = self.get_txt
		txts = [get_txt(str(file), pdf_path=pdf_path, txt_path=txt_path)
				for file in range(1, 50+1)]

		get_paragraphs = self.parser.get_paragraphs

		paragraphs_of_txts = [get_paragraphs(txts[i])
							  for i in range(len(txts))]
		
		print(len(paragraphs_of_txts))
		
		for paragraphs in paragraphs_of_txts:
			print(len(self.analyzer.get_n_gram_analysis_data_vectors(paragraphs)))

	def test_pickle_merged_non_respa_paragraphs_dict(self):
		txt_path = self.test_txts_dir + '/for_training_data/Non-RespAs/paragraphs/'
		pickle_file = 'non_respa_paragraphs_dict.pkl'

		rel_non_respa_paragraphs_path = '..' + txt_path
		rel_pickle_file_path = '..' + '/data/PAOrg_issue_RespA_classifier_resources/paragraph_respa_classifier_data/' + pickle_file

		non_respa_paragraphs = []
		for i in range(1, 669+1):
			with open(rel_non_respa_paragraphs_path + str(i) + '.txt') as txt:
				non_respa_paragraphs.append(txt.read())

		get_clean_words = self.parser.get_clean_words

		non_respa_paragraph_words_list = [get_clean_words(prgrph)[:20] for prgrph in non_respa_paragraphs]

		get_word_n_grams = self.helper.get_word_n_grams
		
		non_respa_paragraph_bigrams_list = [get_word_n_grams(prgrh_words, 2) 
											for prgrh_words in non_respa_paragraph_words_list]

		
		non_respa_paragraph_bigram_dicts = []
		for bigrams in non_respa_paragraph_bigrams_list:
			non_respa_paragraph_bigram_dicts.append([((bigram[0], bigram[1]), 1) for bigram in bigrams])
		
		# Bigrams before merge
		print(sum([len(prgrph_bigrams) for prgrph_bigrams in non_respa_paragraph_bigram_dicts]))

		# Merge possible keys
		merged_non_respa_prgrh_bigrams_dict = defaultdict(int)
		for prgrh_bigrams in non_respa_paragraph_bigram_dicts:
			for bigram in prgrh_bigrams:
				merged_non_respa_prgrh_bigrams_dict[bigram[0]] += bigram[1]

		# Bigrams after merge
		print(len(merged_non_respa_prgrh_bigrams_dict))		

		# Dump to pickle file
		with open(rel_pickle_file_path, 'wb') as handle:
			dump(dict(merged_non_respa_prgrh_bigrams_dict),
				handle, protocol=HIGHEST_PROTOCOL)		

	def test_pickle_merged_respa_paragraphs_dict(self):
		txt_path = self.test_txts_dir + '/for_training_data/RespAs/paragraphs/'
		pickle_file = 'respa_paragraphs_dict.pkl'

		rel_respa_paragraphs_path = '..' + txt_path
		rel_pickle_file_path = '..' + '/data/PAOrg_issue_RespA_classifier_resources/paragraph_respa_classifier_data/' + pickle_file
		
		respa_paragraphs = []
		for i in range(1, 569+1):
			with open(rel_respa_paragraphs_path + str(i) + '.txt') as txt:
				respa_paragraphs.append(txt.read())

		get_clean_words = self.parser.get_clean_words

		respa_paragraph_words_list = [get_clean_words(prgrph)[:20] for prgrph in respa_paragraphs]

		get_word_n_grams = self.helper.get_word_n_grams
		
		respa_paragraph_bigrams_list = [get_word_n_grams(prgrh_words, 2) 
											for prgrh_words in respa_paragraph_words_list]

		
		respa_paragraph_bigram_dicts = []
		for bigrams in respa_paragraph_bigrams_list:
			respa_paragraph_bigram_dicts.append([((bigram[0], bigram[1]), 1) for bigram in bigrams])
		
		# Bigrams before merge
		print(sum([len(prgrph_bigrams) for prgrph_bigrams in respa_paragraph_bigram_dicts]))

		# Merge possible keys
		merged_respa_prgrh_bigrams_dict = defaultdict(int)
		for prgrh_bigrams in respa_paragraph_bigram_dicts:
			for bigram in prgrh_bigrams:
				merged_respa_prgrh_bigrams_dict[bigram[0]] += bigram[1]

		# Bigrams after merge
		print(len(merged_respa_prgrh_bigrams_dict))

		# Dump to pickle file
		with open(rel_pickle_file_path, 'wb') as handle:
			dump(dict(merged_respa_prgrh_bigrams_dict),
				handle, protocol=HIGHEST_PROTOCOL)
	

	def test_cross_validate_respa_clfs(self):
		print("Issue clf data:")
		self.analyzer.cross_validate(self.issue_clf_data_csv, test_size=0.4)
		print("Article clf data:")
		self.analyzer.cross_validate(self.artcl_clf_data_csv, test_size=0.4)
		
	def test_KFold_cross_validate_respa_clfs(self):
		print("Issue clf data:")
		self.analyzer.KFold_cross_validate(self.issue_clf_data_csv)
		print("Article clf data:")
		self.analyzer.KFold_cross_validate(self.artcl_clf_data_csv)

	# def test_respa_classifiers(self):
		
	# 	###################
	# 	# Non-RespA texts #
	# 	###################
	# 	pdf_path = self.test_pdfs_dir + '/Presidential_Decree_Issues/for_test_data/Non-RespAs/'
	# 	txt_path = self.test_txts_dir + '/for_test_data/Non-RespAs/'
		
	# 	txt_1 = self.get_txt('1', pdf_path=pdf_path, txt_path=txt_path)
	# 	txt_2 = self.get_txt('2', pdf_path=pdf_path, txt_path=txt_path)
	# 	txt_3 = self.get_txt('3', pdf_path=pdf_path, txt_path=txt_path)
		
	# 	get_articles = self.parser.get_articles_from_txt
	# 	articles_1 = get_articles(txt_1)
	# 	articles_2 = get_articles(txt_2)
	# 	articles_3 = get_articles(txt_3)
	# 	print(articles_1)
	# 	# Convert any dict to list
	# 	if isinstance(articles_1, dict): articles_1 = list(articles_1.values())
	# 	if isinstance(articles_2, dict): articles_2 = list(articles_2.values())
	# 	if isinstance(articles_3, dict): articles_3 = list(articles_3.values())

	# 	txt_1_data_vec = self.analyzer.get_n_gram_analysis_data_sums_vector(articles_1)
	# 	txt_2_data_vec = self.analyzer.get_n_gram_analysis_data_sums_vector(articles_2)
	# 	txt_3_data_vec = self.analyzer.get_n_gram_analysis_data_sums_vector(articles_3)
		
	# 	issue_clf = self.analyzer.train_clf(self.issue_clf_data_csv)

	# 	# Issue respa classifier results
	# 	print(self.analyzer.has_respa(issue_clf, txt_1_data_vec))
	# 	print(self.analyzer.has_respa(issue_clf, txt_2_data_vec))
	# 	print(self.analyzer.has_respa(issue_clf, txt_3_data_vec))

	# 	articles_1_data_vecs = self.analyzer.get_n_gram_analysis_data_vectors(articles_1)
	# 	articles_2_data_vecs = self.analyzer.get_n_gram_analysis_data_vectors(articles_2)
	# 	articles_3_data_vecs = self.analyzer.get_n_gram_analysis_data_vectors(articles_3)
		
	# 	article_clf = self.analyzer.train_clf(self.artcl_clf_data_csv)

	# 	# Artcl respa classifier results
	# 	for artcl_data_vec in articles_1_data_vecs:
	# 		print(self.analyzer.has_respa(article_clf, artcl_data_vec))

	# 	for artcl_data_vec in articles_2_data_vecs:
	# 		print(self.analyzer.has_respa(article_clf, artcl_data_vec))

	# 	for artcl_data_vec in articles_3_data_vecs:
	# 		print(self.analyzer.has_respa(article_clf, artcl_data_vec))

	# 	###################
	# 	# 	RespA texts   #
	# 	###################
	# 	pdf_path = self.test_pdfs_dir + '/Presidential_Decree_Issues/for_test_data/RespAs/'
	# 	txt_path = self.test_txts_dir + '/for_test_data/RespAs/'
		
	# 	txt_1 = self.get_txt('1', pdf_path=pdf_path, txt_path=txt_path)
	# 	txt_2 = self.get_txt('2', pdf_path=pdf_path, txt_path=txt_path)
	# 	txt_3 = self.get_txt('3', pdf_path=pdf_path, txt_path=txt_path)
	# 	txt_4 = self.get_txt('4', pdf_path=pdf_path, txt_path=txt_path)
	# 	txt_5 = self.get_txt('5', pdf_path=pdf_path, txt_path=txt_path)
	# 	txt_6 = self.get_txt('6', pdf_path=pdf_path, txt_path=txt_path)
		
	# 	get_articles = self.parser.get_articles_from_txt
	# 	articles_1 = get_articles(txt_1)
	# 	articles_2 = get_articles(txt_2)
	# 	articles_3 = get_articles(txt_3)
	# 	articles_4 = get_articles(txt_4)
	# 	articles_5 = get_articles(txt_5)
	# 	articles_6 = get_articles(txt_6)

		
	# 	# Convert any dict to list
	# 	if isinstance(articles_1, dict): articles_1 = list(articles_1.values())
	# 	if isinstance(articles_2, dict): articles_2 = list(articles_2.values())
	# 	if isinstance(articles_3, dict): articles_3 = list(articles_3.values())
	# 	if isinstance(articles_4, dict): articles_4 = list(articles_4.values())
	# 	if isinstance(articles_5, dict): articles_5 = list(articles_5.values())
	# 	if isinstance(articles_6, dict): articles_6 = list(articles_6.values())

	# 	txt_1_data_vec = self.analyzer.get_n_gram_analysis_data_sums_vector(articles_1)
	# 	txt_2_data_vec = self.analyzer.get_n_gram_analysis_data_sums_vector(articles_2)
	# 	txt_3_data_vec = self.analyzer.get_n_gram_analysis_data_sums_vector(articles_3)
	# 	txt_4_data_vec = self.analyzer.get_n_gram_analysis_data_sums_vector(articles_4)
	# 	txt_5_data_vec = self.analyzer.get_n_gram_analysis_data_sums_vector(articles_5)
	# 	txt_6_data_vec = self.analyzer.get_n_gram_analysis_data_sums_vector(articles_6)
		
	# 	issue_clf = self.analyzer.train_clf(self.issue_clf_data_csv)

	# 	# Issue respa classifier results
	# 	print(self.analyzer.has_respa(issue_clf, txt_1_data_vec))
	# 	print(self.analyzer.has_respa(issue_clf, txt_2_data_vec))
	# 	print(self.analyzer.has_respa(issue_clf, txt_3_data_vec))
	# 	print(self.analyzer.has_respa(issue_clf, txt_4_data_vec))
	# 	print(self.analyzer.has_respa(issue_clf, txt_5_data_vec))
	# 	print(self.analyzer.has_respa(issue_clf, txt_6_data_vec))


if __name__ == '__main__':
	unittest.main()