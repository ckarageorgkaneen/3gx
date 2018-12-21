import trie_search as ts
import pandas as pd


def remove_unigrams_contained_in_bigrams(all_patterns):
	unigrams = []
	subpatterns = [] # contains bigrams and unigrams that are not contained in bigrams
	first_pattern = True
	for pat,start_idx in all_patterns:
		if first_pattern:
			first_pattern_offset = start_idx
			first_pattern = False
		parts = pat.split()
		if (len(parts) == 2):
			first_word = parts[0]
			second_word = parts[1]
			unigrams.append(first_word)
			unigrams.append(second_word)
	for pat,start_idx in all_patterns:
		if pat not in unigrams:
			subpatterns.append(pat)
	return subpatterns

def final_prediction(row):
	if row['OrgPrediction'] == 'Org':
		return 'Org'
	elif row['RespAPrediction'] == 'RespA':
		return 'RespA'
	else:
		return 'Irrelevant'

def create_trie_index(most_frequent_non_respas_stems_ordered,most_frequent_respas_stems_ordered,num_non_respa_docs,ratio):
	selected_df = most_frequent_non_respas_stems_ordered[most_frequent_non_respas_stems_ordered['frequency']/num_non_respa_docs>ratio]

	sublist = list(selected_df['stems'])

	subtraction = [x for x in list(most_frequent_respas_stems_ordered['stems']) if x not in sublist] # respas - sublist

	subtraction_df = pd.DataFrame({'stems':subtraction})

	new_df = pd.merge(subtraction_df,most_frequent_respas_stems_ordered,on='stems') # merge subtracted and sorted on stems column

	maxvalue = new_df['frequency'].max() # get max frequency
	meanvalue = new_df['frequency'].mean() # get mean frequency

	most_freq = new_df[new_df.frequency > int(meanvalue)] # get items for which frequency is greater than the mean frequency
	freqstems = most_freq['stems'].values.tolist()
	freqstemscopy=[]

	for s in freqstems:
    	if not tp.hasNumbers(s) and len(s) > 3: 
        	freqstemscopy.append(s)        
	trie = ts.TrieSearch(freqstemscopy) # create trie index from sublist terms that do not contain numbers
	return trie