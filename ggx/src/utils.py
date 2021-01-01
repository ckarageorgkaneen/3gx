def remove_unigrams_contained_in_bigrams(all_patterns):
    unigrams = []
    # Contains bigrams and unigrams that are not contained in bigrams
    subpatterns = []
    first_pattern = True
    for pat, start_idx in all_patterns:
        if first_pattern:
            first_pattern = False
        parts = pat.split()
        if (len(parts) == 2):
            first_word = parts[0]
            second_word = parts[1]
            unigrams.append(first_word)
            unigrams.append(second_word)
    for pat, start_idx in all_patterns:
        if pat not in unigrams:
            subpatterns.append(pat)
    return subpatterns
