def boolean_search_and(term1, term2, inverted_index):
    # intersection of documents containing the two words.
    set1 = set(inverted_index.get(term1, []))
    set2 = set(inverted_index.get(term2, []))
    return sorted(list(set1.intersection(set2)))

def phrase_search(phrase, positional_index):
    # example: "artificial intelligence" -> tokens: ["artifici", "intellig"]
    tokens = preprocess_text(phrase)
    if len(tokens) < 2: return "Please enter at least 2 words"
    
    results = []
    first_term, second_term = tokens[0], tokens[1]
    
    if first_term in positional_index and second_term in positional_index:
        common_docs = set(positional_index[first_term].keys()) & set(positional_index[second_term].keys())
        
        for doc in common_docs:
            positions1 = positional_index[first_term][doc]
            positions2 = positional_index[second_term][doc]
            # if term2 is coming right after the term1 (pos2 = pos1 + 1)
            if any(pos + 1 in positions2 for pos in positions1):
                results.append(doc)
    return results
