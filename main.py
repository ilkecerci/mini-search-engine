from preprocessing import preprocess_text

# testing to read doc01.txt 
try:
    with open("documents/doc01.txt", "r", encoding="utf-8") as file:
        raw_text = file.read()
        
    # since it's too long, let's test the first 500 char only
    sample_content = raw_text[:500]
    clean_content = preprocess_text(sample_content)
    
    print("--- Original (First 500 chars) ---")
    print(sample_content)
    print("\n--- Processed Tokens ---")
    print(clean_content)

except FileNotFoundError:
    print("Entered doc is not found in documents!")

from indexing import build_indexes, get_incidence_matrix

inv_idx, pos_idx, docs = build_indexes("documents")
matrix, terms = get_incidence_matrix(inv_idx, docs)

# sample: for the first word
sample_term = terms[0]
print(f"Term: {sample_term}")
print(f"Inverted Index: {inv_idx[sample_term]}")
print(f"Incidence Matrix Row: {matrix[sample_term]}")