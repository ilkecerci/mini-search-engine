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

from indexing import build_indexes
from search_engine import boolean_search_and, phrase_search
from preprocessing import preprocess_text

def main():
    # Step 1: Initialize and build indexes at startup
    print("Indexing documents... Please wait.")
    inv_idx, pos_idx, doc_list = build_indexes("documents")
    print(f"Total documents indexed: {len(doc_list)}")
    print("-" * 30)

    while True:
        # Step 2: Get user input
        query = input("\nEnter your search (or 'exit' to quit): ").strip()
        
        if query.lower() == 'exit':
            break

        results = []

        # Step 3: Handle Phrase Search (Check for double quotes)
        if query.startswith('"') and query.endswith('"'):
            # Remove quotes and process the phrase
            clean_query = query.replace('"', '')
            print(f"Searching for phrase: {clean_query}")
            results = phrase_search(clean_query, pos_idx)

        # Step 4: Handle Boolean Search (Check for ' AND ' keyword)
        elif " AND " in query:
            parts = query.split(" AND ")
            if len(parts) == 2:
                term1, term2 = parts[0].strip(), parts[1].strip()
                print(f"Searching for: {term1} AND {term2}")
                
                # Apply preprocessing to search terms for consistency
                t1_cleaned = preprocess_text(term1)[0] if preprocess_text(term1) else term1
                t2_cleaned = preprocess_text(term2)[0] if preprocess_text(term2) else term2
                
                results = boolean_search_and(t1_cleaned, t2_cleaned, inv_idx)

        # Step 5: Handle Simple Single-Term Search
        else:
            cleaned_query = preprocess_text(query)
            if cleaned_query:
                term = cleaned_query[0]
                results = inv_idx.get(term, [])

        # Step 6: Display Search Results
        if results:
            print(f"Found in {len(results)} documents: {results}")
        else:
            print("No results found.")

if __name__ == "__main__":
    main()