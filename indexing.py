import os
from preprocessing import preprocess_text

def build_indexes(docs_folder):
    inverted_index = {}
    positional_index = {}
    doc_list = sorted([d for d in os.listdir(docs_folder) if d.endswith('.txt')])
    
    for doc_id in doc_list:
        file_path = os.path.join(docs_folder, doc_id)
        with open(file_path, 'r', encoding='utf-8') as f:
            tokens = preprocess_text(f.read())
            
            for position, term in enumerate(tokens):
                # Inverted Index logic
                if term not in inverted_index:
                    inverted_index[term] = set()
                inverted_index[term].add(doc_id)
                
                # Positional Index logic
                if term not in positional_index:
                    positional_index[term] = {}
                if doc_id not in positional_index[term]:
                    positional_index[term][doc_id] = []
                positional_index[term][doc_id].append(position)
    
    # Convert sets to sorted lists for stability
    for term in inverted_index:
        inverted_index[term] = sorted(list(inverted_index[term]))
        
    return inverted_index, positional_index, doc_list

def get_incidence_matrix(inverted_index, doc_list):
    matrix = {}
    all_terms = sorted(inverted_index.keys())
    
    for term in all_terms:
        # Create a bit-vector (1 if term in doc, else 0)
        matrix[term] = [1 if doc in inverted_index[term] else 0 for doc in doc_list]
        
    return matrix, all_terms
