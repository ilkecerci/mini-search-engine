import math
import os
import preprocessing 

def load_documents(folder_path="documents"):
    """Klasördeki tüm .txt dosyalarını okur ve içeriklerini bir sözlükte toplar."""
    documents = {}
    if not os.path.exists(folder_path):
        print(f"Hata: '{folder_path}' klasörü bulunamadı! Lütfen dökümanların olduğu klasörün adını kontrol et.")
        return documents
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                documents[filename] = f.read()
    return documents

def build_inverted_index_for_idf(docs_tokens):
    """IDF hesaplayabilmek için kelimelerin hangi dökümanlarda geçtiğini belirler[cite: 43]."""
    inverted_index = {}
    for doc_id, tokens in docs_tokens.items():
        for token in set(tokens): 
            if token not in inverted_index:
                inverted_index[token] = []
            inverted_index[token].append(doc_id)
    return inverted_index

def calculate_tfidf_vector(tokens, inverted_index, total_docs):
    """Bir metin (döküman veya sorgu) için TF-IDF vektörü oluşturur[cite: 51, 54]."""
    vector = {}
    counts = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    for term, tf in counts.items():
        if term in inverted_index:
            df = len(inverted_index[term])
            idf = math.log10(total_docs / df)
            vector[term] = tf * idf
        else:
            vector[term] = 0.0
    return vector

def calculate_cosine_similarity(vec1, vec2):
    """İki vektör arasındaki Kosinüs Benzerliğini hesaplar[cite: 55, 56]."""
    common_terms = set(vec1.keys()) & set(vec2.keys())
    dot_product = sum(vec1[t] * vec2[t] for t in common_terms)
    magnitude1 = math.sqrt(sum(val**2 for val in vec1.values()))
    magnitude2 = math.sqrt(sum(val**2 for val in vec2.values()))
    
    if magnitude1 * magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

def get_ranked_results(query):
    """Sorguyu alır ve dökümanları en alakalıdan başlayarak sıralar[cite: 51]."""
    raw_docs = load_documents()
    if not raw_docs: return []
    
    processed_docs = {doc_id: preprocessing.preprocess_text(content) for doc_id, content in raw_docs.items()}
    total_docs = len(raw_docs)
    inverted_index = build_inverted_index_for_idf(processed_docs)
    query_tokens = preprocessing.preprocess_text(query)
    query_vector = calculate_tfidf_vector(query_tokens, inverted_index, total_docs)
    ranked_list = []
    for doc_id, doc_tokens in processed_docs.items():
        doc_vector = calculate_tfidf_vector(doc_tokens, inverted_index, total_docs)
        score = calculate_cosine_similarity(query_vector, doc_vector)
        ranked_list.append((doc_id, score))
    ranked_list.sort(key=lambda x: x[1], reverse=True)
    return ranked_list
if __name__ == "__main__":
    user_query = input("Arama yapmak istediğiniz terimi girin: ")
    print(f"\n'{user_query}' sorgusu için dökümanlar sıralanıyor...\n")
    
    results = get_ranked_results(user_query)
    
    if not results:
        print("Sonuç bulunamadı.")
    else:
        print(f"{'Sıra':<5} {'Döküman ID':<25} {'Benzerlik Skoru'}")
        print("-" * 50)
        for rank, (doc_id, score) in enumerate(results, 1):
            print(f"{rank:<5} {doc_id:<25} {score:.4f}")