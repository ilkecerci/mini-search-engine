import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# Projendeki diğer dosyaları içeri aktarıyoruz
from indexing import build_indexes, get_incidence_matrix
from ranking import get_ranked_results

def draw_heatmap():
    """It plots the Incidence Matrix as a Heat Map."""
    print("Indexes are being scanned, a heat map is being prepared...")
    inv_idx, pos_idx, doc_list = build_indexes("documents")
    matrix, terms = get_incidence_matrix(inv_idx, doc_list)
    sample_terms = terms[:20]
    plot_data = {t: matrix[t][:10] for t in sample_terms}
    
    df = pd.DataFrame(plot_data).transpose()
    df.columns = doc_list[:10]

    plt.figure(figsize=(12, 8))
    sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=False)
    plt.title("Incidence Matrix Heatmap (First 20 Word's)")
    plt.tight_layout()
    print("The Heat Map window is opening...")
    plt.show()

def draw_ranking_chart(query):
    """It plots Ranked Retrieval scores as a bar graph."""
    print(f"'{query}' scores are being calculated for the query...")
    results = get_ranked_results(query)
    
    if results:
        top_docs = [res[0] for res in results[:5]]
        top_scores = [res[1] for res in results[:5]]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_scores, y=top_docs, palette="rocket")
        plt.title(f"TF-IDF similarity scores: '{query}'")
        plt.xlabel("Relevance Score (The closer it is to 1.0, the more relevant it is.)")
        plt.ylabel("Documents")
        plt.tight_layout()
        print("The Ranking Chart window is opening...")
        plt.show()
    else:
        print("No results were found for this word.")

if __name__ == "__main__":
    draw_heatmap()
    user_query = input("\nEnter the search term you want to see the graph of (e.g., computer): ")
    draw_ranking_chart(user_query)