from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def compute_diversity(chains):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(chains)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    avg_similarity = (similarity_matrix.sum() - len(chains)) / (len(chains)**2 - len(chains))
    diversity_score = 1 - avg_similarity  # Higher = more diverse
    return round(diversity_score, 3)
