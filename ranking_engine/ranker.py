# LOCUS: ranker.py
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class LocusRanker:
    def predict(self, query_vector, candidate_vectors):
        """
        Compares the query_vector against all candidate_vectors.
        Returns a sorted list of matches (highest score first).
        """
        # 1. Convert lists to Numpy Arrays (for speed)
        query = np.array([query_vector])
        candidates = np.array(candidate_vectors)

        # 2. Calculate Cosine Similarity
        # This creates a list of scores, e.g., [0.85, 0.10, 0.99]
        scores = cosine_similarity(query, candidates)[0]
        
        # 3. Sort the results
        # argsort returns the indices of the sorted scores (low to high)
        # [::-1] flips it to be High to Low
        sorted_indices = np.argsort(scores)[::-1]
        
        results = []
        for index in sorted_indices:
            results.append({
                "index": int(index),
                "score": float(scores[index])
            })
            
        return results

# Simple test block
if __name__ == "__main__":
    ranker = LocusRanker()
    # A fake query vector
    q = [1, 0, 0] 
    # Fake candidates (Perfect match, Opposite, partial match)
    c = [
        [1, 0, 0], 
        [0, 1, 0], 
        [0.5, 0.5, 0]
    ]
    print(ranker.predict(q, c))