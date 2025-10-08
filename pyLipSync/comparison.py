from enum import Enum
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CompareMethod(Enum):
    L1_NORM = "L1Norm"
    L2_NORM = "L2Norm"
    COSINE_SIMILARITY = "CosineSimilarity"

def calc_l1_norm_score(normalized_mfcc: np.ndarray, normalized_template: np.ndarray):
    distance = np.mean(np.abs(normalized_mfcc - normalized_template))
    return 10.0 ** (-distance)

def calc_l2_norm_score(normalized_mfcc: np.ndarray, normalized_template: np.ndarray):
    diff = normalized_mfcc - normalized_template
    distance = np.sqrt(np.mean(diff ** 2))
    return 10.0 ** (-distance)

def calc_cosine_similarity_score(normalized_mfcc: np.ndarray, normalized_template: np.ndarray):
    """
    Calculate Cosine Similarity score using scikit-learn.
    Based on C# LipSyncJob.CalcCosineSimilarityScore() (lines 148-170)
    """
    # sklearn expects 2D arrays, so reshape
    similarity = cosine_similarity(
        normalized_mfcc.reshape(1, -1), 
        normalized_template.reshape(1, -1)
    )[0][0]
    
    similarity = max(similarity, 0.0)
    return similarity ** 100.0

def calculate_similarity_score(normalized_mfcc: np.ndarray, normalized_template: np.ndarray, compare_method: CompareMethod):
    """Route to appropriate scoring method."""
    if compare_method == CompareMethod.L1_NORM:
        return calc_l1_norm_score(normalized_mfcc, normalized_template)
    elif compare_method == CompareMethod.L2_NORM:
        return calc_l2_norm_score(normalized_mfcc, normalized_template)
    elif compare_method == CompareMethod.COSINE_SIMILARITY:
        return calc_cosine_similarity_score(normalized_mfcc, normalized_template)
    return 0.0