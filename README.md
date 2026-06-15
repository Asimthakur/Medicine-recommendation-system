# Medicine-recommendation-system
Author-Asim Thakur
"""
Medicine Recommendation System
--------------------------------
Given a set of symptoms (user input), the system recommends the most relevant medicines
based on a pre-defined dataset of medicines and their indications.
Uses TF-IDF and cosine similarity for content-based filtering.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# -------------------------------
# 1. Sample Medicine Dataset
# -------------------------------
# Each medicine has:
#   - name: brand/generic name
#   - indications: comma-separated list of symptoms/diseases it treats
#   - category: (optional) for extra info
#   - prescription_required: boolean

