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

data = {
    "name": [
        "Paracetamol",
        "Ibuprofen",
        "Cetirizine",
        "Amoxicillin",
        "Omeprazole",
        "Loperamide",
        "Dextromethorphan",
        "Loratadine",
        "Metformin",
        "Aspirin"
    ],
    "indications": [
        "fever, headache, mild pain, toothache",
        "inflammation, muscle pain, arthritis, fever, headache",
        "allergy, runny nose, sneezing, itchy eyes",
        "bacterial infection, throat infection, pneumonia, UTI",
        "acidity, heartburn, GERD, stomach ulcer",
        "diarrhea, loose motions",
        "dry cough, cough",
        "hay fever, allergy, hives",
        "type 2 diabetes, high blood sugar",
        "pain, fever, inflammation, cardiovascular protection"
    ],
    "category": [
        "Analgesic/Antipyretic",
        "NSAID",
        "Antihistamine",
        "Antibiotic",
        "Proton pump inhibitor",
        "Antidiarrheal",
        "Antitussive",
        "Antihistamine",
        "Antidiabetic",
        "NSAID/Antiplatelet"
    ],
    "prescription_required": [False, False, False, True, False, False, False, False, True, False]
}

df = pd.DataFrame(data)

# -------------------------------
# 2. Preprocess Indications
# -------------------------------
# Convert to lowercase and remove extra spaces
df["processed_indications"] = df["indications"].str.lower().apply(lambda x: re.sub(r'[^\w\s]', '', x))

# -------------------------------
# 3. Build TF-IDF Matrix and Similarity
# -------------------------------
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df["processed_indications"])

def get_recommendations(symptoms, top_n=5):
    """
    symptoms: string, e.g. "fever headache"
    top_n: number of medicine recommendations to return
    Returns: DataFrame with recommended medicines, sorted by similarity score
    """
    # Preprocess symptoms
    symptoms_clean = re.sub(r'[^\w\s]', '', symptoms.lower())
    
    # Transform symptoms into the same TF-IDF space
    symptoms_vec = vectorizer.transform([symptoms_clean])
    
    # Compute cosine similarity between symptoms and all medicines
    similarities = cosine_similarity(symptoms_vec, tfidf_matrix).flatten()
    
    # Get indices sorted by similarity (descending)
    sorted_indices = similarities.argsort()[::-1]
    
    # Prepare recommendations
    recommendations = []
    for idx in sorted_indices[:top_n]:
        recommendations.append({
            "medicine": df.iloc[idx]["name"],
            "indications": df.iloc[idx]["indications"],
            "category": df.iloc[idx]["category"],
            "prescription_required": df.iloc[idx]["prescription_required"],
            "similarity_score": round(similarities[idx], 3)
        })
    
    return pd.DataFrame(recommendations)

# -------------------------------
# 4. Command-Line Interface
# -------------------------------
def main():
    print("\n" + "="*60)
    print(" MEDICINE RECOMMENDATION SYSTEM")
    print("="*60)
    print("\nEnter your symptoms separated by commas or spaces (e.g., fever headache cough).")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input = input("Your symptoms: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye! Stay healthy.")
            break
        if not user_input:
            print("Please enter some symptoms.\n")
            continue
        
        # Get recommendations
        recommendations = get_recommendations(user_input, top_n=5)
        
        if recommendations.empty:
            print("No matching medicines found. Try different symptoms.\n")
            continue
        
        print("\n" + "-"*60)
        print(" RECOMMENDED MEDICINES (most relevant first)")
        print("-"*60)
        for i, row in recommendations.iterrows():
            print(f"{i+1}. {row['medicine']} (similarity: {row['similarity_score']})")
            print(f"   Indications: {row['indications']}")
            print(f"   Category: {row['category']}")
            print(f"   Prescription required: {'Yes' if row['prescription_required'] else 'No'}")
            print()
        
        print("⚠️  IMPORTANT: Always consult a doctor or pharmacist before taking any medicine.\n")

if __name__ == "__main__":
    main()
