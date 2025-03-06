import pandas as pd
from transformers import pipeline
from tqdm import tqdm  # Progress bar

# Load sentiment analysis model 
sentiment_analyzer = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    device=0  # Use GPU if available
)

# Load dataset
csv_file = "/content/sentimentanalysis cse hack.csv"  
df = pd.read_csv(csv_file)

# Ensure 'text' column exists 
text_column = next((col for col in df.columns if col.lower() == "text"), None)
if text_column is None:
    raise ValueError("CSV file must contain a column named 'text' (case-insensitive).")

# Limit dataset to 2000 rows if larger
df = df.head(2000)

# Handle NaN values
df[text_column] = df[text_column].fillna("")

# Convert to list for batch processing
texts = df[text_column].astype(str).tolist()

# Process sentiment analysis in batches
batch_size = 32  # Adjust based on GPU memory
sentiments = []
confidence_scores = []

for i in tqdm(range(0, len(texts), batch_size), desc="Processing Sentiments"):
    batch_texts = texts[i:i + batch_size]
    results = sentiment_analyzer(batch_texts, truncation=True, max_length=512)  # 🔥 Fixes tensor mismatch error

    for result in results:
        label = result['label']
        score = result['score']

        # Convert labels to readable sentiment names
        sentiment_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
        sentiment = sentiment_map.get(label, "Neutral")  # Default to Neutral

        sentiments.append(sentiment)
        confidence_scores.append(score)

# Add results to DataFrame
df["Sentiment"] = sentiments
df["Confidence Score"] = confidence_scores

# Save back to CSV
df.to_csv(csv_file, index=False)

print(f"✅ Sentiment analysis completed! Processed {len(df)} feedback entries. Results saved to '{csv_file}'.")
