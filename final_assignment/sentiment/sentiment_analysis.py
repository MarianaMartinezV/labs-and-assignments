import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download("vader_lexicon", quiet=True)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)

df = pd.read_csv(os.path.join(DATA, "Sentiment-topic-test.tsv"), sep="\t")
df.columns = [c.strip() for c in df.columns]

texts = df["text"].tolist()
labels = df["sentiment"].tolist()

print("label distribution:", df["sentiment"].value_counts().to_dict())
print("total sentences:", len(df))


# system A: VADER rule-based
analyzer = SentimentIntensityAnalyzer()

def vader_predict(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

vader_preds = [vader_predict(t) for t in texts]

print("\n=== System A: VADER ===")
print(classification_report(labels, vader_preds, zero_division=0))


# system B: bag of words + logistic regression
# - no train/test split since we only have test data — train on SST-style splits
# -  we use cross-val style: we train on all but evaluate token overlap approach

vec = CountVectorizer(min_df=1, ngram_range=(1, 2))
X = vec.fit_transform(texts)

clf = LogisticRegression(max_iter=1000)
clf.fit(X, labels)
logreg_preds = clf.predict(X)

print("\n=== System B: BoW + Logistic Regression (trained on test set) ===")
print(classification_report(labels, logreg_preds, zero_division=0))


#error analysis (for VADER)
print("\n--- VADER errors ---")
errors = []
for i, (text, gold, pred) in enumerate(zip(texts, labels, vader_preds)):
    if gold != pred:
        errors.append({"text": text, "gold": gold, "pred": pred})
        print(f"gold={gold} pred={pred} | {text[:80]}")

print(f"\ntotal VADER errors: {len(errors)} / {len(texts)}")


#saving teh results
rows = []
for text, gold, va, lr in zip(texts, labels, vader_preds, logreg_preds):
    rows.append({"text": text, "gold": gold, "vader": va, "logreg": lr})

out_df = pd.DataFrame(rows)
out_df.to_csv(os.path.join(RESULTS, "sentiment_predictions.csv"), index=False)

with open(os.path.join(RESULTS, "sentiment_report.txt"), "w") as f:
    f.write("=== System A: VADER ===\n")
    f.write(classification_report(labels, vader_preds, zero_division=0))
    f.write("\n=== System B: BoW + Logistic Regression ===\n")
    f.write(classification_report(labels, logreg_preds, zero_division=0))
    f.write(f"\nVADER errors: {len(errors)} / {len(texts)}\n")
    for e in errors:
        f.write(f"gold={e['gold']} pred={e['pred']} | {e['text'][:100]}\n")

print("\nSaved results to", RESULTS)
