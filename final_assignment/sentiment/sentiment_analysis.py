import os
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
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
test_counts = df["sentiment"].value_counts().to_dict()

print("label distribution:", test_counts)
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


# system B: BoW + Logistic Regression trained on SST-5
SST5_DATASET = "SetFit/sst5"
SST5_SOURCE = "https://huggingface.co/datasets/SetFit/sst5"
SST3_MAPPING_SOURCE = (
    "https://docs.allennlp.org/models/main/models/classification/"
    "dataset_readers/stanford_sentiment_tree_bank/"
)
SST_LABEL_MAP = {
    0: "negative",  # very negative
    1: "negative",
    2: "neutral",
    3: "positive",
    4: "positive",  # very positive
}


def collapse_sst5_labels(labels):
    return [SST_LABEL_MAP[label] for label in labels]


def load_sst5_data():
    dataset = load_dataset(SST5_DATASET)
    train_texts = list(dataset["train"]["text"])
    train_labels = collapse_sst5_labels(dataset["train"]["label"])
    validation_texts = list(dataset["validation"]["text"])
    validation_labels = collapse_sst5_labels(dataset["validation"]["label"])
    test_labels = collapse_sst5_labels(dataset["test"]["label"])
    stats = {
        "train_size": len(train_texts),
        "validation_size": len(validation_texts),
        "test_size": len(dataset["test"]),
        "train_counts": pd.Series(train_labels).value_counts().to_dict(),
        "validation_counts": pd.Series(validation_labels).value_counts().to_dict(),
        "test_counts": pd.Series(test_labels).value_counts().to_dict(),
    }
    return train_texts, train_labels, validation_texts, validation_labels, stats


def train_bow_logreg(train_texts, train_labels):
    model = Pipeline([
        ("vectorizer", CountVectorizer(min_df=1, ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    model.fit(train_texts, train_labels)
    return model


sst_train_texts, sst_train_labels, sst_val_texts, sst_val_labels, sst_stats = load_sst5_data()
bow_logreg_model = train_bow_logreg(sst_train_texts, sst_train_labels)
bow_logreg_val_preds = bow_logreg_model.predict(sst_val_texts)
bow_logreg_preds = bow_logreg_model.predict(texts)
bow_logreg_scores = bow_logreg_model.predict_proba(texts).max(axis=1)

print("\n=== System B: BoW + Logistic Regression (trained on SST-5) ===")
print("SST-5 collapsed train distribution:", sst_stats["train_counts"])
print("SST-5 collapsed validation distribution:", sst_stats["validation_counts"])
print("\n--- SST-5 validation performance ---")
print(classification_report(sst_val_labels, bow_logreg_val_preds, zero_division=0))
print("\n--- Final test performance ---")
print(classification_report(labels, bow_logreg_preds, zero_division=0))


# system C: Pretrained Transformer
TRANSFORMER_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def transformer_predict(texts):
    from transformers import pipeline

    classifier = pipeline(
        "sentiment-analysis",
        model=TRANSFORMER_MODEL,
        tokenizer=TRANSFORMER_MODEL,
        device=-1,
    )
    raw_predictions = classifier(texts, truncation=True, max_length=512)

    label_map = {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive",
        "NEGATIVE": "negative",
        "NEUTRAL": "neutral",
        "POSITIVE": "positive",
        "negative": "negative",
        "neutral": "neutral",
        "positive": "positive",
    }
    preds = [label_map[pred["label"]] for pred in raw_predictions]
    scores = [pred["score"] for pred in raw_predictions]
    raw_labels = [pred["label"] for pred in raw_predictions]
    return preds, scores, raw_labels


transformer_preds, transformer_scores, transformer_raw_labels = transformer_predict(texts)

print(f"\n=== System C: Pretrained transformer ({TRANSFORMER_MODEL}) ===")
print(classification_report(labels, transformer_preds, zero_division=0))


def collect_errors(system_name, predictions):
    rows = []
    for text, gold, pred in zip(texts, labels, predictions):
        if gold != pred:
            rows.append({"system": system_name, "text": text, "gold": gold, "pred": pred})
    return rows


# error analysis
print("\n--- VADER errors ---")
vader_errors = collect_errors("VADER", vader_preds)
bow_logreg_errors = collect_errors("BoW + Logistic Regression", bow_logreg_preds)
transformer_errors = collect_errors("Pretrained transformer", transformer_preds)
for error in vader_errors:
    print(f"gold={error['gold']} pred={error['pred']} | {error['text'][:80]}")

print(f"\ntotal VADER errors: {len(vader_errors)} / {len(texts)}")
print(f"total BoW + Logistic Regression errors: {len(bow_logreg_errors)} / {len(texts)}")
print(f"total transformer errors: {len(transformer_errors)} / {len(texts)}")


# saving the results
rows = []
for text, gold, va, bow, bow_score, tr, tr_score, tr_raw in zip(
    texts,
    labels,
    vader_preds,
    bow_logreg_preds,
    bow_logreg_scores,
    transformer_preds,
    transformer_scores,
    transformer_raw_labels,
):
    rows.append({
        "text": text,
        "gold": gold,
        "vader": va,
        "bow_logreg_sst5": bow,
        "bow_logreg_sst5_score": round(bow_score, 4),
        "transformer": tr,
        "transformer_raw_label": tr_raw,
        "transformer_score": round(tr_score, 4),
        "vader_transformer_agree": va == tr,
        "all_systems_agree": va == bow == tr,
    })

out_df = pd.DataFrame(rows)
out_df.to_csv(os.path.join(RESULTS, "sentiment_predictions.csv"), index=False)

vader_transformer_disagreements = out_df[out_df["vader"] != out_df["transformer"]]
all_system_disagreements = out_df[~out_df["all_systems_agree"]]
lowest_bow_confidence = out_df.sort_values("bow_logreg_sst5_score").head(3)
lowest_confidence = out_df.sort_values("transformer_score").head(3)

with open(os.path.join(RESULTS, "sentiment_report.txt"), "w") as f:
    f.write("SENTIMENT ANALYSIS\n")
    f.write("==================\n")
    f.write("Task: sentence-level sentiment classification on Sentiment-topic-test.tsv.\n")
    f.write("Labels: negative, neutral, positive.\n\n")
    f.write("Final test data statistics:\n")
    f.write(f"- Sentences: {len(df)}\n")
    f.write(f"- Label distribution: {test_counts}\n")
    f.write(f"- Topic distribution: {df['topic'].value_counts().to_dict()}\n\n")

    f.write("Training data for System B:\n")
    f.write(
        f"- Dataset: Stanford Sentiment Treebank 5-class version via {SST5_DATASET} "
        f"({SST5_SOURCE}).\n"
    )
    f.write(
        "- Label mapping: very negative + negative -> negative; neutral -> neutral; "
        "positive + very positive -> positive.\n"
    )
    f.write(f"- Mapping reference: {SST3_MAPPING_SOURCE}\n")
    f.write(
        f"- Split sizes: train={sst_stats['train_size']}, "
        f"validation={sst_stats['validation_size']}, test={sst_stats['test_size']}.\n"
    )
    f.write(f"- Collapsed train label distribution: {sst_stats['train_counts']}\n")
    f.write(f"- Collapsed validation label distribution: {sst_stats['validation_counts']}\n\n")

    f.write("System A: VADER, the rule-based sentiment system used in Lab 3.\n")
    f.write(
        "System B: BoW + Logistic Regression trained on SST-5. Features are word "
        "unigrams and bigrams from CountVectorizer(min_df=1, ngram_range=(1, 2)); "
        "the classifier uses LogisticRegression(max_iter=2000, class_weight='balanced').\n"
    )
    f.write(
        f"System C: pretrained transformer ({TRANSFORMER_MODEL}), "
        "loaded through the Hugging Face pipeline as in the transformer lab.\n"
    )
    f.write("Neither System B nor System C is trained on the final test set.\n\n")

    f.write("=== System A: VADER ===\n")
    f.write(classification_report(labels, vader_preds, zero_division=0))
    f.write("\n=== System B: BoW + Logistic Regression (SST-5 validation) ===\n")
    f.write(classification_report(sst_val_labels, bow_logreg_val_preds, zero_division=0))
    f.write("\n=== System B: BoW + Logistic Regression (final test) ===\n")
    f.write(classification_report(labels, bow_logreg_preds, zero_division=0))
    f.write(f"\n=== System C: Pretrained transformer ({TRANSFORMER_MODEL}) ===\n")
    f.write(classification_report(labels, transformer_preds, zero_division=0))

    f.write(f"\nVADER errors: {len(vader_errors)} / {len(texts)}\n")
    for e in vader_errors:
        f.write(f"gold={e['gold']} pred={e['pred']} | {e['text'][:100]}\n")

    f.write(f"\nBoW + Logistic Regression errors: {len(bow_logreg_errors)} / {len(texts)}\n")
    for e in bow_logreg_errors:
        f.write(f"gold={e['gold']} pred={e['pred']} | {e['text'][:100]}\n")

    f.write(f"\nTransformer errors: {len(transformer_errors)} / {len(texts)}\n")
    if transformer_errors:
        for e in transformer_errors:
            f.write(f"gold={e['gold']} pred={e['pred']} | {e['text'][:100]}\n")
    else:
        f.write(
            """No errors on this small test set. Because the score is perfect, the result still needs qualitative analysis: the model may be using sentiment-bearing words such as disaster, love, favorite, excellently, and couldn't get into this one, but confidence alone does not prove that it handles contrast, topic neutrality, or domain context for the right reasons.\n"""
        )

    f.write(f"\nVADER-transformer disagreements: {len(vader_transformer_disagreements)} / {len(texts)}\n")
    for _, row in vader_transformer_disagreements.iterrows():
        f.write(
            f"gold={row['gold']} vader={row['vader']} transformer={row['transformer']} "
            f"score={row['transformer_score']} | {row['text'][:100]}\n"
        )

    f.write(f"\nAny-system disagreements: {len(all_system_disagreements)} / {len(texts)}\n")
    for _, row in all_system_disagreements.iterrows():
        f.write(
            f"gold={row['gold']} vader={row['vader']} "
            f"bow={row['bow_logreg_sst5']} transformer={row['transformer']} | "
            f"{row['text'][:100]}\n"
        )

    f.write("\nLowest BoW + Logistic Regression confidence predictions:\n")
    for _, row in lowest_bow_confidence.iterrows():
        f.write(
            f"gold={row['gold']} bow={row['bow_logreg_sst5']} "
            f"score={row['bow_logreg_sst5_score']} | {row['text'][:100]}\n"
        )

    f.write("\nLowest transformer confidence predictions:\n")
    for _, row in lowest_confidence.iterrows():
        f.write(
            f"gold={row['gold']} transformer={row['transformer']} "
            f"score={row['transformer_score']} | {row['text'][:100]}\n"
        )

print("\nSaved results to", RESULTS)
