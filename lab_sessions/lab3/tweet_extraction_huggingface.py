from datasets import load_dataset
import json
import random

random.seed(42)

dataset = load_dataset("tweet_eval", "sentiment", split="train")

label_map = {0: "negative", 1: "neutral", 2: "positive"}

by_label = {"positive": [], "negative": [], "neutral": []}
for item in dataset:
    label = label_map[item["label"]]
    by_label[label].append((item["text"], label))

picks = (
    random.sample(by_label["positive"], 17) +
    random.sample(by_label["neutral"], 17) +
    random.sample(by_label["negative"], 16)
)
random.shuffle(picks)

result = {}
for i, (text, label) in enumerate(picks, start=1):
    result[str(i)] = {
        "sentiment_label": label,
        "text_of_tweet": text,
        "tweet_url": "https://huggingface.co/datasets/tweet_eval"
    }

with open("my_tweets.json", "w") as f:
    json.dump(result, f, indent=4)

print("Done! Label counts:")
for label in ["positive", "neutral", "negative"]:
    count = sum(1 for v in result.values() if v["sentiment_label"] == label)
    print(f"  {label}: {count}")
