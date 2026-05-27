"""
NERC task — final assignment.

Two systems compared on the gold test set (final_assignment/data/NER-test.tsv):

  System A: LinearSVC trained from scratch on CoNLL-2003, token-level features
            (word + POS + simple context/shape features).
  System B: spaCy en_core_web_sm pretrained neural NER, labels mapped to the
            CoNLL tag set (PER / ORG / LOC / MISC).

Both produce BIO predictions aligned token-by-token to the gold file, so they
are evaluated identically with seqeval (span level) and a token-level report.

Run:  python nerc_pipeline.py
Outputs go to ../results/.
"""

import os
from collections import Counter, defaultdict

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from sklearn.metrics import classification_report as sk_report
from seqeval.metrics import classification_report as seq_report
from seqeval.metrics import f1_score as seq_f1

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
RESULTS = os.path.join(HERE, "..", "results")
CONLL = os.path.join(HERE, "..", "..", "lab_sessions", "lab4", "CONLL2003", "CONLL2003")
os.makedirs(RESULTS, exist_ok=True)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def read_conll(path):
    """Read a CoNLL-2003 file -> list of sentences, each a list of (word,pos,tag)."""
    sents, cur = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if not line or line.startswith("-DOCSTART-"):
                if cur:
                    sents.append(cur)
                    cur = []
                continue
            parts = line.split()
            word, pos, tag = parts[0], parts[1], parts[-1]
            cur.append((word, pos, tag))
    if cur:
        sents.append(cur)
    return sents


def read_gold_test(path):
    """Read the assignment test set -> list of sentences of (token, gold_tag)."""
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    sents = []
    for _, grp in df.groupby("sentence id", sort=True):
        sents.append(list(zip(grp["token"].astype(str), grp["BIO NER tag"].str.strip())))
    return sents


# --------------------------------------------------------------------------- #
# System A: feature-based LinearSVC trained on CoNLL-2003
# --------------------------------------------------------------------------- #
def word2features(sent, i):
    """sent is a list of (word, pos). Token features + immediate context + shape."""
    word, pos = sent[i][0], sent[i][1]
    feats = {
        "word.lower": word.lower(),
        "word.istitle": word.istitle(),
        "word.isupper": word.isupper(),
        "word.isdigit": word.isdigit(),
        "word.suffix3": word[-3:].lower(),
        "pos": pos,
    }
    if i > 0:
        feats["-1.word.lower"] = sent[i - 1][0].lower()
        feats["-1.pos"] = sent[i - 1][1]
        feats["-1.istitle"] = sent[i - 1][0].istitle()
    else:
        feats["BOS"] = True
    if i < len(sent) - 1:
        feats["+1.word.lower"] = sent[i + 1][0].lower()
        feats["+1.pos"] = sent[i + 1][1]
        feats["+1.istitle"] = sent[i + 1][0].istitle()
    else:
        feats["EOS"] = True
    return feats


def sent_features(sent):
    wp = [(w, p) for (w, p, *_) in sent]
    return [word2features(wp, i) for i in range(len(wp))]


def run_system_a(train_sents, test_sents_wp):
    train_feats, train_labels = [], []
    for s in train_sents:
        train_feats.extend(sent_features(s))
        train_labels.extend([t for (_, _, t) in s])

    test_feats = []
    for s in test_sents_wp:
        test_feats.extend(sent_features(s))

    vec = DictVectorizer()
    X_all = vec.fit_transform(train_feats + test_feats)
    n = len(train_feats)
    clf = svm.LinearSVC(max_iter=10000, C=1.0)
    clf.fit(X_all[:n], train_labels)
    preds_flat = clf.predict(X_all[n:])

    # re-segment flat predictions back into sentences
    out, k = [], 0
    for s in test_sents_wp:
        out.append(list(preds_flat[k:k + len(s)]))
        k += len(s)
    return out


# --------------------------------------------------------------------------- #
# System B: spaCy pretrained NER, mapped to CoNLL tags
# --------------------------------------------------------------------------- #
SPACY_MAP = {
    "PERSON": "PER",
    "ORG": "ORG",
    "GPE": "LOC", "LOC": "LOC", "FAC": "LOC",
    "NORP": "MISC", "LANGUAGE": "MISC", "EVENT": "MISC",
    "WORK_OF_ART": "MISC", "PRODUCT": "MISC", "LAW": "MISC",
}


def run_system_b(test_sents_tokens):
    """
    TODO (partner): implement System B — a pretrained neural NER, different from
    System A, so the two-system comparison is real.

    Suggested options (pick one, or compare both):
      - spaCy en_core_web_trf  (transformer model; pip install + spacy download)
      - Stanza  (pip install stanza; stanza.download("en"))
      - a different spaCy size (en_core_web_sm / md / lg)

    Contract this function must satisfy:
      INPUT : test_sents_tokens -> list of sentences, each a list of token strings
              (already tokenised; do NOT re-tokenise, feed the given tokens in so
               predictions stay aligned to the gold tags)
      OUTPUT: list of sentences, each a list of BIO tags, SAME length as the input
              sentence, using the CoNLL tag set: O / B-PER / I-PER / B-ORG / I-ORG
              / B-LOC / I-LOC / B-MISC / I-MISC

    Map the model's native entity types to the CoNLL set. SPACY_MAP above is a
    starting point for spaCy; Stanza already emits PER/ORG/LOC/MISC.

    Tip: with spaCy, build a Doc from the given words so tokenisation matches:
        from spacy.tokens import Doc
        doc = Doc(nlp.vocab, words=list(tokens))
        # then run the tok2vec + ner pipes on doc
    """
    raise NotImplementedError(
        "System B not implemented yet — see TODO in run_system_b()."
    )


# --------------------------------------------------------------------------- #
# Evaluation + reporting
# --------------------------------------------------------------------------- #
def evaluate(name, gold_seqs, pred_seqs, tokens_seqs):
    span = seq_report(gold_seqs, pred_seqs, digits=3, zero_division=0)
    f1 = seq_f1(gold_seqs, pred_seqs, zero_division=0)

    gold_flat = [t for s in gold_seqs for t in s]
    pred_flat = [t for s in pred_seqs for t in s]
    token = sk_report(gold_flat, pred_flat, digits=3, zero_division=0)

    # collect token-level disagreements for error analysis
    errors = []
    for toks, g, p in zip(tokens_seqs, gold_seqs, pred_seqs):
        for tk, gg, pp in zip(toks, g, p):
            if gg != pp:
                errors.append((tk, gg, pp))

    report = (
        f"{'='*70}\n{name}\n{'='*70}\n"
        f"Span-level F1 (seqeval): {f1:.3f}\n\n"
        f"--- Span-level (entity) report ---\n{span}\n"
        f"--- Token-level report ---\n{token}\n"
        f"--- Token-level errors ({len(errors)}) ---\n"
        + "\n".join(f"{tk!r}: gold={g} pred={p}" for tk, g, p in errors)
        + "\n"
    )
    return report, f1, errors


def main():
    print("Loading data ...")
    train_sents = read_conll(os.path.join(CONLL, "train.txt"))
    gold = read_gold_test(os.path.join(DATA, "NER-test.tsv"))
    tokens_seqs = [[tok for tok, _ in s] for s in gold]
    gold_seqs = [[tag for _, tag in s] for s in gold]
    print(f"CoNLL train sentences: {len(train_sents)} | test sentences: {len(gold)}")

    # System A needs (word, pos) for the test set; get POS from spaCy once.
    print("Tagging test set with POS (spaCy) for System A features ...")
    import spacy
    from spacy.tokens import Doc
    nlp = spacy.load("en_core_web_sm")
    test_wp = []
    for toks in tokens_seqs:
        doc = Doc(nlp.vocab, words=toks)
        doc = nlp.get_pipe("tagger")(nlp.get_pipe("tok2vec")(doc))
        test_wp.append([(t.text, t.tag_) for t in doc])

    print("Running System A (LinearSVC on CoNLL-2003) ...")
    preds_a = run_system_a(train_sents, test_wp)
    rep_a, f1_a, err_a = evaluate("System A — LinearSVC (CoNLL-2003, word+POS+context)",
                                  gold_seqs, preds_a, tokens_seqs)

    print("Running System B (pretrained NER) ...")
    try:
        preds_b = run_system_b(tokens_seqs)
        rep_b, f1_b, err_b = evaluate("System B — pretrained NER",
                                      gold_seqs, preds_b, tokens_seqs)
    except NotImplementedError as e:
        preds_b, rep_b, f1_b, err_b = None, f"System B skipped: {e}\n", None, None
        print(f"  [skipped] {e}")

    with open(os.path.join(RESULTS, "nerc_report.txt"), "w") as f:
        f.write(rep_a + "\n" + rep_b)
        f.write(f"\n{'='*70}\nSUMMARY\n{'='*70}\n")
        f.write(f"System A span-F1: {f1_a:.3f} ({len(err_a)} token errors)\n")
        if f1_b is not None:
            f.write(f"System B span-F1: {f1_b:.3f} ({len(err_b)} token errors)\n")

    # per-token side-by-side predictions for qualitative analysis
    rows = []
    for i, (toks, g, pa) in enumerate(zip(tokens_seqs, gold_seqs, preds_a)):
        pb = preds_b[i] if preds_b is not None else ["-"] * len(toks)
        for tk, gg, paa, pbb in zip(toks, g, pa, pb):
            rows.append({"token": tk, "gold": gg, "sysA": paa, "sysB": pbb})
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "nerc_predictions.csv"), index=False)

    print(rep_a)
    print(rep_b)
    if f1_b is not None:
        print(f"System A span-F1 {f1_a:.3f} | System B span-F1 {f1_b:.3f}")
    else:
        print(f"System A span-F1 {f1_a:.3f} | System B not yet implemented")
    print(f"Saved: {RESULTS}/nerc_report.txt and nerc_predictions.csv")


if __name__ == "__main__":
    main()
