"""Can B (binding) vs F (functional) be predicted from assay text, and how well?

Labels come only from ChEMBL ASSAYS.ASSAY_TYPE. The model reads the assay description
(BindingDB: assay name + description) and outputs P(F). It is measured, never trusted:

  train / val / test   ChEMBL B/F assays split by DOCUMENT (assays of one paper share
                       wording, so a row split would leak); val only picks the threshold
  bdb_route1           BindingDB assays carrying a ChEMBL assay id (ChEMBL-copied text).
                       Their ChEMBL twins are removed from training. The name is NOT used,
                       it contains the ChEMBL id.
  bdb_route2           BindingDB assays labelled by value match with ChEMBL, split into
                       patent / non-patent; the only labelled set written in patent style
Thresholds are fixed on val before any BindingDB set is scored: t_F is the lowest P(F)
with F precision >= TARGET on val, t_B the highest P(F) with B precision >= TARGET.
Then the 9,328 unclassified BindingDB assays are predicted into F / B / uncertain.

Model: TF-IDF word 1-2 grams + logistic regression (scikit-learn).

Usage: python -I audit_assay_text_classifier_20261104.py <dataset_dir> <audit_dir>
"""
import hashlib
import html
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DATA, AUDIT = Path(sys.argv[1]), Path(sys.argv[2])
LINK = re.compile(r"^Ch(?:EMBL|EBML)_(\d+)")
TARGET = 0.99
started = time.time()


def log(*parts):
    print(f"[{time.time() - started:7.1f}s]", *parts, flush=True)


def clean(text):
    return re.sub(r"\s+", " ", html.unescape(text or "")).strip()


def bucket(document, salt="assay-text-20261104"):
    """Deterministic 0..99 bucket per document."""
    return int(hashlib.sha256(f"{salt}:{document}".encode()).hexdigest(), 16) % 100


# ------------------------------------------------------------------ ChEMBL labelled text
c = ds.dataset(DATA / "deduplicate" / "chembl37_observations_dedup.parquet", format="parquet").to_table(
    columns=["assay_native_id", "assay_type", "assay_description", "assay_context", "document_id"])
g = c.group_by(c.column_names).aggregate([])
# one assay can appear with several contexts; sort so the kept row is the same every run
g = g.sort_by([(n, "ascending") for n in g.column_names])
del c
chembl = {}
for a, t, d, x, doc in zip(*[g[n].to_pylist() for n in g.column_names]):
    if t in ("B", "F") and d:
        chembl[a] = (json.loads(x).get("assay_id"), t, clean(d), doc)
del g
log("ChEMBL B/F assays with description", len(chembl))

labels = pq.read_table(AUDIT / "bindingdb_assay_labels.parquet").to_pylist()
b = ds.dataset(DATA / "deduplicate" / "bindingdb202610_observations_dedup.parquet", format="parquet").to_table(
    columns=["source_database", "assay_entry_id", "assay_native_id", "assay_name", "assay_description"])
g = b.group_by(b.column_names).aggregate([])
del b
bdb_text = {}
for s, e, a, n, d in zip(*[g[x].to_pylist() for x in g.column_names]):
    bdb_text[f"{s}:{e or ''}:{a or ''}"] = (n, d)
del g

route1_ids = set()
for row in labels:
    if row["route"].startswith("route 1"):
        m = LINK.search(bdb_text.get(row["assay_key"], ("", ""))[0] or "")
        if m:
            route1_ids.add(int(m.group(1)))

sets = {"train": ([], []), "val": ([], []), "test": ([], [])}
for _, (number, kind, text, doc) in chembl.items():
    if number in route1_ids:
        continue                                   # twin of a BindingDB test assay
    split = bucket(doc)
    name = "train" if split < 80 else ("val" if split < 90 else "test")
    sets[name][0].append(text)
    sets[name][1].append(1 if kind == "F" else 0)


def bdb_label(group):
    if group.startswith("B "):
        return 0
    if group.startswith("F "):
        return 1
    return None


for name in ("bdb_route1", "bdb_route2_patent", "bdb_route2_nonpatent"):
    sets[name] = ([], [])
unclassified = []
for row in labels:
    n, d = bdb_text.get(row["assay_key"], (None, None))
    y = bdb_label(row["group"])
    patent = row["source_curation"] in ("US Patent", "WIPO")
    if row["route"].startswith("route 1") and y is not None and d:
        sets["bdb_route1"][0].append(clean(d))
        sets["bdb_route1"][1].append(y)
    elif row["route"].startswith("route 2") and y is not None:
        target = "bdb_route2_patent" if patent else "bdb_route2_nonpatent"
        sets[target][0].append(clean(f"{n or ''}. {d or ''}"))
        sets[target][1].append(y)
    elif row["route"] == "unclassified":
        unclassified.append((row["assay_key"], row["source_curation"], row["n_observations"],
                             clean(f"{n or ''}. {d or ''}")))
log({k: (len(v[1]), int(sum(v[1]))) for k, v in sets.items()}, "unclassified", len(unclassified))

# ------------------------------------------------------------------ model
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=400_000,
                             sublinear_tf=True, lowercase=True)
x_train = vectorizer.fit_transform(sets["train"][0])
model = LogisticRegression(C=4.0, max_iter=2000, solver="liblinear")
model.fit(x_train, np.array(sets["train"][1]))
log("trained", x_train.shape)


def proba(texts):
    return model.predict_proba(vectorizer.transform(texts))[:, 1] if texts else np.zeros(0)


# thresholds on val only
pv, yv = proba(sets["val"][0]), np.array(sets["val"][1])
grid = np.linspace(0.01, 0.99, 99)
t_f = next((t for t in grid if (yv[pv >= t] == 1).mean() >= TARGET and (pv >= t).any()), 0.99)
t_b = next((t for t in grid[::-1] if (pv <= t).any() and (yv[pv <= t] == 0).mean() >= TARGET), 0.01)
if t_b >= t_f:
    # bands must not overlap; keep an uncertain zone between them
    raise SystemExit(f"thresholds overlap (B {t_b:.2f} >= F {t_f:.2f}); raise TARGET")
log("thresholds", t_f, t_b)


def scores(y, p):
    y = np.asarray(y)
    if y.size == 0:
        return {"n": 0}
    out = {"n": int(y.size), "n_F": int(y.sum()), "n_B": int((1 - y).sum())}
    if 0 < y.sum() < y.size:
        out["auc"] = float(roc_auc_score(y, p))
    pred = (p >= 0.5).astype(int)
    out["accuracy_at_0.5"] = float((pred == y).mean())
    for label, mask, wanted in (("F", p >= t_f, 1), ("B", p <= t_b, 0)):
        out[f"{label}_called"] = int(mask.sum())
        out[f"{label}_precision"] = float((y[mask] == wanted).mean()) if mask.any() else None
        out[f"{label}_recall"] = float((mask & (y == wanted)).sum() / max(1, (y == wanted).sum()))
    out["uncertain_share"] = float(((p > t_b) & (p < t_f)).mean())
    return out


result = {"model": "tfidf word 1-2gram + logistic regression (liblinear, C=4)",
          "precision_target": TARGET, "threshold_F": float(t_f), "threshold_B": float(t_b), "sets": {}}
for name, (texts, y) in sets.items():
    result["sets"][name] = scores(y, proba(texts))
    log(name, result["sets"][name])

# ------------------------------------------------------------------ unclassified BindingDB
pu = proba([t for *_, t in unclassified])
summary = {}
for (k, cur, n, text), p in zip(unclassified, pu):
    source = "patent" if cur in ("US Patent", "WIPO") else ("PDSP Ki (no text)" if cur == "PDSP Ki" else "non-patent")
    call = "no text" if text in ("", ".") else ("F" if p >= t_f else ("B" if p <= t_b else "uncertain"))
    cell = summary.setdefault(source, {}).setdefault(call, [0, 0])
    cell[0] += 1
    cell[1] += int(n)
result["unclassified_prediction"] = {s: {c: {"assays": v[0], "observations": v[1]} for c, v in d.items()}
                                     for s, d in summary.items()}
log("unclassified", json.dumps(result["unclassified_prediction"]))

terms = np.array(vectorizer.get_feature_names_out())
order = np.argsort(model.coef_[0])
result["top_terms_F"] = terms[order[-25:]][::-1].tolist()
result["top_terms_B"] = terms[order[:25]].tolist()
(AUDIT / "assay_text_classifier.json").write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                                  encoding="utf-8")
pq.write_table(
    __import__("pyarrow").table({"assay_key": [u[0] for u in unclassified],
                                 "source_curation": [u[1] for u in unclassified],
                                 "n_observations": [int(u[2]) for u in unclassified],
                                 "p_functional": pu.astype(float)}),
    AUDIT / "bindingdb_unclassified_text_predictions.parquet")
log("written")
