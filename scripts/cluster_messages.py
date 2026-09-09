"""Step 1 of taxonomy induction: unsupervised structure discovery.

We cluster customer messages with TF-IDF + KMeans and dump top terms + exemplars
per cluster. This is *input* to taxonomy design, not the taxonomy itself -- the
clusters are noisy and some split one intent across several centroids.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from support_agent.config import load_config  # noqa: E402

K = 28
N_TERMS = 12
N_EXEMPLARS = 6


def main() -> None:
    cfg = load_config()
    pool = pd.read_parquet(ROOT / cfg["paths"]["processed"] / "retrieval_pool.parquet")
    n = min(int(cfg["sampling"]["taxonomy_sample"]) * 4, len(pool))
    sample = pool.sample(n, random_state=cfg["sampling"]["seed"]).reset_index(drop=True)

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.4,
        max_features=40_000,
        stop_words="english",
        sublinear_tf=True,
    )
    X = vec.fit_transform(sample["customer_text"])
    km = KMeans(n_clusters=K, random_state=cfg["sampling"]["seed"], n_init=10)
    labels = km.fit_predict(X)
    sample["cluster"] = labels

    terms = np.array(vec.get_feature_names_out())
    order = km.cluster_centers_.argsort()[:, ::-1]

    out = []
    for c in range(K):
        members = sample[sample.cluster == c]
        # Exemplars closest to the centroid read more cleanly than random members.
        idx = members.index.to_numpy()
        if len(idx) == 0:
            continue
        d = np.asarray((X[idx] @ km.cluster_centers_[c].reshape(-1, 1)).ravel())
        top_idx = idx[np.argsort(-d)[:N_EXEMPLARS]]
        out.append(
            {
                "cluster": int(c),
                "size": int(len(members)),
                "top_terms": [str(t) for t in terms[order[c, :N_TERMS]]],
                "exemplars": [sample.loc[i, "customer_text"][:220] for i in top_idx],
            }
        )

    out.sort(key=lambda r: -r["size"])
    dest = ROOT / cfg["paths"]["results"] / "clusters.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    for r in out:
        print(f"\n[{r['cluster']:>2}] n={r['size']:<5} {', '.join(r['top_terms'][:9])}")
        for e in r["exemplars"][:3]:
            print(f"      - {e[:150]}")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
