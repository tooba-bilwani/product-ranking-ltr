"""Learning-to-rank on MovieLens-100k — a public-data demo of personalized
item ranking (same shape as marketplace product-relevancy ranking).

Setup
-----
- Each user is a "query"; items are "documents".
- Relevance: graded from rating -> max(rating - 3, 0)  (so 4->1, 5->2, <=3->0);
  a "positive" is rating >= 4.
- Temporal split PER USER: each user's most recent 20% of interactions -> test,
  the rest -> train. (No future leaks into training.)
- Candidate lists: positives + sampled unseen negatives, so the model must rank
  liked items above items the user never engaged with — exactly the implicit-
  feedback ranking problem you hit in product search / recommendations.

Rigor
-----
- ALL features (item popularity, item/user mean rating, genre affinity) are
  computed from the TRAIN split only. Cold items/users fall back to global means.
  This is the leakage guard: no test interaction informs a feature.

Models
------
- Baseline: rank by global item popularity (non-personalized).
- LTR: XGBoost XGBRanker with objective rank:ndcg (LambdaMART).

Metrics: NDCG@10, MAP@10, MRR — averaged over users.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from xgboost import XGBRanker

MF_DIM = 20   # latent dimensions for the collaborative-filtering embeddings
CONTENT_FEATS = ["item_pop_log", "item_mean", "user_activity", "user_mean", "genre_match"]
FULL_FEATS = CONTENT_FEATS + ["mf_score"]   # content + collaborative-filtering

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ml-100k"
RESULTS = ROOT / "results"
RNG = np.random.default_rng(42)

GENRE_COLS = [f"g{i}" for i in range(19)]
N_TRAIN_NEG = 4      # negatives per positive when training
N_TEST_NEG = 50      # sampled negatives per user to form a test ranking list
K = 10               # cutoff for @k metrics


# ----------------------------- data loading ---------------------------------
def load_data():
    ratings = pd.read_csv(
        DATA / "u.data", sep="\t", names=["user", "item", "rating", "ts"])
    items = pd.read_csv(
        DATA / "u.item", sep="|", encoding="latin-1", header=None,
        names=["item", "title", "rel_date", "vid_date", "url"] + GENRE_COLS)
    genres = items.set_index("item")[GENRE_COLS]
    return ratings, genres


def temporal_split(ratings, test_frac=0.2, min_inter=10):
    """Hold out each user's most-recent interactions as the test set."""
    ratings = ratings.sort_values(["user", "ts"])
    train_parts, test_parts = [], []
    for _, grp in ratings.groupby("user"):
        if len(grp) < min_inter:
            train_parts.append(grp)          # too sparse to test -> train only
            continue
        n_test = max(1, int(len(grp) * test_frac))
        train_parts.append(grp.iloc[:-n_test])
        test_parts.append(grp.iloc[-n_test:])
    return pd.concat(train_parts), pd.concat(test_parts)


# ------------------------- feature engineering ------------------------------
def compute_mf(train, k=MF_DIM):
    """Matrix factorization on the TRAIN positives -> user & item embeddings in
    one shared latent space (collaborative filtering).

    Build the sparse user x item matrix of positives (rating >= 4), factor it
    with truncated SVD, and read off a k-dim vector for each user and each item.
    The dot product of a user vector and an item vector is the 'collaborative
    affinity' — learned purely from co-rating behavior, no genre labels.
    """
    pos = train[train["rating"] >= 4]
    u_ids = sorted(pos["user"].unique())
    i_ids = sorted(pos["item"].unique())
    uidx = {u: i for i, u in enumerate(u_ids)}
    iidx = {it: i for i, it in enumerate(i_ids)}
    rows = pos["user"].map(uidx).to_numpy()
    cols = pos["item"].map(iidx).to_numpy()
    R = csr_matrix((np.ones(len(pos)), (rows, cols)), shape=(len(u_ids), len(i_ids)))
    k = min(k, min(R.shape) - 1)
    U, s, Vt = svds(R, k=k)           # R ~= U @ diag(s) @ Vt
    return {"uidx": uidx, "iidx": iidx,
            "user_emb": U * s,        # (n_users, k)
            "item_emb": Vt.T}         # (n_items, k)


def build_feature_tables(train, genres):
    """Everything here is derived from TRAIN ONLY (leakage guard)."""
    g_mean = train["rating"].mean()

    item_stats = train.groupby("item")["rating"].agg(["count", "mean"])
    item_stats.columns = ["item_pop", "item_mean"]

    user_stats = train.groupby("user")["rating"].agg(["count", "mean"])
    user_stats.columns = ["user_activity", "user_mean"]

    # User genre affinity = mean genre vector of items they rated >= 4 in train.
    pos = train[train["rating"] >= 4].merge(
        genres, left_on="item", right_index=True, how="left")
    user_genre = pos.groupby("user")[GENRE_COLS].mean()

    return {
        "g_mean": g_mean,
        "item_stats": item_stats,
        "user_stats": user_stats,
        "user_genre": user_genre,
        "genres": genres,
        "mf": compute_mf(train),       # collaborative-filtering embeddings
    }


def featurize(pairs, ft):
    """pairs: DataFrame[user, item] -> feature matrix (cold-start safe)."""
    df = pairs.merge(ft["item_stats"], left_on="item", right_index=True, how="left")
    df = df.merge(ft["user_stats"], left_on="user", right_index=True, how="left")
    df["item_pop"] = df["item_pop"].fillna(0.0)
    df["item_mean"] = df["item_mean"].fillna(ft["g_mean"])
    df["user_activity"] = df["user_activity"].fillna(0.0)
    df["user_mean"] = df["user_mean"].fillna(ft["g_mean"])
    df["item_pop_log"] = np.log1p(df["item_pop"])

    # genre affinity = dot(user genre pref, item genre vector)
    ug = ft["user_genre"].reindex(df["user"].values).fillna(0.0).to_numpy()
    ig = ft["genres"].reindex(df["item"].values).fillna(0.0).to_numpy()
    df["genre_match"] = (ug * ig).sum(axis=1)

    # collaborative-filtering affinity = dot(user_emb, item_emb); 0 for cold
    # users/items with no learned embedding (that's when genre_match carries it).
    mf = ft["mf"]
    uix = df["user"].map(mf["uidx"])
    iix = df["item"].map(mf["iidx"])
    score = np.zeros(len(df))
    mask = uix.notna().to_numpy() & iix.notna().to_numpy()
    if mask.any():
        u = uix[mask].astype(int).to_numpy()
        i = iix[mask].astype(int).to_numpy()
        score[mask] = (mf["user_emb"][u] * mf["item_emb"][i]).sum(axis=1)
    df["mf_score"] = score
    return df


# ------------------------- candidate construction ---------------------------
def sample_negs(user_seen, all_items, n, weights=None):
    """Sample n items the user hasn't seen. If `weights` (popularity) is given,
    sample proportional to it -> 'hard' negatives that are popular but not
    relevant to THIS user, so a pure-popularity ranker can't separate them."""
    mask = ~np.isin(all_items, list(user_seen))
    pool = all_items[mask]
    if len(pool) == 0:
        return np.array([], dtype=int)
    p = None
    if weights is not None:
        p = weights[mask]
        p = p / p.sum()
    return RNG.choice(pool, size=min(n, len(pool)), replace=False, p=p)


def popularity_weights(train, all_items):
    counts = train["item"].value_counts()
    return np.array([counts.get(it, 0) + 1.0 for it in all_items])  # +1 smoothing


def build_training_frame(train, all_items, weights):
    """Per user: observed items (graded relevance) + hard negatives (rel 0)."""
    rows = []
    seen_by_user = train.groupby("user")["item"].agg(set)
    for user, grp in train.groupby("user"):
        for _, r in grp.iterrows():
            rows.append((user, r["item"], max(r["rating"] - 3, 0)))
        n_pos = max(1, (grp["rating"] >= 4).sum())
        for it in sample_negs(seen_by_user[user], all_items, n_pos * N_TRAIN_NEG, weights):
            rows.append((user, it, 0))
    df = pd.DataFrame(rows, columns=["user", "item", "rel"]).sort_values("user")
    return df


def build_eval_lists(test, train, all_items, weights):
    """Per user: test positives (graded rel) + hard (popularity) negatives (rel 0)."""
    seen_by_user = train.groupby("user")["item"].agg(set)
    lists = []
    for user, grp in test.groupby("user"):
        pos = grp[grp["rating"] >= 4]
        if len(pos) == 0:
            continue                       # need at least one relevant item
        cand = [(it, max(rt - 3, 0)) for it, rt in zip(pos["item"], pos["rating"])]
        exclude = seen_by_user.get(user, set()) | set(pos["item"])
        for it in sample_negs(exclude, all_items, N_TEST_NEG, weights):
            cand.append((it, 0))
        lists.append((user, cand))
    return lists


# ------------------------------- metrics ------------------------------------
def dcg(rels):
    rels = np.asarray(rels, dtype=float)
    return np.sum(rels / np.log2(np.arange(2, len(rels) + 2)))


def ndcg_at_k(ranked_rels, k):
    ideal = sorted(ranked_rels, reverse=True)
    idcg = dcg(ideal[:k])
    return dcg(ranked_rels[:k]) / idcg if idcg > 0 else 0.0


def ap_at_k(ranked_bin, k):
    hits, score = 0, 0.0
    for i, r in enumerate(ranked_bin[:k]):
        if r:
            hits += 1
            score += hits / (i + 1)
    denom = min(sum(ranked_bin), k)
    return score / denom if denom else 0.0


def rr(ranked_bin):
    for i, r in enumerate(ranked_bin):
        if r:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(lists, score_fn):
    ndcgs, maps, rrs = [], [], []
    for user, cand in lists:
        items = [it for it, _ in cand]
        rels = [rel for _, rel in cand]
        scores = score_fn(user, items)
        order = np.argsort(-np.asarray(scores))
        ranked_rels = [rels[i] for i in order]
        ranked_bin = [1 if r > 0 else 0 for r in ranked_rels]
        ndcgs.append(ndcg_at_k(ranked_rels, K))
        maps.append(ap_at_k(ranked_bin, K))
        rrs.append(rr(ranked_bin))
    return {"NDCG@10": float(np.mean(ndcgs)),
            "MAP@10": float(np.mean(maps)),
            "MRR": float(np.mean(rrs))}


# --------------------------------- main -------------------------------------
def main():
    RESULTS.mkdir(exist_ok=True)
    ratings, genres = load_data()
    all_items = ratings["item"].unique()
    train, test = temporal_split(ratings)
    print(f"users={ratings['user'].nunique()} items={len(all_items)} "
          f"train={len(train)} test={len(test)}")

    ft = build_feature_tables(train, genres)
    pop_w = popularity_weights(train, all_items)   # for hard-negative sampling
    tr = build_training_frame(train, all_items, pop_w)
    groups = tr.groupby("user").size().to_numpy()
    tr_feats = featurize(tr[["user", "item"]], ft)

    def train_ranker(feature_cols):
        r = XGBRanker(objective="rank:ndcg", n_estimators=300, learning_rate=0.1,
                      max_depth=6, subsample=0.8, colsample_bytree=0.8,
                      random_state=42)
        r.fit(tr_feats[feature_cols].to_numpy(), tr["rel"].to_numpy(), group=groups)
        return r

    # v1 = content + popularity features; v2 = v1 + collaborative-filtering embedding
    ranker_v1 = train_ranker(CONTENT_FEATS)
    ranker_v2 = train_ranker(FULL_FEATS)

    eval_lists = build_eval_lists(test, train, all_items, pop_w)
    print(f"eval users with >=1 relevant test item: {len(eval_lists)}")

    def make_scorer(ranker, cols):
        def score(user, items):
            X = featurize(pd.DataFrame({"user": user, "item": items}), ft)
            return ranker.predict(X[cols].to_numpy())
        return score

    pop = ft["item_stats"]["item_pop"]

    def pop_score(user, items):
        return pop.reindex(items).fillna(0.0).to_numpy()

    baseline = evaluate(eval_lists, pop_score)
    v1 = evaluate(eval_lists, make_scorer(ranker_v1, CONTENT_FEATS))
    v2 = evaluate(eval_lists, make_scorer(ranker_v2, FULL_FEATS))

    def lift(a, b):   # % improvement of b over a
        return {m: round(100 * (b[m] - a[m]) / a[m], 1) for m in a}

    out = {
        "baseline_popularity": baseline,
        "v1_ltr_content": v1,
        "v2_ltr_content_plus_cf": v2,
        "lift_v1_vs_baseline": lift(baseline, v1),
        "lift_v2_vs_baseline": lift(baseline, v2),
        "lift_v2_vs_v1": lift(v1, v2),
        "v2_feature_importance": dict(sorted(
            zip(FULL_FEATS, ranker_v2.feature_importances_.tolist()),
            key=lambda x: -x[1])),
    }
    (RESULTS / "metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    # ---- chart: baseline vs v1 vs v2 ----
    metrics = list(baseline)
    x = np.arange(len(metrics))
    plt.figure(figsize=(8, 4.8))
    for off, res, label in [(-0.27, baseline, "Popularity baseline"),
                            (0.0, v1, "LTR: content"),
                            (0.27, v2, "LTR: content + CF embeddings")]:
        plt.bar(x + off, [res[m] for m in metrics], 0.26, label=label)
    plt.xticks(x, metrics)
    plt.ylabel("score")
    plt.title("Personalized ranking on MovieLens-100k")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "ranking_comparison.png", dpi=130)
    print(f"\nWrote {RESULTS/'metrics.json'} and {RESULTS/'ranking_comparison.png'}")


if __name__ == "__main__":
    main()
