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
from xgboost import XGBRanker

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

    feats = ["item_pop_log", "item_mean", "user_activity", "user_mean", "genre_match"]
    return df[feats], feats


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

    # ---- train LambdaMART ranker ----
    tr = build_training_frame(train, all_items, pop_w)
    X, feat_names = featurize(tr[["user", "item"]], ft)
    groups = tr.groupby("user").size().to_numpy()
    ranker = XGBRanker(objective="rank:ndcg", n_estimators=300, learning_rate=0.1,
                       max_depth=6, subsample=0.8, colsample_bytree=0.8,
                       random_state=42)
    ranker.fit(X.to_numpy(), tr["rel"].to_numpy(), group=groups)

    # ---- evaluate (popularity-matched hard negatives) ----
    eval_lists = build_eval_lists(test, train, all_items, pop_w)
    print(f"eval users with >=1 relevant test item: {len(eval_lists)}")

    def model_score(user, items):
        X, _ = featurize(pd.DataFrame({"user": user, "item": items}), ft)
        return ranker.predict(X.to_numpy())

    pop = ft["item_stats"]["item_pop"]

    def pop_score(user, items):
        return pop.reindex(items).fillna(0.0).to_numpy()

    baseline = evaluate(eval_lists, pop_score)
    model = evaluate(eval_lists, model_score)

    lift = {m: round(100 * (model[m] - baseline[m]) / baseline[m], 1)
            for m in baseline}
    out = {"baseline_popularity": baseline, "lambdamart": model,
           "lift_pct": lift, "feature_importance":
           dict(sorted(zip(feat_names, ranker.feature_importances_.tolist()),
                       key=lambda x: -x[1]))}
    (RESULTS / "metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    # ---- chart ----
    metrics = list(baseline)
    x = np.arange(len(metrics))
    plt.figure(figsize=(7, 4.5))
    plt.bar(x - 0.2, [baseline[m] for m in metrics], 0.4, label="Popularity baseline")
    plt.bar(x + 0.2, [model[m] for m in metrics], 0.4, label="LambdaMART (LTR)")
    plt.xticks(x, metrics)
    plt.ylabel("score")
    plt.title("Personalized ranking: LambdaMART vs popularity baseline (MovieLens-100k)")
    plt.legend()
    for i, m in enumerate(metrics):
        plt.text(i + 0.2, model[m] + 0.005, f"+{lift[m]}%", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(RESULTS / "ranking_comparison.png", dpi=130)
    print(f"\nWrote {RESULTS/'metrics.json'} and {RESULTS/'ranking_comparison.png'}")


if __name__ == "__main__":
    main()
