# Learning-to-Rank for Personalized Item Ranking

A compact, reproducible **learning-to-rank** project: rank the items a user will
engage with to the top of their list. This is the same problem shape as **product
search / relevancy ranking on a marketplace** — I built the production version of
this class of system at work; this is a clean public-data demonstration.

**Data:** [MovieLens-100k](https://grouplens.org/datasets/movielens/100k/)
(943 users, 1,682 items, 100k interactions).
**Model:** XGBoost `XGBRanker` (LambdaMART, `rank:ndcg`).
**Built in two versions** to show the impact of collaborative filtering.

## Results

Evaluated with **popularity-matched hard negatives** (see note below), averaged
over 906 users:

| Metric   | Popularity baseline | v1: content features | v2: **+ CF embeddings** |
|----------|--------------------:|---------------------:|------------------------:|
| NDCG@10  | 0.221 | 0.281 (+27%) | **0.443 (+100%)** |
| MAP@10   | 0.133 | 0.178 (+34%) | **0.331 (+150%)** |
| MRR      | 0.365 | 0.452 (+24%) | **0.610 (+67%)** |

*(% vs the popularity baseline. v2 improves on v1 by +58% NDCG / +86% MAP.)*

![comparison](results/ranking_comparison.png)

## v2: adding collaborative-filtering embeddings

v1 uses **content features** — item popularity, average ratings, and `genre_match`
(overlap of the user's genre taste with the item's genres). Genres only capture
taste along **human-labeled** dimensions.

v2 adds a **collaborative-filtering** feature. I factorize the train user×item
matrix (truncated SVD) into **user and item embeddings in one shared latent
space** — 20-dim vectors learned purely from co-rating behavior, no labels. The
dot product of a user vector and an item vector (`mf_score`) is their learned
affinity: *"users who behaved like you also engaged with this."* It captures
structure genres can't — e.g. it separates two same-genre items whose fan bases
differ.

Adding that one feature nearly **doubled** ranking quality, and `mf_score` became
the **most important feature (0.64 importance)** — the behavioral/latent signal
outweighs the hand-labeled content features. `genre_match` is kept because CF is
blind to **cold-start** users/items with no interaction history; there, content
features carry the ranking.

## The other interesting part: I found the bug in my own evaluation

My **first** version sampled test negatives *uniformly at random* — and the LTR
model **lost** to the popularity baseline (NDCG −0.7%).

That's a classic recommender-systems trap: random negatives are mostly unpopular,
so **popularity alone separates them from positives almost perfectly** — an
artificially strong baseline that says nothing about personalization.

The fix is **popularity-matched ("hard") negatives** — sample negatives in
proportion to item popularity, so candidates are *popular items this user didn't
engage with*. Now popularity can't cheat and the model's personalization has to do
the work. Same models, honest evaluation.

## What's modeled

- **Task:** per user (= "query"), rank a candidate list of items (= "documents").
- **Relevance:** graded from rating, `max(rating − 3, 0)` (4→1, 5→2, ≤3→0);
  a "positive" is rating ≥ 4.
- **Temporal split, per user:** each user's most recent 20% of interactions → test.
  No future interaction leaks into training.
- **Leakage guard:** every feature — content stats *and* the CF embeddings — is
  computed from the **train split only**; cold items/users fall back to global
  means / zero CF affinity.
- **Features:** `item_pop_log`, `item_mean`, `user_activity`, `user_mean`,
  `genre_match` (content) + `mf_score` (collaborative filtering).
- **Metrics:** NDCG@10, MAP@10, MRR.

## Run it

```bash
pip install -r requirements.txt
python src/download.py     # fetches MovieLens-100k (~5 MB) into data/
python src/pipeline.py     # trains v1 + v2, evaluates, writes results/
```
Fully reproducible (fixed seed). Runs in well under a minute on a laptop, CPU-only.

## What I'd do next
- Tune the CF rank/regularization and try implicit-feedback ALS (confidence-weighted).
- Two-stage serving: candidate generation (retrieve a few hundred from millions)
  → ranking (score just those) — how this scales in production.
- Cold-start: content/embedding hybrids for brand-new users and items.

## Layout
```
src/download.py   # fetch dataset
src/pipeline.py   # features (leakage-safe) + CF embeddings → train v1/v2 → evaluate → chart
results/          # metrics.json + ranking_comparison.png
```
