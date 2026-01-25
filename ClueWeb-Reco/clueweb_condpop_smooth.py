import argparse
import struct
import math
from collections import defaultdict

# ----------------------------
# IO: ClueWeb ordered format
# ----------------------------
def parse_input_line(line: str):
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 2:
        raise ValueError(f"Bad line: {line[:120]}")
    sid = parts[0]
    seq_str = parts[1].strip()
    toks = seq_str.replace(",", " ").split()
    seq = [int(t) for t in toks] if toks else []
    return sid, seq

def read_ordered_input(path: str):
    sessions = []
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline().strip()
        # skip header if any alpha
        if first and any(c.isalpha() for c in first):
            pass
        else:
            if first:
                sessions.append(parse_input_line(first))
        for line in f:
            line = line.strip()
            if not line:
                continue
            sessions.append(parse_input_line(line))
    return sessions

def read_targets(path: str):
    targets = {}
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline().strip()
        if first and any(c.isalpha() for c in first):
            pass
        else:
            if first:
                sid, y = first.split("\t")[:2]
                targets[sid] = int(y)
        for line in f:
            line = line.strip()
            if not line:
                continue
            sid, y = line.split("\t")[:2]
            targets[sid] = int(y)
    return targets

# ----------------------------
# Model: Conditional Popularity + smoothing
# ----------------------------
def build_counts(train_sessions):
    """
    Build transition counts p->q and global popularity counts.
    """
    next_counts = defaultdict(lambda: defaultdict(int))
    global_counts = defaultdict(int)

    for _, seq in train_sessions:
        if len(seq) < 2:
            continue
        for i in range(len(seq) - 1):
            p, q = seq[i], seq[i + 1]
            next_counts[p][q] += 1
            global_counts[q] += 1

    top_global = [x for x, _ in sorted(global_counts.items(), key=lambda t: -t[1])]
    return next_counts, global_counts, top_global

def recommend_condpop_smooth(history, next_counts, global_counts, top_global, K, lam=0.3, filter_seen=True):
    """
    score(q) = count(last->q) + lam * count(q)
    """
    if not history:
        return top_global[:K] if len(top_global) >= K else (top_global + [0]*(K-len(top_global)))

    last_item = history[-1]
    scores = defaultdict(float)

    # Conditional transitions
    for q, c in next_counts.get(last_item, {}).items():
        scores[q] += float(c)

    # Global backoff (smoothing)
    if lam > 0.0:
        for q, cg in global_counts.items():
            scores[q] += lam * float(cg)

    # Optionally remove already seen items
    seen = set(history) if filter_seen else set()
    for it in seen:
        scores.pop(it, None)

    if not scores:
        # fallback global
        out = [x for x in top_global if x not in seen] if filter_seen else top_global[:]
        if len(out) < K:
            out += [0] * (K - len(out))
        return out[:K]

    ranked = [x for x, _ in sorted(scores.items(), key=lambda t: -t[1])]

    # pad with global if needed
    if len(ranked) < K:
        for x in top_global:
            if x not in seen and x not in ranked:
                ranked.append(x)
            if len(ranked) == K:
                break
    if len(ranked) < K:
        ranked += [0] * (K - len(ranked))
    return ranked[:K]

# ----------------------------
# Metrics
# ----------------------------
def dcg_at_k(rels):
    s = 0.0
    for i, r in enumerate(rels, start=1):
        if r:
            s += 1.0 / math.log2(i + 1)
    return s

def eval_valid(valid_sessions, valid_targets, recommend_fn, Ks):
    res = {k: {"hit": 0, "ndcg": 0.0, "n": 0} for k in Ks}

    for sid, hist in valid_sessions:
        if sid not in valid_targets:
            continue
        y = valid_targets[sid]
        for k in Ks:
            recs = recommend_fn(hist, k)
            hit = 1 if y in recs else 0
            rels = [1 if x == y else 0 for x in recs]
            ndcg = dcg_at_k(rels)  # IDCG=1
            res[k]["hit"] += hit
            res[k]["ndcg"] += ndcg
            res[k]["n"] += 1

    print("\n=== ClueWeb-Reco CondPop + Smoothing (VALID) ===")
    for k in Ks:
        n = res[k]["n"]
        recall = res[k]["hit"] / n if n else 0.0
        ndcg = res[k]["ndcg"] / n if n else 0.0
        print(f"Recall@{k}: {recall:.4f} | NDCG@{k}: {ndcg:.4f} | sessions={n}")

# ----------------------------
# ORBIT bin writer
# ----------------------------
def write_orbit_bin(path, K, predictions):
    """
    ORBIT format:
      int32 num_sessions
      int32 K
      then num_sessions*K int32
    """
    n = len(predictions)
    with open(path, "wb") as f:
        f.write(struct.pack("<i", n))
        f.write(struct.pack("<i", K))
        for recs in predictions:
            if len(recs) != K:
                raise ValueError(f"Each rec list must have length K={K}")
            for x in recs:
                f.write(struct.pack("<i", int(x)))

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_input", required=True, help="ordered_id_splits/valid_input.tsv (used to learn counts)")
    ap.add_argument("--valid_input", default=None, help="ordered_id_splits/valid_input.tsv")
    ap.add_argument("--valid_target", default=None, help="ordered_id_splits/valid_target.tsv")
    ap.add_argument("--test_input", default=None, help="ordered_id_splits/test_input.tsv")
    ap.add_argument("--Ks", default="10,50,100", help="metrics Ks, e.g. 10,50,100")
    ap.add_argument("--submit_K", type=int, default=100, help="K in the output .bin")
    ap.add_argument("--lambda_backoff", type=float, default=0.3, help="smoothing lambda for global popularity")
    ap.add_argument("--out_bin", default="submission.bin", help="output .bin file")
    ap.add_argument("--no_filter_seen", action="store_true", help="do not remove already seen items")
    args = ap.parse_args()

    Ks = [int(x.strip()) for x in args.Ks.split(",") if x.strip()]
    train_sessions = read_ordered_input(args.train_input)

    next_counts, global_counts, top_global = build_counts(train_sessions)
    print(f"✅ Built counts: states={len(next_counts)} | global_items={len(global_counts)} | lambda={args.lambda_backoff}")

    def rec_fn(hist, k):
        return recommend_condpop_smooth(
            hist, next_counts, global_counts, top_global,
            K=k,
            lam=args.lambda_backoff,
            filter_seen=not args.no_filter_seen
        )

    # VALID evaluation (optional)
    if args.valid_input and args.valid_target:
        valid_sessions = read_ordered_input(args.valid_input)
        valid_targets = read_targets(args.valid_target)
        eval_valid(valid_sessions, valid_targets, rec_fn, Ks)

    # TEST submission (optional)
    if args.test_input:
        test_sessions = read_ordered_input(args.test_input)
        preds = [rec_fn(hist, args.submit_K) for _, hist in test_sessions]
        write_orbit_bin(args.out_bin, args.submit_K, preds)
        print(f"\n✅ Wrote bin: {args.out_bin} | num_sessions={len(preds)} | K={args.submit_K}")

if __name__ == "__main__":
    main()
