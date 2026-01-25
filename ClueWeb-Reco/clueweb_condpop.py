import argparse
import struct
from collections import defaultdict
import math

def read_ordered_input(path):
    """
    ordered_id_splits/*_input.tsv:
      columns: [session_id, ordered_history_cw_internal_id]
    history peut être séparé par ',' ou espace.
    Retour: list of (session_id, [int ids])
    """
    sessions = []
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline().strip()
        # skip header if it contains letters
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

def parse_input_line(line):
    parts = line.split("\t")
    if len(parts) < 2:
        raise ValueError(f"Bad line (expected 2 cols): {line[:120]}")
    sid = parts[0]
    seq_str = parts[1].strip()
    if "," in seq_str and " " not in seq_str:
        toks = [t for t in seq_str.split(",") if t]
    else:
        toks = [t for t in seq_str.replace(",", " ").split() if t]
    seq = [int(t) for t in toks]
    return sid, seq

def read_targets(path):
    """
    ordered_id_splits/valid_target.tsv:
      columns: [session_id, target_cw_internal_id]
    """
    targets = {}
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline().strip()
        if first and any(c.isalpha() for c in first):
            pass
        else:
            if first:
                sid, y = parse_target_line(first)
                targets[sid] = y

        for line in f:
            line = line.strip()
            if not line:
                continue
            sid, y = parse_target_line(line)
            targets[sid] = y
    return targets

def parse_target_line(line):
    parts = line.split("\t")
    if len(parts) < 2:
        raise ValueError(f"Bad target line: {line[:120]}")
    return parts[0], int(parts[1])

def topk_from_counts(counts_dict, k):
    return [x for x, _ in sorted(counts_dict.items(), key=lambda t: -t[1])[:k]]

def build_condpop_model(sessions, K_max):
    """
    Apprend:
      - next_counts[p][q] = freq(p->q)
      - global_counts[q]  = popularité globale
    depuis les historiques ordonnés par session.
    """
    next_counts = defaultdict(lambda: defaultdict(int))
    global_counts = defaultdict(int)

    for sid, seq in sessions:
        if len(seq) < 2:
            continue
        for i in range(len(seq) - 1):
            p, q = seq[i], seq[i + 1]
            next_counts[p][q] += 1
            global_counts[q] += 1

    top_global = topk_from_counts(global_counts, K_max)
    top_next = {p: topk_from_counts(q_counts, K_max) for p, q_counts in next_counts.items()}
    return top_next, top_global

def recommend_from_history(history, top_next, top_global, K):
    """
    Reco: top_next[last_item] sinon top_global.
    Assure K items, essaie d'éviter les doublons.
    """
    seed = history[-1] if history else None
    base = top_next.get(seed, top_global) if seed is not None else top_global

    recs = []
    seen = set()
    for x in base:
        if x not in seen:
            recs.append(x)
            seen.add(x)
        if len(recs) == K:
            break

    # pad si jamais (rare)
    if len(recs) < K:
        for x in top_global:
            if x not in seen:
                recs.append(x)
                seen.add(x)
            if len(recs) == K:
                break
    if len(recs) < K:
        recs += [0] * (K - len(recs))
    return recs

def write_bin(path, K, predictions):
    """
    Format ORBIT:
      int32 num_sessions
      int32 K
      then num_sessions*K int32
    """
    num_sessions = len(predictions)
    with open(path, "wb") as f:
        f.write(struct.pack("<i", num_sessions))
        f.write(struct.pack("<i", K))
        for recs in predictions:
            if len(recs) != K:
                raise ValueError("Each rec list must have length K")
            for x in recs:
                f.write(struct.pack("<i", int(x)))

def dcg_at_k(rels):
    s = 0.0
    for i, r in enumerate(rels, start=1):
        if r:
            s += 1.0 / math.log2(i + 1)
    return s

def eval_on_valid(valid_sessions, valid_targets, top_next, top_global, Ks):
    """
    valid_targets: dict session_id -> true_item
    """
    res = {k: {"hit": 0, "ndcg": 0.0, "n": 0} for k in Ks}

    for sid, hist in valid_sessions:
        if sid not in valid_targets:
            continue
        y = valid_targets[sid]
        for k in Ks:
            recs = recommend_from_history(hist, top_next, top_global, k)
            hit = 1 if y in recs else 0
            rels = [1 if x == y else 0 for x in recs]
            ndcg = dcg_at_k(rels)  # IDCG=1 pour 1 item pertinent
            res[k]["hit"] += hit
            res[k]["ndcg"] += ndcg
            res[k]["n"] += 1

    print("\n=== ClueWeb-Reco Conditional Popularity (VALID) ===")
    for k in Ks:
        n = res[k]["n"]
        recall = res[k]["hit"] / n if n else 0.0
        ndcg = res[k]["ndcg"] / n if n else 0.0
        print(f"Recall@{k}: {recall:.4f} | NDCG@{k}: {ndcg:.4f} | sessions={n}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_input", required=True, help="ordered_id_splits/valid_input.tsv (utilisé pour apprendre)")
    ap.add_argument("--valid_input", default=None, help="ordered_id_splits/valid_input.tsv (pour évaluer)")
    ap.add_argument("--valid_target", default=None, help="ordered_id_splits/valid_target.tsv (GT)")
    ap.add_argument("--test_input", default=None, help="ordered_id_splits/test_input.tsv (pour soumission)")
    ap.add_argument("--Ks", default="10,50,100", help="K pour métriques, ex: 10,50,100")
    ap.add_argument("--submit_K", type=int, default=100, help="K exigé par ORBIT pour le .bin")
    ap.add_argument("--out_bin", default="submission.bin", help="nom du fichier .bin à générer")
    args = ap.parse_args()

    Ks = [int(x.strip()) for x in args.Ks.split(",") if x.strip()]
    K_max = max(max(Ks), args.submit_K)

    train_sessions = read_ordered_input(args.train_input)
    top_next, top_global = build_condpop_model(train_sessions, K_max)
    print(f"✅ Model built: states_with_transitions={len(top_next)}, global_top_size={len(top_global)}")

    # Eval (optional)
    if args.valid_input and args.valid_target:
        valid_sessions = read_ordered_input(args.valid_input)
        valid_targets = read_targets(args.valid_target)
        eval_on_valid(valid_sessions, valid_targets, top_next, top_global, Ks)

    # Submission bin (optional)
    if args.test_input:
        test_sessions = read_ordered_input(args.test_input)
        preds = [recommend_from_history(hist, top_next, top_global, args.submit_K) for _, hist in test_sessions]
        write_bin(args.out_bin, args.submit_K, preds)
        print(f"\n✅ Wrote bin: {args.out_bin} | num_sessions={len(preds)} | K={args.submit_K}")

if __name__ == "__main__":
    main()
