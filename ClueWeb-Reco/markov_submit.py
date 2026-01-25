import os
import struct
from collections import defaultdict, Counter

# ---------------- IO ----------------
def read_interactions(path):
    """
    TSV avec header, colonnes:
      session_id \t cw_internal_id \t timestamp
    Retour: dict sid -> liste d'items (ordre du fichier).
    (Si besoin d'ordre temporel strict, il faut trier par timestamp,
     mais souvent le fichier est déjà ordonné.)
    """
    sess = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        _ = f.readline()  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            sid = parts[0]
            iid = int(parts[1])
            sess[sid].append(iid)
    return sess

def parse_hist(s: str):
    s = s.strip()
    if not s:
        return []
    if "," in s:
        return [int(x) for x in s.split(",") if x.strip()]
    return [int(x) for x in s.split() if x.strip()]

def read_ordered_tsv(path):
    """
    TSV avec header, colonnes:
      session_id \t ordered_history_cw_internal_id
    Retour: list[list[int]] dans l'ordre du fichier.
    """
    sessions = []
    with open(path, "r", encoding="utf-8") as f:
        _ = f.readline()  # header
        for line in f:
            sid, hist = line.rstrip("\n").split("\t")
            sessions.append(parse_hist(hist))
    return sessions

def write_bin(out_path, all_preds, K):
    """
    Format binaire:
      uint32 n_sessions
      uint32 K
      puis n_sessions * K int32
    """
    n = len(all_preds)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", n))
        f.write(struct.pack("<I", K))
        for row in all_preds:
            if len(row) != K:
                raise ValueError(f"Row length {len(row)} != K={K}")
            for x in row:
                f.write(struct.pack("<i", int(x)))

# ---------------- Model: Markov + Popularity ----------------
def train_markov_and_pop(train_sessions_dict):
    """
    train_sessions_dict: dict sid -> list[int]
    Retour:
      next_counts: dict last_item -> Counter(next_item)
      pop: Counter(item)
    """
    next_counts = defaultdict(Counter)
    pop = Counter()

    for _sid, seq in train_sessions_dict.items():
        # popularité
        for x in seq:
            pop[x] += 1
        # transitions markov
        for a, b in zip(seq[:-1], seq[1:]):
            next_counts[a][b] += 1

    return next_counts, pop

def predict_markov(seqs, next_counts, pop_list, K, avoid_seen=True):
    """
    seqs: list of sequences (valid_input/test_input)
    next_counts: transitions
    pop_list: items triés par popularité (desc)
    """
    preds = []

    for seq in seqs:
        used = set(seq) if avoid_seen else set()
        row = []

        # 1) Markov depuis le dernier item
        if seq:
            last = seq[-1]
            if last in next_counts:
                for item, _c in next_counts[last].most_common():
                    if item not in used:
                        row.append(item)
                    if len(row) == K:
                        break

        # 2) Fallback popularité
        if len(row) < K:
            for item in pop_list:
                if item not in used and item not in row:
                    row.append(item)
                if len(row) == K:
                    break

        # 3) Sécurité (padding)
        if len(row) < K:
            row += [0] * (K - len(row))

        preds.append(row)

    return preds

# ---------------- Main ----------------
if __name__ == "__main__":
    # Chemins relatifs (lance le script depuis: .../ClueWeb-Reco/)
    train_path = os.path.join("interaction_splits", "valid_inter_input.tsv")
    valid_path = os.path.join("ordered_id_splits", "valid_input.tsv")
    test_path  = os.path.join("ordered_id_splits", "test_input.tsv")

    out_dir = "outputs_markov"
    Ks = [10, 50, 100]

    # 1) "Entraînement" (comptages Markov + popularité)
    train_sessions = read_interactions(train_path)
    next_counts, pop = train_markov_and_pop(train_sessions)
    pop_list = [i for i, _ in pop.most_common()]

    # 2) VALID .bin
    valid_seqs = read_ordered_tsv(valid_path)
    for K in Ks:
        pred_valid = predict_markov(valid_seqs, next_counts, pop_list, K, avoid_seen=True)
        out = os.path.join(out_dir, f"markov_pop_valid_K{K}.bin")
        write_bin(out, pred_valid, K)
        print(" wrote", out, "n=", len(pred_valid), "K=", K)

    # 3) TEST .bin
    # (pas de target pour test)
    if os.path.exists(test_path):
        test_seqs = read_ordered_tsv(test_path)
        for K in Ks:
            pred_test = predict_markov(test_seqs, next_counts, pop_list, K, avoid_seen=True)
            out = os.path.join(out_dir, f"markov_pop_test_K{K}.bin")
            write_bin(out, pred_test, K)
            print(" wrote", out, "n=", len(pred_test), "K=", K)
    else:
        print("ℹ test_input.tsv introuvable -> seulement les .bin de validation ont été générés.")
