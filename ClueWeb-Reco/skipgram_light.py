# skipgram_light.py
# ------------------------------------------------------------
# Ultra-light Skip-gram baseline (ORBIT option A)
# Train corpus: ordered_id_splits/valid_input.tsv
# Submit: valid_input.tsv + test_input.tsv
# Output: 6 bins (valid/test x K in {10,50,100})
# ------------------------------------------------------------

import os, struct, random
from collections import Counter
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------- IO --------------------
def parse_hist(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    if "," in s:
        return [int(x) for x in s.split(",") if x.strip()]
    return [int(x) for x in s.split() if x.strip()]

def read_ordered_histories(path: str) -> List[List[int]]:
    # header: session_id \t ordered_history_cw_internal_id
    seqs = []
    with open(path, "r", encoding="utf-8") as f:
        _ = f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                seqs.append(parse_hist(p[1]))
    return seqs

def write_bin(path: str, preds: List[List[int]], K: int):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(preds)))
        f.write(struct.pack("<i", K))
        for row in preds:
            if len(row) != K:
                raise ValueError(f"Row length {len(row)} != K={K}")
            f.write(struct.pack(f"<{K}i", *row))


# -------------------- Vocab (compact IDs) --------------------
def build_vocab(seqs_list: List[List[List[int]]]) -> Tuple[Dict[int,int], List[int], List[int]]:
    cnt = Counter()
    for seqs in seqs_list:
        for s in seqs:
            cnt.update(s)
    idx2orig = sorted(cnt.keys())
    orig2idx = {it: i for i, it in enumerate(idx2orig)}
    counts = [cnt[it] for it in idx2orig]
    return orig2idx, idx2orig, counts

def remap_seq(seq: List[int], orig2idx: Dict[int,int]) -> List[int]:
    return [orig2idx[x] for x in seq if x in orig2idx]


# -------------------- Skip-gram model --------------------
class SkipGramNeg(nn.Module):
    def __init__(self, V: int, D: int):
        super().__init__()
        self.in_emb = nn.Embedding(V, D)
        self.out_emb = nn.Embedding(V, D)
        nn.init.uniform_(self.in_emb.weight, -0.5 / D, 0.5 / D)
        nn.init.zeros_(self.out_emb.weight)

    def loss(self, center: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        v = self.in_emb(center)              # [B,D]
        u_pos = self.out_emb(pos)            # [B,D]
        u_neg = self.out_emb(neg)            # [B,N,D]
        pos_logits = (v * u_pos).sum(1)      # [B]
        neg_logits = torch.einsum("bd,bnd->bn", v, u_neg)  # [B,N]
        return -(F.logsigmoid(pos_logits) + F.logsigmoid(-neg_logits).sum(1)).mean()


def iter_pairs(seqs: List[List[int]], window: int):
    for s in seqs:
        L = len(s)
        if L < 2:
            continue
        for i in range(L):
            c = s[i]
            for j in range(max(0, i - window), min(L, i + window + 1)):
                if j != i:
                    yield c, s[j]


def train_skipgram(
    model: SkipGramNeg,
    train_seqs: List[List[int]],
    counts: List[int],
    device: str,
    epochs: int = 100,
    window: int = 5,
    neg_k: int = 10,
    lr: float = 1e-3,
    batch_size: int = 4096,
    max_pairs_per_epoch: int = 2_000_000,
    seed: int = 42,
):
    random.seed(seed)
    torch.manual_seed(seed)

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # negative sampling distribution (unigram^0.75)
    freq = torch.tensor(counts, dtype=torch.float)
    prob = (freq.pow(0.75) / freq.pow(0.75).sum()).to(device)

    for ep in range(1, epochs + 1):
        model.train()
        total, steps = 0.0, 0
        bc, bp = [], []
        seen = 0

        for c, p in iter_pairs(train_seqs, window):
            bc.append(c); bp.append(p)
            seen += 1
            if max_pairs_per_epoch and seen >= max_pairs_per_epoch:
                break

            if len(bc) >= batch_size:
                center = torch.tensor(bc, dtype=torch.long, device=device)
                pos    = torch.tensor(bp, dtype=torch.long, device=device)
                neg = torch.multinomial(prob, num_samples=len(bc) * neg_k, replacement=True).view(len(bc), neg_k)

                loss = model.loss(center, pos, neg)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                total += loss.item()
                steps += 1
                bc.clear(); bp.clear()

        if bc:
            center = torch.tensor(bc, dtype=torch.long, device=device)
            pos    = torch.tensor(bp, dtype=torch.long, device=device)
            neg = torch.multinomial(prob, num_samples=len(bc) * neg_k, replacement=True).view(len(bc), neg_k)

            loss = model.loss(center, pos, neg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += loss.item()
            steps += 1

        print(f"Epoch {ep:03d} | avg_loss={total / max(1, steps):.4f} | pairs~{seen:,}")


@torch.no_grad()
def recommend_bins(
    model: SkipGramNeg,
    seqs: List[List[int]],            # compact IDs
    idx2orig: List[int],              # compact->orig internal IDs
    device: str,
    K_list=(10, 50, 100),
    last_m: int = 10,
    avoid_seen: bool = True,
    batch_size: int = 512,
) -> Dict[int, List[List[int]]]:
    model.eval()
    Kmax = max(K_list)
    out = {K: [] for K in K_list}

    M = model.in_emb.weight.to(device)  # [V,D]
    V = M.size(0)

    for start in range(0, len(seqs), batch_size):
        batch = seqs[start:start + batch_size]

        sess_vecs = []
        seen_sets = []
        for s in batch:
            seen_sets.append(set(s) if avoid_seen else set())
            if not s:
                sess_vecs.append(torch.zeros(M.size(1), device=device))
                continue
            tail = s[-last_m:] if len(s) > last_m else s
            sess_vecs.append(M[torch.tensor(tail, device=device)].mean(0))

        S = torch.stack(sess_vecs, 0)    # [B,D]
        scores = S @ M.t()               # [B,V]

        if avoid_seen:
            for i, seen in enumerate(seen_sets):
                for it in seen:
                    if 0 <= it < V:
                        scores[i, it] = -1e9

        top = torch.topk(scores, k=Kmax, dim=1).indices.cpu().tolist()

        for row in top:
            row_orig = [idx2orig[i] for i in row]
            for K in K_list:
                out[K].append(row_orig[:K])

    return out


# -------------------- MAIN --------------------
def main():
    valid_path = "ordered_id_splits/valid_input.tsv"
    test_path  = "ordered_id_splits/test_input.tsv"
    out_dir    = "outputs_skipgram_light"
    K_list     = (10, 50, 100)

    # hyperparams (simple)
    D = 64
    epochs = 100
    window = 5
    neg_k = 10
    lr = 1e-3
    max_pairs_per_epoch = 2_000_000

    last_m = 10
    avoid_seen = True

    random.seed(42)
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # load
    valid_orig = read_ordered_histories(valid_path)
    test_orig  = read_ordered_histories(test_path) if os.path.exists(test_path) else None
    print("valid sessions:", len(valid_orig), "| test sessions:", (len(test_orig) if test_orig else 0))

    # vocab (include test for OOV safety)
    corpora = [valid_orig]
    if test_orig is not None:
        corpora.append(test_orig)
    orig2idx, idx2orig, counts = build_vocab(corpora)
    print("vocab size:", len(idx2orig))

    # remap
    train = [remap_seq(s, orig2idx) for s in valid_orig if len(s) >= 2]   # TRAIN CORPUS = valid histories
    valid = [remap_seq(s, orig2idx) for s in valid_orig]
    test  = [remap_seq(s, orig2idx) for s in test_orig] if test_orig is not None else None
    print("train usable sessions:", len(train))

    # train
    model = SkipGramNeg(V=len(idx2orig), D=D)
    train_skipgram(
        model, train_seqs=train, counts=counts, device=device,
        epochs=epochs, window=window, neg_k=neg_k, lr=lr,
        batch_size=4096, max_pairs_per_epoch=max_pairs_per_epoch
    )

    # valid bins
    predsV = recommend_bins(model, valid, idx2orig, device, K_list=K_list, last_m=last_m, avoid_seen=avoid_seen)
    for K in K_list:
        write_bin(os.path.join(out_dir, f"skipgram_valid_K{K}.bin"), predsV[K], K)
        print("wrote", f"skipgram_valid_K{K}.bin")

    # test bins
    if test is not None:
        predsT = recommend_bins(model, test, idx2orig, device, K_list=K_list, last_m=last_m, avoid_seen=avoid_seen)
        for K in K_list:
            write_bin(os.path.join(out_dir, f"skipgram_test_K{K}.bin"), predsT[K], K)
            print("wrote", f"skipgram_test_K{K}.bin")

    print("done ->", out_dir)


if __name__ == "__main__":
    main()
