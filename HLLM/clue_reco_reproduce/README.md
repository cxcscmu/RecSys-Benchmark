### ClueWeb-Reco Reproduction 

#### Item Encoding
- Use clue_reco_reproduce/HLLM_item_encode.sh to encode item embeddings by shard
- Update code/clever_merge.py for you item embedding file path and shard number
- Use clue_reco_reproduce/HLLM_merge_item_emb.sh to merge item embeddings 

#### Sequence Encoding
- Use clue_reco_reproduce/HLLM_seq_encode.sh to compute sequence embedding by first computing sequence feature input

#### Prediction
- Use clue_reco_reproduce/utils.sh to compute prediction based on cosine similarity 
