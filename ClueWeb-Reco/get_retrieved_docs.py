import requests
import base64
import json
import numpy as np
import pandas as pd
import os
from tqdm import tqdm


URL = "https://clueweb22.us/search"
API_KEY = "YOUR_API_KEY"  # TODO: Replace with your actual API key
FOLDER = "/home/karrym/capstone/RecSys-Benchmark/ClueWeb-Reco/outputs_gpt_41"
MAP_FILE = "/data/group_data/cx_group/REC/ClueWeb-Reco/ClueWeb-Reco_public/cwid_to_id.tsv"

ID_MAP = {}
with open(MAP_FILE, "r") as f:
    for line in tqdm(f, total=87208655, desc="Loading CWID->ID map"):
        cwid, id = line.strip().split("\t")
        ID_MAP[cwid] = int(id)


def write_embed_to_binary(embeddings, output_path):
    """
    Write the embedding array into a binary file in ANN-Indexing (DiskANN, SPTAG) format.
    The content of the output file can be access through: embeds = read_fbin(output_path)
    """
    num, dim = embeddings.shape
    with open(output_path, "wb") as f:
        f.write(num.to_bytes(4, "little"))
        f.write(dim.to_bytes(4, "little"))
        f.write(embeddings.tobytes())


def retrive(query, k=100):
    """
    Retrieve documents from the ClueWeb22 search API.
    Args:
        query (str): The search query.
        k (int): The number of documents to retrieve.
    Returns:
        list: A list of retrieved documents.
    """
    response = requests.get(URL, params={"query": query, "k": k}, headers={"X-API-Key": API_KEY})
    response = response.json()
    docs = []
    for item in response["results"]:
        doc = base64.b64decode(item).decode("utf-8")
        doc = json.loads(doc)
        docs.append(doc)
    return docs


def process_queries(input_file, output_file, k=100):
    """
    Process the input file containing queries and retrieve documents for each query.
    Save ClueWeb IDs of the retrieved documents as a tsv file.
    """
    if os.path.exists(output_file):
        df = pd.read_csv(output_file, sep="\t")
    else:
        df = pd.read_csv(input_file, header=None, names=["session_id", "query"], sep="\t")
        for i in range(k):
            df[f"retrieved_{i}"] = None

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing {os.path.basename(input_file)}"):
        # If the row already has retrieved documents, skip it
        if pd.notna(row.get(f"retrieved_0")):
            continue
        query = row["query"]
        while True:
            try:
                retrieved_docs = retrive(query, k)
                assert len(retrieved_docs) == k, f"Expected {k} documents, but got {len(retrieved_docs)}"
                for i, retrieved_doc in enumerate(retrieved_docs):
                    df.at[index, f"retrieved_{i}"] = retrieved_doc["ClueWeb22-ID"].strip()
                break
            except Exception as e:
                print(f"[ERROR] Query failed at index {index}: {e}")
                continue

        # Incremental save to avoid losing progress
        if index % 10 == 0:
            df.to_csv(output_file, index=False, sep="\t")

    df.to_csv(output_file, index=False, sep="\t")


def save_result(retrieved_df, output_path):
    """
    Save the retrieved documents to a binary file.
    """
    if os.path.exists(output_path):
        print(f"Binary file already exists: {output_path}, skipping.")
        return

    df = pd.read_csv(retrieved_df, sep="\t")
    results = []
    for i in range(len(df)):
        try:
            result = []
            for k in range(100):
                result.append(ID_MAP[df.iloc[i][f"retrieved_{k}"]])
            results.append(result)
        except KeyError as e:
            print(f"[WARN] Missing ID mapping at row {i}: {e}")
            continue

    results = np.array(results, dtype=np.int32)
    print("results shape: ", results.shape)
    write_embed_to_binary(results, output_path)


if __name__ == "__main__":
    for file in os.listdir(FOLDER):
        if file.endswith(".tsv") and not file.startswith("retrieved_"):
            input_file = os.path.join(FOLDER, file)
            retrieved_file = os.path.join(FOLDER, f"retrieved_{file}")
            bin_file = retrieved_file.replace(".tsv", ".bin")
            print(f"\nProcessing {input_file}")
            process_queries(input_file, retrieved_file)
            save_result(retrieved_file, bin_file)
