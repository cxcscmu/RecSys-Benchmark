import requests
import base64
import json
import numpy as np
import pandas as pd
import os
from tqdm import tqdm


URL = "https://clueweb22.us/search"
FOLDER = "/data/group_data/cx_group/REC/data/clueweb_outputs"
MAP_FILE = "/data/group_data/cx_group/REC/ClueWeb-Reco/ClueWeb-Reco_public/cwid_to_id.tsv"

ID_MAP = {}
with open(MAP_FILE, "r") as f:
    for line in tqdm(f, total=87208655):
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
    response = requests.get(URL, params={"query": query, "k": k})
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
    df = pd.read_csv(input_file, header=None, names=["session_id", "query"], sep="\t")

    new_cols = [f"retrieved_{i}" for i in range(k)]
    for col in new_cols:
        if col not in df.columns:
            df[col] = None

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing queries"):
        query = row["query"]
        retrieved_docs = retrive(query, k)
        assert len(retrieved_docs) == k, f"Expected {k} documents, but got {len(retrieved_docs)}"
        for i, retrieved_doc in enumerate(retrieved_docs):
            # retrieved_url = retrieved_doc["URL"].strip()
            # retrieved_text = retrieved_doc["Clean-Text"].strip()
            retrieved_docid = retrieved_doc["ClueWeb22-ID"].strip()
            df.at[index, f"retrieved_{i}"] = retrieved_docid

    df.to_csv(output_file, index=False, sep="\t")


def save_result(retrieved_df, output_path):
    """
    Save the retrieved documents to a binary file.
    """
    df = pd.read_csv(retrieved_df, sep="\t")
    results = []
    for i in range(len(df)):
        result = []
        for k in range(100):
            result.append(ID_MAP[df.iloc[i][f"retrieved_{k}"]])
        results.append(result)
    results = np.array(results, dtype=np.int32)
    print("results shape: ", results.shape)
    write_embed_to_binary(results, output_path)


if __name__ == "__main__":
    for file in os.listdir(FOLDER):
        if file.endswith(".tsv"):
            input_file = os.path.join(FOLDER, file)
            retrieved_file = os.path.join(FOLDER, f"retrieved_{file}")
            bin_file = retrieved_file.replace(".tsv", ".bin")
            print(f"Processing {input_file}, saving to {retrieved_file} and {bin_file}")
            process_queries(input_file, retrieved_file)
            save_result(retrieved_file, bin_file)
