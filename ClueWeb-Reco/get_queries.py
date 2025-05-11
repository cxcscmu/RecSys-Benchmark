import os
import time
import openai
from tqdm import tqdm

from cw_data_processing.ClueWeb22Api import ClueWeb22Api

# ========== CONFIG ==========
DATA_ROOT = "/data/group_data/cx_group/REC/ClueWeb-Reco/ClueWeb-Reco_public"
ID_MAP_PATH = os.path.join(DATA_ROOT, "cwid_to_id.tsv")
VALID_PATH = os.path.join(DATA_ROOT, "ordered_id_splits", "valid_input.tsv")
TEST_PATH = os.path.join(DATA_ROOT, "ordered_id_splits", "test_input.tsv")
CLUEWEB_PATH = "/data/datasets/clueweb22/ClueWeb22_B"
OUTPUT_DIR = "outputs_turbo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== LOAD ID MAPPING ==========
def load_id_mapping(id_map_path):
    id_to_cwid = {}
    with open(id_map_path, "r") as f:
        for line in f:
            official_id, internal_id = line.strip().split("\t")
            id_to_cwid[int(internal_id)] = official_id
    return id_to_cwid

# ========== RETRIEVE TITLES ==========
def get_clueweb_title(cwid, clueweb_path):
    clueweb_api = ClueWeb22Api(cwid, clueweb_path)
    try:
        clean_txt = eval(clueweb_api.get_clean_text())
        content = clean_txt["Clean-Text"]
        title = content.split('\n')[0].replace("\n", "").replace("\t", "").strip()
        return title
    except Exception as e:
        print(f"Error retrieving title for {cwid}: {e}")
        return ""

# ========== LOAD SEQUENCE DATA ==========
def load_data(file_path, id_to_cwid, clueweb_path):
    data = {}
    with open(file_path, "r") as f:
        lines = f.readlines()[1:]  # skip header
        for line in tqdm(lines, desc=f"Loading {os.path.basename(file_path)}"):
            session_id, history_str = line.strip().split("\t")
            history_ids = [int(x) for x in history_str.split(",")]
            titles = [get_clueweb_title(id_to_cwid[i], clueweb_path) for i in history_ids]
            data[session_id] = titles
    return data
# ========== PROMPT CONSTRUCTION ==========
def generate_prompt(history_titles):
    joined_titles = "; ".join(history_titles)
    prompt = f"""
    You are an expert search assistant. A user has visited the following sequence of pages:
    {joined_titles}
    Your task is to infer the user's likely **next intent** based on this browsing history and generate a **concise search query** that represents what they would search for next.
    👉 Important:
    - Do **not** copy or rephrase the titles.
    - **Infer** what the user is looking for, even if it's not explicitly mentioned.
    - Your output should be **only the final search query**. No bullet points, no explanation, no formatting.
    ---
    🔹 Example 1:
    Browsing history: 
    "iPhone 15 Pro Max review"; "best phone cameras 2024"; "Samsung Galaxy S24 specs"
    Search query:
    flagship phone camera comparison
    ---
    🔹 Example 2:
    Browsing history: 
    "how to brew coffee at home"; "aeropress vs french press"; "best coffee beans for espresso"
    Search query:
    best espresso beans for home brewing
    ---
    Now generate a search query based on the browsing history above:
    """
    return prompt.strip()

# ========== OPENAI CLIENT ==========
print("Reading API key...")
api_key = os.getenv("RTGIS")
if not api_key:
    raise ValueError("Set your API key using: export RTGIS='your-key'")
openai.api_key = api_key

def query_openai(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", #Choose "gpt-3.5-turbo" or "gpt-4o" 
            messages=[
                {"role": "system", "content": "You are a helpful assistant that writes search queries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
	return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI error: {e}")
        return ""

# ========== GEMINI CLIENT ==========
print("Reading API key from environment variable 'GEMINI_API_KEY'...")
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Set your Gemini API key using: export GEMINI_API_KEY='your-key'")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
print("✅ Gemini client initialized.")

def query_gemini(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini error: {e}")
        return ""

# ========== MAIN GENERATION FUNCTION ==========
def generate_queries_for_split(split_name, split_path):
    print(f"\nProcessing split: {split_name}")
    id_to_cwid = load_id_mapping(ID_MAP_PATH)
    session_histories = load_data(split_path, id_to_cwid, CLUEWEB_PATH)

    session_queries = {}
    for session_id, titles in tqdm(session_histories.items(), desc=f"Generating queries ({split_name})"):
        prompt = generate_prompt(titles)
        query = query_openai(prompt) #or query_gemini(prompt) to use gemini for query generation
        session_queries[session_id] = query
        time.sleep(1)

    # Save to file
    output_path = os.path.join(OUTPUT_DIR, f"{split_name}_queries.tsv")
    with open(output_path, "w") as f:
        for session_id, query in session_queries.items():
            f.write(f"{session_id}\t{query}\n")

    print(f"{split_name.capitalize()} queries saved to: {output_path}")

# ========== RUN BOTH SPLITS ==========
if __name__ == "__main__":
    generate_queries_for_split("valid", VALID_PATH)
    generate_queries_for_split("test", TEST_PATH)
