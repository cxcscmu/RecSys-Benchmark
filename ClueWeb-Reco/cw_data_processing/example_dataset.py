from ClueWeb22Api import ClueWeb22Api
from tqdm import tqdm 


def get_clueweb_title(cwid, clueweb_path): 
    clueweb_api = ClueWeb22Api(cwid, clueweb_path)
    clean_txt = eval(clueweb_api.get_clean_text())
    content = clean_txt["Clean-Text"]
    title = content.split('\n')[0].replace("\n", "").replace("\t", "").replace("\r", "").replace("\'", "").replace("\"", "").strip()
    return title    

    
def load_data_clueweb(filename, id_map_path, clueweb_path): 
    """
        args: 
            filename: the validation set input (valid.tsv) or the test set input (test.tsv)
            id_map_path: the mapping from official clueweb id to our internal clueweb id (cwid_to_id.tsv) 
            clueweb_path: your clueweb dataset path on disk (ex. /../data/datasets/clueweb22/ClueWeb22_B)
    """

    # load mapping to official clueweb doc ids 
    id_to_cwid = {}
    with open(id_map_path, "r") as f:
        for line in f:  
            parts = line.strip().split("\t")
            id_to_cwid[int(parts[1])] = parts[0]


    data = {}
    lines = open(filename, 'r').readlines()
    for line in tqdm(lines[1:]):
        history_titles = list()
        line = line.strip().split('\t')
        
        session_id = line[0]
        history = line[1].split(",")

        for internal_id in history: 
            internal_id = int(internal_id)
            # get the title of the item by ClueWebApi 
            title = get_clueweb_title(id_to_cwid[internal_id], clueweb_path)
            history_titles.append(title)

        data[session_id] = history_titles

    return data


id_map_path="ClueWeb-Reco/cwid_to_id.tsv"
seq_data_path="ClueWeb-Reco/valid.tsv"
clueweb_path="/data/datasets/clueweb22/ClueWeb22_B"

# dictionary with session_ids as key, list of strings of the titles of historically interacted item
data = load_data_clueweb(filename=seq_data_path, id_map_path=id_map_path, clueweb_path=clueweb_path)


