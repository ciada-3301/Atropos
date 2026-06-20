import json
with open("./data/sequences.json") as f:
    records = json.load(f)
sample = records[11]
print(sample["sequence"])
print(f"genus={sample.get('genus_name')}, family={sample.get('family_name')}, phylum={sample.get('phylum_name')}")