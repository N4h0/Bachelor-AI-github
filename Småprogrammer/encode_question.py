from sentence_transformers import SentenceTransformer  #Bruke modellen på setninger
import json

model_name = "NbAiLab/nb-sbert-base"
model = SentenceTransformer(model_name)

spørsmål = []
sublist = []
with open('txtandCSV-files/Q&A.txt', 'r', encoding='utf-8') as file: #Opne Q&A, les den /r) og enkoder som uft-8 (lar Æ, Ø og å vere med)
    for line in file:
        if line.startswith('Q:'):
            spørsmål.append(sublist)
            sublist = []
            sublist.append(line[3:].strip())  # henter ut alle linjer son starter på q, og tek ut alt frå og med tegn 3. strip fjerner lange mellomrom og linjeskift
        elif line.startswith('AF'):
            sublist.append(line[3:].strip())

spørsmål.pop(0)

print(spørsmål)

encoded_list = []
encoded_sublist = []

for subspørsmål in spørsmål:
    encoded_sublist = []
    for spørsmålsformulering in subspørsmål:
        encoded_sublist.append(model.encode(spørsmålsformulering))
    encoded_list.append(encoded_sublist)

# Convert NumPy arrays to lists for JSON serialization
def convert_to_list(encoded_list):
    return [[[float(i) for i in inner_array] for inner_array in sublist] for sublist in encoded_list]

encoded_list_as_lists = convert_to_list(encoded_list)

# Save the list structure as JSON
with open('txtandCSV-files/Q&A_embedded.json', 'w', encoding='utf-8') as file:
    json.dump(encoded_list_as_lists, file, ensure_ascii=False, indent=4)