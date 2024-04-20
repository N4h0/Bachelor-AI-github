#Sammenligning av setninger: https://huggingface.co/tasks/sentence-similarity
#https://www.sbert.net/ transformers
#Me bruker https://huggingface.co/NbAiLab/nb-sbert-base
#Modellen er bygd opp på ein måte osm gjer at engelske setningar også kan pare bra opp mot norske setningar.

import argparse  #For å kunne køyre programmet med user input i terminalen
from sklearn.metrics.pairwise import cosine_similarity  #For å kunne sammenligne setninger, som cosine_similarity tydeligvis er bra til
from sentence_transformers import SentenceTransformer  #Bruke modellen på setninger
import numpy as np  #For at python skal jobbe med mattegreier
import json
from setfit import SetFitModel


modellnavn = "NbAiLab/nb-sbert-base"  #Modellen me bruker. https://huggingface.co/NbAiLab/nb-sbert-base
sbertbase = SentenceTransformer(modellnavn) #Instansierer BERT modellen . https://huggingface.co/docs/transformers/main_classes/model
modell3 = SetFitModel.from_pretrained("modeller/alpha2")

'''
_______________________________Kunne køyre i terminal med custom input_______________________________
'''

standardspørsmål = "Hei, kan du hjelpe meg?"

parser = argparse.ArgumentParser(description='Et program som tar en input og sammenligner med en liste av spørmsål.')
#E gjer at input ikkje er påkrevd, slik at programmet kan køyrast både frå komandlinje enten med eller uten argument, og direkte i VSC.
parser.add_argument('input', nargs='?', type=str, help='Spørsmål som skal bli samenlignet med liste med spørsmål', default=standardspørsmål)
parser.add_argument('--model', type=str, help='Which model do you want to use? Options: sbertbase, alpha', default='sbertbase')


args = parser.parse_args()
user_question = args.input

if args.model == '1':
    modell = sbertbase
elif args.model == '2':
    modell = modell3
else:
    modell = sbertbase



'''
_______________________________Formater spørsmål med modellen_______________________________
'''

spørsmål = [] #Tom liste

with open('txtandCSV-files/Q&A.txt', 'r', encoding='utf-8') as file: #Opne Q&A, les den /r) og enkoder som uft-8 (lar Æ, Ø og å vere med)
    for line in file:
        if line.startswith('Q:'):
            spørsmål.append(line[3:].strip())  # henter ut alle linjer son starter på q, og tek ut alt frå og med tegn 3. strip fjerner lange mellomrom og linjeskift.

# Load the structure from JSON
with open('txtandCSV-files/Q&A_embedded.json', 'r', encoding='utf-8') as file:
    loaded_list_as_lists = json.load(file)

# Convert lists back to NumPy arrays if necessary
def convert_to_arrays(loaded_list_as_lists):
    return [np.array(sublist) for sublist in loaded_list_as_lists]

print(len(loaded_list_as_lists))

loaded_list = [convert_to_arrays(sublist) for sublist in loaded_list_as_lists]

encoded_user_question = modell.encode([args.input])[0]

'''
_______________________________Sammenlign spørmsål_______________________________
'''

# Similarity scores
# https://huggingface.co/tasks/sentence-similarity "The similarity of the embeddings is evaluated mainly on cosine similarity. 
# It is calculated as the cosine of the angle between two vectors. It is particularly useful when your texts are not the same length"
# Initialize an empty list to hold the similarity scores

similarity_scores = []

for sublist in loaded_list:
    similarity_scores.append(max(cosine_similarity([encoded_user_question], sublist)[0]))

# Identify the most similar question
most_similar_question_index = np.argmax(similarity_scores)
most_similar_question = spørsmål[most_similar_question_index]

'''
_______________________________Printing og formattering_______________________________
'''

print("Spørsmål gitt:", user_question)
print("Spørsmål som ligner mest:", most_similar_question)

nested_list = [[a, b] for a, b in zip(spørsmål, similarity_scores)]

# Sort the nested list based on the first value of each inner list, in descending order
sorted_nested_list = sorted(nested_list, key=lambda x: x[1], reverse=True)

print("Spørsmål som ligner fra mest til minst:")
# Print each nested list as one line
for item in sorted_nested_list:
    print(' '.join(map(str, item)))