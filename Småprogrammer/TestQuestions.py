#Sammenligning av setninger: https://huggingface.co/tasks/sentence-similarity
#https://www.sbert.net/ transformers
#Me bruker https://huggingface.co/NbAiLab/nb-sbert-base
#Modellen er bygd opp på ein måte osm gjer at engelske setningar også kan pare bra opp mot norske setningar.

from sklearn.metrics.pairwise import cosine_similarity  #For å kunne sammenligne setninger, som cosine_similarity tydeligvis er bra til
from sentence_transformers import SentenceTransformer  #Bruke modellen på setninger
import numpy as np  #For at python skal jobbe med mattegreier

modellnavn = "NbAiLab/nb-sbert-base"  #Modellen me bruker. https://huggingface.co/NbAiLab/nb-sbert-base
modell = SentenceTransformer(modellnavn) #Instansierer BERT modellen . https://huggingface.co/docs/transformers/main_classes/model

'''
_______________________________Formater spørsmål med modellen_______________________________
'''

fasit = [] #Lista somk inneheld fasiten aka spørsmåla chatGPT har generert i teorien skal ligne mest på
spørsmål = [] #Liste med spørmsåla chatGPT har laga
søkeliste = [] #Liste som programmet søker gjennom for å finne spørsmål som ligner mest.


with open('txtandCSV-files/chatQA.txt', 'r', encoding='utf-8') as file: #Opne Q&A, les den /r) og enkoder som uft-8 (lar Æ, Ø og å vere med)
    for line in file:
        if line.startswith('Q:'):
            fasit.append(line[3:].strip())  # henter ut alle linjer son starter på q, og tek ut alt frå og med tegn 3. strip fjerner lange mellomrom og linjeskift.

with open('txtandCSV-files/chatQA.txt', 'r', encoding='utf-8') as file: #Opne Q&A, les den /r) og enkoder som uft-8 (lar Æ, Ø og å vere med)
    for line in file:
        if line.startswith('I:'):
            spørsmål.append(line[3:].strip())  # henter ut alle linjer son starter på q, og tek ut alt frå og med tegn 3. strip fjerner lange mellomrom og linjeskift.

with open('txtandCSV-files/Q&A.txt', 'r', encoding='utf-8') as file: #Opne Q&A, les den /r) og enkoder som uft-8 (lar Æ, Ø og å vere med)
    for line in file:
        if line.startswith('Q:'):
            søkeliste.append(line[3:].strip())  # henter ut alle linjer son starter på q, og tek ut alt frå og med tegn 3. strip fjerner lange mellomrom og linjeskift.

encoded_questions = np.loadtxt('txtandCSV-files/Q&A_embedded.csv', delimiter=',') #henter ut lista med spørsmål som alt er "encoda".
encoded_user_questions = modell.encode(spørsmål)

'''
_______________________________Sammenlign spørmsål_______________________________
'''

# Similarity scores
# https://huggingface.co/tasks/sentence-similarity "The similarity of the embeddings is evaluated mainly on cosine similarity. 
# It is calculated as the cosine of the angle between two vectors. It is particularly useful when your texts are not the same length"
# Initialize an empty list to hold the similarity scores

IandAandQ = [] #Inquiry, fasit og det modellen kom fram til.

with open('txtandCSV-files/testresults.txt', 'w', encoding='utf-8') as file:
    file.write("")

for i, question in enumerate(encoded_user_questions, start = 0):
    
    similarity_scores = cosine_similarity([question], encoded_questions)[0]
    most_similar_question_index = np.argmax(similarity_scores)
    most_similar_question = søkeliste[most_similar_question_index]
    
    inquiry = spørsmål[i] #Spørsmål generert av chatGPT.
    fasiten = fasit[i] #Fasiten, spørsmålet chatGPT har generet skal i teorien ligne mest på dette.
    løsning = most_similar_question #Spørsmålet modellen vår kjem fram til at ligner mest.

    with open('txtandCSV-files/testresults.txt', 'a', encoding='utf-8') as file:
        file.write(f"""
            Spørsmål generert: {inquiry}
            Spørsmål som ligner mest: {løsning}
            Fasit: {fasiten}
            """)
    
    nested_list = [[a, b] for a, b in zip(spørsmål, similarity_scores)]

    # Sort the nested list based on the first value of each inner list, in descending order
    sorted_nested_list = sorted(nested_list, key=lambda x: x[1], reverse=True)
    
    if (løsning != fasiten): #Gir output hvis løsningen modellen har kommet fram til ikke er lik fasiten
        print("Spørsmål generert: " + inquiry + "\n" + "løsning: " + str(max(similarity_scores)) + løsning + "\n" + "fasit: " + fasiten +"\n")
        print(sorted_nested_list)