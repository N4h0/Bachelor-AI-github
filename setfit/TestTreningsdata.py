#Sammenligning av setninger: https://huggingface.co/tasks/sentence-similarity
#https://www.sbert.net/ transformers
#Me bruker https://huggingface.co/NbAiLab/nb-sbert-base
#Modellen er bygd opp på ein måte osm gjer at engelske setningar også kan pare bra opp mot norske setningar.

from sklearn.metrics.pairwise import cosine_similarity  #For å kunne sammenligne setninger, som cosine_similarity tydeligvis er bra til
from sentence_transformers import SentenceTransformer  #Bruke modellen på setninger
import numpy as np  #For at python skal jobbe med mattegreier
import pandas as pd
import matplotlib.pyplot as plt
import json
from setfit import SetFitModel #For å kunne teste med en modell treng med setFit

modellnavn = "NbAiLab/nb-sbert-base"  #Modellen me bruker. https://huggingface.co/NbAiLab/nb-sbert-base
#modell = SetFitModel.from_pretrained("modeller/alpha2") #Instansierer BERT modellen . https://huggingface.co/docs/transformers/main_classes/model

#modell = SetFitModel.from_pretrained("modeller/alpha1") #For å teste med en trent modell.
modell = SentenceTransformer(modellnavn) #For å teste utrent modell
modell = SetFitModel.from_pretrained("modeller/alpha5")
'''
_______________________________Formater spørsmål med modellen_______________________________
'''

fasit = [] #Lista somk inneheld fasiten aka spørsmåla chatGPT har generert i teorien skal ligne mest på
spørsmål = [] #Liste med spørmsåla chatGPT har laga
søkeliste = [] #Liste som programmet søker gjennom for å finne spørsmål som ligner mest.

with open('SetFit/treningsdata3.txt', 'r', encoding='utf-8') as file: #Opne Q&A, les den og enkoder som uft-8 (lar Æ, Ø og å vere med)
    for line in file:
        if line.startswith('text:'):
            spørsmål.append(line[5:].strip())  # henter ut alle linjer son starter på q, og tek ut alt frå og med tegn 3. strip fjerner lange mellomrom og linjeskift.
        elif line.startswith('label'):
            fasit.append(line[6:].strip())

with open('txtandCSV-files/Q&A.txt', 'r', encoding='utf-8') as file: #Opne Q&A, les den og enkoder som uft-8 (lar Æ, Ø og å vere med)
    for line in file:
        if line.startswith('Q:'):
            søkeliste.append(line[3:].strip())  # henter ut alle linjer son starter på q, og tek ut alt frå og med tegn 3. strip fjerner lange mellomrom og linjeskift.

# Load the structure from JSON
with open('txtandCSV-files/Q&A_embedded.json', 'r', encoding='utf-8') as file:
    c = json.load(file)

# Convert lists back to NumPy arrays if necessary
def convert_to_arrays(loaded_list_as_lists):
    return [np.array(sublist) for sublist in loaded_list_as_lists]

loaded_list = [convert_to_arrays(sublist) for sublist in c]

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

Resultat = []
rettsvar = []
feilsvar =[]
highestWrongCoSim = None
lowestCorrectCoSim = None

#Looper gjennom heile lista med chatbpt-spørsmål
for i, question in enumerate(encoded_user_questions, start = 0):
    
    #Finn cosine similarity for spørsmålet
    similarity_scores = []

    for sublist in loaded_list:
        similarity_scores.append(max(cosine_similarity([question], sublist)[0]))
    most_similar_question_index = np.argmax(similarity_scores)
    most_similar_question = søkeliste[most_similar_question_index]
    
    inquiry = spørsmål[i] #Spørsmål generert av chatGPT.
    fasiten = fasit[i] #Fasiten, spørsmålet chatGPT har generet skal i teorien ligne mest på dette.
    løsning = most_similar_question #Spørsmålet modellen vår kjem fram til at ligner mest.
    
    #RESTEN ER BARE DATABEHANDING.
    
    #legg til resultatet i resultatsliast
    Resultat.append({
        "spørringsnummer": i+1,
        "inquiry": inquiry,
        "løsning": løsning,
        "fasit": fasiten,
        "similarity_score": max(similarity_scores)
    })
    #Hvis løsningen er feil blir det lagt til i ei eigen liste (i tillegg til listen med alle resultat).
    if løsning != fasiten:
        feilsvar.append({
        "spørringsnummer": i+1,
        "inquiry": inquiry,
        "løsning": løsning,
        "fasit": fasiten,
        "similarity_score": max(similarity_scores)
        })
        if highestWrongCoSim is None or max(similarity_scores) < highestWrongCoSim:
            highestWrongCoSim = max(similarity_scores)
    else:
        rettsvar.append({
        "spørringsnummer": i+1,
        "inquiry": inquiry,
        "løsning": løsning,
        "fasit": fasiten,
        "similarity_score": max(similarity_scores)
        })
        if lowestCorrectCoSim is None or max(similarity_scores) < lowestCorrectCoSim:
            lowestCorrectCoSim = max(similarity_scores)

    #Lagrer resultatet i ei tekstfil
    with open('txtandCSV-files/testresults.txt', 'a', encoding='utf-8') as file:
        file.write(f"""
            Spørsmål {i}
            Spørsmål generert: {inquiry}
            Spørsmål som ligner mest: {løsning}
            Fasit: {fasiten}
            CoSim-Score: {max(similarity_scores)}
            """)

#Tekstfil som oppsummerer de viktiste resultatene. 
with open('txtandCSV-files/testresultsoppsumert.txt', 'w', encoding='utf-8') as file:
    file.write(f"{str(len(feilsvar))} av {str(len(Resultat))} av spørringer ga feil resultat. Dette gjaldt følgende spørringer: \n")
    for i, svar in enumerate(feilsvar):
        file.write(f"{str(svar['spørringsnummer'])}")
        if i < len(feilsvar) - 1:
            if i == len(feilsvar) - 2:
                file.write(" og ")
            else:
                file.write(", ")
    file.write(f"\nHøyest cosine similarity der løsning og fasit ikke er like fasit: {highestWrongCoSim}")
    file.write(f"\nLaveste cosine similiarty der fasit og løsning er like: {lowestCorrectCoSim}")
            
    file.write("\n\n")
    for i, svar in enumerate(feilsvar):
        file.write(f"Spørringsnummer: {svar['spørringsnummer']}, ")
        file.write(f"\nInquiry: {svar['inquiry']}")
        file.write(f"\nLøsning: {svar['løsning']}")
        file.write(f"\nFasit: {svar['fasit']}")
        if i < len(feilsvar)-1:
            file.write(f"\nSimilarity Score: {svar['similarity_score']}\n\n\n")
        else:
            file.write(f"\nSimilarity Score: {svar['similarity_score']}")

df = pd.DataFrame(feilsvar)

print(df)

resultat_scores = [item['similarity_score'] for item in Resultat]
rettsvar_scores = [item['similarity_score'] for item in rettsvar]
feilsvar_scores = [item['similarity_score'] for item in feilsvar]


# Creating the boxplots
fig, ax = plt.subplots()
ax.boxplot([resultat_scores, rettsvar_scores, feilsvar_scores], labels=['Resultat', 'Rett svar', 'Feil svar'])
ax.set_title('Boxplot av Resultat, Rett svar, og Feil svar')
ax.set_ylabel('Cosine Similarity Scores')
plt.savefig("figures/result-boxplot")