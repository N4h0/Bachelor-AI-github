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
from setfit import SetFitModel


modellnavn = "NbAiLab/nb-sbert-base"  #Modellen me bruker. https://huggingface.co/NbAiLab/nb-sbert-base
modell = SentenceTransformer(modellnavn) #Instansierer BERT modellen . https://huggingface.co/docs/transformers/main_classes/model
modell2 = SetFitModel.from_pretrained("modeller/alpha1")



'''
_______________________________Formater spørsmål med modellen_______________________________
'''

fasit = [] #Lista somk inneheld fasiten aka spørsmåla chatGPT har generert i teorien skal ligne mest på
spørsmål = [] #Liste med spørmsåla chatGPT har laga
søkeliste = [] #Liste som programmet søker gjennom for å finne spørsmål som ligner mest.

with open('txtandCSV-files/chatQA.txt', 'r', encoding='utf-8') as file: #Opne Q&A, les den og enkoder som uft-8 (lar Æ, Ø og å vere med)
    for line in file:
        if line.startswith('Q:'):
            fasit.append(line[3:].strip())  # henter ut alle linjer son starter på q, og tek ut alt frå og med tegn 3. strip fjerner lange mellomrom og linjeskift.

with open('txtandCSV-files/chatQA.txt', 'r', encoding='utf-8') as file: #Opne Q&A, les den og enkoder som uft-8 (lar Æ, Ø og å vere med)
    for line in file:
        if line.startswith('I:'):
            spørsmål.append(line[3:].strip())  # henter ut alle linjer son starter på q, og tek ut alt frå og med tegn 3. strip fjerner lange mellomrom og linjeskift.

with open('txtandCSV-files/Q&A.txt', 'r', encoding='utf-8') as file: #Opne Q&A, les den og enkoder som uft-8 (lar Æ, Ø og å vere med)
    for line in file:
        if line.startswith('Q:'):
            søkeliste.append(line[3:].strip())  # henter ut alle linjer son starter på q, og tek ut alt frå og med tegn 3. strip fjerner lange mellomrom og linjeskift.

# Load the structure from JSON
with open('txtandCSV-files/Q&A_embedded.json', 'r', encoding='utf-8') as file:
    loaded_list_as_lists = json.load(file)

# Convert lists back to NumPy arrays if necessary
def convert_to_arrays(loaded_list_as_lists):
    return [np.array(sublist) for sublist in loaded_list_as_lists]

loaded_list = [convert_to_arrays(sublist) for sublist in loaded_list_as_lists]

encoded_user_questions = modell.encode(spørsmål)
encoded_user_questions2 = modell2.encode(spørsmål)

'''
_______________________________Sammenlign spørmsål_______________________________
'''

# Similarity scores
# https://huggingface.co/tasks/sentence-similarity "The similarity of the embeddings is evaluated mainly on cosine similarity. 
# It is calculated as the cosine of the angle between two vectors. It is particularly useful when your texts are not the same length"
# Initialize an empty list to hold the similarity scores

IandAandQ = [] #Inquiry, fasit og det modellen kom fram til.
IandAandQ2 = [] #Inquiry, fasit og det modellen kom fram til.


with open('results/comparisonresults.txt', 'w', encoding='utf-8') as file:
    file.write("")

Resultat = []
Resultat2 = []
rettsvar = []
rettsvar2 = []
feilsvar =[]
feilsvar2 = []
highestWrongCoSim = None
highestWrongCoSim2 = None
lowestCorrectCoSim = None
lowestCorrectCoSim2 = None

with open('results/defaultmodelresults.txt', 'w', encoding='utf-8') as file:
    file.write("")
with open('results/trainedmodelresults.txt', 'w', encoding='utf-8') as file:
    file.write("")

#Looper gjennom heile lista med chatbpt-spørsmål
for i, (question,question2) in enumerate(zip(encoded_user_questions,encoded_user_questions2), start = 0):
    
    #Finn cosine similarity for spørsmålet
    similarity_scores = []
    similarity_scores2 = []

    for sublist in loaded_list: #Looper gjennom lista med spørsmål og finner cosine similarity scores.
        similarity_scores.append(max(cosine_similarity([question], sublist)[0]))
        similarity_scores2.append(max(cosine_similarity([question2], sublist)[0]))
    most_similar_question_index = np.argmax(similarity_scores)
    most_similar_question_index2 = np.argmax(similarity_scores2)
    most_similar_question = søkeliste[most_similar_question_index]
    most_similar_question2 = søkeliste[most_similar_question_index2]
    
    inquiry = spørsmål[i] #Spørsmål generert av chatGPT.
    fasiten = fasit[i] #Fasiten, spørsmålet chatGPT har generet skal i teorien ligne mest på dette.
    løsning = most_similar_question #Spørsmålet modellen vår kjem fram til at ligner mest.
    løsning2 = most_similar_question2
    
    #RESTEN ER BARE DATABEHANDING.
    
    #legg til resultatet i resultatsliast
    Resultat.append({
        "spørringsnummer": i+1,
        "inquiry": inquiry,
        "løsning": løsning,
        "fasit": fasiten,
        "similarity_score": max(similarity_scores)
    })
    Resultat2.append({
        "spørringsnummer": i+1,
        "inquiry": inquiry,
        "løsning": løsning,
        "fasit": fasiten,
        "similarity_score": max(similarity_scores2)
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
            
    if løsning2 != fasiten:
        feilsvar2.append({
        "spørringsnummer": i+1,
        "inquiry": inquiry,
        "løsning": løsning2,
        "fasit": fasiten,
        "similarity_score": max(similarity_scores2)
        })
        if highestWrongCoSim2 is None or max(similarity_scores2) < highestWrongCoSim2:
            highestWrongCoSim2 = max(similarity_scores2)
    else:
        rettsvar.append({
        "spørringsnummer": i+1,
        "inquiry": inquiry,
        "løsning": løsning2,
        "fasit": fasiten,
        "similarity_score": max(similarity_scores2)
        })
        if lowestCorrectCoSim2 is None or max(similarity_scores2) < lowestCorrectCoSim2:
            lowestCorrectCoSim2 = max(similarity_scores2)

    #Lagrer resultatet i ei tekstfil
    with open('results/defaultmodelresults.txt', 'a', encoding='utf-8') as file:
        file.write(f"""
            Spørsmål {i}
            Spørsmål generert: {inquiry}
            Spørsmål som ligner mest: {løsning2}
            Fasit: {fasiten}
            CoSim-Score: {max(similarity_scores2)}
            """)
        #Lagrer resultatet i ei tekstfil
    with open('results/trainedmodelresults.txt', 'a', encoding='utf-8') as file:
        file.write(f"""
            Spørsmål {i}
            Spørsmål generert: {inquiry}
            Spørsmål som ligner mest: {løsning}
            Fasit: {fasiten}
            CoSim-Score: {max(similarity_scores2)}
            """)

# Extract spørringsnummer from both lists
sporringsnummer_feilsvar = {d['spørringsnummer'] for d in feilsvar}
sporringsnummer_feilsvar2 = {d['spørringsnummer'] for d in feilsvar2}

# Find spørringsnummer unique to feilsvar
unique_to_feilsvar = sporringsnummer_feilsvar - sporringsnummer_feilsvar2

# Find spørringsnummer unique to feilsvar2
unique_to_feilsvar2 = sporringsnummer_feilsvar2 - sporringsnummer_feilsvar

# Find spørringsnummer present in both
common_to_both = sporringsnummer_feilsvar & sporringsnummer_feilsvar2


#Tekstfil som oppsummerer de viktiste resultatene når det gjelder sammenligning av modeller
with open('results/comparisonresults.txt', 'w', encoding='utf-8') as file:
    file.write(f"{str(len(feilsvar))} av {str(len(Resultat))} av spørringer ga feil resultat ved bruk av utrent modell. ")
    file.write(f"{str(len(feilsvar2))} av {str(len(Resultat2))} av spørringer ga feil resultat ved bruk av trent modell.\n")
    file.write(f"Unquique to default model: {sorted(unique_to_feilsvar)}\n")
    file.write(f"Unquique to trained model: {sorted(unique_to_feilsvar2)}\n")
    file.write(f"Common to both: {sorted(common_to_both)}")
    file.write("\n\n_____Feil unike til trent modell_____\n\n")
    for i in feilsvar2:
        if i['spørringsnummer'] in unique_to_feilsvar2:
            file.write(f"Spørringsnummer: {i['spørringsnummer']}, ")
            file.write(f"\nInquiry: {i['inquiry']}")
            file.write(f"\nLøsning: {i['løsning']}")
            file.write(f"\nFasit: {i['fasit']}")
            file.write(f"\nScore:{i['similarity_score']}")
            file.write("\n\n")
    file.write("_____Feil unike til utrent modell_____\n\n")
    for i in feilsvar:
        if i['spørringsnummer'] in unique_to_feilsvar:
            file.write(f"Spørringsnummer: {i['spørringsnummer']}, ")
            file.write(f"\nInquiry: {i['inquiry']}")
            file.write(f"\nLøsning: {i['løsning']}")
            file.write(f"\nFasit: {i['fasit']}")
            file.write(f"\nScore:{i['similarity_score']}")
            file.write("\n\n")
    file.write("_____Feil felles for begge modellene_____\n\n")
    for svar1, svar2 in zip(feilsvar, feilsvar2):
        if svar1['spørringsnummer'] in common_to_both:
            file.write(f"Spørringsnummer: {svar1['spørringsnummer']}, ")
            file.write(f"\nInquiry: {svar1['inquiry']}")
            file.write(f"\nDefaultLøsning: {svar1['løsning']}")
            file.write(f"\nTrainedLøsning: {svar2['løsning']}")
            file.write(f"\nFasit: {svar1['fasit']}")
            file.write(f"\nDefaultScore: {svar1['similarity_score']}")
            file.write(f"\nTrainedScore: {svar2['similarity_score']}")
            file.write("\n\n")
            
feilspørsmål = []    
#Tekstfil som oppsummerer de viktiste resultatene innad i hver modell.
with open('results/defaultmodelresultsummary.txt', 'w', encoding='utf-8') as file:
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
            
#Tekstfil som oppsummerer de viktiste resultatene innad i hver modell.
with open('results/trainedmodelresultssummary.txt', 'w', encoding='utf-8') as file:
    file.write(f"{str(len(feilsvar2))} av {str(len(Resultat2))} av spørringer ga feil resultat. Dette gjaldt følgende spørringer: \n")
    for i, svar in enumerate(feilsvar2):
        file.write(f"{str(svar['spørringsnummer'])}")
        if i < len(feilsvar2) - 1:
            if i == len(feilsvar2) - 2:
                file.write(" og ")
            else:
                file.write(", ")
    file.write(f"\nHøyest cosine similarity der løsning og fasit ikke er like fasit: {highestWrongCoSim2}")
    file.write(f"\nLaveste cosine similiarty der fasit og løsning er like: {lowestCorrectCoSim2}")
            
    file.write("\n\n")
    for i, svar in enumerate(feilsvar2):
        file.write(f"Spørringsnummer: {svar['spørringsnummer']}, ")
        file.write(f"\nInquiry: {svar['inquiry']}")
        file.write(f"\nLøsning: {svar['løsning']}")
        file.write(f"\nFasit: {svar['fasit']}")
        if i < len(feilsvar2)-1:
            file.write(f"\nSimilarity Score: {svar['similarity_score']}\n\n\n")
        else:
            file.write(f"\nSimilarity Score: {svar['similarity_score']}")