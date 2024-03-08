from sentence_transformers import SentenceTransformer  #Bruke modellen på setninger
import numpy as np

model_name = "NbAiLab/nb-sbert-base"
model = SentenceTransformer(model_name)

questions = []
with open('txtandCSV-files/Q&A.txt', 'r', encoding='utf-8') as file:
    for line in file:
        if line.startswith('Q:'):
            questions.append(line[3:].strip())

encoded_questions = model.encode(questions)

np.savetxt("txtandCSV-files/Q&A_embedded.csv", encoded_questions, delimiter=",")