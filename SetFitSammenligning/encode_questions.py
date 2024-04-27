from sentence_transformers import SentenceTransformer  #Bruke modellen på setninger
import json
from setfit import SetFitModel #For å kunne teste med en modell treng med setFit

def embedOrdbok(setfitmodel):
    model_name = "NbAiLab/nb-sbert-base"
    model = SentenceTransformer(model_name)
    model2 = SetFitModel.from_pretrained(setfitmodel)
    spørsmål = []
    sublist = []
    with open('txtandCSV-files/Q&A.txt', 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith('Q:'):
                if sublist:
                    spørsmål.append(sublist)
                sublist = [line[3:].strip()]
            elif line.startswith('AF'):
                sublist.append(line[3:].strip())

    spørsmål.append(sublist)

    encoded_list = []
    encoded_list2 = []

    def encode_questions(questions, model):
        encoded_questions = []
        for sub_questions in questions:
            encoded_sub_questions = [model.encode(question).tolist() for question in sub_questions]
            encoded_questions.append(encoded_sub_questions)
        return encoded_questions

    encoded_list = encode_questions(spørsmål, model)
    encoded_list2 = encode_questions(spørsmål, model2)

    # Convert NumPy arrays to lists for JSON serialization
    def convert_to_list(encoded_list):
        return [[[float(i) for i in inner_array] for inner_array in sublist] for sublist in encoded_list]

    encoded_list_as_lists = convert_to_list(encoded_list)
    encoded_list_as_lists2 = convert_to_list(encoded_list2)

    # Save the list structure as JSON

    with open('txtandCSV-files/Q&A_embedded.json', 'w', encoding='utf-8') as file:
        json.dump(encoded_list_as_lists, file, ensure_ascii=False, indent=4)

    with open('txtandCSV-files/Q&A_embeddedetFitModel.json', 'w', encoding='utf-8') as file:
        json.dump(encoded_list_as_lists2, file, ensure_ascii=False, indent=4)