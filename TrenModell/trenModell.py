#SetFit dokumentasjon: https://huggingface.co/docs/setfit/index
#Sjekk ut feks https://huggingface.co/docs/setfit/en/quickstart og spesielt https://huggingface.co/blog/setfit
#https://www.sbert.net/docs/training/overview.html#training-overview og
#https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/95_Training_Sentence_Transformers.ipynb#scrollTo=iErisVnE5sCa - GOAT
#https://huggingface.co/blog/how-to-train-sentence-transformers Endå meir GOAT!

from datasets import Dataset, DatasetDict
from sentence_transformers import InputExample, losses, SentenceTransformer
from torch.utils.data import DataLoader

#https://www.sbert.net/docs/package_reference/losses.html#multiplenegativesrankingloss
#Bruker MultipleNegativesRankingLoss https://www.sbert.net/docs/package_reference/losses.html#multiplenegativesrankingloss
# "This loss expects as input a batch consisting of sentence pairs (a_1, p_1), (a_2, p_2)..., (a_n, p_n) where we assume that (a_i, p_i) are a positive pair and (a_i, p_j) for i != j a negative pair."

"""__________Hente modell og data__________"""

model_name = "NbAiLab/nb-sbert-base"
utgangsmodell = SentenceTransformer(model_name)
data_path = 'TrenModell/Treningsdata/alpha14.txt'
Text= []

"""__________Henter dataen som skal brukes til å formatere dataen__________"""
#Tomt array som inneholder alle set med setninger som blir brukt for å trene modellen.
sets = []
#Et sett med setinger som ligner på hverandre og betyr det samme. 
set = []

#Case 2: The Sentence Compression dataset has examples made up of positive pairs. If your dataset has more than two positive sentences per example, for example quintets as in the COCO Captions or the Flickr30k Captions datasets, you can format the examples as to have different combinations of positive pairs.
#https://huggingface.co/blog/how-to-train-sentence-transformers

with open(data_path, 'r', encoding='utf-8') as file:
    for line in file:
        # Check if the line starts with 'text' and remove the first 5 characters
        if line.startswith('text.'):
            set.append(line[6:].strip())
        # Check if the line starts with 'label' and remove the first 6 characters
        if line.startswith('__________NYTT SPØRSMÅL') and set !=[] or len(set) > 1:
            if len(set) > 1:
                sets.append(set)
            set = []

#Legge til siste settet. 
if set != []:
    sets.append(set)

for set in sets:
    print(set)


"""__________Formaterer treningsdataen__________"""
treningsdata = []
#Looper gjennom alle setta i lista med set av treningsdata.
for set in sets:
    for text in set:
        treningsdata.append(InputExample(texts=[text] + [other for other in set if other != text]))    

# Må konvertere dataen til ein "dataloader" for å kunne lese dataen. Dataloader enderer dataen til det formatet den må vere i for å kunne brukas til trening. 
train_dataloader = DataLoader(treningsdata, shuffle=True, batch_size=64)

# Dette er linja som trener dataen. 
utgangsmodell.fit(train_objectives=[(train_dataloader, losses.MultipleNegativesRankingLoss(model=utgangsmodell))], epochs=10)

utgangsmodell.save("modeller/alpha14") #https://discuss.huggingface.co/t/how-to-save-my-model-to-use-it-later/20568