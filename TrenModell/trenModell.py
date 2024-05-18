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
data_path = 'TrenModell/Treningsdata/alpha16.txt'
model = SentenceTransformer(model_name)
Text= []

"""__________Henter dataen som skal brukes til å formatere dataen__________"""
#Tomt array som inneholder alle set med setninger som blir brukt for å trene modellen.
sets = []
#Et sett med setinger som ligner på hverandre og betyr det samme. 
set = []

#Case 2: The Sentence Compression dataset has examples made up of positive pairs. If your dataset has more than two positive sentences per example, for example quintets as in the COCO Captions or the Flickr30k Captions datasets, you can format the examples as to have different combinations of positive pairs.
#https://huggingface.co/blog/how-to-train-sentence-transformers

Text = []
Labels = []

#Litt ufnky kode for å unngå å endre på formatet til dataen :P
x=False
with open(data_path, 'r', encoding='utf-8') as file:
    for line in file:
        # Check if the line starts with 'text' and remove the first 5 characters
        if x == True:
            label = line[6:].strip()
            x = False
        elif line.startswith('text.'):
            Labels.append(label)
            Text.append(line[5:].strip())
        # Check if the line starts with 'label' and remove the first 6 characters
        if line.startswith('__________NYTT SPØRSMÅL'):
            x = True
            
merged = [[text, label] for text, label in zip(Text, Labels)]

print(merged)

dataset = merged

# Assuming 'merged' is already defined as a list of [text, label] pairs
# First, convert 'merged' into a list of dictionaries, which is a format that can be directly converted into a Dataset
texts, labels = zip(*merged)  # This separates the merged list into two lists of texts and labels
data_dict = {'text': texts, 'label': labels}

# Convert this dictionary into a Hugging Face Dataset
train_dataset = Dataset.from_dict(data_dict)

# Now, create a DatasetDict containing your train_dataset under the 'train' key
dataset = DatasetDict({'train': train_dataset})

# Your dataset is now correctly structured
train_dataB = dataset['train']
texts, labels = zip(*merged)  # This separates the merged list into two lists of texts and labels
data_dict = {'text': texts, 'label': labels}

# Convert this dictionary into a Hugging Face Dataset
train_dataset = Dataset.from_dict(data_dict)

# Now, create a DatasetDict containing your train_dataset under the 'train' key
dataset = DatasetDict({'train': train_dataset})

# Your dataset is now correctly structured
train_dataB = dataset['train']
n_examples = len(train_dataB)

train_examples = []
# Loop over all examples in the training set
for i in range(n_examples):
    # Retrieve the ith example from the training data
    example = train_dataB[i]
    
    # Create an InputExample object from the example, assuming it has two fields to be used as texts
    # and append it to the list of training examples
    train_examples.append(InputExample(texts=[example['text'], example['label']]))
    
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=64)
train_loss = losses.MultipleNegativesRankingLoss(model=model)  #https://www.sbert.net/docs/package_reference/losses.html

model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=10) 

model.save("modeller/alpha16")