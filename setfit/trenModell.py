#SetFit dokumentasjon: https://huggingface.co/docs/setfit/index
#Sjekk ut feks https://huggingface.co/docs/setfit/en/quickstart og spesielt https://huggingface.co/blog/setfit
#https://www.sbert.net/docs/training/overview.html#training-overview og
#https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/95_Training_Sentence_Transformers.ipynb#scrollTo=iErisVnE5sCa - GOAT
#https://huggingface.co/blog/how-to-train-sentence-transformers Endå meir GOAT!

from datasets import Dataset, DatasetDict
from sentence_transformers import InputExample, losses, SentenceTransformer
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader

#https://www.sbert.net/docs/package_reference/losses.html#multiplenegativesrankingloss
#Bruker MultipleNegativesRankingLoss https://www.sbert.net/docs/package_reference/losses.html#multiplenegativesrankingloss
# "This loss expects as input a batch consisting of sentence pairs (a_1, p_1), (a_2, p_2)..., (a_n, p_n) where we assume that (a_i, p_i) are a positive pair and (a_i, p_j) for i != j a negative pair."



data_path = 'SetFit/treningsdata.txt'
model_name = "NbAiLab/nb-sbert-base"
utgangsmodell = SentenceTransformer(model_name)

data_path = 'SetFit/treningsdata3.txt'
Text= []
Labels = []

with open(data_path, 'r', encoding='utf-8') as file:
    for line in file:
        # Check if the line starts with 'text' and remove the first 5 characters
        if line.startswith('text'):
            Text.append(line[4:].strip())
        # Check if the line starts with 'label' and remove the first 6 characters
        elif line.startswith('label'):
            Labels.append(line[5:].strip())

#Må bruke dictionary for å trene modellen.
data_dict = {'text': Text, 'label': Labels}

#Til huggingface dataset!
train_dataset = Dataset.from_dict(data_dict)

dataset = DatasetDict({'train': train_dataset})

train_dataB = dataset['train']

train_examplesB = []
for i in range(len(train_dataB)):
    example = train_dataB[i]
    
    train_examplesB.append(InputExample(texts=[example['text'], example['label']]))
    
train_dataloaderB = DataLoader(train_examplesB, shuffle=True, batch_size=64)
loss_function = losses.MultipleNegativesRankingLoss(model=utgangsmodell)  #https://www.sbert.net/docs/package_reference/losses.html
num_epochsB = 10
warmup_stepsB = int(len(train_dataloaderB) * num_epochsB * 0.1) #10% of train data

#"Remember that if you are fine-tuning an existing Sentence Transformers model (see Notebook Companion), you can directly call the fit method from it." https://huggingface.co/blog/how-to-train-sentence-transformers
utgangsmodell.fit(train_objectives=[(train_dataloaderB, loss_function)],
          epochs=num_epochsB,
          warmup_steps=warmup_stepsB) 

utgangsmodell.save("modeller/alpha6") #https://discuss.huggingface.co/t/how-to-save-my-model-to-use-it-later/20568