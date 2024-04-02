#SetFit dokumentasjon: https://huggingface.co/docs/setfit/index
#Sjekk ut feks https://huggingface.co/docs/setfit/en/quickstart og spesielt https://huggingface.co/blog/setfit
#https://www.sbert.net/docs/training/overview.html#training-overview 

from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sentence_transformers.losses import CosineSimilarityLoss
from sentence_transformers import InputExample, losses, SentenceTransformer
from setfit import SetFitModel, SetFitTrainer
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from setfit import TrainingArguments
from datasets import load_dataset
from torch.utils.data import DataLoader


data_path = 'SetFit/treningsdata.txt'
model_name = "NbAiLab/nb-sbert-base"
modelB = SentenceTransformer(model_name)

data_path = 'SetFit/treningsdata.txt'
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

merged = [[text, label] for text, label in zip(Text, Labels)]

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

train_examplesB = []
# Loop over all examples in the training set
for i in range(n_examples):
    # Retrieve the ith example from the training data
    example = train_dataB[i]
    
    # Create an InputExample object from the example, assuming it has two fields to be used as texts
    # and append it to the list of training examples
    train_examplesB.append(InputExample(texts=[example['text'], example['label']]))
    
train_dataloaderB = DataLoader(train_examplesB, shuffle=True, batch_size=64)
train_lossB = losses.MultipleNegativesRankingLoss(model=modelB)  #https://www.sbert.net/docs/package_reference/losses.html
num_epochsB = 10
warmup_stepsB = int(len(train_dataloaderB) * num_epochsB * 0.1) #10% of train data

modelB.fit(train_objectives=[(train_dataloaderB, train_lossB)],
          epochs=num_epochsB,
          warmup_steps=warmup_stepsB) 

modelB.save("modeller/alpha2")