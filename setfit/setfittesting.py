#SetFit dokumentasjon: https://huggingface.co/docs/setfit/index
#Sjekk ut feks https://huggingface.co/docs/setfit/en/quickstart og spesielt https://huggingface.co/blog/setfit
#Testing 

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sentence_transformers.losses import CosineSimilarityLoss
from sentence_transformers import SentenceTransformer
from setfit import SetFitModel, SetFitTrainer
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from setfit import TrainingArguments

data_path = 'SetFit/treningsdata.txt'
model_name = "NbAiLab/nb-sbert-base"
model = SetFitModel.from_pretrained(model_name)

data_path = 'SetFit/treningsdata.txt'
data_lines = []

with open(data_path, 'r', encoding='utf-8') as file:
    while True:
        text_line = file.readline().strip()
        label_line = file.readline().strip()
        if not text_line or not label_line:
            break  # Exit loop if end of file or if there's an empty line.
        text = text_line.split(" ", 1)[1]  # Skip the 'text' prefix.
        label = label_line.split(" ", 1)[1]  # Skip the 'label' prefix.
        data_lines.append({'text': text, 'label': label})

df = pd.DataFrame(data_lines)

# Split data into train+val and test
train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Split train+val into train and val
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
val_dataset = Dataset.from_pandas(val_df)

args = TrainingArguments(
    batch_size=32,
    num_epochs=10,
)

trainer = SetFitTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    loss_class=CosineSimilarityLoss,
    batch_size=16,
    num_iterations=20, # Number of text pairs to generate for contrastive learning
    num_epochs=1 # Number of epochs to use for contrastive learning
)

trainer.train()
model.save_pretrained('modeller/alpha1')
