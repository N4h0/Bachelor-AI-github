#Me er for late for å lage data for å trene modellen sjølv :D
import openai
import os
import re

question = "*Her skal spørsmålet som skal omformuleres komme*"

prompt =    f"""Lag 10 setninger som er en omforumlering av følgende spørsmål: {question}. Spørsmålene skal brukes til å trene en AI-modell."""

reformulted_list = []

def clean_string(s):
    # Remove all numbers
    s = re.sub(r'\d+', '', s)
    # Remove the first period
    s = re.sub(r'\.', '', s, count=1)
    # Replace multiple spaces with a single space and strip leading/trailing spaces
    s = re.sub(r'\s+', ' ', s).strip()
    return s

#Gjerne bruk dene linja for å beskrive treningsdata
with open('TrenModell/Treningsdata/alpha11.txt', 'w', encoding='utf-8') as file:
    file.write("Denne treningsdataen var laget ved å bruke følgende prompt:\n")
    file.write(prompt)
    file.write("\n\n")

try:
    client = openai.Client(api_key=os.getenv('CHATGPT_API_KEY'))
except Exception as e:
    print(f"En feil skjedde ved henting av API-nøkkel: {e}")


questions = []
with open('txtandCSV-files/Q&A.txt', 'r', encoding='utf-8') as file:
    for line in file:
        if line.startswith('Q:'):
            questions.append(line[3:].strip())

#Generer et spørsmål for hvert spørsmål i spørsmålslisten.
i=0
for question in questions:
    spørsmålsliste = ""
    message = [
                {"role": "user", "content": f"""
Lag 10 setninger som er omformuleringer av følgende spørsmål: {question}. Spørsmålene skal brukes til å trene en AI-modell.
"""}
    ]
    stream = client.chat.completions.create(
    temperature=1.0, #https://platform.openai.com/docs/api-reference/audio
    #sjekk og
    #https://community.openai.com/t/cheat-sheet-mastering-temperature-and-top-p-in-chatgpt-api/172683
    #https://aimresearch.co/leaders-opinion/leaders-opinion-how-temperature-affects-chatgpt-with-rachael-chudoba
    model="gpt-4-turbo-2024-04-09", #https://platform.openai.com/docs/models
    messages=message,
    stream=True,
    )
    for chunk in stream: #returnerer ein og ein token, så må loope gjennom heile streamen og legge til tokens ein og ein
        content = chunk.choices[0].delta.content
        if content:
            spørsmålsliste += content 
    
    #Fjern alle tall
    spørsmålsliste = re.sub(r'\d+|\.\s', '', spørsmålsliste)
    #Formater outputtet med å legge til label og text. 
    spørsmålsliste = ''.join(f"text {line}\nlabel {question}\n" for line in spørsmålsliste.splitlines() if line.strip())
    #Legg til label-linje    
    #Etter hver linje det det står text må det være en linje der det bare står "Label".
    
    with open('TrenModell/Treningsdata/alpha11.txt', 'a', encoding='utf-8') as file:
        file.write(f"__________NYTT SPØRSMÅL: {question}__________\n")
        file.write(spørsmålsliste)
        file.write("\n\n")
    i += 1