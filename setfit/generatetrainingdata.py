#Me er for late for å lage data for å trene modellen sjølv :D
from openai import OpenAI

with open('setfit/treningsdata.csv', 'w', encoding='utf-8') as file:
    file.write("text,label_text\n")

client = OpenAI(
    # API-nøkkel burde ikke vere public i de fleste scripts, men denne kjører bare lokalt og er bare delt innad i gruppen.
    api_key="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"
)

questions = []
with open('txtandCSV-files/Q&A.txt', 'r', encoding='utf-8') as file:
    for line in file:
        if line.startswith('Q:'):
            questions.append(line[3:].strip())

#Generer et spørsmål for hvert spørsmål i spørsmålslisten.
i=0
for question in questions:
    promptquestion = question
    for j in range(0,6):
        message = [
                    {"role": "user", "content": f"""
Lag 1 setninger som er en omforumlering av følgende spørsmål: {promptquestion}
"""}
        ]
        promptquestion = ""
        stream = client.chat.completions.create(
        temperature=1.0, #https://platform.openai.com/docs/api-reference/audio
        #sjekk og
        #https://community.openai.com/t/cheat-sheet-mastering-temperature-and-top-p-in-chatgpt-api/172683
        #https://aimresearch.co/leaders-opinion/leaders-opinion-how-temperature-affects-chatgpt-with-rachael-chudoba
        model="gpt-3.5-turbo-1106",
        messages=message,
        stream=True,
        )
        promptquestion = ""
        for chunk in stream: #returnerer ein og ein token, så må loope gjennom heile streamen og legge til tokens ein og ein
            content = chunk.choices[0].delta.content
            if content:
                promptquestion += content 
        with open('setfit/treningsdata.csv', 'a', encoding='utf-8') as file:
            file.write("\"" + promptquestion + "\""+ "," + question + "\n")
        i += 1