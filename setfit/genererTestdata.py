#Me er for late for å lage data for å trene modellen sjølv :D
from openai import OpenAI

with open('setfit/treningsdata3.txt', 'w', encoding='utf-8') as file:
    file.write("")

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
    reformulated_questions = []
    for j in range(0,16):
        reformulted_list = '\n'.join(reformulated_questions)
        message = [
                    {"role": "user", "content": f"""
Lag 1 setninger som er en omforumlering av følgende spørsmål: {question}. Skriv korte setninger. Spørsmålet kan IKKE være likt noen av disse spørsmålene: \n\n
{reformulted_list}
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
        omformulert_spørsmål = ""
        for chunk in stream: #returnerer ein og ein token, så må loope gjennom heile streamen og legge til tokens ein og ein
            content = chunk.choices[0].delta.content
            if content:
                omformulert_spørsmål += content 
        reformulated_questions.append(omformulert_spørsmål)
        with open('setfit/treningsdata3.txt', 'a', encoding='utf-8') as file:
            file.write("text " + omformulert_spørsmål + " \n"+ "label " + question + "\n")
        i += 1