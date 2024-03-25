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
prompts = [
    "Lag 1 setning som er en omformulering av følgende spørsmål. Skriv kort:",
    "Omformuler dette spørsmålet. Svar som en kort setning:",
    "Omformuler dette spørsmålet på en kort og sint måte.",
    "Omformuler dette spørsmålet kort og konsist.",
    "Du er en kunde som chatter med chatbotten til et regnskapsfirma. Skriv om følgende spørsmål, men fullstendig omformulert.",
    "Skriv om følgende spørsmål. Bruk helt andre ord.",
    "Du er en kunde som chatter med chatbotten til et regnskapsfirma. Du ønsker svar på følgende spørsmål, men skriv kort og konsist som om du snakker med en chatbot.",
    "Skriv om spørsmålet som en kunde typisk ville ha skrevet det til en chatbot. Skriv en kort setning. Inkluder skrivefeil i spørsmålet.",
    "Skriv dette spørsmålet på nytt, med samme betydydning, men omformulert og kortfattet: ",
    "Skriv om dette spørsmålet på en humoristisk og kortfattet måte",
    "Skriv om dette spørsmålet, men på nynorsk.",
    "Skriv dette spørsmålet som en kunde som ønsker svar. Skriv det på en annen måte.",
    "Skriv om dette spørmsålet må en kritisk måte. Kort og konsist.",
    "Skriv om dette spørsmålet men med helt andre ord. Vær kortfattet.",
    "Skriv om dette spørsmålet men med helt andre ord og utrykk. Vær kortfattet."
]

with open('setfit/treningsdata.txt', 'w', encoding='utf-8') as file:
    file.write("")

for question in questions:
    for prompt in prompts:
        message = [
                    {"role": "user", "content": f"{prompt} {question}"}
        ]
        stream = client.chat.completions.create(
        temperature=1.0, #https://platform.openai.com/docs/api-reference/audio
        #sjekk og
        #https://community.openai.com/t/cheat-sheet-mastering-temperature-and-top-p-in-chatgpt-api/172683
        #https://aimresearch.co/leaders-opinion/leaders-opinion-how-temperature-affects-chatgpt-with-rachael-chudoba
        model="gpt-4-turbo-preview",
        messages=message,
        stream=True,
        )
        answer = ""
        for chunk in stream: #returnerer ein og ein token, så må loope gjennom heile streamen og legge til tokens ein og ein
            content = chunk.choices[0].delta.content
            if content:
                answer += content 
        with open('setfit/treningsdata.txt', 'a', encoding='utf-8') as file:
            file.write("text " + answer + " \n"+ "label " + question + "\n")
        print(answer)
        i += 1
        