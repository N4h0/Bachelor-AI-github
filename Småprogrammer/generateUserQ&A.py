from openai import OpenAI

with open('txtandCSV-files/chatQA.txt', 'w', encoding='utf-8') as file:
    file.write("\n")

client = OpenAI(
    # API-nøkkel burde ikke vere public i de fleste scripts, men denne kjører bare lokalt og er bare delt innad i gruppen.
    api_key="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"
)

questions = []
with open('txtandCSV-files/Q&A.txt', 'r', encoding='utf-8') as file:
    for line in file:
        if line.startswith('Q:'):
            questions.append(line[3:].strip())

print(questions)

#Generer et spørsmål for hvert spørsmål i spørsmålslisten.
i=0
for question in questions:
    promptquestion = question
    if i != 0:
        with open('txtandCSV-files/chatQA.txt', 'a', encoding='utf-8') as file:
            file.write(f"\n_____________NYTT SPØRSMÅL: {question}_____________\n")
    for j in range(1,15):
        message = [
                    {"role": "user", "content": f"""
Du er en kunde for et regnskapsfirma, og ønsker informasjon fra en chatbot.
Spørsmålet du stiller stilles annerledes enn dette spørsmålet, men har samme betydning: {promptquestion}. 
Spørsmålet har samme betydning som {question}.
"""}
        ]
        
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
        with open('txtandCSV-files/chatQA.txt', 'a', encoding='utf-8') as file:
            file.write("Spørringsnummer: " + str(i) + "\n")
            file.write("I: " + promptquestion)
            file.write("\n" + "Q: " + question + "\n\n")
        print(promptquestion)
        i += 1