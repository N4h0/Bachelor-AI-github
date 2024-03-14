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



#Generer et spørsmål for hvert spørsmål i spørsmålslisten.
for question in questions:
    promptquestion = question
    for i in range(1,15):
        message = [
                    {"role": "user", "content": f"""
Du er en kunde for et regnskapsfirma, og ønsker informasjon fra en chatbot.
Spørsmålet du stiller stilles HELT annerledes enn dette spørsmålet: {promptquestion}. 
"""}
        ]
        stream = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=message,
        stream=True,
        )
        with open('txtandCSV-files/chatQA.txt', 'a', encoding='utf-8') as file:
            file.write("I: ")
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    file.write(content)
                    promptquestion = content
            file.write("\n" + "Q: " + question + "\n\n")