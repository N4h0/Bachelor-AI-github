from openai import OpenAI

with open('txtandCSV-files/chatQA.txt', 'w', encoding='utf-8') as file:
    file.write("\n")

client = OpenAI(
    # This is the default and can be omitted
    api_key="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"
)

questions = []
with open('txtandCSV-files/Q&A.txt', 'r', encoding='utf-8') as file:
    for line in file:
        if line.startswith('Q:'):
            questions.append(line[3:].strip())

for question in questions:
    message = [
                {"role": "user", "content": f"Generer et spørsmål som er likt dette spørsmålet, men skrevet på en annen måte. Du er en kunde for et regnskapsfirma, og undersøker nettsiden deres: {question}"}
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
                print(content, end = "")
                file.write(content)
        file.write("\n" + "Q: " + question + "\n\n")