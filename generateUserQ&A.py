from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key="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"
)

questions = []
with open('Q&A.txt', 'r', encoding='utf-8') as file:
    for line in file:
        if line.startswith('Q:'):
            questions.append(line[3:].strip())

stream = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[
        {"role": "user", "content": 
            """ 
            Generer 1 spørsmål som passer til disse spørsmålene, stilt som en kunde til et regnskapsfirma. Generer listen uten numerering.: 
            {questions}
            """}
        ],
    stream=True,
)

with open('chatQA.txt', 'w', encoding='utf-8') as file:
    newQ=False
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end = "")
            file.write(content)
