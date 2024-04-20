import openai
import os

try:
    client = openai.Client(api_key=os.getenv('CHATGPT_API_KEY'))
except Exception as e:
    print(f"En feil skjedde ved henting av API-nøkkel: {e}")


with open('txtandCSV-files/randomquestions.txt', 'w', encoding='utf-8') as file:
    file.write("")
        
for j in range(1,300):
    message = [
                {"role": "user", "content": f"""
Generer en tilfeldig setning. 
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
    with open('txtandCSV-files/randomquestions.txt', 'a', encoding='utf-8') as file:
        file.write("Q: " + promptquestion + "\n")
    print(promptquestion)
