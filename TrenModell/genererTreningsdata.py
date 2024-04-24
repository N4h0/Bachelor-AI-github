#Me er for late for å lage data for å trene modellen sjølv :D
import openai
import os

reformulted_list = "Her vil listen av alle spørsmål som er blit stillt innenfor det nåverende spørsmålet komme."
question = "Her vil brukerspørsmålet komme."

prompt =    f"""Lag 1 setninger som er en omforumlering av følgende spørsmål: {question}. Skriv korte setninger og bruk andre ord og skrivemåte. Spørsmålet kan IKKE være likt noen av disse spørsmålene: \n\n
    {reformulted_list}"""

reformulted_list = []

#Gjerne bruk dene linja for å beskrive treningsdata
with open('TrenModell/Treningsdata/alpha10.txt', 'w', encoding='utf-8') as file:
    file.write("This training data was made using the following prompt:\n")
    # Write the prompt on the next line
    file.write(prompt + "\n\n")

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
        reformulated_questions = []
        for j in range(0,7):
            reformulted_list = '\n'.join(reformulated_questions)
            message = [
                        {"role": "user", "content": f"""
    Lag 1 setninger som er en omforumlering av følgende spørsmål: {question}. Spørsmålet MÅ ha samme mening men forskjellig struktur og orbruk fra disse spørsmålene: \n\n
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
            with open('TrenModell/Treningsdata/alpha10.txt', 'a', encoding='utf-8') as file:
                file.write("text " + omformulert_spørsmål + " \n"+ "label " + question + "\n")
            i += 1