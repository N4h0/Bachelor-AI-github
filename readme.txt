For å kjøre disse programmene lokalt kreves det en innstalert versjon av python, python 3.08 eller nyere. Miniconda er et forslag til en python-distribisjon som kan innstalleres,
da denne er er minimal og komme uten unødvendige grafisk UI eller andre tillegg. 
For å kjøre programmet må det opprettest et python envirnomen (I VSC: ctrl+shift+P og skriv inn python: Create envirnoment). Deretter må en del pakker installeres:
numpy, sentence_transformers, scikit-learn, openai, flask_cors og flask. Disse kan installeres i terminal med pip install XXXXX. Pass på at du er i rett envirnoment når du installerer pakkene.


Dette repositoriet består av 3 deler: 2 mapper og ett program uten mappe, app.py. 

App.py starter opp en server som kan kobles til med postman, flask API eller direkte i terminalen. 

Under småprogrammer ligger ulike programmer som ble brukt i ulik fase av testing. 

txtandCSV-files
chatQA.txt er spørsmål opprettet av chatGPT
Q&A_embedded er en embedded versjon av Q&A.txt
testresults.txt er resultater fra testing i txt.format
testresultasoppsumert oppsummerer resultatene fra sist test kjørt

Småprogrammer
Encode_question.py omformaterer spørsmålene i filen Q&A.txt til embedded format, dvs vektorer som kan samenlignes med hverande ved feks cosine similarity.
generateUserQ&A bruker chatGPT til å generere spørsmål som ligner på spørsmålene i Q&A.txt
RunTestsFromScratch kjører alle programmene som er en del av testingen fra bunnen av.
sentence.py sammenligner ett valgrfitt input med sqørsmålene som ligger i Q&A.txt, og returnerer spørsmålet som ligner mest.
TestQuestions.py tester hvor godt modellen klarer å koble spørsmplene som ligger i chatQA.txt med rett svar.

App.py: server som køyrer chatbotten, kan bli kobla til via flask API, eller testa via command line eller postman.
Sentence.py: Køyrer koden uten server etc.
Encode_questions.py: transformerer spørsmåla slik at sentence.py forstår dei. 
generateUserQ&A: bruker chatGPT av openAI til å generere spørsmål me kan bruke til å teste AIen vår. 