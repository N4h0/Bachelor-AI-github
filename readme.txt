For å kjøre disse programmene lokalt kreves det en innstalert versjon av python, python 3.08 eller nyere. Miniconda er et forslag til en python-distribisjon som kan innstalleres,
da denne er er minimal og komme uten unødvendige grafisk UI eller andre tillegg. 
For å kjøre programmet må det opprettest et python envirnomen (I VSC: ctrl+shift+P og skriv inn python: Create envirnoment). Deretter må en del pakker installeres:
numpy, sentence_transformers, scikit-learn, openai, flask_cors og flask. Disse kan installeres i terminal med pip install XXXXX. Pass på at du er i rett envirnoment når du installerer pakkene.


Dette repositoriet består av flere mapper med ulike funksjoner. overfitting-test omhander å teste en modell for overfitting, SetFitSammenligning sammenligner to modeller, trenmodell trener modeller og txt/CSV-filer inneholder tekst og CSV-filer.
