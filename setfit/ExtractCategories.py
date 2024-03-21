#Enkelt program som henter ut alle kategoriene våre (dvs alle unike setninger).

kategorier = []
with open('txtandCSV-files/Q&A.txt', 'r', encoding='utf-8') as file: #Opne Q&A, les den /r) og enkoder som uft-8 (lar Æ, Ø og å vere med)
    for line in file:
        if line.startswith('Q:'):
            kategorier.append(line[3:])
            
with open('setfit/kategorier', 'w', encoding='utf-8') as file:
    for kategori in kategorier:
        file.write(f"{kategori}")