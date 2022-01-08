import csv
import pandas as pd


# 100% CSV-Datei einlesen und zum Variable df zuweisen
df = pd.read_csv('online_shoppers_intention_test.csv', delimiter=',')

print('---------------')
# Liste erzeugen mit IndexNr. für 80% CSV_Datei
lst80p = [i for i in range(0,len(df),5)]

# Liste erzeugen mit unbenutzte Index Nr
lst20p = [i for i in range(0,len(df))]
for i in lst80p:
    for j in lst20p:
        if(i==j):
            lst20p.remove(j)
   
# Löschen von 100% df jeder 5 Zeile
df_80prozent = df.drop(lst80p, axis = 0)

# Löschen bentzte IndexNr in 80% CSV-Datei
df_20prozent = df.drop(lst20p, axis = 0)

# Schreiben neue CSV-Datei, 80% 
writer = csv.writer(open("df_80prozent.csv", "w"))

# Schreiben neue  CSV-Datei, 20%
writer = csv.writer(open("df_20prozent.csv", "w"))
    
print(df_80prozent)
print('---------------')

print(df_20prozent)
print('---------------')

