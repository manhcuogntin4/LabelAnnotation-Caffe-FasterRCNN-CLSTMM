import difflib
words = []
f = open('noms.txt', 'r')
for line in f:
 words.append(line)
f.close()
simi=difflib.get_close_matches('CHISTOPHUE', words,5)
print(simi)
