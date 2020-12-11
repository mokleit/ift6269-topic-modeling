import os
import re

folder = "speech_recordings"
files = [filename for filename in os.listdir(folder) if filename.isdigit()]
files.sort(key=int)
sorted_files = [folder + '/' + filename for filename in files]
output_file = "corpus.txt"

for index, filename in enumerate(sorted_files):
    file = open(filename, "r")
    document = file.read().replace("\n", " ")
    document = re.sub(r'Lacoste-Julien Simon', "", document)
    document = re.sub(r'Unknown Speaker', "", document)
    document = re.sub(r'Simon Demeule', "", document)
    document = re.sub(r'\d\d:\d\d:\d\d', '', document)
    output = open(output_file, "a")
    output.write(document)
    if index != (len(files) - 1):
        output.write("\n")
    output.close()
