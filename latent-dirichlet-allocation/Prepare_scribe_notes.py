# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:02:42 2020

@author: rober
"""
import pandas as pd
import numpy as np
#import resource
import os
import re


folder = "C:/Users/rober/OneDrive/Bureau/etude/graph models udem/lectures/text"



#prepare data

files = [filename for filename in os.listdir(folder) if filename.isdigit()]
files.sort(key=int)
sorted_files = [folder + '/' + filename for filename in files]
output_file = folder+"/corpus.txt"

for index, filename in enumerate(sorted_files):
    print(filename)
    file = open(filename,"r", encoding="utf8")
    document = file.read().replace("\n", " ")
    document =document.lower()
    document= re.sub(r'[^a-z]',' ',document )   
    document = re.sub(r'\b\w{1,3}\b', '', document)
    document = re.sub(r'\s+', " ", document)
    
    output = open(output_file, "a")
    output.write(document)
    #if index != (len(files) - 1):
        #print(124)
    output.write("\n")
    output.close()
    



    