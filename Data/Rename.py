# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 18:10:30 2021

@author: User
"""

import glob
import pandas
import os

Z=[]
for file in glob.glob("Provisional_COVID-19_Death_Counts_in_the_United_States_by_County*.csv"):
    print (file)
    df = pandas.read_csv(file)
    End_Date = pandas.to_datetime(df[df.columns[2]]).astype(str)[0]
    os.renames(file,"Provisional_COVID-19_Death_Counts_in_the_United_States_by_County_" + End_Date +".csv")
    Z.append(End_Date)
    

