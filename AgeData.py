#Creates combined Ages data series from each individual state data file to "Ages.csv".
import pandas

A=["01","02","04","05","06","08","09","10","11","12","13","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35","36","37","38","39","40","41","42","44","45","46","47","48","49","50","51","53","54","55","56"]
for i in range(51):
    D = pandas.read_csv("Data/Age/CC-EST2020-AGESEX-"+A[i]+".csv")
    D=D[D["YEAR"]==13]
    D=D.drop("YEAR",axis=1).reset_index(drop=True)
    if i==0:
        dff=D[D["STATE"]==-1]
    dff=pandas.concat([dff,D]).reset_index(drop=True)

dff["FIPS"]=dff["STATE"].astype(str).str.zfill(2)+dff["COUNTY"].astype(str).str.zfill(3)

dff.to_csv('Data/Ages.csv')