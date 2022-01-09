import numpy
import seaborn as sns
import pandas
import matplotlib.pyplot as plt
import sklearn.impute
from operator import add 

start_date="2021-06-01"
end_date="2021-09-05"

start_dateC = "2021-06-01"   #Start and end dates of the comparison period (lagged). (i.e. set 2021-01-01, 2021-12-31 to compare to the period of full year.)
end_dateC = "2021-09-05"

Mode="Aggregate" #Choose aggregate or pool. Aggregate := all weeks in the period are summed together. Pool := all weeks/months are pooled for plotting.

#Load data:
#Weekly:
data1 = pandas.read_csv("Data/COVID-19_Vaccinations_in_the_United_States_Jurisdiction.csv")
data2 = pandas.read_csv("Data/Provisional_COVID-19_Death_Counts_by_Week_Ending_Date_and_State.csv")
data3 = pandas.read_csv("Data/Weekly_Counts_of_Death_by_Jurisdiction_and_Select_Causes_of_Death.csv")
data4 = pandas.read_csv("Data/Weekly_Counts_of_Deaths_by_Jurisdiction_and_Age.csv")

#Monthly:
data5 = pandas.read_csv("Data/Provisional_COVID-19_Deaths_by_Place_of_Death_and_Age.csv")


#Edit data 1 : Vaccination data
data1["Date"]=pandas.to_datetime(data1["Date"])
data1=data1.sort_values(["Location","Date"]).reset_index(drop=True)
data1=data1.set_index('Date')
data1=data1.groupby("Location",as_index=False).resample('W').first().reset_index(drop=False)
data1["Population"]=data1["Series_Complete_Yes"]/data1["Series_Complete_Pop_Pct"]
data1["Population65+"]=data1["Series_Complete_65Plus"]/data1["Series_Complete_65PlusPop_Pct"]
data1["Population18+"]=data1["Series_Complete_18Plus"]/data1["Series_Complete_18PlusPop_Pct"]
A=["AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"]
B=["Alabama","Alaska","Arizona","Arkansas","California","Colorado","Conneticut","Delaware","Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa","Kansas","Kentucky","Lousiana","Maine","Maryland","Massachusetts","Michigan","Minnesota","Mississippi","Missouri","Montana","Nebraska","Nevada","New Hampshire","New Jersey","New Mexico","New York","North Caarolina","North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania","Rhode Island","South Caronlina","South Dakohta","Tennessee","Texas","Utah","Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming"]
data1["State"]=data1["Location"].replace(dict(zip(A, B)))


#Edit data 2 : Covid and non-covid deaths.
data2=data2[data2["Group"]=="By Week"]
data2["Date"]=pandas.to_datetime(data2["End Date"])
data2=data2.sort_values(["State","Date"]).reset_index(drop=True)
data2=data2.set_index('Date')
data2=data2.groupby("State",as_index=False).resample('W').first().reset_index(drop=False)
data2["State"] = data2.groupby('level_0')['State'].transform('first')
data2=data2[["Date","State","Total Deaths","COVID-19 Deaths"]]


#Edit data 3 : Cause of death.
data3=data3.rename(columns={"Number of Deaths" : "Deaths", "Jurisdiction" : "State"})
data3=data3.drop("Cause Group",axis=1)
data3=data3[data3["Type"]=="Predicted (weighted)"]  #Week 31-35
data3["Date"]=pandas.to_datetime(data3["Week Ending Date"])
data3=data3.sort_values(["State","Cause Subgroup","Date"]).reset_index(drop=True)
data3=data3.set_index('Date')
data3=data3.groupby(['State','Cause Subgroup'],as_index=False).resample('7D').first().reset_index(drop=False)
data3["State"] = data3.groupby('level_0')['State'].transform('first')
data3["Cause Subgroup"] = data3.groupby('level_0')['Cause Subgroup'].transform('first')
dataM=data3[data3["State"]=="Montana"]
dataM=dataM[dataM["Cause Subgroup"]=="Chronic lower respiratory disease"]


#Edit data 4 : Deaths by age.
data4=data4.rename(columns={"Number of Deaths" : "Deaths", "Jurisdiction" : "State"})
data4=data4[data4["Type"]=="Predicted (weighted)"]
data4["Date"]=pandas.to_datetime(data4["Week Ending Date"])
data4=data4.sort_values(["State","Age Group","Date"]).reset_index(drop=True)
data4=data4.set_index('Date')
data4=data4.groupby(['State','Age Group'],as_index=False).resample('W').first().reset_index(drop=False)
data4["State"] = data4.groupby('level_0')['State'].transform('first')
data4["Age Group"] = data4.groupby('level_0')['Age Group'].transform('first')


#Edit data 5 : Deaths by age (monthly).
data5 = data5[data5["Place of Death"]=="Total - All Places of Death"]
data5 = data5.drop("Place of Death",axis=1)
data5 = data5[data5["Group"]=="By Month"]
data5["Date"]=pandas.to_datetime(data5["End Date"])
data5 = data5.sort_values(["State","Age group","Date"]).reset_index(drop=True)
data5=data5[["Date","State","Age group","Total Deaths","COVID-19 Deaths"]]


#Reshape to wide form:
dataTable=data3.pivot_table(index=["State","Date"],columns="Cause Subgroup",values=["Deaths"]).reset_index()
D=data4.pivot_table(index=["State","Date"],columns="Age Group",values=["Deaths"]).reset_index()
S=["Total Deaths","COVID-19 Deaths"]
df=pandas.DataFrame()
for i in range(len(S)):
    df=pandas.concat([df,data5.pivot_table(index=["State","Date"],columns="Age group",values=[S[i]])],axis=1)
data5=df.reset_index()


#Missing data:
#Impute:    
def impute(data,IDVar,n=10,Min=1,Max=0): 
    df=data.isnull().iloc[:,2:]
    imputer=sklearn.impute.KNNImputer(n_neighbors=n)
    data["Cost"]=numpy.arange(len(data))
    for i in range(len(data.columns)-3):
        for j in range(len(data[IDVar].unique())):
                if data[data[IDVar]==data[IDVar].unique()[j]][data.columns[[i+2]]].isna().all()[0]==False:
                    data.loc[data[IDVar]==data[IDVar].unique()[j],data.columns[i+2]]=imputer.fit_transform(data[data[IDVar]==data[IDVar].unique()[j]][data.columns[[data.shape[1]-1,i+2]]])[:,1]
    data = data.drop("Cost", axis = 1)
    if Min<Max:
        data.iloc[:,2:]=numpy.minimum(numpy.maximum(data.iloc[:,2:],Min),Max)*df+(1-df)*data.iloc[:,2:]
    return data

dataTable=impute(dataTable,"State")
D=impute(D,"State")
data2=impute(data2,"State",4,1,9)
data5=impute(data5,"State",4,1,9)


#Create non-covid deaths variable:
data2["Non-Covid Deaths"]=data2["Total Deaths"]-data2["COVID-19 Deaths"]
data5[list(zip(["Non-Covid Deaths",]*9,list(zip(*data5.columns[2:11]))[1]))]=data5["Total Deaths"]-data5["COVID-19 Deaths"]


#Lagged vars:
def lags(data,IDVar,A,Name,L):
    S=data.columns
    for i in range(L):
        Y=(data.groupby(IDVar).shift(A[0])[S[2+i]])
        for j in range(1,len(A)):
            Y+=(data.groupby(IDVar).shift(A[j])[S[2+i]])
        Y=Y/len(A)
        data[(S[2+i][0]+ Name, S[2+i][1])]=Y
    return data

l0=[52*2,52*3,52*4,52*5]
l1=[52]
L=dataTable.shape[1]-2
dataTable=lags(dataTable,"State",l0,"BP",L)
dataTable=lags(dataTable,"State",l1,"Y1",L)
L=D.shape[1]-2
D=lags(D,"State",l0,"BP",L)
D=lags(D,"State",l1,"Y1",L)
data2[["Total DeathsY1","COVID-19 DeathsY1","Non-Covid DeathsY1"]]=data2[["State","Total Deaths","COVID-19 Deaths","Non-Covid Deaths"]].groupby("State").shift(52)[["Total Deaths","COVID-19 Deaths","Non-Covid Deaths"]]
data5[list(zip(list(map(add, list(zip(*data5.columns[2:len(data5.columns)]))[0], ["Y1M"]*(len(data5.columns)-1))),list(zip(*data5.columns[2:len(data5.columns)]))[1]))]=data5.groupby("State").shift(12)[data5.columns[2:len(data5.columns)]]


#Merge:
dataTable = dataTable.merge(D,how="outer",on=["State","Date"])
dataTable = dataTable.merge(data2,how="outer", on=["State","Date"]).reset_index(drop=True)
dataTable = dataTable.drop(dataTable.columns[[2,3]],axis=1)


#Level:
def levelC(data,mode=0):
    dataC=data[(pandas.to_datetime(start_dateC)<=data["Date"]) & (data["Date"]<=pandas.to_datetime(end_dateC))].reset_index(drop=True)
    data=data[(pandas.to_datetime(start_date)<=data["Date"]) & (data["Date"]<=pandas.to_datetime(end_date))].reset_index(drop=True)
    C=data["Date"].count()/dataC["Date"].count()
    if mode==0:
        data=data.groupby(["State"],as_index=False).sum()
        dataC=dataC.groupby(["State"],as_index=False).sum()
    cols= [i for i in range(len(list(zip(*data.columns))[0])) if ('Y1' in list(zip(*data.columns))[0][i] or "BP" in list(zip(*data.columns))[0][i])]
    dataC.iloc[:,cols] = dataC.iloc[:,cols] * C
    data.iloc[:,cols] = dataC.iloc[:,cols]
    return data.reset_index(drop=True)

if Mode == "Aggregate":
#Weekly table:
    dataTable=levelC(dataTable)
    DV=data1[(start_date<=data1["Date"]) & (data1["Date"]<=end_date)].reset_index(drop=True)
    DV=data1.groupby(["State"],as_index=False).last()  
    dataTable=dataTable.merge(DV,on=["State"]).reset_index(drop=True)
    
#Monthly table:
    dataTableM=levelC(data5)
    dataTableM=data5[(start_date<=data5["Date"]) & (data5["Date"]<=end_date)].reset_index(drop=True)
    dataTableM=dataTableM.groupby(["State"],as_index=False).sum()
    dataTableM=dataTableM.merge(DV,on=["State"]).reset_index(drop=True)
    dataTableM=dataTableM.drop(dataTableM.columns[[1]],axis=1)

if Mode == "Pool":
#Weekly table:
    dataTable=levelC(dataTable,1)
    DV=data1[(start_date<=data1["Date"]) & (data1["Date"]<=end_date)].reset_index(drop=True)
    dataTable=dataTable.merge(DV,on=["Date","State"]).reset_index(drop=True)

#Monthly table:
    dataTableM=levelC(data5,1)
    dataTableM=dataTableM.sort_values(["Date","State"]).reset_index(drop=True)
    DV=DV.sort_values(["Date","State"]).reset_index(drop=True)
    dataTableM = pandas.merge_asof(dataTableM,DV,on=["Date"],by=["State"],direction="nearest").reset_index(drop=True) 
    dataTableM=dataTableM.sort_values(["State","Date"]).reset_index(drop=True).set_index("Date").drop(dataTableM.columns[[2,3]],axis=1)
    DV=DV.sort_values(["State","Date"]).reset_index(drop=True)

#for i in range(60):
#    print(i,dataTableM.columns[i])


#Compute Excess Deaths:
#Weekly:
for i in range(1,14):
    dataTable["Excess " + str(dataTable.columns[i+13])]=dataTable[dataTable.columns[i]]/dataTable[dataTable.columns[i+13]]-1
for i in range(1,14):
    dataTable["Excess " + str(dataTable.columns[i+26])]=dataTable[dataTable.columns[i]]/dataTable[dataTable.columns[i+26]]-1
for i in range(40,46):
    dataTable["Excess " + str(dataTable.columns[i+6])]=dataTable[dataTable.columns[i]]/dataTable[dataTable.columns[i+6]]-1
for i in range(40,46):
    dataTable["Excess " + str(dataTable.columns[i+12])]=dataTable[dataTable.columns[i]]/dataTable[dataTable.columns[i+12]]-1
for i in range(58,61):
    dataTable["Excess " + str(dataTable.columns[i+3])]=dataTable[dataTable.columns[i]]/dataTable[dataTable.columns[i+3]]-1

dataTable["ExcessAllDeathsAltBP"]= dataTable[dataTable.columns[40:46]].sum(axis=1)/dataTable[dataTable.columns[46:52]].sum(axis=1)-1
dataTable["ExcessAllDeathsAltY1"] = dataTable[dataTable.columns[40:46]].sum(axis=1)/dataTable[dataTable.columns[52:58]].sum(axis=1)-1

dataTable["Excess0to65BP"]=dataTable[dataTable.columns[[40,41,45]]].sum(axis=1)/dataTable[dataTable.columns[[45,46,47]]].sum(axis=1)-1
dataTable["Excess0to65Y1"]=dataTable[dataTable.columns[[40,41,45]]].sum(axis=1)/dataTable[dataTable.columns[[51,52,53]]].sum(axis=1)-1

dataTable["ExcessOtherBP"]=(dataTable["Total Deaths"]-dataTable[dataTable.columns[1:14]].sum(axis=1))/(dataTable[dataTable.columns[46:52]].sum(axis=1)-dataTable[dataTable.columns[14:27]].sum(axis=1))-1
dataTable["ExcessOtherY1"]=(dataTable["Total Deaths"]-dataTable[dataTable.columns[1:14]].sum(axis=1))/(dataTable["Total DeathsY1"]-dataTable[dataTable.columns[27:40]].sum(axis=1))-1


#Monthly:
for i in range(27):
    dataTableM["Excess " + str(dataTableM.columns[i+28])] = dataTableM[dataTableM.columns[i+1]]/dataTableM[dataTableM.columns[i+28]]-1


dataTableM["Excess Total 18-65 Y1M"] = (dataTableM[dataTableM.columns[9]]-dataTableM[dataTableM.columns[[1,6,7,8]]].sum(axis=1)) / (dataTableM[dataTableM.columns[9+27]]-dataTableM[dataTableM.columns[[1+27,6+27,7+27,8+27]]].sum(axis=1))-1
dataTableM["Excess Covid 18-65 Y1M"] = (dataTableM[dataTableM.columns[18]]-dataTableM[dataTableM.columns[[10,15,16,17]]].sum(axis=1) )/ (dataTableM[dataTableM.columns[18+27]]-dataTableM[dataTableM.columns[[10+27,15+27,16+27,17+27]]].sum(axis=1))-1
dataTableM["Excess Non-Covid 18-65 Y1M"] = (dataTableM[dataTableM.columns[27]]-dataTableM[dataTableM.columns[[19,24,25,26]]].sum(axis=1) )/ (dataTableM[dataTableM.columns[27+27]]-dataTableM[dataTableM.columns[[19+27,24+27,25+27,26+27]]].sum(axis=1))-1

dataTableM["Excess Total 65+ Y1M"] = dataTableM[dataTableM.columns[[6,7,8]]].sum(axis=1)/dataTableM[dataTableM.columns[[6+27,7+27,8+27]]].sum(axis=1)-1
dataTableM["Excess Covid 65+ Y1M"] = dataTableM[dataTableM.columns[[15,16,17]]].sum(axis=1)/dataTableM[dataTableM.columns[[15+27,16+27,17+27]]].sum(axis=1)-1
dataTableM["Excess Non-Covid 65+ Y1M"] = dataTableM[dataTableM.columns[[24,25,26]]].sum(axis=1)/dataTableM[dataTableM.columns[[24+27,25+27,26+27]]].sum(axis=1)-1

dataTableM["Excess Total 0-18 Y1M"] = dataTableM[dataTableM.columns[[1]]].sum(axis=1)/dataTableM[dataTableM.columns[[1+27]]].sum(axis=1)-1
dataTableM["Excess Covid 0-18 Y1M"] = dataTableM[dataTableM.columns[[10]]].sum(axis=1)/dataTableM[dataTableM.columns[[10+27]]].sum(axis=1)-1
dataTableM["Excess Non-Covid 0-18 Y1M"] = dataTableM[dataTableM.columns[[19]]].sum(axis=1)/dataTableM[dataTableM.columns[[19+27]]].sum(axis=1)-1



#Plot and save:
#Based on weekly:
dataTable["Full Vaccination 65 Below"]=(dataTable["Series_Complete_Yes"]-dataTable["Series_Complete_65Plus"])/(dataTable["Population"]-dataTable["Population65+"])
dataTable["Full Vaccination 18-65"]=(dataTable["Series_Complete_18Plus"]-dataTable["Series_Complete_65Plus"])/(dataTable["Population"]-dataTable["Population65+"]-dataTable["Population18+"])    
sns.set_theme(color_codes=True,style='darkgrid', palette='deep')
for i in range(13+13+6+6+3+2):
    sns.regplot(dataTable["Series_Complete_Pop_Pct"],dataTable[dataTable.columns[i+148]],line_kws={'lw': 1.5, 'color': 'red'},lowess=True)
    plt.savefig("Plots/ExcessDeaths/"+dataTable.columns[i+148]+".png",bbox_inches="tight")
    plt.figure()

A=["Full Vaccination 65 Below","Full Vaccination 65 Below","Series_Complete_Pop_Pct","Series_Complete_Pop_Pct"]
B=["Excess0to65BP","Excess0to65Y1","ExcessOtherBP","ExcessOtherY1"]
C=["Excess Full Vaccination Below 65 BP","Excess Full Vaccination Below 65 Y1","Excess Other Deaths BP","Excess Other Deaths Y1"]
for i in range(len(A)):
    sns.regplot(dataTable[A[i]],dataTable[B[i]],line_kws={'lw': 1.5, 'color': 'red'},lowess=True)
    plt.savefig("Plots/ExcessDeaths/"+C[i]+".png",bbox_inches="tight")
    plt.figure()

#Based on monthly:
dataTableM["Full Vaccination 65 Below"]=(dataTableM["Series_Complete_Yes"]-dataTableM["Series_Complete_65Plus"])/(dataTableM["Population"]-dataTableM["Population65+"])
dataTableM["Full Vaccination 18-65"]=(dataTableM["Series_Complete_18Plus"]-dataTableM["Series_Complete_65Plus"])/(dataTableM["Population18+"]-dataTableM["Population65+"])
dataTableM["Full Vaccination 0-18"]=(dataTableM["Series_Complete_Yes"]-dataTableM["Series_Complete_18Plus"])/(dataTableM["Population"]-dataTableM["Population18+"])

for i in range(28):
    sns.regplot(dataTableM["Series_Complete_Pop_Pct"],dataTableM[dataTableM.columns[i+139]],line_kws={'lw': 1.5, 'color': 'red'},lowess=True)
    plt.savefig("Plots/ExcessDeaths/"+dataTable.columns[i+148]+".png",bbox_inches="tight")
    plt.figure()
    
A=["Full Vaccination 18-65","Full Vaccination 18-65","Full Vaccination 18-65","Series_Complete_65PlusPop_Pct","Series_Complete_65PlusPop_Pct","Series_Complete_65PlusPop_Pct","Full Vaccination 0-18","Full Vaccination 0-18","Full Vaccination 0-18"]
B=["Excess Total 18-65 Y1M","Excess Non-Covid 18-65 Y1M","Excess Covid 18-65 Y1M","Excess Total 65+ Y1M","Excess Non-Covid 65+ Y1M","Excess Covid 65+ Y1M","Excess Total 0-18 Y1M","Excess Non-Covid 0-18 Y1M","Excess Covid 0-18 Y1M"]
C=["Excess Deaths 18-65 Y1M","Excess Non-Covid Deaths 18-65 Y1M","Excess Covid Deaths 18-65 Y1M","Excess Deaths 65Plus Y1M","Excess Non-Covid Deaths 65Plus Y1M","Excess Covid Deaths 65Plus Y1M","Excess Deaths 0-18 Y1M","Excess Non-Covid Deaths 0-18 Y1M","Excess Covid Deaths 0-18 Y1M"]
for i in range(9):
    sns.regplot(dataTableM[A[i]],dataTableM[B[i]],line_kws={'lw': 1.5, 'color': 'red'},lowess=True)
    plt.savefig("Plots/ExcessDeaths/"+C[i]+".png",bbox_inches="tight")
    plt.figure()
