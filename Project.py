import numpy
import seaborn as sns
import pandas
import statsmodels.api as sm
import matplotlib.pyplot as plt

VE=numpy.array([0.95,0.8,0.6,0.4,0.2,0.05])

StartDate="2021-10-23"       #Start date for variables not including deaths. Year-Month-Day.
EndDate="2022-01-02"

StartDateDeaths=StartDate    #Possible death dates: '2021-07-10', '2021-08-28', '2021-10-02', '2021-10-23', '2021-12-11'
EndDateDeaths=EndDate

# %% Data section:
#Load datasets: 
data = pandas.read_csv("Data/United_States_COVID-19_County_Level_of_Community_Transmission_Historical_Changes.csv",dtype={'fips_code':str})
data2 = pandas.read_csv("Data/COVID-19_Vaccinations_in_the_United_States_County.csv",dtype={'FIPS':str})
data3 = pandas.read_csv("Data/Provisional_COVID-19_Death_Counts_in_the_United_States_by_County_"+ StartDateDeaths +".csv",dtype={'FIPS County Code':str})
data4 = pandas.read_csv("Data/Provisional_COVID-19_Death_Counts_in_the_United_States_by_County_"+ EndDateDeaths+".csv",dtype={'FIPS County Code':str})
data5 = pandas.read_table("Data/LND01.csv",delimiter=",",dtype={'STCOU':str})
data6 = pandas.read_table("Data/Ages.csv",delimiter=",",dtype={'FIPS':str})
data7 = pandas.read_table("Data/AH_County_of_Occurrence_COVID-19_Deaths_Counts__2020_Provisional.csv",delimiter=",",dtype={'Fips Code':str})

#Fix FIPS code:
data3['FIPS County Code']=data3['FIPS County Code'].str.zfill(5)
data4['FIPS County Code']=data4['FIPS County Code'].str.zfill(5)
data7['Fips Code']=data7['Fips Code'].str.zfill(5)

#Create "deaths" series:
deaths=data3.merge(data4, how='inner', on=["FIPS County Code"]) 
deaths=deaths.fillna(0)
deaths["deaths"]=deaths["Deaths involving COVID-19_y"]-deaths["Deaths involving COVID-19_x"]
deaths["alldeaths"]=deaths["Deaths from All Causes_y"]-deaths["Deaths from All Causes_x"]
deaths["noncoviddeaths"]=deaths["alldeaths"]-deaths["deaths"]


#
#Data editing:
#
#Vaccine data:
data2["Administered_Dose1_Pop_Pct"]=data2["Administered_Dose1_Pop_Pct"].replace(0, numpy.nan)
data2["Administered_Dose1_Recip_65PlusPop_Pct"]=data2["Administered_Dose1_Recip_65PlusPop_Pct"].replace(0, numpy.nan)
data2["Administered_Dose1_Recip_18PlusPop_Pct"]=data2["Administered_Dose1_Recip_18PlusPop_Pct"].replace(0, numpy.nan)
data2["Administered_Dose1_Recip_12PlusPop_Pct"]=data2["Administered_Dose1_Recip_12PlusPop_Pct"].replace("12.{",12.7).astype(float)
data2["Administered_Dose1_Recip_12PlusPop_Pct"]=data2["Administered_Dose1_Recip_12PlusPop_Pct"].replace(0, numpy.nan)

data2["Series_Complete_Pop_Pct"]=data2["Series_Complete_Pop_Pct"].replace(0, numpy.nan)
data2["Series_Complete_12PlusPop_Pct"]=data2["Series_Complete_12PlusPop_Pct"].replace(0, numpy.nan)
data2["Series_Complete_18PlusPop_Pct"]=data2["Series_Complete_18PlusPop_Pct"].replace(0, numpy.nan)
data2["Series_Complete_65PlusPop_Pct"]=data2["Series_Complete_65PlusPop_Pct"].replace(0, numpy.nan)

data2 = data2.drop(data2[data2.Recip_County ==  "Unknown County"].index)
data2 = data2.drop(data2[data2.Series_Complete_Pop_Pct ==  0].index)
data2 = data2.replace('suppressed', '0')


#Cases data:
data=data.replace('suppressed', '0')
data["cases_per_100K_7_day_count_change"]=data["cases_per_100K_7_day_count_change"].str.replace(',', '').astype(float)
data["Date"]=pandas.to_datetime(data["date"])
data=data.sort_values(["fips_code","Date"]).reset_index(drop=True)

data["CumCases"]=0
for i in range(7):
    I=(1-numpy.minimum((data.groupby(["fips_code"]).cumcount()+i)%7,1))
    data["C"]=data["cases_per_100K_7_day_count_change"].fillna(0).astype(float) * I
    data["CumCases"]+=  data[["fips_code","C"]].groupby(["fips_code"]).cumsum()["C"] * I


#Age data: 
data6["PO85"]=data6["AGE85PLUS_TOT"]/data6["POPESTIMATE"]
data6["PO65"]=data6["AGE65PLUS_TOT"]/data6["POPESTIMATE"]
data6["P4564"]=data6["AGE4564_TOT"]/data6["POPESTIMATE"]


#Merge datasets:
Data=data.merge(data2, how='inner', left_on=["date","fips_code"], right_on=["Date","FIPS"]).reset_index(drop=True)
Data["date"]=pandas.to_datetime(Data['date']) 
Data=Data.sort_values(["fips_code","date"]).reset_index(drop=True).reset_index(drop=True)
Data=Data.loc[(pandas.to_datetime(Data['date']) <= EndDate) & (pandas.to_datetime(Data['date']) >= StartDate)]
Data = Data.reset_index(drop=True)
Data["Population"]=(Data["Series_Complete_Yes"]/(Data["Series_Complete_Pop_Pct"]/100)).round().astype(float)


#Create leveled datasets:
df=Data.groupby(["fips_code"],as_index=False).mean()
df[["Metro_status","state_name"]]=Data.groupby(["fips_code"],as_index=False).first()[["Metro_status","state_name"]]
df["prevcases"]=Data[["fips_code","CumCases"]].groupby(["fips_code"],as_index=False).first()["CumCases"]
#df["cases_per_100K (est)"] = df["cases_per_100K_7_day_count_change"]
df["cases_per_100K (est)"]=Data[["fips_code","CumCases"]].groupby(["fips_code"],as_index=False).last()["CumCases"]-df["prevcases"]
df["C"]=numpy.ones(len(df))
df["Population"]=df["Population"].astype(float)
df[["Series_Complete_Pop_Pct","Series_Complete_12PlusPop_Pct","Series_Complete_18PlusPop_Pct","Series_Complete_65PlusPop_Pct","Administered_Dose1_Pop_Pct","Administered_Dose1_Recip_12PlusPop_Pct","Administered_Dose1_Recip_18PlusPop_Pct","Administered_Dose1_Recip_65PlusPop_Pct"]]=Data.groupby(["fips_code"],as_index=False).last()[["Series_Complete_Pop_Pct","Series_Complete_12PlusPop_Pct","Series_Complete_18PlusPop_Pct","Series_Complete_65PlusPop_Pct","Administered_Dose1_Pop_Pct","Administered_Dose1_Recip_12PlusPop_Pct","Administered_Dose1_Recip_18PlusPop_Pct","Administered_Dose1_Recip_65PlusPop_Pct"]]
df[["Series_Complete_Yes","Series_Complete_12Plus","Series_Complete_18Plus","Series_Complete_65Plus","Administered_Dose1_Recip","Administered_Dose1_Recip_12Plus","Administered_Dose1_Recip_18Plus","Administered_Dose1_Recip_65Plus"]]=Data.groupby(["fips_code"],as_index=False).last()[["Series_Complete_Yes","Series_Complete_12Plus","Series_Complete_18Plus","Series_Complete_65Plus","Administered_Dose1_Recip","Administered_Dose1_Recip_12Plus","Administered_Dose1_Recip_18Plus","Administered_Dose1_Recip_65Plus"]]

df[["S1","S2","S3","S4","S5","S6","S7","S8"]]=Data.groupby(["fips_code"],as_index=False).last()[["Series_Complete_Pop_Pct","Series_Complete_12PlusPop_Pct","Series_Complete_18PlusPop_Pct","Series_Complete_65PlusPop_Pct","Administered_Dose1_Pop_Pct","Administered_Dose1_Recip_12PlusPop_Pct","Administered_Dose1_Recip_18PlusPop_Pct","Administered_Dose1_Recip_65PlusPop_Pct"]]-Data.groupby(["fips_code"],as_index=False).first()[["Series_Complete_Pop_Pct","Series_Complete_12PlusPop_Pct","Series_Complete_18PlusPop_Pct","Series_Complete_65PlusPop_Pct","Administered_Dose1_Pop_Pct","Administered_Dose1_Recip_12PlusPop_Pct","Administered_Dose1_Recip_18PlusPop_Pct","Administered_Dose1_Recip_65PlusPop_Pct"]]

df = df.merge(data5[["STCOU","LND110210D"]], left_on=["fips_code"],right_on="STCOU").reset_index(drop=True) 
df["density"]=df["Population"]/df["LND110210D"]
df=df.merge(data6,left_on=["fips_code"],right_on=["FIPS"]).reset_index(drop=True) 
df=df[df["cases_per_100K (est)"]>0].reset_index(drop=True)                               #Delete zeros (cases).

dfD=df.merge(deaths, how='inner', left_on=["fips_code"], right_on=["FIPS County Code"]).reset_index(drop=True)
dfD["deathspercap"]=dfD["deaths"]/dfD["Population"]
dfD["alldeathspercap"]=dfD["alldeaths"]/dfD["Population"]
dfD["noncoviddeathspercap"]=dfD["alldeathspercap"]-dfD["deathspercap"]
dfD["CFR"]=dfD["deathspercap"]/(dfD["cases_per_100K (est)"]/100000)
dfD["FCFR"]=dfD["noncoviddeathspercap"]/dfD["cases_per_100K (est)"]*100000
dfD["prevdeaths"]=dfD["Deaths involving COVID-19_y"]/dfD["Population"]
dfD=dfD[dfD["deathspercap"]>0].reset_index(drop=True)                                     #Delete zeros (deaths).

M=dfD[["CFR","cases_per_100K (est)","deathspercap"]]


#Construct additional data for simulation (not used by default):
DataSim=data.loc[(data['date'] <= '12/31/2020')]
DataSim["cases_per_100K (est)"]=DataSim["cases_per_100K_7_day_count_change"].astype("float")
DataSim=DataSim.groupby(["fips_code"],as_index=False).mean()
DataSim=DataSim[DataSim.fips_code.isin(df.fips_code.values)==True]

DataSimD=DataSim.copy()
DataSimD=DataSimD.merge(data7[["Total Deaths","COVID-19 Deaths","Fips Code"]], how='inner', left_on=["fips_code"], right_on=["Fips Code"])
DataSimD=DataSimD.merge(dfD[["Population","fips_code"]], how='inner', on=["fips_code"])
DataSimD=DataSimD.rename(columns={"COVID-19 Deaths" : "deaths","Total Deaths" : "alldeaths"})
DataSimD["deathspercap"]=DataSimD["deaths"]/DataSimD["Population"]
DataSimD["alldeathspercap"]=DataSimD["alldeaths"]/DataSimD["Population"]
DataSimD=DataSimD[DataSimD.fips_code.isin(dfD.fips_code.values)==True]


#Create additional variables:
df[["TwoDoses","TwoDoses12+","TwoDoses18+","TwoDoses65+"]]=(df[["Series_Complete_Pop_Pct","Series_Complete_12PlusPop_Pct","Series_Complete_18PlusPop_Pct","Series_Complete_65PlusPop_Pct"]].to_numpy()+df[["Administered_Dose1_Pop_Pct","Administered_Dose1_Recip_12PlusPop_Pct","Administered_Dose1_Recip_18PlusPop_Pct","Administered_Dose1_Recip_65PlusPop_Pct"]].to_numpy())/2
df[["ThreeDoses","ThreeDoses18+","ThreeDoses65+"]]=(df[["TwoDoses","TwoDoses18+","TwoDoses65+"]].to_numpy()+df[["Booster_Doses_Vax_Pct","Booster_Doses_18Plus_Vax_Pct","Booster_Doses_65Plus_Vax_Pct"]].to_numpy())/3
df[["FullBooster","FullBooster18+","FullBooster65+"]]=(df[["Series_Complete_Pop_Pct","Series_Complete_18PlusPop_Pct","Series_Complete_65PlusPop_Pct"]].to_numpy()+df[["Booster_Doses_Vax_Pct","Booster_Doses_18Plus_Vax_Pct","Booster_Doses_65Plus_Vax_Pct"]].to_numpy())/2

dfD[["TwoDoses","TwoDoses12+","TwoDoses18+","TwoDoses65+"]]=(dfD[["Series_Complete_Pop_Pct","Series_Complete_12PlusPop_Pct","Series_Complete_18PlusPop_Pct","Series_Complete_65PlusPop_Pct"]].to_numpy()+dfD[["Administered_Dose1_Pop_Pct","Administered_Dose1_Recip_12PlusPop_Pct","Administered_Dose1_Recip_18PlusPop_Pct","Administered_Dose1_Recip_65PlusPop_Pct"]].to_numpy())/2
dfD[["ThreeDoses","ThreeDoses18+","ThreeDoses65+"]]=(dfD[["TwoDoses","TwoDoses18+","TwoDoses65+"]].to_numpy()+dfD[["Booster_Doses_Vax_Pct","Booster_Doses_18Plus_Vax_Pct","Booster_Doses_65Plus_Vax_Pct"]].to_numpy())/3
dfD[["FullBooster","FullBooster18+","FullBooster65+"]]=(dfD[["Series_Complete_Pop_Pct","Series_Complete_18PlusPop_Pct","Series_Complete_65PlusPop_Pct"]].to_numpy()+dfD[["Booster_Doses_Vax_Pct","Booster_Doses_18Plus_Vax_Pct","Booster_Doses_65Plus_Vax_Pct"]].to_numpy())/2
dfD["deathspercap2"]=dfD["deathspercap"]**2
dfD["noncoviddeathspercap2"]=dfD["noncoviddeathspercap"]**2
dfD["noncoviddeathspercap3"]=dfD["noncoviddeathspercap"]**3
dfD["noncoviddeathspercap4"]=dfD["noncoviddeathspercap"]**4
dfD["noncoviddeathspercap5"]=dfD["noncoviddeathspercap"]**5
dfD["Series_Complete_Pop_Pct2"]=dfD["Series_Complete_Pop_Pct"]**2
dfD["Series_Complete_Pop_Pct3"]=dfD["Series_Complete_Pop_Pct"]**3
dfD["Series_Complete_Pop_Pct4"]=dfD["Series_Complete_Pop_Pct"]**4
dfD["prevcases2"]=dfD["prevcases"]**2
dfD["prevcases3"]=dfD["prevcases"]**3
dfD["prevdeaths2"]=dfD["prevdeaths"]**2
dfD["prevdeaths3"]=dfD["prevdeaths"]**3
df["density2"]=df["density"]**2
df["density3"]=df["density"]**3
dfD["density2"]=dfD["density"]**2
dfD["density3"]=dfD["density"]**3
dfD["PO852"]=dfD["PO85"]**2
dfD["PO652"]=dfD["PO65"]**2
dfD["P45642"]=dfD["P4564"]**2
dfD["PO853"]=dfD["PO85"]**3
dfD["PO653"]=dfD["PO65"]**3
dfD["P45643"]=dfD["P4564"]**3
dfD["0"]=numpy.zeros(len(dfD))
dfD["Series_Complete_65MinusPop_Pct"]=100*(dfD["Series_Complete_Yes"]-dfD["Series_Complete_65Plus"])/(dfD["POPESTIMATE"]-dfD["PO65"])


#Create dummies:
D1=pandas.get_dummies(df["state_name"],drop_first=True)
D2=pandas.get_dummies(df["Metro_status"],drop_first=True)
D3=pandas.get_dummies(df[["state_name","Metro_status"]],drop_first=True)
D4=pandas.get_dummies(dfD["state_name"],drop_first=True)
D5=pandas.get_dummies(dfD["Urban Rural Code_x"],drop_first=True)
D6=pandas.get_dummies(dfD[["state_name","Urban Rural Code_x"]],drop_first=True)
dfD["Metro_status"]=pandas.get_dummies(dfD["Metro_status"],drop_first=True)
# %% Regression models:
if (True==True):
    Y1="cases_per_100K (est)"
    Y2="deathspercap"
    Y3="noncoviddeathspercap"
    #Y4="CFR"
    Vars1 = ["C","Series_Complete_Pop_Pct","PO65","PO652","PO85","PO852"]
    Vars2 = ["C","Series_Complete_Pop_Pct","PO65","PO652","PO85","PO852"]
    Vars3 = ["C","Series_Complete_Pop_Pct"]
    #Vars4 = ["C","Series_Complete_Pop_Pct","PO65","PO652","PO85","PO852"]
    n=24

    model1=sm.OLS(dfD[Y1], dfD[Vars1],missing="drop")
    model2=sm.OLS(dfD[Y1], pandas.concat([dfD[Vars1],D4],axis=1),missing="drop")
    model3=sm.OLS(dfD[Y1], pandas.concat([dfD[Vars1],D5],axis=1),missing="drop")
    model4=sm.OLS(dfD[Y1], pandas.concat([dfD[Vars1],D6],axis=1),missing="drop")
    model5=sm.WLS(dfD[Y1], dfD[Vars1],weights=dfD["Population"],missing="drop")
    model6=sm.WLS(dfD[Y1], pandas.concat([dfD[Vars1],D4],axis=1),missing="drop",weights=dfD["Population"])
    model7=sm.WLS(dfD[Y1], pandas.concat([dfD[Vars1],D5],axis=1),missing="drop",weights=dfD["Population"])
    model8=sm.WLS(dfD[Y1], pandas.concat([dfD[Vars1],D6],axis=1),missing="drop",weights=dfD["Population"])

    model9=sm.OLS(dfD[Y2]*100000, dfD[Vars2],missing="drop")
    model10=sm.OLS(dfD[Y2]*100000, pandas.concat([dfD[Vars2],D4],axis=1),missing="drop")
    model11=sm.OLS(dfD[Y2]*100000, pandas.concat([dfD[Vars2],D5],axis=1),missing="drop")
    model12=sm.OLS(dfD[Y2]*100000, pandas.concat([dfD[Vars2],D6],axis=1),missing="drop")
    model13=sm.WLS(dfD[Y2]*100000, dfD[Vars2],weights=dfD["Population"],missing="drop")
    model14=sm.WLS(dfD[Y2]*100000, pandas.concat([dfD[Vars2],D4],axis=1),missing="drop",weights=dfD["Population"])
    model15=sm.WLS(dfD[Y2]*100000, pandas.concat([dfD[Vars2],D5],axis=1),missing="drop",weights=dfD["Population"])
    model16=sm.WLS(dfD[Y2]*100000, pandas.concat([dfD[Vars2],D6],axis=1),missing="drop",weights=dfD["Population"])
    
    model17=sm.OLS(dfD[Y3]*100000, dfD[Vars3],missing="drop")
    model18=sm.OLS(dfD[Y3]*100000, pandas.concat([dfD[Vars3],D4],axis=1),missing="drop")
    model19=sm.OLS(dfD[Y3]*100000, pandas.concat([dfD[Vars3],D5],axis=1),missing="drop")
    model20=sm.OLS(dfD[Y3]*100000, pandas.concat([dfD[Vars3],D6],axis=1),missing="drop")
    model21=sm.WLS(dfD[Y3]*100000, dfD[Vars3],weights=dfD["Population"],missing="drop")
    model22=sm.WLS(dfD[Y3]*100000, pandas.concat([dfD[Vars3],D4],axis=1),missing="drop",weights=dfD["Population"])
    model23=sm.WLS(dfD[Y3]*100000, pandas.concat([dfD[Vars3],D5],axis=1),missing="drop",weights=dfD["Population"])
    model24=sm.WLS(dfD[Y3]*100000, pandas.concat([dfD[Vars3],D6],axis=1),missing="drop",weights=dfD["Population"])

    #model25=sm.OLS(dfD[Y4]*100000, dfD[Vars4],missing="drop")
    #model26=sm.OLS(dfD[Y4]*100000, pandas.concat([dfD[Vars4],D4],axis=1),missing="drop")
    #model27=sm.OLS(dfD[Y4]*100000, pandas.concat([dfD[Vars4],D5],axis=1),missing="drop")
    #model28=sm.OLS(dfD[Y4]*100000, pandas.concat([dfD[Vars4],D6],axis=1),missing="drop")
    #model29=sm.WLS(dfD[Y4]*100000, dfD[Vars4],weights=dfD["Population"],missing="drop")
    #model30=sm.WLS(dfD[Y4]*100000, pandas.concat([dfD[Vars4],D4],axis=1),missing="drop",weights=dfD["Population"])
    #model31=sm.WLS(dfD[Y4]*100000, pandas.concat([dfD[Vars4],D5],axis=1),missing="drop",weights=dfD["Population"])
    #model32=sm.WLS(dfD[Y4]*100000, pandas.concat([dfD[Vars4],D6],axis=1),missing="drop",weights=dfD["Population"])


    results=numpy.zeros([n,5])
    for i in range(n):
        print("")
        print("")
        print("Model"+(str(i+1)) + "result:")
        M=eval("model"+ str(i+1)+".fit()")
        print(M.summary())
        results[i,0]=M.params[1]
        results[i,1]=M.pvalues[1]
        
        MData = eval("model"+str(i+1)+".exog")
        MParam0 = numpy.copy(MData)
        MParam1 = numpy.copy(MData)
        MParam0[:,1]=0
        MParam1[:,1]=100
        #results[i,2]=1-numpy.mean(M.predict(MParam1))/numpy.mean(M.predict(MParam0))
        results[i,2]=(1-numpy.mean(numpy.maximum(M.predict(MParam1),0)/(M.predict(MParam0))))*100
        results[i,3]=M.rsquared
        results[i,4]=M.aic

#Analysis for each state.
    results1 = []
    results2 = []
    results3 = []
    for i in range(len(dfD["state_name"].unique())):
        datas=dfD[(dfD['state_name'] == dfD["state_name"].unique()[i])]
        if (len(datas))>=20:
            Model=sm.WLS(datas["cases_per_100K (est)"], datas[["C","Series_Complete_Pop_Pct"]])
            M=Model.fit()
            VE0=-M.params[1]*100/M.params[0]
            results1.append((df["state_name"].unique()[i],M.params[1],M.pvalues[1],VE0*100,int(numpy.round(datas["Population"].sum())),len(datas)))
            print(M.summary())


    for i in range(len(dfD["state_name"].unique())):
        datas=dfD[(dfD['state_name'] == dfD["state_name"].unique()[i])]
        if (len(datas))>=20:
            Model=sm.WLS(datas["deathspercap"]*100000, datas[["C","Series_Complete_Pop_Pct"]])
            M=Model.fit()
            VE0=-M.params[1]*100/M.params[0]
            results2.append((dfD["state_name"].unique()[i],M.params[1],M.pvalues[1],VE0*100,int(numpy.round(datas["Population"].sum())),len(datas)))
            
    for i in range(len(dfD["state_name"].unique())):
        datas=dfD[(dfD['state_name'] == dfD["state_name"].unique()[i])]
        if (len(datas))>=20:
            Model=sm.WLS(datas["noncoviddeathspercap"]*100000, datas[["C","Series_Complete_Pop_Pct"]])
            M=Model.fit()
            VE0=-M.params[1]*100/M.params[0]
            results3.append((dfD["state_name"].unique()[i],M.params[1],M.pvalues[1],VE0*100,int(numpy.round(datas["Population"].sum())),len(datas)))


#Create text file:
    myText0 = open(r'Model results\Model results.txt','w')
    myText1 = open(r'Model results\All states.txt','w')
    Str0 = ['Results for different models', "    M1 : Cases per 100K explained by full V%, 2nd order polynomial of population % over 65 and 85 as control.","    M2 : Controlled for state.", "    M3 : Controlled for metropolian area.", "    M4 : Controlled for state and metropolian area.", "    M5-M8 : Same as M1-M4, weighted by population.", "    M9-M16 : Same as M1-M8, but covid deaths as response","    M17-M24 : Same as M1-M8, but non-covid deaths as response.""","        Coeff:            p:            VE:            R^2:          AIC:"]
    Str1 = ["Cases and deaths per capita explained by overall full vaccination percentage independently in each state. No controls.", "      State:                        Coef:                          p-val:                             VE:                     Population:       Number of counties:"]
    S=["Cases:","Deaths:", "Non-Covid Deaths:"]
    
    z=0
    for i in Str0:
        myText0.write(i + '\n')
    for i in range(n):
        if i%8==0:
            myText0.write("\n" + S[z] + "\n")
            z+=1
        myText0.write(("M"+ str(i+1)).rjust(3," "))
        for j in range(results.shape[1]):
            myText0.write("         " + "{:.3f}".format(results[i,j]))
        myText0.write('\n')
    
    
    for i in Str1:
        myText1.write(i + '\n')
    myText1.write("\nCASES: \n \n")
    for i in range(len(results1)):
        for j in range(len(results1[0])):
            if type(results1[i][j])==numpy.float64:
                myText1.write("         " + "{:.5f}".format(results1[i][j]).ljust(25," "))
            else:
                myText1.write(str(results1[i][j]).ljust(25," "))
        myText1.write('\n')
    
    
    myText1.write("\nCOVID-19 DEATHS: \n \n")
    for i in range(len(results2)):
        for j in range(len(results2[0])):
            if type(results2[i][j])==numpy.float64:
                myText1.write("         " + "{:.5f}".format(results2[i][j]).ljust(25," "))
            else:
                myText1.write(str(results2[i][j]).ljust(25," "))
        myText1.write('\n')
        
    myText1.write("\nNON-COVID DEATHS: \n \n")
    for i in range(len(results3)):
        for j in range(len(results3[0])):
            if type(results3[i][j])==numpy.float64:
                myText1.write("         " + "{:.5f}".format(results3[i][j]).ljust(25," "))
            else:
                myText1.write(str(results3[i][j]).ljust(25," "))
        myText1.write('\n')


# %% Requested analysis:

def scatter1(x,y,VE,name,df=df,dfSim=df,scale=1,n=4):
    fig, ax = plt.subplots(nrows=1,ncols=n+1, figsize=(20,7))
    fig.suptitle(name,fontsize=24)
    fig.tight_layout(w_pad=3)
    ax[0].set_ylim([0, max(df[y])*scale])
    ax[0].title.set_text('Actual')
    sns.regplot(x,y,data=df,ax=ax[0],line_kws={'lw': 1.5, 'color': 'red'},scatter_kws={"s": 20})
    for i in range(n):
        ax[i+1].set_ylim([0, max(df[y])*scale])
        ax[i+1].title.set_text("Simulation "+str(i+1))
        df[y+" permute"]=numpy.random.choice(dfSim[y].values/(1-df[x]/100*VE),size=len(dfSim[y].values),replace=False)*(1-df[x]/100*VE)
        df[y+" permute"]=df[y+" permute"]*df[y].mean()/df[y+" permute"].mean()
        sns.regplot(x,y+" permute",data=df,ax=ax[i+1],line_kws={'lw': 1.5, 'color': 'red'},x_ci=0.95,scatter_kws={"s": 20})
    return fig
#Set theme:
sns.set_theme(color_codes=True,style='darkgrid', palette='deep')

for i in range(len(VE)):
    name="Cases and Full Vaccination VE: "+str(int(VE[i]*100))+"%"
    fig=scatter1("Series_Complete_Pop_Pct","cases_per_100K (est)",VE[i],name) #Push dfSim for the "dfSim" argument to use alternative data for cases.
    fig.savefig("Plots/All vaccinations/" + "Cases Full VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")

for i in range(len(VE)):
    name="COVID Deaths and Full Vaccination VE: "+str(int(VE[i]*100))+"%"
    fig=scatter1("Series_Complete_Pop_Pct","deathspercap",VE[i],name,dfD,dfD,0.2) #Push dfSimD for the argument to use alternative data for deaths.
    fig.savefig("Plots/All vaccinations/" +"Deaths Full VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")

for i in range(len(VE)):
    name="Cases and One Vaccination VE: "+str(int(VE[i]*100))+"%"
    fig=scatter1("Administered_Dose1_Pop_Pct","cases_per_100K (est)",VE[i],name)
    fig.savefig("Plots/All vaccinations/" +"Cases One VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")

for i in range((len(VE))):
    name="COVID Deaths and One Vaccination VE: "+str(int(VE[i]*100))+"%"
    fig=scatter1("Administered_Dose1_Pop_Pct","deathspercap",VE[i],name,dfD,dfD,0.2)
    fig.savefig("Plots/All vaccinations/" +"Deaths One VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")


# %% : Analysis for differnt vaccination ages:

if (True==True):
    #12+
    for i in range(len(VE)):
        name="Cases and Full Vaccination 12+ VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("Series_Complete_12PlusPop_Pct","cases_per_100K (est)",VE[i],name)
        fig.savefig("Plots/12+/"+"Cases Full 12+ VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")
    
    for i in range(len(VE)):
        name="COVID Deaths and Full Vaccination 12+ VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("Series_Complete_12PlusPop_Pct","deathspercap",VE[i],name,dfD,dfD,0.2)
        fig.savefig("Plots/12+/"+"Deaths Full 12+ VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")
        
    for i in range(len(VE)):
        name="Cases and One Vaccination 12+ VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("Administered_Dose1_Recip_12PlusPop_Pct","cases_per_100K (est)",VE[i],name)
        fig.savefig("Plots/12+/"+"Cases One 12+ VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")
    
    for i in range((len(VE))):
        name="COVID Deaths and One Vaccination 12+ VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("Administered_Dose1_Recip_12PlusPop_Pct","deathspercap",VE[i],name,dfD,dfD,0.2)
        fig.savefig("Plots/12+/"+"Deaths One 12+ VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")

    
    #18+
    for i in range(len(VE)):
        name="Cases and Full Vaccination 18+ VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("Series_Complete_18PlusPop_Pct","cases_per_100K (est)",VE[i],name)
    
    for i in range(len(VE)):
        name="COVID Deaths and Full Vaccination 18+ VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("Series_Complete_18PlusPop_Pct","deathspercap",VE[i],name,dfD,dfD,0.2)
        fig.savefig("Plots/18+/"+"Deaths Full 18+ VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")
    
    for i in range(len(VE)):
        name="Cases and One Vaccination 18+ VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("Administered_Dose1_Recip_18PlusPop_Pct","cases_per_100K (est)",VE[i],name)
        fig.savefig("Plots/18+/"+"Cases One 18+ VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")
    
    for i in range((len(VE))):
        name="COVID Deaths and One Vaccination 18+ VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("Administered_Dose1_Recip_18PlusPop_Pct","deathspercap",VE[i],name,dfD,dfD,0.2)
        fig.savefig("Plots/18+/"+"Deaths One 18+ VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")
    
    
    #65+
    for i in range(len(VE)):
        name="Cases and Full Vaccination 65+ VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("Series_Complete_65PlusPop_Pct","cases_per_100K (est)",VE[i],name)
        fig.savefig("Plots/65+/"+"Cases Full 65+ VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")
    
    for i in range(len(VE)):
        name="COVID Deaths and Full Vaccination 65+ VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("Series_Complete_65PlusPop_Pct","deathspercap",VE[i],name,dfD,dfD,0.2)
        fig.savefig("Plots/65+/"+"Deaths Full 65+ VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")
    
    for i in range(len(VE)):
        name="Cases and One Vaccination 65+ VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("Administered_Dose1_Recip_65PlusPop_Pct","cases_per_100K (est)",VE[i],name)
        fig.savefig("Plots/65+/"+"Cases One 65+ VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")
    
    for i in range((len(VE))):
        name="COVID Deaths and One Vaccination 65+ VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("Administered_Dose1_Recip_65PlusPop_Pct","deathspercap",VE[i],name,dfD,dfD,0.2)
        fig.savefig("Plots/65+/"+"Deaths One 65+ VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")
    
    #BOOSTERS:
    for i in range(len(VE)):
        name="Cases and Boosters All Vaccinations VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("Booster_Doses_Vax_Pct","cases_per_100K (est)",VE[i],name)
        fig.savefig("Plots/Boosters/"+"Cases Boosters VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")
    
    for i in range(len(VE)):
        name="Cases and Booster Vaccination 18+ VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("Booster_Doses_18Plus_Vax_Pct","cases_per_100K (est)",VE[i],name)
        fig.savefig("Plots/Boosters/"+"Cases Booster 18+ VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")
        
    for i in range(len(VE)):
        name="Cases and Booster Vaccination 50+ VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("Booster_Doses_50Plus_Vax_Pct","cases_per_100K (est)",VE[i],name)
        fig.savefig("Plots/Boosters/"+"Cases Booster 50+ VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")
    
    for i in range((len(VE))):
        name="Cases and Booster Vaccination 65+ VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("Booster_Doses_65Plus_Vax_Pct","cases_per_100K (est)",VE[i],name)
        fig.savefig("Plots/Boosters/"+"Cases Booster 65+ VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")


    for i in range(len(VE)):
        name="Deaths and Boosters All Vaccinations VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("Booster_Doses_Vax_Pct","deathspercap",VE[i],name,dfD,dfD,0.2) 
        fig.savefig("Plots/Boosters/"+"Deaths Boosters VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")
    
    for i in range(len(VE)):
        name="Deaths and Booster Vaccination 18+ VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("Booster_Doses_18Plus_Vax_Pct","deathspercap",VE[i],name,dfD,dfD,0.2)
        fig.savefig("Plots/Boosters/"+"Deaths Booster 18+ VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")
        
    for i in range(len(VE)):
        name="Deaths and Booster Vaccination 50+ VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("Booster_Doses_50Plus_Vax_Pct","deathspercap",VE[i],name,dfD,dfD,0.2)
        fig.savefig("Plots/Boosters/"+"Deaths Booster 50+ VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")
    
    for i in range((len(VE))):
        name="Deaths and Booster Vaccination 65+ VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("Booster_Doses_65Plus_Vax_Pct","deathspercap",VE[i],name,dfD,dfD,0.2)
        fig.savefig("Plots/Boosters/"+"Deaths Booster 65+ VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")


    #Doses Sum Total:
    for i in range(len(VE)):
        name="Cases and Two Doses Sum All Vaccinations VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("TwoDoses","cases_per_100K (est)",VE[i],name)
        fig.savefig("Plots/DosesSum/"+"Cases Two Doses Sum VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")
    
    for i in range(len(VE)):
        name="Cases and Two Doses Sum 12+ VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("TwoDoses12+","cases_per_100K (est)",VE[i],name)
        fig.savefig("Plots/DosesSum/"+"Cases Two Doses Sum 12+ VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")
        
    for i in range(len(VE)):
        name="Cases and Two Doses Sum 18+ VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("TwoDoses18+","cases_per_100K (est)",VE[i],name)
        fig.savefig("Plots/DosesSum/"+"Cases Two Doses Sum 18+ VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")
    
    for i in range((len(VE))):
        name="Cases and Two Doses Sum 65+ VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("TwoDoses65+","cases_per_100K (est)",VE[i],name)
        fig.savefig("Plots/DosesSum/"+"Cases Two Doses Sum 65+ VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")


    for i in range(len(VE)):
        name="Deaths and Two Doses Sum All Vaccinations VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("TwoDoses","deathspercap",VE[i],name,dfD,dfD,0.2) 
        fig.savefig("Plots/DosesSum/"+"Deaths Two Doses Sum VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")
    
    for i in range(len(VE)):
        name="Deaths and Two Doses Sum 12+ VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("TwoDoses12+","deathspercap",VE[i],name,dfD,dfD,0.2) 
        fig.savefig("Plots/DosesSum/"+"Deaths Two Doses Sum 12+ VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")
        
    for i in range(len(VE)):
        name="Deaths and Two Doses Sum 18+ VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("TwoDoses18+","deathspercap",VE[i],name,dfD,dfD,0.2)
        fig.savefig("Plots/DosesSum/"+"Deaths Two Doses Sum 18+ VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")
    
    for i in range((len(VE))):
        name="Deaths and Two Doses Sum 65+ VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("TwoDoses65+","deathspercap",VE[i],name,dfD,dfD,0.2)
        fig.savefig("Plots/DosesSum/"+"Deaths Two Doses Sum 65+ VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")



    for i in range(len(VE)):
        name="Cases and Three Doses Sum All Vaccinations VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("ThreeDoses","cases_per_100K (est)",VE[i],name) 
        fig.savefig("Plots/DosesSum/"+"Cases Three Doses Sum VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")
            
    for i in range(len(VE)):
        name="Cases and Three Doses Sum 18+ VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("ThreeDoses18+","cases_per_100K (est)",VE[i],name)
        fig.savefig("Plots/DosesSum/"+"Cases Three Doses Sum 18+ VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")
    
    for i in range((len(VE))):
        name="Cases and Three Doses Sum 65+ VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("ThreeDoses65+","cases_per_100K (est)",VE[i],name)
        fig.savefig("Plots/DosesSum/"+"Cases Three Doses Sum 65+ VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")


    for i in range(len(VE)):
        name="Deaths and Three Doses Sum All Vaccinations VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("ThreeDoses","deathspercap",VE[i],name,dfD,dfD,0.2) 
        fig.savefig("Plots/DosesSum/"+"Deaths Three Doses Sum VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")
        
    for i in range(len(VE)):
        name="Deaths and Three Doses Sum 18+ VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("ThreeDoses18+","deathspercap",VE[i],name,dfD,dfD,0.2)
        fig.savefig("Plots/DosesSum/"+"Deaths Three Doses Sum 18+ VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")
    
    for i in range((len(VE))):
        name="Deaths and Three Doses Sum 65+ VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("ThreeDoses65+","deathspercap",VE[i],name,dfD,dfD,0.2)
        fig.savefig("Plots/DosesSum/"+"Deaths Three Doses Sum 65+ VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")


    for i in range(len(VE)):
        name="Cases and Full Booster Sum All Vaccinations VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("FullBooster","cases_per_100K (est)",VE[i],name) 
        fig.savefig("Plots/DosesSum/"+"Cases Full Booster Sum VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")
            
    for i in range(len(VE)):
        name="Cases and Full Booster Sum 18+ VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("FullBooster18+","cases_per_100K (est)",VE[i],name)
        fig.savefig("Plots/DosesSum/"+"Cases Full Booster Sum 18+ VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")
    
    for i in range((len(VE))):
        name="Cases and Full Buuster Sum 65+ VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("FullBooster65+","cases_per_100K (est)",VE[i],name)
        fig.savefig("Plots/DosesSum/"+"Cases Full Booster Sum 65+ VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")


    for i in range(len(VE)):
        name="Deaths and Full Booster Sum All Vaccinations VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("FullBooster","deathspercap",VE[i],name,dfD,dfD,0.2) 
        fig.savefig("Plots/DosesSum/"+"Deaths Full Booster Sum VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")
        
    for i in range(len(VE)):
        name="Deaths and Full Booster Sum 18+ VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("FullBooster18+","deathspercap",VE[i],name,dfD,dfD,0.2)
        fig.savefig("Plots/DosesSum/"+"Deaths Full Booster Sum 18+ VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")
    
    for i in range((len(VE))):
        name="Deaths and Full Booster Sum Sum 65+ VE: "+str(int(VE[i]*100))+"%"
        fig=scatter1("FullBooster65+","deathspercap",VE[i],name,dfD,dfD,0.2)
        fig.savefig("Plots/DosesSum/"+"Deaths Full Booster Sum 65+ VE "+str(int(VE[i]*100))+".png",bbox_inches="tight")
