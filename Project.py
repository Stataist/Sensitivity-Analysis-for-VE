import numpy
import seaborn as sns
import pandas
import statsmodels.api as sm
import matplotlib.pyplot as plt

VE=numpy.array([0.95,0.8,0.6,0.4,0.2,0.05])


# %% Data section:
#Load datasets: 
data = pandas.read_csv("Data/United_States_COVID-19_County_Level_of_Community_Transmission_Historical_Changes.csv",dtype={'fips_code':str})
data2 = pandas.read_csv("Data/COVID-19_Vaccinations_in_the_United_States_County_Dec16.csv",dtype={'FIPS':str})
data3 = pandas.read_csv("Data/Provisional_COVID-19_Death_Counts_in_the_United_States_by_County_Dec15.csv",dtype={'FIPS County Code':str})
data4 = pandas.read_csv("Data/Provisional_COVID-19_Death_Counts_in_the_United_States_by_County_Oct6.csv",dtype={'FIPS County Code':str})
data5 = pandas.read_table("Data/LND01.csv",delimiter=",",dtype={'STCOU':str})
data6 = pandas.read_table("Data/Ages.csv",delimiter=",",dtype={'FIPS':str})

#17099 LASALLE COUNTY
#Create "deaths" series:
data3['FIPS County Code']=data3['FIPS County Code'].str.zfill(5)
data4['FIPS County Code']=data4['FIPS County Code'].str.zfill(5)
deaths=data3.merge(data4, how='inner', on=["FIPS County Code"]) 
deaths=deaths.fillna(0)
deaths["deaths"]=deaths["Deaths involving COVID-19_x"]-deaths["Deaths involving COVID-19_y"]
deaths["alldeaths"]=deaths["Deaths from All Causes_x"]-deaths["Deaths from All Causes_y"]
deaths["noncoviddeaths"]=deaths["alldeaths"]-deaths["deaths"]

#Data editing:
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
data2["FIPS"]=data2.FIPS.astype(str)
data2=data2.replace('suppressed', '0')

data=data.replace('suppressed', '0')
data["cases_per_100K_7_day_count_change"]=data["cases_per_100K_7_day_count_change"].str.replace(',', '').astype(float)
data["Date"]=pandas.to_datetime(data["date"])
data=data.sort_values(["fips_code","Date"]).reset_index(drop=True)
data["CumCases"]=0
i=0
for i in range(7):
    I=(1-numpy.minimum((data.groupby(["fips_code"]).cumcount()+i)%7,1))
    data["C"]=data["cases_per_100K_7_day_count_change"].fillna(0).astype(float) * I
    data["CumCases"]+=  data[["fips_code","C"]].groupby(["fips_code"]).cumsum()["C"] * I

data6["PO85"]=data6["AGE85PLUS_TOT"]/data6["POPESTIMATE"]
data6["PO65"]=data6["AGE65PLUS_TOT"]/data6["POPESTIMATE"]
data6["P4564"]=data6["AGE4564_TOT"]/data6["POPESTIMATE"]
data6["PO852"]=(data6["AGE85PLUS_TOT"]/data6["POPESTIMATE"])**2
data6["PO652"]=(data6["AGE65PLUS_TOT"]/data6["POPESTIMATE"])**2
data6["P45642"]=(data6["AGE4564_TOT"]/data6["POPESTIMATE"])**2


#Merge datasets:
Data=data.merge(data2, how='inner', left_on=["date","fips_code"], right_on=["Date","FIPS"])
Data["date"]=pandas.to_datetime(Data['date']) 
Data=Data.sort_values(["fips_code","date"]).reset_index(drop=True)
Data=Data.loc[(pandas.to_datetime(Data['date']) <= '12/15/2021') & (pandas.to_datetime(Data['date']) >= '10/06/2021')]
Data = Data.reset_index(drop=True)
Data["Population"]=(Data["Series_Complete_Yes"]/(Data["Series_Complete_Pop_Pct"]/100)).round().astype(float)


#Data=Data.fillna(0)
#Create leveled cases and deaths data: 
df=Data.groupby(["fips_code"],as_index=False).mean()
df[["Metro_status","state_name"]]=Data.groupby(["fips_code"],as_index=False).first()[["Metro_status","state_name"]]
df["prevcases"]=Data[["fips_code","CumCases"]].groupby(["fips_code"],as_index=False).first()["CumCases"]
df["prevcases2"]=df["prevcases"]**2
df["cases_per_100K (est)"]=Data[["fips_code","CumCases"]].groupby(["fips_code"],as_index=False).last()["CumCases"]-df["prevcases"]
df["C"]=numpy.ones(len(df))
df["Population"]=df["Population"].astype(float)
df[["S1","S2","S3","S4","S5","S6","S7","S8"]]=Data.groupby(["fips_code"],as_index=False).last()[["Series_Complete_Pop_Pct","Series_Complete_12PlusPop_Pct","Series_Complete_18PlusPop_Pct","Series_Complete_65PlusPop_Pct","Administered_Dose1_Pop_Pct","Administered_Dose1_Recip_12PlusPop_Pct","Administered_Dose1_Recip_18PlusPop_Pct","Administered_Dose1_Recip_65PlusPop_Pct"]]-Data.groupby(["fips_code"],as_index=False).first()[["Series_Complete_Pop_Pct","Series_Complete_12PlusPop_Pct","Series_Complete_18PlusPop_Pct","Series_Complete_65PlusPop_Pct","Administered_Dose1_Pop_Pct","Administered_Dose1_Recip_12PlusPop_Pct","Administered_Dose1_Recip_18PlusPop_Pct","Administered_Dose1_Recip_65PlusPop_Pct"]]


df = df.merge(data5[["STCOU","LND110210D"]], left_on=["fips_code"],right_on="STCOU")
df["density"]=df["Population"]/df["LND110210D"]
df=df.merge(data6,left_on=["fips_code"],right_on=["FIPS"])
df=df[df["cases_per_100K (est)"]>0] #Delete zeros (cases).

dfD=df.merge(deaths, how='inner', left_on=["fips_code"], right_on=["FIPS County Code"])
dfD["deathspercap"]=dfD["deaths"]/dfD["Population"]
dfD["alldeathspercap"]=dfD["alldeaths"]/dfD["Population"]
dfD["noncoviddeathspercap"]=dfD["alldeathspercap"]-dfD["deathspercap"]

dfD["prevdeaths"]=dfD["Deaths involving COVID-19_y"]/dfD["Population"]

dfD=dfD[dfD["deathspercap"]>0]      #Delete zeros (deaths).


#Construct additional data for simulation (not used by default):
DataSim=data.loc[(data['date'] <= '12/31/2020')]
DataSim["cases_per_100K (est)"]=DataSim["cases_per_100K_7_day_count_change"].astype("float")
DataSim=DataSim.groupby(["fips_code"],as_index=False).mean()
DataSim=DataSim[DataSim.fips_code.isin(df.fips_code.values)==True]

#DataSimD=DataSim.copy()
#data5["Fips Code"]=data5["Fips Code"].astype(str)
#DataSimD=DataSimD.merge(data5[["Total Deaths","COVID-19 Deaths","Fips Code"]], how='inner', left_on=["fips_code"], right_on=["Fips Code"])
#DataSimD=DataSimD.merge(dfD[["Population","fips_code"]], how='inner', on=["fips_code"])
#DataSimD=DataSimD.rename(columns={"COVID-19 Deaths" : "deaths","Total Deaths" : "alldeaths"})
#DataSimD["deathspercap"]=DataSimD["deaths"]/DataSimD["Population"]
#DataSimD["alldeathspercap"]=DataSimD["alldeaths"]/DataSimD["Population"]
#DataSimD=DataSimD[DataSimD.fips_code.isin(dfD.fips_code.values)==True]

#Create additional variables:
dfD["noncoviddeathspercap2"]=dfD["noncoviddeathspercap"]**2
dfD["noncoviddeathspercap3"]=dfD["noncoviddeathspercap"]**3
dfD["noncoviddeathspercap4"]=dfD["noncoviddeathspercap"]**4
dfD["noncoviddeathspercap5"]=dfD["noncoviddeathspercap"]**5
dfD["Series_Complete_Pop_Pct2"]=dfD["Series_Complete_Pop_Pct"]**2
dfD["Series_Complete_Pop_Pct3"]=dfD["Series_Complete_Pop_Pct"]**3
dfD["Series_Complete_Pop_Pct4"]=dfD["Series_Complete_Pop_Pct"]**4
dfD["prevdeaths2"]=dfD["prevdeaths"]**2
df["density2"]=df["density"]**2
df["density3"]=df["density"]**3
dfD["density2"]=dfD["density"]**2
dfD["density3"]=dfD["density"]**3


#Create dummies:
D1=pandas.get_dummies(df["state_name"],drop_first=True)
D2=pandas.get_dummies(df["Metro_status"],drop_first=True)
D3=pandas.get_dummies(df[["state_name","Metro_status"]],drop_first=True)
D4=pandas.get_dummies(dfD["state_name"],drop_first=True)
D5=pandas.get_dummies(dfD["Urban Rural Code_x"],drop_first=True)
D6=pandas.get_dummies(dfD[["state_name","Urban Rural Code_x"]],drop_first=True)
# %% Regression models:
if (False==True):
    Y1="cases_per_100K (est)"
    Y2="deathspercap"
    Vars1 = ["C","Series_Complete_Pop_Pct","density","density2","noncoviddeathspercap","noncoviddeathspercap2","noncoviddeathspercap3","noncoviddeathspercap4","noncoviddeathspercap5","PO85","PO65","P4564","PO852","PO652","P45642","prevcases","prevcases2"]
    Vars2 = ["C","Series_Complete_65PlusPop_Pct","density","density2","PO85","PO65","P4564","PO852","PO652","P45642","prevcases","prevcases2"]
    n=16
    #19.7,-0.4
    #Vars1 = ["C","Series_Complete_Pop_Pct"]
    #Vars2 = ["C","Series_Complete_65Plus"]

    
    #df=df[df["cases_per_100K (est)"] >0]
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

    results=numpy.zeros([n,4])
    for i in range(n):
        print("")
        print("")
        print("Model"+(str(i+1)) + "result:")
        M=eval("model"+ str(i+1)+".fit()")
        print(M.summary())
        results[i,0]=M.params[1]
        results[i,1]=M.pvalues[1]
        results[i,2]=M.rsquared
        results[i,3]=M.aic
        

#Analysis for each state.
    results1 = []
    results2 = []
    for i in range(len(df["state_name"].unique())):
        datas=df.loc[(df['state_name'] == df["state_name"].unique()[i])]
        if (len(datas))>30:
            M=sm.WLS(datas["cases_per_100K (est)"], datas[["C","Series_Complete_Pop_Pct","Metro_status"]],weights=datas["Population"])
            M=M.fit()
            results1.append((df["state_name"].unique()[i],M.params[1],M.pvalues[1],int(numpy.round(datas["Population"].sum())),len(datas)))

    for i in range(len(dfD["state_name"].unique())):
        datas=dfD.loc[(dfD['state_name'] == dfD["state_name"].unique()[i])]
        if (len(datas))>5:
            M=sm.WLS(datas["deaths"]/datas["Population"]*100000, datas[["C","Series_Complete_Pop_Pct","Metro_status"]],weights=datas["Population"])
            M=M.fit()
            results2.append((dfD["state_name"].unique()[i],M.params[1],M.pvalues[1],int(numpy.round(datas["Population"].sum())),len(datas)))
        
    #sns.regplot(dfD["S1"],dfD["noncoviddeathspercap"],line_kws={'lw': 1.5, 'color': 'red'},lowess=True)



#Create text file:
    myText0 = open(r'Model results\Model results.txt','w')
    myText1 = open(r'Model results\All states.txt','w')
    Str0 = ['Results for different models', "    M1 : Cases per 100K explained by complete V%.","    M2 : Controlled for state.", "    M3 : Controlled for metropolian area.", "    M4 : Controlled for state and metropolian area.", "    M5-M8 : Same as M1-M4, weighted by population.", "    M9-M16 : Same as M1-M8, but deaths explained by V% and non-covid deaths.","","        Coeff:           p:           R^2:           AIC:"]
    Str1 = ["Cases and deaths per capita explained independently in each state weighted by population. Metropolian area and in the case of deaths also non-covid deaths as controls.", "      State:                      Coef:                            p-val:                   Population:       Number of counties:"]
    
    for i in Str0:
        myText0.write(i + '\n')
    for i in range(n):
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
        
        
    myText1.write("\n \n \nDEATHS: \n \n")
    for i in range(len(results2)):
        for j in range(len(results2[0])):
            if type(results2[i][j])==numpy.float64:
                myText1.write("         " + "{:.5f}".format(results2[i][j]).ljust(25," "))
            else:
                myText1.write(str(results2[i][j]).ljust(25," "))
        myText1.write('\n')

# %% Requested analysis:

def scatter1(x,y,VE,name,df=df,dfSim=df,n=4):
    fig, ax = plt.subplots(nrows=1,ncols=n+1, figsize=(20,7))
    fig.suptitle(name,fontsize=24)
    fig.tight_layout(w_pad=3)
    ax[0].set_ylim([0, max(df[y])])
    ax[0].title.set_text('Actual')
    sns.regplot(x,y,data=df,ax=ax[0],line_kws={'lw': 1.5, 'color': 'red'})
    for i in range(n):
        ax[i+1].set_ylim([0, max(df[y])])
        ax[i+1].title.set_text("Simulation "+str(i+1))
        df[y+" permute"]=numpy.random.choice(dfSim[y].values,size=len(dfSim[y].values),replace=False)*(1-df[x]/100*VE)
        df[y+" permute"]=df[y+" permute"]*df[y].mean()/df[y+" permute"].mean()
        sns.regplot(x,y+" permute",data=df,ax=ax[i+1],line_kws={'lw': 1.5, 'color': 'red'},x_ci=0.95)
    return fig


for i in range(len(VE)):
    name="Cases and Full Vaccination VE: "+str(VE[i])+""
    fig=scatter1("Series_Complete_Pop_Pct","cases_per_100K (est)",VE[i],name) #Push dfSim for the "dfSim" argument to use alternative data for cases.
    fig.savefig("Plots/All vaccinations/" + "Cases Full VE "+str(VE[i])+".png",bbox_inches="tight")

for i in range(len(VE)):
    name="COVID Deaths and Full Vaccination VE: "+str(VE[i])+""
    fig=scatter1("Series_Complete_Pop_Pct","deathspercap",VE[i],name,dfD,dfD) #Push dfSimD for the argument to use alternative data for deaths.
    fig.savefig("Plots/All vaccinations/" +"Deaths Full VE "+str(VE[i])+".png",bbox_inches="tight")

for i in range(len(VE)):
    name="Cases and One Vaccination VE: "+str(VE[i])+""
    fig=scatter1("Administered_Dose1_Pop_Pct","cases_per_100K (est)",VE[i],name)
    fig.savefig("Plots/All vaccinations/" +"Cases One VE "+str(VE[i])+".png",bbox_inches="tight")

for i in range((len(VE))):
    name="COVID Deaths and One Vaccination VE: "+str(VE[i]) +""
    fig=scatter1("Administered_Dose1_Pop_Pct","deathspercap",VE[i],name,dfD,dfD)
    fig.savefig("Plots/All vaccinations/" +"Deaths One VE "+str(VE[i])+".png",bbox_inches="tight")


# %% : Analysis for differnt vaccination ages:

if (True==False):
    #12+
    for i in range(len(VE)):
        name="Cases and Full Vaccination 12+ VE: "+str(VE[i])+""
        fig=scatter1("Series_Complete_12PlusPop_Pct","cases_per_100K (est)",VE[i],name) #Push dfSim for the "dfSim" argument to use alternative data.
        fig.savefig("Plots/12+/"+"Cases Full 12+ VE "+str(VE[i])+".png",bbox_inches="tight")
    
    for i in range(len(VE)):
        name="COVID Deaths and Full Vaccination 12+ VE: "+str(VE[i])+""
        fig=scatter1("Series_Complete_12PlusPop_Pct","deathspercap",VE[i],name,dfD,dfD) #Push dfSimD for the argument to use alternative data for deaths.
        fig.savefig("Plots/12+/"+"Deaths Full 12+ VE "+str(VE[i])+".png",bbox_inches="tight")
        
    for i in range(len(VE)):
        name="Cases and One Vaccination 12+ VE: "+str(VE[i])+""
        fig=scatter1("Administered_Dose1_Recip_12PlusPop_Pct","cases_per_100K (est)",VE[i],name)
        fig.savefig("Plots/12+/"+"Cases One 12+ VE "+str(VE[i])+".png",bbox_inches="tight")
    
    for i in range((len(VE))):
        name="COVID Deaths and One Vaccination 12+ VE: "+str(VE[i]) +""
        fig=scatter1("Administered_Dose1_Recip_12PlusPop_Pct","deathspercap",VE[i],name,dfD,dfD)
        fig.savefig("Plots/12+/"+"Deaths One 12+ VE "+str(VE[i])+".png",bbox_inches="tight")

    
    #18+
    for i in range(len(VE)):
        name="Cases and Full Vaccination 18+ VE: "+str(VE[i])+""
        fig=scatter1("Series_Complete_18PlusPop_Pct","cases_per_100K (est)",VE[i],name) #Push dfSim for the "dfSim" argument to use alternative data.
        fig.savefig("Plots/18+/"+"Cases Full 18+ VE "+str(VE[i])+".png",bbox_inches="tight")
    
    for i in range(len(VE)):
        name="COVID Deaths and Full Vaccination 18+ VE: "+str(VE[i])+""
        fig=scatter1("Series_Complete_18PlusPop_Pct","deathspercap",VE[i],name,dfD,dfD) #Push dfSimD for the argument to use alternative data for deaths.
        fig.savefig("Plots/18+/"+"Deaths Full 18+ VE "+str(VE[i])+".png",bbox_inches="tight")
    
    for i in range(len(VE)):
        name="Cases and One Vaccination 18+ VE: "+str(VE[i])+""
        fig=scatter1("Administered_Dose1_Recip_18PlusPop_Pct","cases_per_100K (est)",VE[i],name)
        fig.savefig("Plots/18+/"+"Cases One 18+ VE "+str(VE[i])+".png",bbox_inches="tight")
    
    for i in range((len(VE))):
        name="COVID Deaths and One Vaccination 18+ VE: "+str(VE[i]) +""
        fig=scatter1("Administered_Dose1_Recip_18PlusPop_Pct","deathspercap",VE[i],name,dfD,dfD)
        fig.savefig("Plots/18+/"+"Deaths One 18+ VE "+str(VE[i])+".png",bbox_inches="tight")
    
    
    #65+
    for i in range(len(VE)):
        name="Cases and Full Vaccination 65+ VE: "+str(VE[i])+""
        fig=scatter1("Series_Complete_65PlusPop_Pct","cases_per_100K (est)",VE[i],name) #Push dfSim for the "dfSim" argument to use alternative data.
        fig.savefig("Plots/65+/"+"Cases Full 65+ VE "+str(VE[i])+".png",bbox_inches="tight")
    
    for i in range(len(VE)):
        name="COVID Deaths and Full Vaccination 65+ VE: "+str(VE[i])+""
        fig=scatter1("Series_Complete_65PlusPop_Pct","deathspercap",VE[i],name,dfD,dfD) #Push dfSimD for the argument to use alternative data for deaths.
        fig.savefig("Plots/65+/"+"Deaths Full 65+ VE "+str(VE[i])+".png",bbox_inches="tight")
    
    for i in range(len(VE)):
        name="Cases and One Vaccination 65+ VE: "+str(VE[i])+""
        fig=scatter1("Administered_Dose1_Recip_65PlusPop_Pct","cases_per_100K (est)",VE[i],name)
        fig.savefig("Plots/65+/"+"Cases One 65+ VE "+str(VE[i])+".png",bbox_inches="tight")
    
    for i in range((len(VE))):
        name="COVID Deaths and One Vaccination 65+ VE: "+str(VE[i]) +""
        fig=scatter1("Administered_Dose1_Recip_65PlusPop_Pct","deathspercap",VE[i],name,dfD,dfD)
        fig.savefig("Plots/65+/"+"Deaths One 65+ VE "+str(VE[i])+".png",bbox_inches="tight")
    






def scatterD(x,y,VE,name,df=df,dfSim=df,n=4): ##Removes effect of V% on other deaths. ADD CONTROLS FOR STATE, METROPOLIAN
    fig, ax = plt.subplots(nrows=1,ncols=n+1, figsize=(20,7))
    fig.suptitle(name,fontsize=24)
    fig.tight_layout(w_pad=3)
    ax[0].set_ylim([0, max(df[y])])
    ax[0].title.set_text('Actual')
    
    
    M=sm.WLS(df["noncoviddeathspercap"], df[["C",x]],missing="drop") #weights=dfD["Population"]
    M=M.fit()
    b=M.params[1]
    M0=df[y].sum()
    M1=df["noncoviddeathspercap"].sum()
    df[y+ "2"]=df[y]-(df[x]*b)*M1/M0
    df[y+ "2"]=df[y+"2"]/df[y+"2"].mean()*df[y].mean()
    
    sns.regplot(x,y+"2",data=df,ax=ax[0],line_kws={'lw': 1.5, 'color': 'red'})
    for i in range(n):
        ax[i+1].set_ylim([0, max(df[y])])
        ax[i+1].title.set_text("Simulation "+str(i+1))
        df[y+" permute"]=numpy.random.choice(dfSim[y].values,size=len(dfSim[y].values),replace=False)*(1-df[x]/100*VE *(M0-M1)/M0 ) #* (M0-M1)/M0 dfSim["deathspercap"].values/dfSim["alldeathspercap"].values
        df[y+" permute"]=df[y+" permute"]*df[y].mean()/df[y+" permute"].mean()
        sns.regplot(x,y+" permute",data=df,ax=ax[i+1],line_kws={'lw': 1.5, 'color': 'red'},x_ci=0.95)
    return fig

#fig=scatterD("Series_Complete_Pop_Pct","alldeathspercap",0.95,"name",dfD,dfD)

