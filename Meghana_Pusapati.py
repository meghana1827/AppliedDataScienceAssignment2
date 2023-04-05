


import numpy as np   # for data analysis
import pandas as pd    # for reading data and analysis
import matplotlib.pyplot as mpbt    # data visualization
import seaborn as sns    # data visualization


# # Function to Read Data and Return Dataframes

def readWBData(nmfl):    # function for data reading

"""
The function readWBData reads World Bank data from a given file, cleans it, and returns two dataframes - one for the actual data and the other for the time series. The function filters the data by selecting the indicators of "Urban population" and "CO2 emissions (kt)" and the countries India, China, Brazil, Ireland, Iraq, and World. The missing values in the data are cleaned using the mean value. The function then combines the Country Name and Indicator Name columns and inserts it into a new column named "Country with Indicator". The data is then transposed, and the first row is set as column names. The years are added to the data, and the year column is set as the index for the time series. The function takes one argument, nmfl, which is the name of the World Bank data file. The function returns two dataframes - the first dataframe contains the actual data with selected indicators and countries, and the second dataframe is the time series of the selected indicators for the selected countries.
"""
    wbd=open(nmfl,"r+").read()     # open data in read mode  
    wbd=wbd[85:]     # skip 85 location to get data
    wbdt=open("wbt.csv","w+").write(wbd)    # write the data in drive with CSV extension
    wbdt=pd.read_csv("wbt.csv")    # read the stored data (CSV)
    wbdt=wbdt.fillna(wbdt.mean())     # clenaing missing values with mean
    wbyrs=wbdt.columns.tolist()[4:-1]      # take years from data
    # filter data with indicators
    wbdt=wbdt[(wbdt['Indicator Name']=="Urban population")|((wbdt['Indicator Name']=="CO2 emissions (kt)"))]
    global nations   # set nations to global variable so taht it can be accessed from anywgere in the code
    nations=["India", "China", "Brazil", "Ireland", "Iraq", "World"]    # selecetd countries
    fltr=wbdt['Country Name'].isin(nations)    # checking whether the countrie are present in the data and take all records
    wbdt=wbdt[fltr]   # filter the data with the taken records
    cntry=wbdt['Country Name'].tolist()    # take the Country Names into a list
    ind=wbdt['Indicator Name'].tolist()    # take the Indicator Names into a list
    merged_wbdt=[]
    for i in range(len(ind)):
        merged_wbdt.append(cntry[i]+"-"+ind[i])    # store all data regarding indicators and countries
    wbdt.insert(4,"Country with Indicator",merged_wbdt)   # inser column name
    wbdt1=wbdt.T.iloc[4:][:-1]        # subset data 
    wbdt=wbdt.drop("Country with Indicator",axis=1)    # drop 'Country with Indicator' column from data
    wbdt1.columns=wbdt1.iloc[0]    # take 1st row for column names
    wbdt1=wbdt1.iloc[1:]    # filter data starting from 1st row (skipping 0th row)
    wbdt1['Year']=wbyrs    # insert years into data
    wbdt1=wbdt1.set_index("Year")    # set year and index (fruitful for time series oplotting)
    return wbdt.reset_index(drop=True).drop(['Unnamed: 65','Country Code','Indicator Code'],axis=1),wbdt1   # return data
print(readWBData.__doc__)


wbd,wbd1=readWBData("CO2_Emission_Population_World_Bank.csv")   # call function to read data


wbd.head()   # First Data with Year Column


wbd1.head()   # second data with country column


# Data Statistics

wbd.describe().T   # data description (transposed to get complete view)


wbd1.describe().T      # data description (transposed to get omplete view)


wbd.info()   # data information


wbd1.info()       # data information

kss=wbd1.kurtosis()   # calculation of kurtosis
fet_kss=kss.index.tolist()   # take the features into list
val_kss=kss.tolist()   # take the kurtosis values into list
kssdf=pd.DataFrame({"Features":fet_kss,"Kurtosis":val_kss})    # create dataframe for kurtosis
kssdf=kssdf.set_index("Features")   # set features as index
kssdf.plot(kind='barh',figsize=(6,4),color=["g"])    # plot the dataframe with horizontal bar type
mpbt.title("Visualization of Kurtosis of Features",fontsize=22,color="g")    # title
mpbt.ylabel("Features",fontsize=18,color="g")    # ylabel assignment
mpbt.xlabel("Kurtosis",fontsize=18,color="g")      # xlabel assignment
mpbt.grid()     # gridding the plot
mpbt.show() 

skw=wbd1.skew()   # calculation of kurtosis
fet_skw=skw.index.tolist()   # take the features into list
val_skw=skw.tolist()   # take the kurtosis values into list
skwdf=pd.DataFrame({"Features":fet_skw,"Skewness":val_skw})    # create dataframe for kurtosis
skwdf=skwdf.set_index("Features")   # set features as index
skwdf.plot(kind='barh',figsize=(6,4),color=["m"])    # plot the dataframe with horizontal bar type
mpbt.title("Visualization of Skewness of Features",fontsize=22,color="m")    # title
mpbt.ylabel("Features",fontsize=18,color="m")    # ylabel assignment
mpbt.xlabel("Skewness",fontsize=18,color="m")      # xlabel assignment
mpbt.grid()     # gridding the plot
mpbt.show() 


# Data Analysis

vr_1,vr_2=[],[]    # take list to store Urban population and CO2 emissions for taken countries
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
for i in range(len(nations)):
    wbdt=wbd[wbd['Country Name']==nations[i]]   # prepare data with country name in each loop
    wbdt=wbdt.drop('Indicator Name',axis=1)
    print("               Statistics for {}".format(nations[i]))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Urban population (Average) for {} => {}".format(nations[i],round(wbdt.iloc[0,1:].mean(),2)))  # print average values for Urban population
    print("CO2 emissions (Average) for {} => {}".format(nations[i],round(wbdt.iloc[1,1:].mean(),2)))  # print average values for CO2 emissions
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    vr_1.append(round(wbdt.iloc[0,1:].mean(),2))    # store average values for Urban population
    vr_2.append(round(wbdt.iloc[1,1:].mean(),2))    # store average values for CO2 emissions
statwbd=pd.DataFrame({"Country":nations,"Urban Population":vr_1,"CO2 emissions":vr_2})   # create data
statwbd=statwbd.sort_values(by="Urban Population")    # sort data by population
ccf=np.corrcoef(statwbd['Urban Population'],statwbd['CO2 emissions'])[0,1]    # Calculating correlation between Urban population and CO2 emissions
print("Correlation between Population and CO2 Emition: {}".format(round(ccf,3)))
statwbd


def colsep(dt):    # function to preapre data for urban population and co2 emission

"""
This function separates the columns of a given dataset into two separate datasets, one for urban population and the other for CO2 emissions. The input parameter 'dt' represents the dataset to be separated. The function first converts the column names of the dataset to a list and then iterates through each column. If the column name contains the string "Urban", it is added to the 'urb' list. Otherwise, it is added to the 'co2' list. The function then prints the list of urban columns and creates two new datasets 'dt1' and 'dt2', which contain only the urban and CO2 columns respectively (except for the "World" column). Finally, the function returns the two new datasets 'dt1' and 'dt2'.
"""
    cls=dt.columns.tolist()   # taking columns into list
    urb,co2=[],[]
    for c in cls:
        if "Urban" in c:   # check if "Urban" into the list element (actually data columns)
            urb.append(c)   # if yes, store the column to urb list
        else:
            co2.append(c)    # if no, store the column to co2 list
    print(urb)
    dt1=wbd1[urb[:-1]]   # create data with urb columns except World
    dt2=wbd1[co2[:-1]]   # create data with co2 columns except World
    return dt1,dt2

print(colsep.__doc__)


def timeseries(dt,ind):
"""
This function creates a time series plot from a given dataset 'dt'. The input parameter 'ind' represents the feature to be plotted against the years. The function first creates a line plot using the 'plot' function of the pandas library and specifies the plot type as 'line'. The plot is then displayed with the specified figure size. The title, xlabel, and ylabel of the plot are assigned using the 'title', 'xlabel', and 'ylabel' functions of the matplotlib library respectively. Finally, the plot is gridded using the 'grid' function and displayed using the 'show' function of the matplotlib library.
"""
    dt.plot(kind='line',figsize=(9,5))    # plotting by features (nm) from dataframe by years
    mpbt.title("{} for Countries by Year".format(ind),fontsize=22,color="b")    # title
    mpbt.ylabel("{}".format(ind),fontsize=18,color="b")    # ylabel assignment
    mpbt.xlabel("Year",fontsize=18,color="b")      # xlabel assignment
    mpbt.grid()     # gridding the plot
    mpbt.show()    # show plot
print(timeseries.__doc__)


wbd11,wbd12=colsep(wbd1)
timeseries(wbd11,"Urban population")
timeseries(wbd12,"CO2 emissions")


def barchart(dt,cl,fet,ci):
"""
This function creates a bar chart from a given dataset 'dt' using a specific column 'ci'. The input parameter 'cl' represents the color of the bars in the plot, 'fet' represents the feature to be plotted against the countries, and 'ci' represents the index of the column to be plotted. If the 'Country' column is present in the dataset, it is set as the index of the dataset using the 'set_index' function of the pandas library. The function then selects the specified column using the 'iloc' function and creates a bar plot using the 'plot' function of the pandas library with the plot type as 'bar'. The plot is then displayed with the specified figure size. The title, xlabel, and ylabel of the plot are assigned using the 'title', 'xlabel', and 'ylabel' functions of the matplotlib library respectively. Finally, the plot is gridded using the 'grid' function and displayed using the 'show' function of the matplotlib library.
"""
    try:
        dt=dt.set_index("Country")
    except:
        pass
    dt.iloc[:,ci].plot(kind="bar",color=cl,figsize=(7,3))
    mpbt.title("{} by Country".format(fet),fontsize=18,color="#000080")   # title
    mpbt.xlabel("Country",fontsize=14,color="#000080")   # x-label of plot
    mpbt.ylabel("{}".format(fet),fontsize=14,color="#000080")   # y-label of plot
    mpbt.grid()   # grid the plot
    mpbt.show()    # show the plot
print(barchart.__doc__)

barchart(statwbd,"#00FFFF","Urban Population",0)
barchart(statwbd,"#FDBD01","CO2 Emissions",1)


wbd1cols=wbd1.columns   # take column names
wbd1idx=wbd1.index   # take data index
wbd1arr=np.array(wbd1.values,float)   # convert data to array
wbd1=pd.DataFrame(wbd1arr,columns=wbd1cols,index=wbd1idx)   # craete datadframe
wbd1.info()

mpbt.figure(figsize=(10,7))
mpbt.title("Correlation Among Indicators",fontsize=20,color="b")
sns.heatmap(wbd1.corr(),annot=True,fmt="0.2f",cmap="mako")   # show visuaization for data correlation
mpbt.show()

for i in range(len(nations)):
    wbdt=wbd[wbd['Country Name']==nations[i]]     # subset data with country name
    wbdt=wbdt.drop('Indicator Name',axis=1)      # remove Indicator Name 
    cls=["Urban population","CO2 emissions"]    # declare indicator names
    wbdt1=wbdt.T     # transpose data
    wbdt1.columns=cls    # take columns
    wbdt1=wbdt1.iloc[1:]    # take data  from 1st row
    wbdt1.plot(kind='area',color=["#87CEFA","#FD1C03"],figsize=(5,3))    # Area Plot
    mpbt.title("Analysis for {}".format(nations[i]),fontsize=18,color="#000080")    # title
    mpbt.xlabel("Year",fontsize=16,color="#000080")    # ylabel assignment
    mpbt.ylabel("Value",fontsize=16,color="#000080")      # xlabel assignment
    mpbt.grid()     # gridding the plot
    mpbt.show()    # show plot



