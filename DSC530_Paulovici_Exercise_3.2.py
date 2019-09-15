#%% [markdown]
# # Week 3: Excercise 3.2
# File: DSC530_Paulovici_Excercise_3.2.py (.ipynb)<br> 
# Name: Kevin Paulovici<br>
# Date: 9/15/2019<br>
# Course: DSC 530 Data Exploration and Analysis<br>
# Assignment: Chapter 1: 1.2 & Chapter 2: 2.2, 2.4


#%% [markdown]
# ## Chapter 1: Excercise 1.2

#%% [markdown]
# ### Directions: 
# Exercise 1.2 In the repository you downloaded, you should find a file named chap01ex.py; using this file as a starting place, write a function that reads the respondent file, 2002FemResp.dat.gz. <br><br>
# The variable pregnum is a recode that indicates how many times each respondent has been pregnant. Print the value counts for this variable and compare them to the published results in the NSFG codebook. <br><br>
# You can also cross-validate the respondent and pregnancy files by comparing pregnum for each respondent with the number of records in the pregnancy file. <br><br> 
# You can use nsfg.MakePregMap to make a dictionary that maps from each caseid to a list of indices into the pregnancy DataFrame.

#%%
import numpy as np
import pandas as pd
import nsfg
import thinkstats2

#%% [markdown]
# ### Read in the data

#%%
def ReadFemResp(file1, file2, nrows=None):
    """ This function takes the dct and dat files (NSFG respondent data),
    reads them and returns a dataframe of the data.

    :param file1: (str) dct filename
    :param file2: (str) dat filename
    :param nrows: option
    :return: df - dataframe of NSFG respondent data
    """
    dct = thinkstats2.ReadStataDct(file1)
    df = dct.ReadFixedWidth(file2, compression="gzip", nrows=nrows)

    return df

dct_file='2002FemResp.dct'
dat_file = '2002FemResp.dat.gz'
resp_df = ReadFemResp(dct_file, dat_file)
resp_df.head()

#%% [markdown]
# ### Count pregnancies and cross-validate

#%%
def PregnumCheck(df):
    """ This function counts the number of pregnancies and
        cross-validate the respondent and pregnancy files by comparing
        pregnum for each respondent with the number of records in the
        pregnancy file (df)

    :param df: dataframe of NSFG respondent data
    """
    # data series of: index (# of pregnancies) | values (participants)
    preg_num = df.pregnum.value_counts().sort_index()

    # sum of pregnancy respondents
    total = df.pregnum.value_counts().sum()

    # used for higher # of pregnancies
    count = 0

    # loop through the preg_num data series to print statement
    # formatted for comparison to NSFG codebook
    print("***** Pregnancy Data *****")
    for index, value in preg_num.items():
        if index < 7:
            print("{} pregnancies = {} respondents".format(index, value))
        else:
            count += value
    else:
        print("7 or more pregnancies = {} respondents".format(count))

    # cross-validation
    if total == len(df):
        print("***** cross-validation *****")
        print("The respondents match the pregnancy data, which is: {}".format(total))

PregnumCheck(resp_df)

#%% [markdown]
# ## Chapter 2: Excercise 2.1

#%% [markdown]
# ### Directions:
# Exercise 2.1 Based on the results in this chapter, suppose you were asked to summarize what you learned about whether first babies arrive late. Which summary statistics would you use if you wanted to get a story on the evening news? Which ones would you use if you wanted to reassure an anxious patient?<br> <br>

# To reassure an anxious patient I would use the standard deviation. I would also put it into contex for them so they understood the impact. To make the evening news I would probably use the mean. However, I would also want to include outliers inorder to get extreme values.<br><br>

# Finally, imagine that you are Cecil Adams, author of The Straight Dope (http://straightdope.com), and your job is to answer the question, “Do first babies arrive late?” Write a paragraph that uses the results in this chapter to answer the question clearly, precisely, and honestly.<br><br>

# From the 7643 participants in the National Survey of Family Growth (NSFG), we can finally answer whether first babies arrive late! So, do they? Yes... and no, our research indicates that it depends. There is no hard evident to support that first born babies will always arrive late. A standard deviation of 0.029 was determined between first born and non-first born babies; which is fairly insignificant. With additional data we can continue to minimize the standard deviation. Further studies and funding can provide additional insight how often first born babies arrive late. 

#%% [markdown]
# ## Chapter 2: Excercise 2.4

#%% [markdown]
# ### Directions:
# Exercise 2.4 Using the variable totalwgt_lb, investigate whether first babies are lighter or heavier than others. Compute Cohen’s d to quantify the difference between the groups. How does it compare to the difference in pregnancy length?


#%%
import numpy as np
import pandas as pd
import nsfg
import thinkstats2
import math

#%% [markdown]
# ### Read in data

#%%
# Read in the dataset
df = nsfg.ReadFemPreg()
preg = df[df.outcome == 1]
preg.head()

#%%
# filter dataframe for cols of interest (easier to work with)
preg_filtered = preg.filter(items = ["caseid", "outcome", "birthord", 
"prglngth", "totalwgt_lb"])
preg_filtered.head()

#%%
# split preg_filtered to get first babies
first = preg_filtered[preg_filtered.birthord == 1]
first.head()
#%%
# split preg_filtered to get other babies
other = preg_filtered[preg_filtered.birthord != 1]
other.head()

#%% [markdown]
# ### Histogram

#%%
hist_plot = first["totalwgt_lb"].hist()
hist_plot.set_title("totalwgt_lb for first born")
hist_plot.set_xlabel("Weight (lb)")
hist_plot.set_ylabel("Frequency")

#%%
hist_plot = first["totalwgt_lb"].hist()
hist_plot.set_title("totalwgt_lb for other born")
hist_plot.set_xlabel("Weight (lb)")
hist_plot.set_ylabel("Frequency")

#%% [markdown]
# ### Cohen's d Calculation

#%% 
# function for cohens d
def CohenD(f, o, col):
    """ function calc the cohen's d statistic 

    @param: f (dataframe) - first babies
    @param: o (dataframe) - other babies
    @param: col (str) - column of interest 
    returns: d (float) - cohens d rounded to 4 sig. figs
    """
    diff = f[col].mean() - o[col].mean()

    var1 = f[col].var()
    var2 = o[col].var()
    n1, n2 = len(f), len(o)

    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    d = diff / math.sqrt(pooled_var)

    return round(d, 4)

#%%
# cohen's d for totalwgt_lb
cohen_d = CohenD(first, other, "totalwgt_lb")
cohen_d

#%%
# cohen's d for prglngth
cohen_d = CohenD(first, other, "prglngth")
cohen_d

#%% [markdown]
# ### Cohen's d Summary
# The cohen's d value for totalwgt_lb gives us a value of -0.0887. The cohen's d value for prglngth gives us a values of 0.0289. This tells us the pregnency length does not vary much between first and other born babies. However, the totalwgt_lb varies about three times as much compared to prglength.  
