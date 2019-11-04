#%% [markdown]
# # Week 10: Exercise 10.2
# File: DSC530_Paulovici_Exercise_10.2.py (.ipynb)<br> 
# Name: Kevin Paulovici<br>
# Date: 11/3/2019<br>
# Course: DSC 530 Data Exploration and Analysis<br>
# Assignment: Chapter 12: 12.1 & 12.2

#%%
import pandas
import numpy as np
import statsmodels.formula.api as smf

import thinkplot
import thinkstats2
import timeseries


#%% [markdown]
# ## Chapter 12

#%% [markdown]
# ### Exercise 12.2
# The linear model I used in this chapter has the obvious drawback that it is linear, and there is no reason to expect prices to change linearly over time. We can add flexibility to the model by adding a quadratic term, as we did in Section 11.3.
# 
# Use a quadratic model to fit the time series of daily prices, and use the model to generate predictions. You will have to write a version of `RunLinearModel` that runs that quadratic model, but after that you should be able to reuse code from the chapter to generate predictions.

#%%
# create the quadratic model from section 12.3 
def RunQuadraticModel(daily):
    """Runs a linear model of prices verus years

    @param: daily (dataframe) - daily prices 

    returns:
    @param: model
    @param: resuls 
    """
    # quadratic term
    daily['years2'] = daily.years**2
    model = smf.ols('ppg ~ years + years2', data = daily)
    results = model.fit()

    return model, results

#%%
# read data from timeseries.py
df = timeseries.ReadData()
df.head()

#%%
# group by quality
dailies = timeseries.GroupByQualityAndDay(df)

# select high for comparisons
name = 'high'
daily = dailies[name] 

#%%
# run the quadratic model
model, results = RunQuadraticModel(daily)
results.summary()

#%%
# plot fitted values
timeseries.PlotFittedValues(model, results, label=name)
thinkplot.Config(title='Fitted Values', xlabel='years',
                xlim=[-0.1, 3.8], ylabel='price ($)/gram')


#%%
# plot predictions

# set linear spacing of years
years = np.linspace(0, 5, 101)

thinkplot.Scatter(daily.years, daily.ppg, alpha=0.1, label=name)
timeseries.PlotPredictions(daily, years, func=RunQuadraticModel)

thinkplot.Config(title='predictions',
                 xlabel='years',
                 xlim=[years[0]-0.1, years[-1]+0.1],
                 ylabel='price ($)/gram')


#%% [markdown]
# ### Exercise 12.2
# Write a definition for a class named `SerialCorrelationTest` that extends `HypothesisTest` from Section 9.2. It should take a series and a lag as data, compute the serial correlation of the series with the given lag, and then compute the p-value of the observed correlation.
# 
# Use this class to test whether the serial correlation in raw price data is statistically significant. Also test the residuals of the linear model and (if you did the previous exercise), the quadratic model.

#%%
# create SerialCorrelationTest class to extends HypothesisTest from Section 9.2
class SerialCorrelationTest(thinkstats2.HypothesisTest):
    """ Test seial correlations
    """

    def TestStatistic(self, data):
        """ computes the test statistic

        @param: data (tuple) - x and y values 

        returns
        @param: test_stat
        """
        series, lag = data
        test_stat = abs(thinkstats2.SerialCorr(series, lag))

        return test_stat

    def RunModel(self):
        """ Run the model of the null hypothesis

        returns:
        @param: permutation, lag - simulated data
        """
        series, lag = self.data
        permutation = series.reindex(np.random.permutation(series.index))

        return permutation, lag

#%%
# correlation test b/w consecutive prices

# group by quality
dailies = timeseries.GroupByQualityAndDay(df)

# select high for comparisons
name = 'high'
daily = dailies[name] 

series = daily.ppg
test = SerialCorrelationTest((series, 1))
p_val = test.PValue()

print("test.actual: {:.3f}, P value: {:.3f}".format(test.actual, p_val))

#%%
# correlation test for residuals on linear model

_, results = timeseries.RunLinearModel(daily)
series = results.resid
test = SerialCorrelationTest((series, 1))
p_val = test.PValue()

print("test.actual: {:.3f}, P value: {:.3f}".format(test.actual, p_val))


#%%
# correlation test for residuals on quadratic model

_, results = RunQuadraticModel(daily)
series = results.resid
test = SerialCorrelationTest((series, 1))
p_val = test.PValue()

print("test.actual: {:.3f}, P value: {:.3f}".format(test.actual, p_val))
