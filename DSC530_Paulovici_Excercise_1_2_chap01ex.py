"""This file contains code for use with "Think Stats",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
############################################################
File: Excercise_1_2_Paulovici_chap01ex.py
Name: Kevin Paulovici
Date: 9/13/2019
Course: DSC 530 Data Exploration and Analysis
Assignment: Chapter 1, 1.2
"""
import numpy as np
import pandas as pd

import nsfg
import thinkstats2


def main():
    """
    """
    dct_file='2002FemResp.dct'
    dat_file = '2002FemResp.dat.gz'

    resp_df = ReadFemResp(dct_file, dat_file)
    PregnumCheck(resp_df)


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


if __name__ == '__main__':
    main()
