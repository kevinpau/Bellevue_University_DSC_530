"""
File: DSC_530_Paulovici_excercise_2_1.py
Name: Kevin Paulovici
Date: 9/2/2019
Course: DSC 530 Data Exploration and Analysis
Assignment: Exercise 2.1
Description: Python refresher to prepare for EDA
"""
import random

####################################################################################
""" 1) Display the text “Hello World! I wonder why that is always the
 default coding text to start with” """

hello_world = "Hello World! I wonder why that is always the default coding text to start with."
print("1) {}".format(hello_world))

####################################################################################
""" 2) Add two numbers together """

num1 = int(random.uniform(1, 100))
num2 = int(random.uniform(1, 100))
add = num1 + num2

print("2) The sum of {} + {} is: {}".format(num1, num2, add))

####################################################################################
""" 3) Subtract a number from another number """

num3 = int(random.uniform(1, 100))
num4 = int(random.uniform(1, 100))
sub = max(num3, num4) - min(num3, num4)

print("3) {} minus {} is: {}".format(max(num3, num4), min(num3, num4), sub))

####################################################################################
""" 4) Multiply two numbers """

num5 = int(random.uniform(1, 100))
num6 = int(random.uniform(1, 100))
mult = num5 * num6

print("4) {} * {} is: {}".format(num5, num6, mult))

####################################################################################
""" 5) Divide between two numbers """

num7 = int(random.uniform(1, 100))
num8 = int(random.uniform(1, 100))
div = num7 / num8

print("5) {} / {} is: {:.2f}".format(num7, num8, div))

####################################################################################
""" 6) Concatenate two strings together (any words) """

str1 = "I'll have an order of "
str2 = "Spam, Spam, Spam, egg and Spam"

print("6) str1 is: '{}' and str2 is: '{}', \n   concatenating them makes: '{}'.".format
      (str1, str2, str1 + str2))

####################################################################################
""" 7) Create a list of 4 items (can be strings, numbers, both) """

rand_list = [num1, num2, num3, num4]

print("7) The rand_list consists of: {} which is of type {}.".format(rand_list, type(rand_list)))

####################################################################################
""" 8) Append an item to your list (again, can be a string, number) """

rand_list.append(num5)

print("8) The rand_list now consists of: {}.".format(rand_list))

####################################################################################
""" 9) Create a tuple with 4 items (can be strings, numbers, both) """

rand_tuple = (num6, num7, num8, num1)

print("9) The rand_tuple consists of: {} which is of type {}.".format(rand_tuple, type(rand_tuple)))
