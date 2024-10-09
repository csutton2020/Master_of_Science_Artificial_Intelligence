#!/usr/bin/env python
# coding: utf-8

# In[2]:


import random
from typing import List, Dict, Tuple, Callable
import streamlit as st
import numpy as np


# In[3]:


@st.cache_data
def parse_data(file_name: str) -> List[List]:
    data = []
    file = open(file_name, "r")
    for line in file:
        datum = [float(value) for value in line.rstrip().split(",")]
        data.append(datum)
    random.shuffle(data)
    return data


# In[4]:


data = parse_data("concrete_compressive_strength.csv")


# In[7]:


def data_boundry(data):
    d = np.array(data).T
    min_max =[]
    for component in d:
        min_max.append((np.min(component),np.max(component)))
    return min_max


# In[9]:


boundry = data_boundry(data)


# In[1]:


# build table for displaying results. This table is not used for the full data set because it's too slow
def write_markdown_table(data):
    if not data:
        return ""
    # Determine the number of columns based on the length of the first row
    num_columns = len(data[0])
    
    names= ["Cement (kg/m^3)", "Slag (kg/m^3)", "Ash (kg/m^3)", "Water (kg/m^3)", "Superplasticizer (kg/m^3)", "Coarse Aggregate (kg/m^3)", "Fine Aggregate (kg/m^3)", "Age (Days)", "Compressive Stength (MPa)"]
    # Create the table header
    header = "| " + " | ".join(names) + " |\n"
    separator = "| " + " | ".join(["---"] * num_columns) + " |\n"

    # Create the table rows
    rows = ""
    for row in data:
        row_str = "| " + " | ".join([f"{cell:.2f}" for cell in row]) + " |\n"
        rows += row_str
    markdown_table = header + separator + rows
    return markdown_table


# In[418]:


def distance(point1: List[float], point2: List[float])-> float:
        return np.sqrt(np.sum((np.array(point1[:7]) - np.array(point2[:7])) ** 2))


# In[420]:


def find_all_distances(queryPoint: List[float], train_set: List[List[float]])->List[float] :
        distances = [distance(queryPoint, train_x) for train_x in train_set]
        return distances


# In[422]:


def k_best(k: int, list_of_distance: List[float], train_set: List[List[float]]):
        exported_data= []
        k_sorted = np.argsort(list_of_distance)[:k]
        for i in  k_sorted:
            exported_data.append(train_set[i][:9])
        return ([train_set[i][8] for i in  k_sorted], exported_data)
        


# In[424]:


def knn(k: int, train_set: List[List[float]], query_point: List[float]):
    return (np.mean(k_best(k, find_all_distances(query_point, train_set), train_set)[0]), k_best(k, find_all_distances(query_point, train_set), train_set)[1])


# In[ ]:


st.title("Concrete Compressive Strength Estimation")
# place controls into the sidebar
with st.sidebar:
#     place user input into the expander tab
    with st.expander("Enter your own cement mixture"):
#       inputs are bounded by the data min and max of a given feature. Ex: 100Kg of only water should not have a compressive strength
        cement_number = st.number_input('cement (kg/m^3)', min_value= boundry[0][0], max_value= boundry[0][1])
        slag_number = st.number_input('slag (kg/m^3)',  min_value= boundry[1][0], max_value= boundry[1][1])
        ash_number = st.number_input('ash (kg/m^3)', min_value= boundry[2][0], max_value= boundry[2][1])
        water_number = st.number_input('water (kg/m^3)', min_value= boundry[3][0], max_value= boundry[3][1])
        superplasticizer_number = st.number_input('superplasticizer (kg/m^3)', min_value= boundry[4][0], max_value= boundry[4][1])
        coarse_aggregate_number = st.number_input('coarse aggregate (kg/m^3)',min_value= boundry[5][0], max_value= boundry[5][1])
        fine_aggregate_number = st.number_input('fine aggregate ( kg/m^3)', min_value= boundry[6][0], max_value= boundry[6][1])
        age_number = st.number_input('age (days)', min_value= boundry[7][0], max_value= boundry[7][1])
#     button is in the sidebar and below the expander tab
    button1 = st.button("Calculate Compressive Strength")
    
# collect inputs
new_calculation =[cement_number,
                  slag_number, 
                  ash_number,
                  water_number,
                  superplasticizer_number,
                  coarse_aggregate_number,
                  fine_aggregate_number,
                  age_number
                 ]

np.set_printoptions(threshold=np.inf)

# gather compressive strength estimate and KNN used
estimate, data_points_used = knn(4, data, new_calculation)
new_calculation.append(estimate)

# main window has two tabs
tab1, tab2 = st.tabs(["Estimated Results", "Raw Concrete Data"])
# build tab 1 on button click
with tab1:
    if button1:
        st.header("Concrete Compressive Strength (Mpa)")
        st.metric("",f'{estimate:.2f}')
        st.markdown(write_markdown_table([new_calculation]))
        st.header("Data Points Used for Estimate Calculation ")
        st.markdown(write_markdown_table(data_points_used))
        
# tab2 has full data set for inspection         
with tab2:
    st.header("Raw Concrete Data")
    st.write(np.array(data))


