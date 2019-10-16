import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import csv
from operator import itemgetter

#This code file is written for TCD CS Machine Learning individual contest, but the score is only 144086. By TLIN 2019.10.16

#If an item is existed in the statistic array, the number of that classification is increased by 1, otherwise, the item will be added to the statistic array.
def count_item(statistic_array,income,value):   
    has_it = False
    for row in statistic_array: 
        if value.upper() == row[0].upper():
            has_it = True
            row[1] += 1
            row[2] += income
            break
    if has_it == False:
        statistic_array.append([value.upper(),          #the name of item
                                1,                      #the number of instances
                                income,                 #total income
                                0.0])                   #average income
    return statistic_array

#Sorting the income and calculating the average for each classification.
def sort_item(statistic_array): 
    for row in statistic_array:
        row[3] = row[2]/row[1]
    statistic_array.sort(key=lambda x: x[3])
    number_count    = 0
    weight_count    = 0
    for i in range(len(statistic_array)):
        number = statistic_array[i][1]
        number_count            += number
        weight_count            += i*number
    
    average = weight_count/number_count
    return statistic_array,average

#Reading the CSV file.
def read_csv(file_name):
    with open(file_name, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        headers = next(reader)
        data = list(reader)
    return data

#If the value is #N/A, the value is replaced by average.
def covert_NA(value,average,index):
    if value == '#N/A':
        result = average
    else:
        result = value
    return result   

#Converting text to number, for example: male->1, female->0.
def covert_number(data_row,item_statistic): 
    column_count = len(item_statistic)
    row_x = np.zeros(column_count)
    for i in range(column_count):
        average         = item_statistic[i][1]
        column_index    = item_statistic[i][2]     
        if i in [1,3,5,6,7,8]:
            has_it          = False
            value           = 0
            for j in range(len(item_statistic[i][0])):
                item_name = item_statistic[i][0][j][0]
                if item_name.upper() == data_row[column_index].upper():
                    has_it  = True
                    value   = j
            row_x[i] = value if has_it else average
        if i in [0,2,4,9]:
            row_x[i] = covert_NA(data_row[column_index],average,i)
        else:
            row_x[i] = 0
        
    return row_x

#Writing the CSV file for uploading to Kaggle.
def write_csv(option,array_y,test_data):
    count = len(array_y)
    file_name = 'output.csv'
    with open(file_name, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')
        if option == 0:
            writer.writerow(['Instance', 'Year of Record','Gender','Age','Country','Size of City','Profession','University Degree','Wears Glasses','Hair Color','Body Height [cm]','Income'])
            for i in range(count):
                writer.writerow([test_data[i][0],test_data[i][1],test_data[i][2],test_data[i][3],test_data[i][4],test_data[i][5],
                                 test_data[i][6],test_data[i][7],test_data[i][8],test_data[i][9],test_data[i][10],array_y[i]])
        elif option == 1:
            writer.writerow(['Instance','Income'])
            for i in range(count):
                writer.writerow([test_data[i][0],array_y[i]])

def main():

    origin_data = read_csv('tcdml1920-income-ind/tcd ml 2019-20 income prediction training (with labels).csv')
    test_data   = read_csv('tcdml1920-income-ind/tcd ml 2019-20 income prediction test (without labels).csv')


    #This part is removing the instances with '0' or '#N/A' values.
    training_count      = len(origin_data)
    i = 0
    while i < training_count:
        row = origin_data[i]       
        row[11] = float(row[11])

        if '0' in [row[2],row[7],row[9]]:
            origin_data.remove(row)
            training_count-=1
            continue       
        if '#N/A' in row:
            origin_data.remove(row)
            training_count-=1
            continue
        
        i+=1
        
    training_count = len(origin_data)

    #Calculating the average and doing ranking for gender, country, profession, university, wears glass, and hair color.
    # I store those data in an array.
    item_statistic = []     #The array for storing statistic data.
    for i in range(1,11):
        item_statistic.append([
                    [],     #The detail of tatistic data.
                    0,      #The average of classification.
                    i       #The index of column for the classification in the CSV file.
                    ])     

    for row in origin_data:
        income = row[11]
        for i in range(len(item_statistic)):
            row2 = item_statistic[i]
            if i in [0,2,4,9]:              #0:year 2:age 4:city size 9:body height, just skipping it and doing it in next part
                row2[0] = [['',1,0,0]]
            else:
                row2[0] = count_item(row2[0],income,row[row2[2]])    

    for row in item_statistic:
        row[0],row[1] = sort_item(row[0])

    non_zero_count = 0

    training_array      = [np.zeros(shape=(training_count,10)),np.zeros(training_count)]
    #converting text to number.
    for i in range(training_count):
        row = origin_data[i]
        training_array[1][i] = row[11]      #income
        training_array[0][i] = covert_number(row,item_statistic)

    #this part is calculating average value for year,age,city size and body height.
    year_total = 0
    age_total = 0
    size_total = 0
    height_total = 0

    for row in origin_data:
        year_total+=float(row[1])
        age_total+=float(row[3])
        size_total+=float(row[5])
        height_total+=float(row[10])

    item_statistic[0][1] = year_total / training_count
    item_statistic[2][1] = age_total / training_count
    item_statistic[4][1] = size_total / training_count
    item_statistic[9][1] = height_total / training_count

    #Converting the text value to be number value in testing data.
    test_count = len(test_data)
    test_array = np.zeros(shape=(test_count,10))
    for i in range(test_count):
        test_array[i] = covert_number(test_data[i],item_statistic)

    #Using linear regression to predict income.
    regression = make_pipeline(PolynomialFeatures(3),linear_model.LinearRegression())
    regression.fit(training_array[0],training_array[1])
    array_y = regression.predict(test_array)

    #generating a CSV file.
    write_csv(0,array_y,test_data)
    
if __name__ == '__main__':
    main()
