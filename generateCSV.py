import os
import csv

#FilePaths 
root_path = "/home/tdawg/Desktop/ML_Project/data/data/data_reduced/"        #Root Path is where you store your data in the positive and negative folders 
hel_path = "/home/tdawg/Desktop/ML_Project/data/hel_filenames_labels.csv"   #This is the path for the Healthy filenames and labels
dis_path = "/home/tdawg/Desktop/ML_Project/data/dis_filenames_labels.csv"   #This is the path for the Diseased filenames and labels


#Prparing the CSV writer for the healthy names
f=open(hel_path,'w')
w=csv.writer(f)

#Loop that writes a row to the csv for each file in healthy. 
in_path = root_path + "Healthy"         #Change this to your label. 
for path, dirs, files in os.walk(in_path):
    for filename in files:
        data = [in_path+"/"+filename, 0] #Healthy = 0 
        w.writerow(data)

#Prparing the CSV writer for the diseased names
f=open(dis_path,'w')
w=csv.writer(f)


#Loop that writes a row to the csv for each file in diseased
in_path = root_path + "Diseased"     #Change this to your label. 
for path, dirs, files in os.walk(in_path):
    for filename in files:
        data = [in_path+"/"+filename, 1] #Diseased = 1
        w.writerow(data)



