import matplotlib.pyplot as plt # for plotting
import numpy as np # doin math ? prolly won't use this
import pandas as pd # reading csv

# read csv file. 
# check how the csv file is formatted
data = pd.read_csv('testData.csv')

# GROUP BY receivers
groups = data['Reciever'].unique()
group_data = {receiver: data[data['Reciever'] == receiver] for receiver in groups}

# plot each receiver with receiver data
for receiver, receiver_data in group_data.items():
    plt.plot(receiver_data['Day'], receiver_data['# of sturgeon'])

plt.show()
