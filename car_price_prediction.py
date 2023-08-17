import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')

import matplotlib
print(np.__version__, pd.__version__, sns.__version__, matplotlib.__version__)

# Load the dataset
df = pd.read_csv('car_price_dataset.csv')

# Explore the data
#print(df.head())
#print(df.shape())
#print(df.describe())
#print(df.info())
print(df.columns)

# rename columns
df.rename(columns = {'name': 'Name', 'year': 'Year', 'selling_price' : 'Sell-price', 'km_driven' : 'Driven(KM)', 'fuel' : 'Fuel', 'seller_type' : 'Sell-type',
       'transmission' : 'Tran', 'owner' : 'Own', 'mileage' : 'Mil', 'engine' : 'eng', 'max_power' : 'M-power', 'torque' : 'Torque',
       'seats' : 'Seats'}, inplace = True)

print(df.columns)


# Let's see how many developing and developed countries there are
sns.countplot(data = df, x = 'M-power')
sns.displot(data = df, x = 'Driven(KM)')
plt.show()




