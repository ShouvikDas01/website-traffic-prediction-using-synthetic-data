import pandas as pd
import numpy as np

# Create date range
start = '2017-01-01'
end = '2022-12-31'  
dates = pd.date_range(start, end, freq='H')

# Generate data points
total_data_points = np.random.randint(50000, 60001)
unique_dates = np.random.choice(dates, total_data_points, replace=True)
timestamps = np.unique(unique_dates)

# Initialize dataframe  
df = pd.DataFrame({'Date': timestamps})

# Add PageLoad column
base_page_load = np.random.randint(1000, 10000, len(timestamps))
fluctuations = np.random.normal(loc=0, scale=0.05, size=len(timestamps))
df['PageLoad'] = base_page_load + fluctuations * base_page_load
df['PageLoad'] = df['PageLoad'].round().astype(int) 

# Add seasonal effects  
daily_seasonality = np.sin(np.linspace(0, 2*np.pi, len(timestamps))) * 50
weekly_seasonality = np.sin(np.linspace(0, 2*np.pi, len(timestamps))) * 100 
monthly_seasonality = np.sin(np.linspace(0, 2*np.pi, len(timestamps))) * 150
df['PageLoad'] += daily_seasonality + weekly_seasonality + monthly_seasonality

# Unique and first time visits
df['UniqueVisits'] = df['PageLoad'] ** 0.8
df['FirstVisits'] = (df['UniqueVisits'] ** 0.5) * 100

# Return visits 
df['ReturnVisits'] = np.random.randint(100, 999, len(timestamps)) - df['UniqueVisits'] 

# To ensure only positive values
mask = df['ReturnVisits'] < 1
df.loc[mask, 'ReturnVisits'] = 100

# Add DayOfWeek
df['DayOfWeek'] = df['Date'].dt.day_name() 


# Add outliers and round integers
outliers_indices = np.random.choice(len(timestamps), 50, replace=False)
df.loc[outliers_indices, 'PageLoad'] *= np.random.uniform(2, 5, 50)

visits_columns = ['PageLoad', 'UniqueVisits', 'ReturnVisits', 'FirstVisits'] 
df[visits_columns] = df[visits_columns].round().astype(int)

# Shuffle and output to CSV
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv('data/web_traffic_data.csv', index=False)
