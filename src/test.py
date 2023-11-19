from datetime import timedelta, datetime
import pandas as pd
import numpy as np
import sys, os

# ------------------------------------- . ------------------------------------ #

df1 = pd.read_parquet("../data/transformed/transformed_dataset.parquet", engine = 'pyarrow')
print(df1.shape)
print(df1)

df2 = pd.read_csv("../data/transformed/transformed_dataset_2.csv", sep=',', decimal='.', encoding='utf-8')
print(df2.shape)
print(df2)

for col in df1.columns: 
    if (df1[col] == df2[col]).all():
        print(f'Columna [{col}] OK')

sys.exit(0)
# ------------------------------------- . ------------------------------------ #

datatypes = {
    'StartTime': str,
    'EndTime': str,
    'AreaID': object,    
    'UnitName': object,
    'PsrType': object,
    'quantity': int, 
    'Load': int
}

def load_data(filepath):
    # Reading the file
    df = pd.read_csv(filepath, sep=',', decimal='.', dtype = datatypes, encoding='utf-8')

    # Converting the dates to datetime format
    df.StartTime = pd.to_datetime(df.StartTime, format='%Y-%m-%dT%H:%M%zZ')
    df.EndTime = pd.to_datetime(df.EndTime, format='%Y-%m-%dT%H:%M%zZ')
    
    # Generating Hour and Date columns for posterior simplicity
    df['Hour'] = df['StartTime'].dt.hour
    df['Date'] = df['StartTime'].dt.date
    
    return df

df = load_data('../data/temporal/gen_HU_B01.csv')

print('Duplicats?')
print(df[df.duplicated()], end = '\n----------------\n')

print('Alguna entrada amb diferència > 15 mins?')
start_times = df.StartTime.to_list()
for i, time in enumerate(start_times[1:]):
    diff = (time - start_times[i]).total_seconds()/60
    if diff != 15:
        print('  Ep! Aquí hi ha una diferència diferent de 15 mins! (%d) %s -> %s' % (diff, start_times[i], time))
print('-----------------')

# sys.exit(0)

# ------------------------------------- . ------------------------------------ #

df = pd.DataFrame({'hour':    ['45',   '00',   '15',  '30', '45'],
                   'value_1': [np.NaN, np.NaN, np.NaN, 4,    np.NaN],
                   'value_2': [8, np.NaN, np.NaN, 4, np.NaN],
                   'value_3': [0,0,0,4,0]})
                   

print(any(df.isna()))

df.interpolate(method='linear', limit_direction='both', inplace=True)

print(df)
print(df.isna().any().any())

sys.exit(0)

# ------------------------------------- . ------------------------------------ #

energy_types = {
    "B01": "Biomass",
    "B09": "Geothermal",
    # "B10": "Hydro Pumped Storage",
    "B11": "Hydro Run-of-river and poundage",
    "B12": "Hydro Water Reservoir",
    "B13": "Marine",
    "B15": "Other renewable",
    "B16": "Solar",
    # "B17": "Waste",
    "B18": "Wind Offshore",
    "B19": "Wind Onshore"
}

regions = {
    'HU': '10YHU-MAVIR----U',
    'IT': '10YIT-GRTN-----B',
    'PO': '10YPL-AREA-----S',
    'SP': '10YES-REE------0',
    'UK': '10Y1001A1001A92E',
    'DE': '10Y1001A1001A83F',
    'DK': '10Y1001A1001A65H',
    'SE': '10YSE-1--------K',
    'NE': '10YNL----------L',
}

filename = '../data/temporal/gen_%s_%s.csv'

datatypes = {
    'StartTime': str,
    'EndTime': str,
    'AreaID': object,    
    'UnitName': object,
    'PsrType': object,
    'quantity': int
}

# ----------------------------------------------------------------------------------------

# DataFrame de l'esquerra (df_merged)
df_merged = pd.DataFrame({
    'Date': ['2023-01-01', '2023-01-02'],
    'Hour': [1, 2],
    'Value_left': [10, 20]
})

# DataFrame de la dreta (df_grouped)
df_grouped = pd.DataFrame({
    'Date': ['2023-01-01', '2023-01-03'],
    'Hour': [1, 3],
    'Value_right': [30, 40]
})

# Right join
result_df = df_merged.merge(df_grouped, how='right', on=['Date', 'Hour'])

# Imprimeix el resultat
print(result_df)

sys.exit(0)

# ----------------------------------------------------------------------------------------

for region in regions: 
    for energy_type in energy_types:
        
        input_file = filename % (region, energy_type)
        
        if os.path.exists(input_file):
        
            df = pd.read_csv(input_file, sep=',', decimal='.', dtype = datatypes, encoding='utf-8')
            
            df['StartTime'] = pd.to_datetime(df['StartTime'].str[:-1], format='%Y-%m-%dT%H:%M%z')
            df['EndTime'] = pd.to_datetime(df['EndTime'].str[:-1], format='%Y-%m-%dT%H:%M%z')
            
            # Deleting previous timepoints
            # print(df.shape)
            df = df[df.StartTime.dt.year > 2021]
            print(df.shape)
            
                    
            df['diffs'] = df.EndTime - df.StartTime
            df.diffs = df.diffs.dt.total_seconds()/60
            # print(df.diffs.value_counts())
            # print(len(df.diffs.value_counts()))
            # for x in df.diffs.value_counts():
            #     print(x)
            
            d = dict(df.diffs.value_counts())
            keys = list(d.keys())
            print('%s   %s   %.1f  %d' % (region, energy_type, keys[0], d[keys[0]]))
            for key in keys[1:]:
                print('           %.1f  %d' % (key, d[key]))
            
            # print(dict(df.diffs.value_counts()))
            
            # print(df.StartTime[:5])
            
            # sys.exit(0)
print('--------------------')

