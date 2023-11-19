import argparse, os, sys
import pandas as pd
import numpy as np
from utils import regions
import matplotlib.pyplot as plt

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
    # df = pd.read_csv(filepath, sep=',', decimal='.', dtype = datatypes, encoding='utf-8')
    df = pd.read_parquet(filepath, engine = 'pyarrow')

    # # Setting the correct types
    df.StartTime = pd.to_datetime(df.StartTime, format='%Y-%m-%dT%H:%M%zZ')
    df.EndTime = pd.to_datetime(df.EndTime, format='%Y-%m-%dT%H:%M%zZ')
    df.AreaID = df.AreaID.astype(object)
    df.UnitName = df.UnitName.astype(object)
    if 'PsrType' in df.columns:
        df.PsrType = df.PsrType.astype(object)
    if 'quantity' in df.columns:
        df.quantity = df.quantity.astype(int)
    if 'Load' in df.columns:
        df.Load = df.Load.astype(int)

    # Generating Hour and Date columns for posterior simplicity
    df['Hour'] = df['StartTime'].dt.hour
    df['Date'] = df['StartTime'].dt.date
    return df

def compute_max_region(row):
    max_region = max(regions, key=lambda region: row[f'green_energy_{region}'] - row[f'{region}_Load'])
    return max_region

def transform_merge_data(file_path):
    # ----------------------------- Merging GEN data ----------------------------- #
            
    # We create an empty dataframe to merge it with the others
    df_merged = pd.DataFrame(columns=['Date','Hour'])#columns=column_names)
    
    # Collecting the files regarding the generation of energy
    # gen_filenames = [ filename for filename in os.listdir(file_path) if 'gen' in filename and filename.endswith('.csv')]    
    gen_filenames = [ filename for filename in os.listdir(file_path) if 'gen' in filename and filename.endswith('.parquet')]
    
    for filename in gen_filenames:
        
        # Loading Gen data
        df = load_data(os.path.join(file_path, filename))
        
        # ---------------------------------------------------------------------------- #
        #                                MOURE AL CLEAN                                #
        # ---------------------------------------------------------------------------- #
        df.dropna(subset=['AreaID'], inplace=True)
        # ------------------------------------- . ------------------------------------ #
        
        df_grouped = df.groupby(['Date', 'Hour']).agg({'quantity': np.sum}).reset_index()
        df_merged = df_merged.merge(df_grouped, how='outer', on=['Date','Hour'])

        # tag = filename[4:-4]
        tag = filename[4:-8]
        df_merged = df_merged.rename(columns = {'quantity': tag})
        
        # umbral = len(df_merged.columns) - 1  # Se mantiene la fila si tiene al menos len(df.columns) - umbral valores no nulos
        # nuevo_df = df_merged.dropna(thresh=umbral)
    
    # ----------------------------- Merging LOAD data ---------------------------- #
    
    # Collecting the files regarding the energy load
    # load_filenames = [ filename for filename in os.listdir(file_path) if 'load' in filename and filename.endswith('.csv')]
    load_filenames = [ filename for filename in os.listdir(file_path) if 'load' in filename and filename.endswith('.parquet')]
    
    for filename in load_filenames:
        
        # Loading Load data
        df = load_data(os.path.join(file_path, filename))
        
        # ---------------------------------------------------------------------------- #
        #                                MOURE AL CLEAN                                #
        # ---------------------------------------------------------------------------- #
        df.dropna(subset=['AreaID'], inplace=True)
        # ------------------------------------- . ------------------------------------ #
        
        df_grouped = df.groupby(['Date', 'Hour']).agg({'Load': np.sum}).reset_index()

        df_merged = df_merged.merge(df_grouped, how='outer', on=['Date','Hour'])

        # tag = filename[5:-4] + '_Load'
        tag = filename[5:-8] + '_Load'
        df_merged = df_merged.rename(columns = {'Load': tag})

        # umbral = len(df_merged.columns) - 1  # Se mantiene la fila si tiene al menos len(df.columns) - umbral valores no nulos
        # nuevo_df = df_merged.dropna(thresh=umbral)
    
    # ---------------------------- Generating outcomes --------------------------- #
    
    print(df_merged.head(5))
    
    df_sorted = df_merged.sort_values(by=['Date', 'Hour']).reset_index(drop = True)
    print('-----------------')
    print(df_sorted.head(5))
    
    # df_merged.to_csv('../data/transformed/transformed_dataset_2.csv', index = False, sep = ',', encoding = 'utf-8')
    df_sorted.to_parquet('../data/transformed/transformed_dataset.parquet', index = False, engine = 'pyarrow')

    return df_merged

def clean_data(df):
    # TODO: Handle missing values, outliers, etc.
    # We interpolate the missing values
    # df_merged.interpolate(method='linear', limit_direction='both', inplace=True)
    
    # ---------------------------------------------------------------------------- #
    #                           PER A L'ETPA DE CLEANING                           #
    # ---------------------------------------------------------------------------- #
    # for region in regions: 
    #     cols = df_merged.filter(regex = f'^{region}_').columns
    #     df_merged[f'green_energy_{region}'] = df_merged[cols].sum(axis = 1)
    #     df_merged.drop(cols, axis = 1, inplace = True)
    
    # Generating the targets
    # df_merged['target'] = df_merged.apply(compute_max_region, axis=1).shift(-1)
    df_clean = df

    return df_clean
    
def preprocess_data(df):
    # TODO: Generate new features, transform existing features, resampling, etc.
    print(df)
    print(df.dtypes)
    
    # Creating the new feature for the day of the week
    df.Date = pd.to_datetime(df.Date, format='%Y-%m-%d')
    df['Weekday'] = df['Date'].dt.day_name()

    # Creating the new feature for the week of the year
    df['Week_of_Year'] = df['Date'].dt.isocalendar().week

    # Creating the new feature for the month of the year
    df['Month_of_Year'] = df['Date'].dt.month

    # Creating the new feature to consider the Office hours from 9-5
    df['Office_Hours'] = df['Hour'].between(9, 17, inclusive = 'both')

    df['date_time'] = df['Date'] + pd.to_timedelta(df['Hour'], unit='h')

    # Plotting
    plt.figure(figsize=(10, 6))

    # Separate data based on 'Office_Hours' value
    office_hours = df[df['Office_Hours'] == True]
    non_office_hours = df[df['Office_Hours'] == False]

    # Plotting points for Office Hours in red and Non-Office Hours in blue
    plt.scatter(office_hours['date_time'], office_hours['SE_Load'], color='red', label='Office Hours')
    plt.scatter(non_office_hours['date_time'], non_office_hours['SE_Load'], color='blue', label='Non-Office Hours')

    plt.xlabel('Date')
    plt.ylabel('SE_Load')
    plt.title('SE_Load over Time')
    plt.legend()
    plt.show()

    # si hay sol o no?
    # donde geograficamente, distancias epicentro o algo?
    # si es festivo o laborable ?
    # si es vispera de festivo ?
    
    print(df.iloc[:, :2].join(df.iloc[:, -4:]))
    sys.exit(0)
    return df_processed

def save_data(df, output_file):
    # TODO: Save processed data to a CSV file
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Data processing script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file',
        type=str,
        default='data/raw_data.csv',
        help='Path to the raw data file to process'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='data/processed_data.csv', 
        help='Path to save the processed data'
    )
    return parser.parse_args()

def main(input_file, output_file):
    # DANIEL
    # df = transform_merge_data(input_file)
    # df_clean = clean_data(df)
    # df_processed = preprocess_data(df_clean)

    df_clean = pd.read_parquet("../data/transformed/transformed_dataset.parquet", engine = 'pyarrow')
    df_processed = preprocess_data(df_clean)

    save_data(df_processed, output_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.output_file)