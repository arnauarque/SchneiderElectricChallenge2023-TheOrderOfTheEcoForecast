import argparse, os, sys
import pandas as pd
import numpy as np
from utils import regions
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

datatypes = {
    'StartTime': str,
    'EndTime': str,
    'AreaID': object,    
    'UnitName': object,
    'PsrType': object,
    'quantity': int, 
    'Load': int
}

# This method plots the 'var' attribute of the 'df' DataFrame
# in original scale and log-scale (side by side)
def plot_transform(df, var, trans, bins = 20):
    fig, axes = plt.subplots(nrows = 1, ncols = 2, 
                             gridspec_kw = {'width_ratios': [2,2]}, 
                             figsize = (10,5))
    # Original scale
    df.hist(column = var, ax = axes[0], bins = bins, 
            density = True, color = 'cornflowerblue')
    sns.kdeplot(data = df, x = var, ax = axes[0], color = 'dimgray');
    axes[0].set_title(None)
    # Transform
    newvar = None
    if trans == 'log':
        if (df[var] <= 0).any():
            print("Warning: Data contains non-positive values, log transformation not applied.")
            newvar = df[var]
        else:
            newvar = np.log10(df[var])
    elif trans == 'boxcox':
        newvar, _ = stats.boxcox(df[var])
        newvar = pd.Series(newvar)
    elif trans == 'sqrt':
        newvar = np.sqrt(df[var])
    else:
        newvar = df[var]
        print('Warning: transformation not recognized.',
              'Showing the originalDF-originalDF plot.')
    newvar.hist(ax = axes[1], bins = bins, 
                density = True, color = 'lightcoral')
    plt.xlabel('%s (%s)' % (var, trans))
    sns.kdeplot(data = newvar, ax = axes[1], color = 'dimgray');
    plt.show()

def plot_evolution(df, column):
    """
    Shows the evolution of an specific column of a pandas dataframe.

    Parameters:
    - df: Pandas DataFrame
    - column: Name of the column for which to plot its evolution
    """
    # Ensure the column exists in the DataFrame
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in the DataFrame.")
    
    fig = plt.figure(figsize=(10, 6))
    
    # Filter rows with non-null values in the current column
    df_filtered = df[['Date', 'Hour', column]].dropna()

    # Plot the points with filled circles and connect them with thin lines
    _ = plt.plot(df_filtered['Date'] + pd.to_timedelta(df_filtered['Hour'], unit='h'), df_filtered[column],
             marker='o', linestyle='-', markersize=2, linewidth=0.8, label=column)

    # Customize the appearance of the plot
    _ = plt.title(f'Evolution in time of column {column}')
    _ = plt.xlabel('Date and Hour')
    _ = plt.ylabel(column)
    _ = plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    fig.axes[0].set_xlim([datetime.date(2022, 1, 1), datetime.date(2023, 1, 1)])
    
    # Show the plot
    plt.show()


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
    

    ###################################################################
    #                            WEEKDAY                              #
    ###################################################################

    # Creating the new feature for the day of the week
    df.Date = pd.to_datetime(df.Date, format='%Y-%m-%d')
    df['Weekday'] = df['Date'].dt.day_name()

    # Create a figure and axis to plot
    plt.figure(figsize=(8, 6))

    # Plot boxplots for office hours and non-office hours
    sns.boxplot(x='Weekday', y='SE_Load', data=df, palette=sns.color_palette("husl", 7) )

    # Set plot labels and title
    plt.xlabel('Weekday')
    plt.ylabel('SE Load')
    plt.title('SE Load Comparison for different Weekdays')

    # Show the plot
    plt.show()

    ###################################################################
    #                         WEEK OF YEAR                            #
    ###################################################################

    # Creating the new feature for the week of the year
    df['Week_of_Year'] = df['Date'].dt.isocalendar().week

    ###################################################################
    #                         MONTH OF YEAR                           #
    ###################################################################

    # Creating the new feature for the month of the year
    df['Month_of_Year'] = df['Date'].dt.month

    ###################################################################
    #                         OFFICE HOURS                            #
    ###################################################################

    # Creating the new feature to consider the Office hours from 9-5
    df['Office_Hours'] = df['Hour'].between(9, 17, inclusive = 'both')
    
    # Create a figure and axis to plot
    plt.figure(figsize=(8, 6))

    # Plot boxplots for office hours and non-office hours
    sns.boxplot(x='Office_Hours', y='SE_Load', data=df, palette={"True": "orange", "False": "blue"})

    # Set plot labels and title
    plt.xlabel('Hour type')
    plt.ylabel('SE Load')
    plt.title('SE Load Comparison between Office Hours and Non-Office Hours')

    # Show the plot
    plt.show()

    df['NE_B01'] = df['NE_B01'] + 1
    df['NE_B01_log'] = np.log10(df['NE_B01'])
    # plot_transform(df, 'NE_B01', 'log', bins = 10)
    
    plot_evolution(df, 'NE_B01_log')
    
    
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