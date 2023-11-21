import argparse, os, sys, datetime
import pandas as pd
import numpy as np
from utils import regions
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    return df

def compute_max_region(row):
    max_region = max(regions, key=lambda region: row[f'green_energy_{region}'] - row[f'{region}_Load'])
    return max_region

def get_finest_granularity(df):
    """
    Returns the minimum difference of time (EndTime-StartTime) of a dataframe 
    containing two datetime columns StartTime and EndTime.
    
    Parameters: 
    - df: Pandas DataFrame containing the aforementioned columns
    """
    
    diffs = (df['EndTime'] - df['StartTime']).dt.total_seconds() / 60
    return diffs.min()

def impute_missing_values(df, column):
    """
    Imputes the missing values in a column of a DataFrame, taking into account 
    the minimum granularity of time intervals (StartTime, EndTime).

    Parameters:
    - df: Pandas DataFrame containing columns StartTime, EndTime, and column
    - column: The column containing the missing values to be imputed
    
    Return: 
    Returns a Pandas DataFrame with columns Date, Hour, and column. The missing 
    values in the column have been imputed.
    """
    
    # Data range considered in this problem
    date_range = pd.date_range(start='2022-01-01 00:00', end='2023-01-01 00:00', freq='15T')
    
    # Create a DataFrame that includes all slots with minimum granularity 
    # time intervals
    df_full = pd.DataFrame(date_range, columns=['FullDate'])
    
    # Set the correct format for StartTime and EndTime
    df.StartTime = df.StartTime.dt.strftime('%Y-%m-%d %H:%M:%S')
    df.StartTime = pd.to_datetime(df.StartTime, format='%Y-%m-%d %H:%M:%S')
    
    # Merge with the DataFrame that contains the necessary slots
    df_merged = df_full.merge(df, how='left', left_on='FullDate', right_on='StartTime')
    
    # Generate Date and Hour variables and drop unnecessary columns
    df_merged['Date'] = df_merged.FullDate.dt.date
    df_merged['Hour'] = df_merged.FullDate.dt.hour
    df_merged = df_merged[['Date', 'Hour', column]]
    
    # Filter rows in the DataFrame to eliminate those
    # that have no value in the one-hour interval. We only want
    # to impute missing values that belong to an hour where
    # there is some known value.
    # Example 1: We will impute values (2, 3, 4) because in the hour
    # 00:00 there is a known value (at 00:15 -> 10)
    #  1) 00:00 -> NaN
    #  2) 00:15 -> 10
    #  3) 00:30 -> NaN
    #  4) 00:45 -> NaN
    # Example 2: We do not impute any values and remove rows 
    # from the hour 01:00 because in the one-hour interval there is 
    # no known value
    #  1) 01:00 -> NaN
    #  2) 01:15 -> 10
    #  3) 01:30 -> NaN
    #  4) 01:45 -> NaN
    df_filtered = df_merged.groupby([df_merged.Date, df_merged.Hour])\
        .filter(lambda x: x[column].notnull().any())
    
    # Interpolate missing values using linear method
    df_filtered.interpolate(method='linear', limit_direction='both', inplace=True)
    
    return df_filtered

def myplot(df, column):
    fig = plt.figure(figsize=(10, 6))
    
    # Filter rows with non-null values in the current column
    # df_filtered = df[['Date', 'Hour', column]].dropna()

    # Plot the points with filled circles and connect them with thin lines
    _ = plt.plot(df['Date'] + pd.to_timedelta(df['Hour'], unit='h'), df[column],
             marker='o', linestyle='-', markersize=2, linewidth=0.8, label=column)

    # Customize the appearance of the plot
    _ = plt.title(f'Evolution in time of column {column}')
    _ = plt.xlabel('Date and Hour')
    _ = plt.ylabel(column)
    _ = plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    serie_ordenada = df[column].sort_values()
    # Aconsegueix els valors mínim, dos intermig i màxim en una llista
    valors_seleccionats = [serie_ordenada.min(),
                            serie_ordenada.quantile(0.25),
                            serie_ordenada.quantile(0.75),
                            serie_ordenada.max()]
    plt.yticks(valors_seleccionats)
    
    fig.axes[0].set_xlim([datetime.date(2022, 1, 1), datetime.date(2023, 1, 1)])
    
    # Show the plot
    plt.show()

def transform_merge_data(data_path):
    """
    Reads the data from the temporary directory, transforms it considering 
    its nature (GeneratedEnergy or Load), and merges it into a single 
    DataFrame with hourly granularity. If there are missing values between 
    the intervals of each data source, they are imputed using linear 
    interpolation. If there is a complete one-hour interval without data, 
    it is set as a missing value.

    Parameters:
    - data_path: Path where the temporary data is located

    Return:
    Returns a Pandas DataFrame with all the data (GeneratedEnergy and Load) 
    grouped into one-hour intervals.
    """
    
    # ---------------------------------------------------------------------------- #
    #                         Merging GeneratedEnergy data                         #
    # ---------------------------------------------------------------------------- #
            
    # We create an empty dataframe to merge it with the others
    df_merged = pd.DataFrame(columns=['Date','Hour'])#columns=column_names)
    
    # Collecting the files regarding the generation of energy 
    gen_filenames = [ filename for filename in os.listdir(data_path) if 'gen' in filename and filename.endswith('.parquet')]
    
    for filename in gen_filenames:
        
        # Loading Gen data
        df = load_data(os.path.join(data_path, filename))
        
        # ------------------------- Preliminar Data Cleaning ------------------------- #
        
        # We drop the rows which have no AreaID, since they are duplicated
        df.dropna(subset=['AreaID'], inplace=True)
        
        # Imputing missing values
        df_clean = impute_missing_values(df, 'quantity')
        
        # ---------------------- End of Preliminar Data Cleaning --------------------- #
        
        # Uncomment to show the difference between imputing vs not imputing missing values
        # df['Date'] = df.StartTime.dt.date
        # df['Hour'] = df.StartTime.dt.hour
        # df_grouped_old = df.groupby(['Date', 'Hour']).agg({'quantity': np.sum}).reset_index()
        
        # Grouping the data by Hour and aggregating the 'quantity' column
        df_grouped = df_clean.groupby(['Date', 'Hour']).agg({'quantity': np.sum}).reset_index()
        
        # Uncomment to show the difference between imputing vs not imputing missing values
        # if 'SP' in filename:
        #     myplot(df_grouped_old, 'quantity')
        #     myplot(df_grouped, 'quantity')
                
        # Merging the grouped dataset with imputed missing values with the global one
        df_merged = df_merged.merge(df_grouped, how='outer', on=['Date','Hour'])

        # Renaming the merged column
        tag = filename[4:-8]
        df_merged = df_merged.rename(columns = {'quantity': tag})
    
    # ---------------------------------------------------------------------------- #
    #                               Merging Load data                              #
    # ---------------------------------------------------------------------------- #
    
    # Collecting the files regarding the energy load
    load_filenames = [ filename for filename in os.listdir(data_path) if 'load' in filename and filename.endswith('.parquet')]
    
    for filename in load_filenames:
        
        # Loading Load data
        df = load_data(os.path.join(data_path, filename))
        
        # ------------------------- Preliminar Data Cleaning ------------------------- #
        
        # We drop the rows which have no AreaID, since they are duplicated
        df.dropna(subset=['AreaID'], inplace=True)
        
        # Imputing missing values
        df_clean = impute_missing_values(df, 'Load')
        
        # ---------------------- End of Preliminar Data Cleaning --------------------- #
        
        # Grouping the data by hour by aggregating the Load column
        df_grouped = df_clean.groupby(['Date', 'Hour']).agg({'Load': np.sum}).reset_index()
        
        # Merging the grouped dataset with imputed missing values with the global one
        df_merged = df_merged.merge(df_grouped, how='outer', on=['Date','Hour'])

        # Renaming the merged column
        tag = filename[5:-8] + '_Load'
        df_merged = df_merged.rename(columns = {'Load': tag})
    
    df_sorted = df_merged.sort_values(by=['Date', 'Hour']).reset_index(drop = True)
    
    df_sorted.to_parquet('../data/transformed/transformed_dataset.parquet', index = False, engine = 'pyarrow')

    return df_sorted

def clean_data(df):
    
    # We interpolate the missing values using Linear interpolation
    df.interpolate(method='linear', limit_direction='both', inplace=True)
    
    # Creating green_energy_REGION and deleting REGION_Bxx varibles. 
    for region in regions:
        df[f'green_energy_{region}'] = df.filter(like=f'{region}_B').sum(axis=1)
        df.drop(df.filter(like=f'{region}_B').columns, axis=1, inplace=True)
    
    
    
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
    # df['Week_of_Year'] = df['Date'].dt.isocalendar().week
    df['Week_of_Year_sin'] = np.sin(2 * np.pi * df['Date'].dt.isocalendar().week / 52)
    df['Week_of_Year_cos'] = np.cos(2 * np.pi * df['Date'].dt.isocalendar().week / 52)

    ###################################################################
    #                         MONTH OF YEAR                           #
    ###################################################################

    # Creating the new feature for the month of the year
    # df['Month_of_Year'] = df['Date'].dt.month

    # Making a Sinusoidal Transformation to keep the cyclical meaning
    df['Month_of_Year_sin'] = np.sin(2 * np.pi * df['Date'].dt.month / 12)
    df['Month_of_Year_cos'] = np.cos(2 * np.pi * df['Date'].dt.month / 12)


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
    
    print(df.iloc[:, :2].join(df.iloc[:, -4:]))
    sys.exit(0)
    return df_processed

def save_data(df, output_file):
    # TODO: Save processed data to a CSV file
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Data processing script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_data_path',
        type=str,
        default='../data/',
        help='Path to the raw data directory to process'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='../data/preprocessed/processed_data.csv', 
        help='Path to save the processed data'
    )
    return parser.parse_args()

def main(input_file, output_file):
    # DANIEL
    _ = transform_merge_data(input_file)
    sys.exit(0)
    # df_clean = clean_data(df)
    # df_processed = preprocess_data(df_clean)

    df_clean = pd.read_parquet("../data/transformed/transformed_dataset.parquet", engine = 'pyarrow')
    df_processed = preprocess_data(df_clean)
    save_data(df_processed, output_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.output_file)