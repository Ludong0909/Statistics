# %%
import pandas as pd
import os

# %%
def try_read_file(file_path):
    for encoding in ['ISO-8859-1', 'utf-8']:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                for line_num, line in enumerate(file):
                    if line.startswith('#'):
                        # Remove the '#' symbol and the following space
                        cleaned_header = line[1:].strip().split()
                        return line_num, cleaned_header
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to decode file {file_path} with available encodings.")

def process_monthly_files(base_dir, start_year, end_year, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    all_data = {}  # Dictionary to store data for each station
    
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Define the directory path based on year
            year_dir = os.path.join(base_dir, f'{year}9999_cwb_dy')
            # Define the file path based on year and month
            file_path = os.path.join(year_dir, f'{year}{str(month).zfill(2)}99.cwb_dy.txt')
            
            if os.path.exists(file_path):
                print(f'Processing file: {file_path}')
                
                # Find and clean the header line
                header_line, cleaned_header = try_read_file(file_path)
                if header_line is None:
                    print(f'Header line not found in file {file_path}')
                    continue
                
                # Read the data file starting from the line after the header line
                try:
                    df = pd.read_csv(file_path, delim_whitespace=True, skiprows=header_line + 1, names=cleaned_header, encoding='utf-8')
                except Exception as e:
                    print(f'Error reading file {file_path}: {e}')
                    continue
                
                # Extract the necessary columns
                required_columns = ['stno', 'yyyymmdd', 'TX01', 'PP01']
                df_filtered = df[required_columns]
                
                # Get unique station codes
                stations = df_filtered['stno'].unique()
                
                # Iterate through each station and store its data in the dictionary
                for station in stations:
                    station_data = df_filtered[df_filtered['stno'] == station]
                    
                    if station not in all_data:
                        all_data[station] = []
                    
                    all_data[station].append(station_data)
            else:
                print(f'File not found: {file_path}')
    
    # Process and save data for each station
    for station, data_list in all_data.items():
        # Concatenate all data for the station
        station_df = pd.concat(data_list)
        
        # Extract year and day from 'yyyymmdd'
        station_df['year'] = station_df['yyyymmdd'].astype(str).str[:4].astype(int)
        station_df['day'] = station_df['yyyymmdd'].astype(str).str[4:].astype(int)
        
        # Pivot the data to get the desired shape
        temperature_pivot = station_df.pivot(index='year', columns='day', values='TX01')
        precipitation_pivot = station_df.pivot(index='year', columns='day', values='PP01')
        
        # Save the reshaped data
        temp_output_file = os.path.join(output_dir, f'{station}_temperature.csv')
        precip_output_file = os.path.join(output_dir, f'{station}_precipitation.csv')
        
        temperature_pivot.to_csv(temp_output_file)
        precipitation_pivot.to_csv(precip_output_file)

# %%
if __name__ == "__main__":
    # Define the base directory containing the data files and the output directory
    # get the current working directory
    current_dir = os.getcwd()
    base_dir = current_dir + '/raw/cwb_dy'
    output_dir = current_dir + '/st_dy'

    # Define the range of years to process
    start_year = 1897
    end_year = 2023

    # Run the function to process files and extract data
    process_monthly_files(base_dir, start_year, end_year, output_dir)