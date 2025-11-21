# ToDO:
# -document check_and_combine_humidity_temperature
# -mover datos de /home/jalvarezbe/TFG-CALIDAD-DEL-AIRE/Datos/Datos_originales
#   a /home/jalvarezbe/TFG-CALIDAD-DEL-AIRE/Datos/Datos_originales/humidity_temperature
# -exportar dataframes finales de cada cosa




import os
import re
import pandas as pd
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import numpy as np

# GLOBAL CONFIGURATION
'''The `CHUNK_SIZE` parameter represents the number of rows to read at a time when processing the data files. It is used for
    chunking the data to manage memory efficiently, especially when dealing with large datasets. This
    parameter allows the function to read and write by parts.
    '''
CHUNK_SIZE = 20000
HOUR_COLS = [f"H{h:02d}" for h in range(1, 25)]  # H01..H24


def check_and_combine_air_quality_per_type(data_dir,
                                  years_list,
                                  output_dir):
    """
    The function `check_and_combine_air_quality_per_type` processes air quality data files by finding,
    organizing, checking for consistency, and combining them into a single file per type over specified
    years.
    
    :param data_dir: `data_dir` is the directory where the air quality data files are located
    :param years_list: The `years_list` parameter is a list of years for which you want to check and
    combine air quality data files. It is used to filter and process files specific to the years
    provided in the list
    :param output_dir: The `output_dir` parameter in the `check_and_combine_air_quality_per_type` function refers
    to the directory where the combined air quality data files will be saved. This directory should be
    specified as a path where the function will store the final combined data files with the name type_2016_2019.tsv
    """
    # === STEP 1. FIND ALL FILES ===
    filename_pattern = re.compile(r"(?P<type>.+?)_HH_(?P<year>\d{4})\.(csv|xlsx)$", re.IGNORECASE)


    files_found = []

    for year in years_list:
        for root, _, files in os.walk(data_dir / year):
            for f in files:
                if f.lower().endswith((".csv", ".xlsx")):
                    match = filename_pattern.match(f)
                    if match:
                        file_type = match.group("type")
                        # Normalize inconsistent type names
                        if file_type.upper() == "PM2.5":
                            file_type = "PM25"
                        files_found.append({
                            "year": year,
                            "type": file_type,
                            "path": os.path.join(root, f)
                        })

    # === STEP 2. BUILD FILE MAPPING BY TYPE ===
    from collections import defaultdict
    files_by_type = defaultdict(dict)

    for entry in files_found:
        files_by_type[entry["type"]][entry["year"]] = entry["path"]

    # === STEP 3. REPORT MISSING FILES ===
    print("🔍 Checking for missing files per type...\n")

    missing_report = []
    for t, year_map in files_by_type.items():
        missing_years = [y for y in years_list if y not in year_map]
        if missing_years:
            print(f"⚠ {t}: missing years {', '.join(missing_years)}")
            missing_report.append((t, missing_years))
        else:
            print(f"✅ {t}: all years present")


    # === STEP 4. COMBINE FILES PER TYPE ===
    print("\n📦 Verifying and combining data for each complete type...\n")

    output_dir.mkdir(exist_ok=True)

    for t, year_map in files_by_type.items():
        if any(y not in year_map for y in years_list):
            continue  # skip incomplete types

        # --- STEP 4a: Verify column consistency first (lightweight check)
        columns_by_year = {}
        consistent = True

        for y in sorted(year_map.keys()):
            path = year_map[y]
            print(f"   → Checking columns in {path}")
            try:
                if path.lower().endswith(".csv"):
                    cols = pd.read_csv(path, nrows=0).columns.tolist()
                else:
                    cols = pd.read_excel(path, nrows=0).columns.tolist()
            except Exception as e:
                print(f"⚠ Error reading columns from {path}: {e}")
                consistent = False
                break
            
            # Normalize semicolon-separated single strings 
            if len(cols) == 1 and ";" in cols[0]:
                columns_by_year[y] = [c.strip() for c in cols[0].split(";")]
            
            else:
                columns_by_year[y] = cols


    # Compare all lists to the reference (first year's columns)
    years_sorted = sorted(columns_by_year.keys())
    ref_year = years_sorted[0]
    ref_cols = columns_by_year[ref_year]

    mismatch_found = False

    for year in years_sorted[1:]:
        cols = columns_by_year[year]
        if cols != ref_cols:
            mismatch_found = True
            print(f"⚠ Column mismatch in {year}:")
            # Show side-by-side difference
            print(f"   Expected ({ref_year}): {ref_cols}")
            print(f"   Found ({year}):     {cols}\n")

    if not mismatch_found:
        print("✅ All years have identical column names and order.\n")
    else:
        print("⛔ Some years have inconsistent columns.\n")



    # Concatenate files of the same type chronologically by year
    for t, year_map in files_by_type.items():
        # Sort years numerically to ensure chronological order
        ordered_years = sorted(year_map.keys())
        output_path = output_dir / f"{t}_2016_2019.tsv"

        # Delete file if exists 
        if os.path.exists(output_path):
            os.remove(output_path)

        print(f"\n📦 Combining type: {t}")
        print(f"   → Output: {output_path}")

        # If file already exists, remove to avoid appending to old data
        if output_path.exists():
            output_path.unlink()

        write_header = True  # write header only for the first chunk

        # === Iterate through years in chronological order ===
        for y in ordered_years:
            file_path = Path(year_map[y])
            print(f"   → Reading {file_path}")

            # --- Case 1: CSV files ---
            if file_path.suffix.lower() == ".csv":
                for chunk in pd.read_csv(file_path, sep=";", chunksize=CHUNK_SIZE):
                    chunk.to_csv(
                        output_path,
                        sep="\t",
                        index=False,
                        header=write_header,
                        mode="a"
                    )
                    write_header = False

            # --- Case 2: Excel files (.xlsx) ---
            elif file_path.suffix.lower() == ".xlsx":
                # Pandas cannot stream Excel directly, so read once then chunk manually
                df = pd.read_excel(file_path)
                for start in range(0, len(df), CHUNK_SIZE):
                    chunk = df.iloc[start:start + CHUNK_SIZE]
                    chunk.to_csv(
                        output_path,
                        sep="\t",
                        index=False,
                        header=write_header,
                        mode="a"
                    )
                    write_header = False

            else:
                print(f"⚠ Unsupported file type for {file_path.suffix}")

        print(f"✅ Saved combined file for {t}: {output_path}")



def filter_province(data_file, column_name, numerical_value, output_file):
    """
    This function filters a data file based on a numerical value in a specified column and writes the
    filtered data to an output file.
    
    :param data_file: The `data_file` parameter in the `filter_province` function is the file path to
    the file that contains the tab separated data you want to filter. This function reads the data from this file,
    filters it based on the specified column and numerical value, and then writes the filtered data to
    an output file
    :param column_name: The `column_name` parameter in the `filter_province` function refers to the name
    of the column in the dataset that you want to filter on based on a numerical value. This function
    reads a data file in chunks, converts the specified column to numeric values, filters the data based
    on the numerical value
    :param numerical_value: The `numerical_value` parameter in the `filter_province` function represents
    the specific numerical value that you want to filter the data on. This value will be used to filter
    the data based on the specified `column_name` in the input `data_file`. Only rows with the value in
    equal to the specified will be outputted.
    :param output_file: The `output_file` parameter in the `filter_province` function is used to specify
    the file path where the filtered data will be saved after applying the filtering condition based on
    the `column_name` and `numerical_value`. This parameter should be a string representing the file
    path where the filtered data will be outputted.
    """
    
    print(f"Filtering {data_file}: {column_name} = {numerical_value}")
    write_header = True

    # Delete file if exists 
    if os.path.exists(output_file):
        os.remove(output_file)

    for chunk in pd.read_csv(data_file,
                             sep="\t",
                             chunksize=CHUNK_SIZE):
        
        if column_name in chunk.columns:
            chunk[column_name] = pd.to_numeric(chunk[column_name], errors="coerce")
            chunk = chunk[chunk[column_name] == numerical_value]
        else:
            raise ValueError(f"{column_name}: Column not found")
        
        chunk.to_csv(
            output_file,
            sep="\t",
            index=False,
            header=write_header,
            mode="a"
        )
        write_header = False

    df = pd.read_csv(output_file,
                sep="\t")
    
    if df.empty:
        raise ValueError(f"{data_file}: No rows found for {column_name} = {numerical_value}")


def check_numerical_empty_rows_any_column(df, columns):
    """
    This function checks for rows in a DataFrame where any of the specified columns contain non-numeric
    or empty values.
    
    :param df: A pandas DataFrame containing the data you want to check for empty or non-numeric rows
    :param columns: The `columns` parameter in the `check_numerical_empty_rows_any_column` function is a
    list of column names that you want to check for numerical or empty values in a pandas DataFrame `df`
    :return: The function `check_numerical_empty_rows_any_column` returns the count of rows in the
    DataFrame `df` where there is at least one invalid entry (non-numeric or missing) in any of the
    specified `columns`.
    """
    # Step 1: Select only the relevant columns
    subset_df = df[columns]

    # Step 2: Try to convert to numeric, forcing non-numeric to NaN
    subset_converted_df = subset_df.apply(pd.to_numeric, errors='coerce')

    # Step 3: Identify where conversion failed (NaN means non-numeric or missing)
    mask_invalid = subset_converted_df.isna() | subset_df.isin(['', ' '])

    # Step 4: Find rows with ANY invalid entries in the needed columns
    rows_with_any_invalid = mask_invalid.any(axis=1)

    # Step 5: Count them
    count_invalid_rows = rows_with_any_invalid.sum()

    return count_invalid_rows

def check_and_format_date_columns(df, histogram_empty_hourly_data_file):
    """
    The function `check_and_format_date_columns` checks for non-numerical or empty values in columns specifying
    day, month, year. Creates a date out of them and checks for empty or non-valid dates.
    It also checks for non-numerical or empty values in hourly columns 
    of a DataFrame, formats the date columns, and generates a histogram of invalid hourly
    column counts per row.
    
    :param df: The function `check_and_format_date_columns` takes a DataFrame `df` and a file path
    `histogram_empty_hourly_data_file` as input parameters. The DataFrame `df` is expected to contain
    columns representing date information such as year, month, and day, as well as hourly data columns
    :param histogram_empty_hourly_data_file: The parameter `histogram_empty_hourly_data_file` is a file
    path where the histogram of empty hourly data will be saved as a PNG image. This function
    :returns: the formatted dataframe with renamed columns and int type for day, month, year. 
    Proper date type in the column "date" and int type for hourly values 
    """

    needed_cols = {"ANNO", "MES", "DIA"}
    if not needed_cols.issubset(set(df.columns)):
        raise ValueError(f"Missing date columns {needed_cols - set(df.columns)}")

    # Check non-numerical or empty values in date columns
    count_invalid_rows = check_numerical_empty_rows_any_column(df, list(needed_cols))

    print("Number of rows with any non-numerical or empty values in date columns:", count_invalid_rows)

    # Format date
    for c in ("ANNO", "MES", "DIA"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    
    df["date"] = pd.to_datetime(
        df[["ANNO", "MES", "DIA"]].rename(columns={"ANNO": "year", "MES": "month", "DIA": "day"}),
        errors="coerce",
    )

    # Validate date column

    date_series = df["date"]

    # Convert to datetime (errors produce NaT)
    date_converted = pd.to_datetime(date_series, errors="coerce")

    # Identify invalid values:
    #   - NaT after conversion  → non-date or wrong format
    #   - Explicit empty strings or single spaces
    mask_invalid_fecha = (
        date_converted.isna() | 
        date_series.isin(["", " "])
    )

    # Count invalid rows
    invalid_date_count = mask_invalid_fecha.sum()

    print("Number of rows with invalid or empty date:", invalid_date_count)
    percentage_invalid_fecha = (invalid_date_count / len(df)) * 100
    print(f"Percentage of rows with invalid or empty date: {percentage_invalid_fecha:.2f}%")


    hour_cols_present = [c for c in HOUR_COLS if c in df.columns]
    if len(hour_cols_present) != len(HOUR_COLS):
        missing_cols = [c for c in HOUR_COLS if c not in df.columns]
        raise ValueError(f"Missing hourly columns: {missing_cols}")
    
    # Check non-numerical or empty values in hourly data columns
    # Step 1: Select only the relevant columns
    subset_df = df[HOUR_COLS]

    # Step 2: Try to convert to numeric, forcing non-numeric to NaN
    subset_converted_df = subset_df.apply(pd.to_numeric, errors='coerce')

    # Step 3: Identify where conversion failed (NaN means non-numeric or missing)
    mask_invalid = subset_converted_df.isna() | subset_df.isin(['', ' '])

    # Step 4: Count number of invalid hourly columns per row
    invalid_count_per_row = mask_invalid.sum(axis=1)
    
    # Step 5: Find rows with ANY invalid entries in the needed columns
    rows_with_any_invalid = invalid_count_per_row > 0

    # Step 6: Count them
    count_invalid_rows = rows_with_any_invalid.sum()

    print("Number of rows with any non-numerical or empty values in hourly columns:", count_invalid_rows)
    percentage_invalid_rows = (count_invalid_rows / len(df)) * 100
    print(f"Percentage of rows with any non-numerical or empty hourly values: {percentage_invalid_rows:.2f}%")

    # Step 7: Plot histogram of invalid column counts (excluding 0-invalid rows)
    plt.figure(figsize=(8, 5))

    # Calculate histogram data (excluding zeros)
    data = invalid_count_per_row[invalid_count_per_row > 0]
    counts, bins = np.histogram(data, bins=range(1, 26))

    # Convert frequencies to percentage of total rows
    percentages = (counts / len(df)) * 100

    # Plot as a bar chart (so we can directly control heights)
    plt.bar(bins[:-1], percentages, width=0.8, edgecolor='black', align='center')

    plt.title(f"Distribution of Invalid Hourly Column Counts per Row\n(Total data rows: {len(df)})")
    plt.xlabel("Number of Invalid Hourly Columns in Row")
    plt.ylabel("Percentage of Total Rows (%)")
    plt.xticks(range(1, 25))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(histogram_empty_hourly_data_file, dpi=300, format="png")
    plt.close()
    
    
    
    # Format hourly columns    
    for c in HOUR_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        
    return df
    

def calculate_daily_averages_per_row_and_location(df):
    """
    The function calculates daily averages per row and location based on hourly data per station.
    
    :param df: A pandas DataFrame containing data with columns for hours and dates. The function
    `calculate_daily_averages_per_row_and_location` calculates the daily average per row and location
    based on the hourly data provided in the DataFrame. It first calculates the mean of hourly data for
    each row ignoring NaN values, then removes invalid or empty values date and mean daily 
    values columns and calculates the mean per day (including all valid stations). 
    :return: The function `calculate_daily_averages_per_row_and_location` returns a pandas Series
    containing the daily averages of the calculated row-hour means for each date in the input DataFrame
    `df`.
    """
    df["row_hour_mean"] = np.nanmean(df[HOUR_COLS].to_numpy(dtype=float), axis=1)
    
    # Check non-numerical or empty values in the row hour mean
    count_invalid_rows = check_numerical_empty_rows_any_column(df, ["row_hour_mean"])

    if count_invalid_rows > 0:
        print(f"After averaging data per hour, there are {count_invalid_rows} non-numeric or empty row")

    # Ensure "date" is a valid datetime and drop invalid values ---
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])


    # Ensure "row_hour_mean" is numeric and drop invalid values ---
    df["row_hour_mean"] = pd.to_numeric(df["row_hour_mean"], errors="coerce")
    df = df.dropna(subset=["row_hour_mean"])


    # Group by date and compute daily mean ---
    daily_series = (
        df.groupby("date", dropna=True)["row_hour_mean"]
        .mean()
        .sort_index()
    )
    
    return daily_series


def check_and_combine_humidity_temperature(data_dir, output_file):
    """
    The function `check_and_combine_humidity_temperature` verifies if multiple CSV files have identical
    columns and order before concatenating them chronologically by year into an output file.
    
    :param data_dir: The `data_dir` parameter in the `check_and_combine_humidity_temperature` function
    refers to the directory where the CSV files containing humidity and temperature data are located.
    This function is designed to search for CSV files in the specified directory and perform operations
    on them to combine the data
    :param output_file: The `output_file` parameter in the `check_and_combine_humidity_temperature`
    function is the path to the file where the concatenated data will be saved. This file will contain
    the combined data from all the CSV files found in the `data_dir`. If the `output_file` already
    exists, it will be overwritten.
    """

    # === STEP 1. FIND ALL FILES ===

    # Example: 'M01_Center Finca experimental_01_01_2016_31_12_2016.csv'
    # We capture the year twice (start and end) but only need one.
    filename_pattern = re.compile(
        r"^(?P<basename>.+?)_01_01_(?P<year>\d{4})_31_12_(?P=year)\.csv$",
        re.IGNORECASE
    )


    files_found = []

    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(".csv"):
                match = filename_pattern.match(f)
                if match:
                    files_found.append({
                        "year": int(match.group("year")),
                        "path": os.path.join(root, f)
                    })


    # ------------------------------------------------------------------------------
    # 1️⃣ Verify all files have the same columns and order
    # ------------------------------------------------------------------------------
    reference_cols = None
    differences_found = False

    for i, entry in enumerate(sorted(files_found, key=lambda x: x['year'])):
        path = entry['path']

        try:
            df_head = pd.read_csv(path,
                                  header = 0,
                                  encoding='utf-16le',
                                  decimal=",",
                                  sep=";",
                                  nrows=5)
        except Exception as e:
            print(f"❌ Error reading {path}: {e}")
            differences_found = True
            continue

        cols = list(df_head.columns)

        if reference_cols is None:
            reference_cols = cols
            print(f"✅ Reference columns taken from {os.path.basename(path)}")
        else:
            if cols != reference_cols:
                differences_found = True
                print(f"⚠️ Column mismatch in {os.path.basename(path)}:")
                
                # Detailed comparison
                if len(cols) != len(reference_cols):
                    print(f"   → Different number of columns ({len(cols)} vs {len(reference_cols)})")
                for idx, (c1, c2) in enumerate(zip(cols, reference_cols)):
                    if c1 != c2:
                        print(f"   → Position {idx+1}: '{c1}' vs '{c2}'")

                # Check missing/extra
                missing = set(reference_cols) - set(cols)
                extra = set(cols) - set(reference_cols)
                if missing:
                    print(f"   → Missing in file: {missing}")
                if extra:
                    print(f"   → Extra columns in file: {extra}")

    if differences_found:
        print("\n❌ Differences detected — review above before concatenating.\n")
    else:
        print("\n✅ All files have identical columns and order.\n")

    # ------------------------------------------------------------------------------
    # 2️⃣ Concatenate all files chronologically by year, using chunks
    # ------------------------------------------------------------------------------
    # Delete old output file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"🗑️ Existing output file deleted: {output_file}")

    # Process and append chunk by chunk
    for entry in sorted(files_found, key=lambda x: x['year']):
        path = entry['path']
        print(f"📂 Processing {os.path.basename(path)}...")

        for i, chunk in enumerate(pd.read_csv(path,
                                  header = 0,
                                  encoding='utf-16le',
                                  decimal=",",
                                  sep=";",
                                  chunksize=CHUNK_SIZE)):
            # Append to output file (header only for the first chunk of the first file)
            chunk.to_csv(
                output_file,
                mode='a',
                index=False,
                sep="\t",
                header=not os.path.exists(output_file) and i == 0,
            )
        print(f"   → Finished {path}")

    print(f"\n✅ All files concatenated successfully into: {output_file}")


def check_and_format_temperature_humidity(df, histogram_missing_temperature_humidity):
    """
    The function `check_and_format_temperature_humidity` validates and formats the date column and
    temperature/humidity data in a DataFrame, and generates a histogram showing the distribution of rows
    with invalid temperature or humidity values. Finally, it returns a dataframe where the rows with invalid 
    dates and temperature/humidity columns are removed.
    
    :param df: The function `check_and_format_temperature_humidity` takes a DataFrame `df` and a file
    path `histogram_missing_temperature_humidity` as input parameters. The function performs the
    following tasks:
    :param histogram_missing_temperature_humidity: The parameter
    `histogram_missing_temperature_humidity` is the file path where you want to save the histogram image
    showing the distribution of rows with missing or invalid temperature and humidity data. This
    function `check_and_format_temperature_humidity` takes a DataFrame `df` and generates this histogram
    based on the data
    :return: The function `check_and_format_temperature_humidity` returns the DataFrame `df` after
    performing validation and formatting operations on the "Fecha" column and the columns "Temp Media
    (ºC)" and "Humedad Media (%)" in the DataFrame. The function also generates a histogram visualizing
    the distribution of rows with invalid temperature and humidity data, and saves it as an image file
    specified by the `histogram_missing_temperature_humidity` parameter
    """
    
    # VALIDATE FECHA COLUMN ----

    # Extract original Fecha column
    fecha_series = df["Fecha"]

    # Convert to datetime (errors produce NaT)
    fecha_converted = pd.to_datetime(fecha_series, format="%d/%m/%Y", errors="coerce")

    # Identify invalid values:
    #   - NaT after conversion  → non-date or wrong format
    #   - Explicit empty strings or single spaces
    mask_invalid_fecha = (
        fecha_converted.isna() | 
        fecha_series.isin(["", " "])
    )

    # Count invalid rows
    invalid_fecha_count = mask_invalid_fecha.sum()

    print("Number of rows with invalid or empty Fecha:", invalid_fecha_count)
    percentage_invalid_fecha = (invalid_fecha_count / len(df)) * 100
    print(f"Percentage of rows with invalid or empty Fecha: {percentage_invalid_fecha:.2f}%")

    
    data_columns = ["Temp Media (ºC)", "Humedad Media (%)"]  

    subset_df = df[data_columns]

    # Convert to numeric → non-numeric becomes NaN
    subset_converted_df = subset_df.apply(pd.to_numeric, errors='coerce')

    # Identify empty or invalid data
    mask_invalid = subset_converted_df.isna() | subset_df.isin(["", " "])

    # Count rows with ANY invalid entry in the two columns
    invalid_count_per_row = mask_invalid.sum(axis=1)
    rows_with_any_invalid = invalid_count_per_row > 0
    count_invalid_rows = rows_with_any_invalid.sum()

    print("Number of rows with invalid Temp/Humedad data:", count_invalid_rows)
    percentage_invalid_rows = (count_invalid_rows / len(df)) * 100
    print(f"Percentage of rows with invalid Temp/Humedad data: {percentage_invalid_rows:.2f}%")

    # HISTOGRAM OF INVALID COUNTS (0, 1, or 2 invalid columns per row) ----
    plt.figure(figsize=(8, 5))

    # Only take rows that have at least 1 invalid column
    data = invalid_count_per_row[invalid_count_per_row > 0]

    # Histogram for values 1 or 2 invalid columns
    counts, bins = np.histogram(data, bins=[1, 2, 3])  # bins: 1,2

    # Convert to percentages of full dataset (not only invalid rows)
    percentages = (counts / len(df)) * 100

    # Plot as bar chart
    plt.bar([1, 2], percentages, width=0.6, edgecolor="black")

    plt.title(f"Distribution of Invalid Temp and/or Humidity Column Counts per Row\n(Total rows: {len(df)})")
    plt.xlabel("Number of Invalid Columns in Row")
    plt.ylabel("Percentage of Total Rows (%)")
    plt.xticks([1, 2])
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(histogram_missing_temperature_humidity, dpi=300, format="png")
    plt.close()

    # Ensure "Fecha" is a valid datetime and drop invalid values ---
    df["Fecha"] = pd.to_datetime(df["Fecha"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["Fecha"])

    
    # Convert each column in data_columns to numeric, invalid → NaN
    # Drop any rows which contain any of the columns in the list "data_columns" as NaN
    rows_before_humid_temp_check = len(df)

    df[data_columns] = df[data_columns].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=data_columns, how='any')

    if rows_before_humid_temp_check != len(df):
        print(f"Rows dropped for having any of the columns non-numeric or empty {rows_before_humid_temp_check-len(df)}")

    return df


def fill_data_all_dates_in_range(df, years_list, date_column, fill_in_cols, fill_in_value):
    """
    The function `fill_data_all_dates_in_range` validates that the input DataFrame contains only the
    specified date column and the list of fill-in columns, ensures that the date column contains valid
    dates in strict YYYY-MM-DD format without time components, filters the data to include only the
    years specified in `years_list`, and completes the dataset by inserting missing days and filling
    the corresponding values in all fill-in columns with the value provided in `fill_in_value`.

    :param df: The function `fill_data_all_dates_in_range` takes a DataFrame `df` as input.
    This DataFrame must contain **exactly the date column specified by `date_column` plus the list of
    numeric columns specified in `fill_in_cols`**. The function validates the format of the date column,
    ensures that no timestamps contain time components, filters rows according to `years_list`, and
    generates the full daily date range for those years. Missing dates are automatically added, and all
    fill-in columns are populated with the specified `fill_in_value` for those inserted rows.

    :param years_list: The parameter `years_list` contains the list of years (either integers or strings
    convertible to integers) that must be present in the final output. These years determine both which
    rows are kept from the input DataFrame and the range of dates generated for alignment.

    :param date_column: The name of the column in `df` containing the dates. This column must contain
    values in strict YYYY-MM-DD format with no time component. The function checks for incorrect formats,
    invalid dates, and timestamps containing time information.

    :param fill_in_cols: A list of column names whose values should be aligned with the completed daily
    date range. When missing dates are detected, all columns in `fill_in_cols` will be filled using the
    value provided in `fill_in_value`.

    :param fill_in_value: A scalar value used to populate the columns in `fill_in_cols` for all dates
    that are missing from the original dataset but fall within the years specified in `years_list`.

    :return: The function `fill_data_all_dates_in_range` returns a DataFrame indexed by date and
    containing **all days** for the years given in `years_list`. The returned DataFrame includes only
    validated dates, and missing days are added with the value `fill_in_value` in all columns listed in
    `fill_in_cols`. The index name matches the provided `date_column`.
    """

    # -------------------------------------------------
    # Validate column set
    # -------------------------------------------------
    expected_cols = {date_column, *fill_in_cols}

    if set(df.columns) != expected_cols:
        raise ValueError(
            f"When generating values for all dates in range, DataFrame must contain ONLY these "
            f"columns: {expected_cols}, but found: {set(df.columns)}"
        )
    
    # -------------------------------------------------
    # Validate date column: strict YYYY-MM-DD format
    # -------------------------------------------------
    parsed = pd.to_datetime(df[date_column], format="%Y-%m-%d", errors="coerce")

    invalid_mask = parsed.isna()
    if invalid_mask.any():
        bad_vals = df.loc[invalid_mask, date_column]
        raise ValueError(
            f"Invalid date format found in '{date_column}'. Expected format YYYY-MM-DD. "
            f"Problematic values:\n{bad_vals}"
        )

    # Check for unwanted time components
    has_time = parsed.dt.time != pd.to_datetime("00:00:00").time()
    if has_time.any():
        bad_vals = df.loc[has_time, date_column]
        raise ValueError(
            f"Column '{date_column}' contains timestamps with time components. "
            f"Only YYYY-MM-DD is allowed. Problematic values:\n{bad_vals}"
        )

    # -------------------------------------------------
    # Index by date and filter by year
    # -------------------------------------------------
    df = df.set_index(date_column).sort_index()

    years_int = [int(y) for y in years_list]
    df = df[df.index.year.isin(years_int)]
    
    # -------------------------------------------------
    # Build full date range
    # -------------------------------------------------
    min_year, max_year = min(years_int), max(years_int)

    full_index = pd.date_range(
        start=f"{min_year}-01-01",
        end=f"{max_year}-12-31",
        freq="D"
    )

    # Filter only requested years (in case years_list is non-continuous)
    full_index = full_index[full_index.year.isin(years_int)]

    # -------------------------------------------------
    # Reindex and fill missing dates
    # -------------------------------------------------
    if len(full_index) != len(df):
        missing = len(full_index) - len(df)
        print(f"{missing} rows missing — filling with {fill_in_value}")

        df = df.reindex(full_index, fill_value=fill_in_value)

    df.index.name = date_column

    return df



def get_average_temperature_humidity(df, years_list):
    """
    Computes validated daily averages of temperature and humidity from a raw meteorological dataset.

    The function `get_average_temperature_humidity` processes an input DataFrame containing a datetime
    column `'Fecha'` and two numeric measurement columns, `'Temp Media (ºC)'` and `'Humedad Media (%)'`.
    The dataset may contain missing or invalid values. The function first selects only these required
    columns, removes any rows where **either the date or any of the numeric fields are missing**
    (`NaT` in `'Fecha'` or `NaN` in any numeric column), and reports the number of rows removed. It then
    computes the daily mean temperature and humidity by grouping by date, automatically ignoring any
    NaN values within the aggregation. After aggregation, the resulting daily averages are validated to
    ensure that no non-numeric or empty entries remain. Finally, the function guarantees completeness
    of the dataset by inserting all calendar dates corresponding to the years listed in `years_list`,
    filling missing values with `NaN`.

    :param df: The input DataFrame from which daily averages will be computed. It must contain at least
        the following three columns:

        - `'Fecha'`: A datetime-like column representing timestamps. Any rows where `'Fecha'` is
          missing or contains `NaT` are removed.
        - `'Temp Media (ºC)'`: A numeric column representing mean temperature. Rows with missing
          (`NaN`) values are removed before aggregation.
        - `'Humedad Media (%)'`: A numeric column representing mean humidity. Rows with missing
          (`NaN`) values are removed before aggregation.

        All other columns in the input DataFrame are ignored. The function reports how many rows were
        excluded due to missing dates or missing numeric values.

    :param years_list: A list of integer years (e.g., `[2022, 2023, 2024]`) that define the expected
        date range in the final output. After computing daily averages, the function delegates to
        `fill_data_all_dates_in_range`, which ensures that **all days** belonging to these years are
        present in the result. Missing dates are inserted, and the numeric columns are filled with
        `NaN` for those rows.

    :return: A cleaned, validated, and complete DataFrame containing one row per calendar date for the
        years specified in `years_list`. The returned DataFrame includes:

        - A `'Fecha'` date column with no missing values and no time components.
        - `'Temp Media (ºC)'` and `'Humedad Media (%)'`, holding the computed daily averages or `NaN`
          for dates where no data was originally present.
        
        The function also prints diagnostic messages indicating how many rows were removed due to
        missing/invalid values and whether any invalid entries were detected after aggregation. The
        final output is sorted by date and contains a complete daily sequence for the requested years.
    """
    
    # Subset only columns of interest
    df = df[["Fecha", "Temp Media (ºC)", "Humedad Media (%)"]] 

    numeric_cols = ["Temp Media (ºC)", "Humedad Media (%)"]
    
    number_of_initial_rows = len(df)

    df = df.dropna(subset=["Fecha", "Temp Media (ºC)", "Humedad Media (%)"])
    
    if number_of_initial_rows != len(df):
        print(f"{number_of_initial_rows - len(df)} rows have been removed due to not having proper dates, humidity and/or temperature values")

    # Group by Fecha and compute mean (automatically ignores NaN)
    daily_avg = (
        df.groupby("Fecha", as_index=False)[numeric_cols]
          .mean()
          .sort_values("Fecha")
    )
    # Check non-numerical or empty values in the row hour mean
    count_invalid_rows = check_numerical_empty_rows_any_column(daily_avg, numeric_cols)

    if count_invalid_rows > 0:
        print(f"After averaging data per day, there are {count_invalid_rows} non-numeric or empty row")

    # Make sure data for all dates in the years are present, if they don't exist, they
    # are filled with NaN
    daily_avg = fill_data_all_dates_in_range(daily_avg,
                                                years_list,
                                                date_column = "Fecha",
                                                fill_in_cols = numeric_cols,
                                                fill_in_value = np.nan)



    return daily_avg


def custom_date_parser(date):
    """
    The custom_date_parser function converts date strings to datetime objects handling different formats
    and empty values.
    
    :param date: The `custom_date_parser` function is designed to parse dates in different
    formats using the `pd.to_datetime` function from the pandas library. It first tries to parse the
    date with the format '%Y-%m-%d' and if that fails, it tries with the format '%Y-%m-%d %H:%M:%S'
    :return: returns a dataframe with date formatted columns. If the input
    date string does not match any of these formats, the function will return `pd.NaT`,
    """
    try:
        # For columns with format dd-mm-yyyy
        return pd.to_datetime(date, format='%Y-%m-%d')
    except ValueError:
        try:
            # For columns with format dd-mm-yyyy hh:mm:ss
            return pd.to_datetime(date, format='%Y-%m-%d %H:%M:%S')
        except ValueError:
            # For empty values
            return pd.NaT  # This represents a missing or undefined datetime value


def check_and_format_hospitalizations(input_file, date_column, years_list, autonomous_community_code):
    """
    The function `check_and_format_hospitalizations` reads a tab-separated file containing 
    hospitalization records, uses the column `date_column` as the hospitalization date, checks 
    and formats this date column, identifies and handles invalid or empty values, filters the 
    dataset by the specified autonomous community, groups the records by date to obtain the 
    number of hospitalizations per day, and returns a DataFrame with hospitalization counts for 
    all days in the selected years (days with no hospitalizations are assigned a count of 0).

    :param input_file: The `input_file` parameter is the file path to the tab-separated file 
    containing hospitalization data. The function loads this file, parsing the dates in the 
    column specified by `date_column`, and performs validation and normalization operations on 
    these dates.

    :param date_column: The `date_column` parameter is the name of the column in the input file 
    that contains the hospitalization dates. This column is parsed as datetime during the file 
    read operation and later normalized. Invalid or empty values in this column are identified 
    and removed.

    :param years_list: The `years_list` parameter is a list of years (integers) used to filter 
    the hospitalization data. Only records whose date belongs to one of the years in 
    `years_list` are retained. All days in those years are included in the output, and days 
    without hospitalizations are assigned a count of 0.

    :param autonomous_community_code: The `autonomous_community_code` parameter specifies the 
    value of the `"Comunidad Autónoma"` column that will be used to filter the data. Only rows 
    belonging to the selected autonomous community are processed.

    :return: The function returns a DataFrame `df_counts` that contains the count of 
    hospitalizations grouped by the column provided by `date_column`. The resulting DataFrame 
    is sorted chronologically by date and includes all days from the years specified in 
    `years_list`. If a given day has no hospitalization records, its count is set to 0.
    """
    df = pd.read_csv(input_file,
                 sep="\t",
                 parse_dates=[date_column],
                 decimal=',',
                 date_parser=custom_date_parser)
    
    # Subselect only the corresponding autonomous community 
    df = df[df["Comunidad Autónoma"] == autonomous_community_code]
    
    # Extract original column
    fecha_series = df[date_column]

    # Convert to datetime (errors produce NaT)
    fecha_converted = pd.to_datetime(fecha_series, format="%d/%m/%Y", errors="coerce")

    # Identify invalid values:
    #   - NaT after conversion  → non-date or wrong format
    #   - Explicit empty strings or single spaces
    mask_invalid_fecha = (
        fecha_converted.isna() | 
        fecha_series.isin(["", " "])
    )

    # Count invalid rows
    invalid_fecha_count = mask_invalid_fecha.sum()

    print(f"Number of rows with invalid or empty {date_column}: {invalid_fecha_count}")
    percentage_invalid_fecha = (invalid_fecha_count / len(df)) * 100
    print(f"Percentage of rows with invalid or empty {date_column}: {percentage_invalid_fecha:.2f}%")

    # Ensure "Fecha" is a valid datetime (remove the timestamp information) and drop invalid values ---
    df[date_column] = df[date_column].dt.normalize()
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    
    rows_before_date_check = len(df)
    df = df.dropna(subset=[date_column])
      
    if rows_before_date_check != len(df):
        print(f"Rows dropped for having date columns invalid or empty {rows_before_date_check-len(df)}")


    # Group by calendar date (time set to 00:00:00) and count
    df_counts = (
        df
        .groupby(date_column, dropna=True)                                   # dropna=True avoids grouping NaT
        .size()
        .reset_index(name='Count')
        .rename(columns={date_column: date_column})
        .sort_values(date_column)                          # ascending by date
        .reset_index(drop=True)
    )

    count_invalid_rows = check_numerical_empty_rows_any_column(df_counts, ['Count'])
    print(f"Number of dates with invalid count of {date_column} data: {count_invalid_rows}")
    percentage_invalid_rows = (count_invalid_rows / len(df_counts)) * 100
    print(f"Percentage of rows with invalid {date_column} data: {percentage_invalid_rows:.2f}%")

    # Remove invalid or empty count rows
    df_counts['Count'] = pd.to_numeric(df_counts['Count'], errors="coerce")
    df_counts = df_counts.dropna(subset=['Count'])


    # Make sure all dates in the years specified have data
    # if they were empty, they are filled with 0  
    df_counts = fill_data_all_dates_in_range(df = df_counts,
                                             years_list = years_list,
                                             date_column = date_column,
                                             fill_in_cols =["Count"],
                                             fill_in_value = 0)

    return df_counts


def get_pollutant_dataframe(source_files_dir,
                            years_list,
                            output_dir,
                            province_name,
                            province_code):
    
    """
    Loads, filters, validates, and aggregates air-quality pollutant measurements for a given province,
    producing a unified daily dataset containing one column per pollutant type.

    The function `get_pollutant_dataframe` scans the directory `output_dir` for air-quality data files
    matching the pattern `*_2016_2019.tsv`. For each file, the data is filtered to include only rows
    corresponding to the specified province, using the `province_code` applied to the `'PROVINCIA'`
    column. After filtering, the function loads the resulting file, validates and standardizes its
    datetime columns, and computes daily averages for each measurement location using the helper
    function `calculate_daily_averages_per_row_and_location`.

    Each file corresponds to a single pollutant type. The pollutant name is inferred from the filename
    and used as the column name of the resulting daily-average series. The daily averages are converted
    into a DataFrame with a `'date'` column and one measurement column named after the pollutant. The
    function then ensures that the resulting dataset contains **all dates** for the years specified in
    `years_list` by calling `fill_data_all_dates_in_range`, which inserts missing dates and fills the
    pollutant column with `NaN` for those empty entries.

    After all pollutant-specific dataframes are processed, they are concatenated column-wise and aligned
    by their `'date'` index, producing a final dataset where each column represents a pollutant and each
    row a unique calendar date.

    :param source_files_dir: Directory where the original, unfiltered air-quality source files are
        stored. This parameter is not directly used in the current implementation but is kept for
        compatibility with upstream workflows.

    :param years_list: A list of integer years (e.g., `[2016, 2017, 2018, 2019]`) defining the complete
        daily date range expected in the final output. For every pollutant, the dataset is expanded to
        include all dates within these years, with missing measurements filled using `NaN`.

    :param output_dir: Directory where intermediate filtered files are located and where province-specific
        files are written. The function searches this directory for files matching `*_2016_2019.tsv` and
        processes each as a separate pollutant dataset.

    :param province_name: The name of the province (string) used for labeling output filenames produced
        during filtering. It is appended to each filename to distinguish province-specific subsets.

    :param province_code: The numeric or string identifier of the province. Rows in each dataset are
        filtered by matching the `'PROVINCIA'` column to this value.

    :return: A combined daily dataset containing one column per pollutant type and a complete sequence
        of dates for the years in `years_list`. The returned DataFrame:

        - Contains a `'date'` column with no missing values.
        - Includes one column per pollutant, named according to the base filename.
        - Is aligned by date across all pollutants.
        - Uses `NaN` for missing pollutant values on dates where no measurements were available.
        
        The final dataset is suitable for time-series analysis, visualization, or further aggregation.
    """
    
    check_and_combine_air_quality_per_type(source_files_dir,
                                  years_list,
                                  output_dir)
    
    pattern = os.path.join(output_dir, "*_2016_2019.tsv")
    data_files = glob.glob(pattern)
    
    # Filter data by province
    # List of dataframes to combine all pollutants
    pollutant_df_list = []   
    
    for data_file in data_files:
        
        base, ext = os.path.splitext(data_file)  # Split into filename and extension
        output_filtered_file = f"{base}_{province_name}{ext}"  

        filter_province(data_file,
                        "PROVINCIA",
                        province_code,
                        output_filtered_file)

        print(f"\n📦 Processing dates in: {output_filtered_file}")
        
        histogram_empty_hourly_data_file = f"{base}_{province_name}_missing_hourly_data_histogram.png"
        df = pd.read_csv(output_filtered_file,
                    sep="\t")
        df = check_and_format_date_columns(df, histogram_empty_hourly_data_file)

        series_location_daily_averages = calculate_daily_averages_per_row_and_location(df)

        # Get data type measured
        data_type_measured = os.path.basename(base).split("_2016_2019")[0]

        series_location_daily_averages.name = data_type_measured  # rename the series
        
        # Convert into a dataframe
        df_location_daily_averages = (
            series_location_daily_averages
                .rename_axis("date")
                .reset_index()
        ) 

        # Make sure all dates are included (filling in with NA in empty ones) 
        df_location_daily_averages = fill_data_all_dates_in_range(df = df_location_daily_averages,
                                                                  years_list = years_list,
                                                                  date_column = "date",
                                                                  fill_in_cols = [data_type_measured],
                                                                  fill_in_value = np.nan)

        pollutant_df_list.append(df_location_daily_averages)
        


    df_combined = pd.concat(pollutant_df_list, axis=1)

    return df_combined



def get_humidity_temp_dataframe(source_files_dir,
                            years_list,
                            output_file):
    
    check_and_combine_humidity_temperature(source_files_dir, output_file)

    df_humidity_temp = pd.read_csv(output_file, sep="\t")

    base, ext = os.path.splitext(output_file)  # Split into filename and extension
    
    histogram_missing_humidity_temp = f"{base}_missing_humidity_temp_data_histogram.png"

    df_humidity_temp = check_and_format_temperature_humidity(df_humidity_temp, histogram_missing_humidity_temp)

    df_humidity_temp = get_average_temperature_humidity(df_humidity_temp, years_list)
    
    return df_humidity_temp
        
    
    

if __name__=="__main__":
    ### Pollutant data
    # === CONFIGURATION ===
    BASE_DIR = Path("/home/jalvarezbe/TFG-CALIDAD-DEL-AIRE/Datos/Datos_originales")
    YEARS = ["2016", "2017", "2018", "2019"] # For the rolling mean, we need late 2016 data 
    OUTPUT_DIR = BASE_DIR  # where to save combined .tsv files
    PROVINCE_NAME = "Madrid" # "Madrid" or "Valencia"
    PROVINCE_CODE = 28 # Code for Madrid or Valencia 
    OUTPUT_TABLE_FILE = os.path.join(OUTPUT_DIR, ("pollutants_"+PROVINCE_NAME+"_"+YEARS[0]+"_"+YEARS[-1]+".tsv"))

    df_pollutant = get_pollutant_dataframe(source_files_dir = BASE_DIR,
                                           years_list = YEARS,
                                           output_dir = OUTPUT_DIR,
                                           province_name = PROVINCE_NAME,
                                           province_code = PROVINCE_CODE)
    
    
    
    ### Temperature and humidity data
    # === CONFIGURATION ===
    PROVINCE_NAME = 'Madrid'  
    BASE_DIR = Path("/home/jalvarezbe/TFG-CALIDAD-DEL-AIRE/Datos/Datos_originales/humidity_temperature/" + PROVINCE_NAME)
    OUTPUT_FILE = os.path.join(BASE_DIR, ("humidity_temp_"+PROVINCE_NAME+"_2016_2019.tsv"))


    df_humidity_temp = get_humidity_temp_dataframe(source_files_dir = BASE_DIR,
                                                   years_list = YEARS,
                                                   output_file = OUTPUT_FILE)
    
    # Rename columns
    df_humidity_temp = df_humidity_temp.rename_axis("date") \
                                   .rename(columns={
                                       'Temp Media (ºC)': 'Temperature',
                                       'Humedad Media (%)': 'Humidity'
                                   }) 

    # Combine pollutants and humidity temperature
    OUTPUT_ENVIRONMENTAL_TABLE = os.path.join(OUTPUT_DIR, ("environmental_features_"+PROVINCE_NAME+"_"+YEARS[0]+"_"+YEARS[-1]+".tsv"))
    
    df_combined_environmental = pd.concat([df_humidity_temp, df_pollutant], axis=1)

    df_combined_environmental.to_csv(OUTPUT_ENVIRONMENTAL_TABLE,
        sep="\t",        # tab-separated
        index=True,      # include the 'date' index
        header=True,     # include column names
        date_format="%Y-%m-%d"  # format dates nicely
    )

    ### Number of hospitalizations
    # === CONFIGURATION ===
    INPUT_TABLE = "/home/jalvarezbe/TFG-CALIDAD-DEL-AIRE/Datos/Datos_originales/hospitalizations/final_data_with_freq_diagnostics.tsv"
    DATE_COLUMN = "Fecha de Inicio contacto"
    YEARS = ["2017", "2018", "2019"]
    PROVINCE_NAME = "Madrid"
    PROVINCE_CODE = 28
    AUTONOMOUS_COMMUNITY_CODE = 13 # Comunidad de Madrid   

    hospitalizations_df = check_and_format_hospitalizations(INPUT_TABLE, DATE_COLUMN, YEARS, AUTONOMOUS_COMMUNITY_CODE)
    
    OUTPUT_HOSPITALIZATIONS_TABLE = os.path.join(OUTPUT_DIR, ("hospitalizations_"+PROVINCE_NAME+"_"+YEARS[0]+"_"+YEARS[-1]+".tsv"))
    
    hospitalizations_df.to_csv(OUTPUT_HOSPITALIZATIONS_TABLE,
        sep="\t",        # tab-separated
        index=True,      # include the 'date' index
        header=True,     # include column names
        date_format="%Y-%m-%d"  # format dates nicely
    )