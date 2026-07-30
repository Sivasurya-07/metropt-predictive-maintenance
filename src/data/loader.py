import os
import urllib.request
import zipfile
import polars as pl
from pathlib import Path
from src import config

def download_dataset(url: str, dest_path: Path):
    """
    Downloads the dataset from a URL with basic progress reporting.
    """
    print(f"Downloading from {url}...")
    
    def report_progress(block_num, block_size, total_size):
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = min(100, (read_so_far * 100) // total_size)
            print(f"Progress: {percent}% ({read_so_far // (1024*1024)}MB / {total_size // (1024*1024)}MB)", end="\r")
        else:
            print(f"Downloaded: {read_so_far // (1024*1024)}MB", end="\r")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest_path, reporthook=report_progress)
    print("\nDownload completed successfully.")

def extract_zip(zip_path: Path, target_dir: Path, target_csv_name: str):
    """
    Extracts the first CSV file found inside the ZIP and renames it.
    """
    print(f"Extracting {zip_path} to {target_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        csv_files = [f for f in zip_ref.namelist() if f.lower().endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("No CSV file found inside the ZIP archive.")
        
        extracted_name = csv_files[0]
        print(f"Found CSV inside ZIP: {extracted_name}")
        
        # Extract the file
        zip_ref.extract(extracted_name, path=target_dir)
        
        # Rename to target_csv_name
        extracted_file_path = target_dir / extracted_name
        final_file_path = target_dir / target_csv_name
        
        if extracted_file_path != final_file_path:
            if final_file_path.exists():
                final_file_path.unlink()
            extracted_file_path.rename(final_file_path)
            print(f"Renamed extracted CSV to {final_file_path}")
            
            # Clean up empty parent directories if extracted file was in a subfolder
            if extracted_file_path.parent != target_dir:
                try:
                    extracted_file_path.parent.rmdir()
                except OSError:
                    pass

def get_dataset(force_download: bool = False) -> Path:
    """
    Acquires the dataset: downloads and extracts it if it doesn't exist,
    then converts it to Parquet for highly optimized reading.
    Returns the path to the Parquet file.
    """
    zip_path = config.RAW_DATA_DIR / config.ZIP_FILENAME
    csv_path = config.RAW_DATA_DIR / config.CSV_FILENAME
    
    parquet_path = config.RAW_DATA_DIR / "metropt.parquet"
    
    if parquet_path.exists() and not force_download:
        print(f"Parquet dataset already exists at: {parquet_path}")
        return parquet_path
        
    if force_download or not zip_path.exists():
        download_dataset(config.DATASET_URL, zip_path)
        
    extract_zip(zip_path, config.RAW_DATA_DIR, config.CSV_FILENAME)
    
    # Convert extracted CSV to Parquet using Polars
    print(f"Converting CSV ({csv_path}) to Parquet ({parquet_path}) for optimized I/O...")
    df_csv = pl.read_csv(csv_path, try_parse_dates=True)
    df_csv.write_parquet(parquet_path)
    
    # Clean up massive raw CSV and ZIP files to save disk space
    if csv_path.exists():
        csv_path.unlink()
        print("Removed raw CSV file.")
        
    if zip_path.exists():
        zip_path.unlink()
        print("Removed downloaded ZIP file.")
        
    return parquet_path

def load_raw_data(data_path: Path) -> pl.DataFrame:
    """
    Loads raw Parquet data using Polars for high performance.
    """
    print(f"Loading data from {data_path}...")
    # Read Parquet
    df = pl.read_parquet(data_path)
    print(f"Successfully loaded dataframe. Shape: {df.shape}")
    return df

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    # Test script execution
    csv_file = get_dataset()
    df = load_raw_data(csv_file)
    print(df.head())
