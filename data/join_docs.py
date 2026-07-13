"""
file: data/join_docs.py
author: Samuel Kudla <xkudlas00@stud.fit.vutbr.cz>
date: 12.05.2026
"""
import sys
import pandas as pd
from pathlib import Path

def _load_csv_for_join(file_path: str, join_attr: str, filter_new: bool = False) -> pd.DataFrame:
    """
    @param file_path: path to CSV file
    @param join_attr: name of attribute to join on
    @param filter_new: whether to filter out rows where 'New' is True. This filter is used since the pretraining is used, so results of pretraining of models is not relevant.
    
    Function loading a CSV file into a pandas DataFrame and preparing it for joining. It checks if the join attribute exists, optionally filters out rows where 'New' is True.
    Function also adds prefix to non-key columns to keep them distinct after joining, using the file name as prefix.
    """
    import pandas as pd
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    if join_attr not in df.columns:
        raise ValueError(f"Join attribute '{join_attr}' not found in {file_path}")

    if filter_new and 'New' in df.columns:
        df = df[df['New'] == False]

    prefix = Path(file_path).stem
    rename_map = {col: f"{prefix}__{col}" for col in df.columns if col != join_attr}
    return df.rename(columns=rename_map)

def join_csv_documents(files, join_attr, output_name='joined.csv', filter_new=False):
    """
    @param files: list of file paths to CSV files to join
    @param join_attr: name of attribute to join on
    @param output_name: name of output CSV file
    @param filter_new: whether to filter out rows where 'New' is True. This
    
    Function joining multiple CSV files on a specified attrubute. Suffixes are added to non-key columns to keep them distinct. 
    If filter_new is True, rows where 'New' is True in any file are filtered out before joining.
    """
    merged_df = None
    
    for file_path in files:
        current_df = _load_csv_for_join(file_path, join_attr, filter_new=filter_new)
        if merged_df is None:
            merged_df = current_df
        else:
            merged_df = pd.merge(merged_df, current_df, on=join_attr, how='inner')

    if merged_df is None or merged_df.empty:
        print("No match found")
        return

    try:
        merged_df = merged_df.sort_values(by=join_attr)
    except Exception:
        pass

    merged_df.to_csv(output_name, index=False)
    print(f"Done! Joined {len(merged_df)} records into {output_name}")
    
if __name__ == '__main__':
  filter_new = '--loaded' in sys.argv

  if len(sys.argv) < 4 or '-h' in sys.argv or '--help' in sys.argv:
      print("Usage: python3 join_docs.py --files <file1.csv,file2.csv,file3.csv...> --attrib <join_attr> [output_name] [--loaded]")
      print('Script joins all files containing same value specified by --attrib, and keeps all non-key columns distinct by prefixing them with the file name.')
      print('If --loaded is specified, rows where "New" is True in any file are filtered out before joining.')
      exit(1)
      
  if '--files' not in sys.argv or '--attrib' not in sys.argv:
      print("Error: Missing required arguments --files and/or --attrib.")
      exit(1)
  
  files_arg_index = sys.argv.index('--files') + 1
  attrib_arg_index = sys.argv.index('--attrib') + 1
  
  files = sys.argv[files_arg_index].split(',')
  join_attr = sys.argv[attrib_arg_index]
  output_name = 'joined.csv' if len(sys.argv) <= max(files_arg_index, attrib_arg_index) + 1 else sys.argv[max(files_arg_index, attrib_arg_index) + 1]
  
  join_csv_documents(files, join_attr, output_name=output_name, filter_new=filter_new)