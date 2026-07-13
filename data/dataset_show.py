"""
file: data/dataset_show.py
author: Samuel Kudla <xkudlas00@stud.fit.vutbr.cz>
date: 12.05.2026
"""
import sys
import csv

def is_stdout_csv() -> bool:
    if hasattr(sys.stdout, 'name'):
        filename = sys.stdout.name
        return filename.lower().endswith('.csv')
    return False

def apply_filters(file, col_filter, col_filter_val = "", display_col = "", delimiter=';') -> None:
    """
    @param file: input .csv file to process
    @param col_filter: column name to filter by
    @param col_filter_val: value to filter by in col_filter column (if empty, all values from col_filter are printed)
    @param display_col: column name to display (if empty, whole row is printed)
    @param delimiter: delimiter used in the .csv file (default is ';')
    
    Function filtering .csv file based on given column and value. 
    If display_col is given, only values from this column are printed, otherwise whole row is printed. 
    If col_filter_val is not given, all values from col_filter column are printed.
    Output is printed in csv format.
    """
    reader = csv.reader(open(file, 'r'), delimiter=delimiter)
    headers = next(reader)
    col_index = headers.index(col_filter) if col_filter in headers else -1
    display_index = headers.index(display_col) if (display_col and display_col in headers) else -1

    if col_index < 0:
        print(f"Column '{col_filter}' not found in the file.")
        return
    if display_index < 0 and display_col:
        print(f"Display column '{display_col}' not found in the file.")
        return
    if not col_filter_val:
        for row in reader:
            print(row[col_index])
    else:
        for row in reader:
            if row[col_index] == col_filter_val:
                if display_index >= 0:
                    print(row[display_index])
    

def dataset_show(file, col_filter="", col_filter_val="", display_col="", delimiter=';', unique=False) -> None:
    """
    @param file: input .csv file to process
    @param col_filter: column name to filter by
    @param col_filter_val: value to filter by in col_filter column (if empty, all values from col_filter are printed)
    @param display_col: column name to display (if empty, whole row is printed)
    @param delimiter: delimiter used in the .csv file (default is ';')
    @param unique: if True, display number of unique entries
    
    Function filtering .csv file based on given column and value.
    This function prints output in formatted style: \"name:value\" for each attribute in the row, separated by newlines.
    """
    with open(file, 'r') as f:
        unique_set = set()
        lines = [line.strip() for line in f]
        text_with_val = {}

        attribs = lines[0].split(delimiter)

        for line in lines[1:]:
            data = line.split(delimiter)
            text_with_val = dict(zip(attribs, data))
            
            if not col_filter:
                unique_set.add(line)
                for h, v in text_with_val.items():
                    print(f"{h}: {v}")
                print()
                
            elif col_filter in text_with_val:
                if not col_filter_val:
                    unique_set.add(text_with_val[col_filter])
                    print(text_with_val[col_filter])
                elif text_with_val[col_filter] == col_filter_val:
                    if display_col:
                        if display_col not in text_with_val:
                            continue
                        unique_set.add(text_with_val[display_col])
                        print(f"{display_col}: {text_with_val[display_col]}")
                    else:
                        for h, v in text_with_val.items():
                            print(f"{h}: {v}")
                        unique_set.add(line)
                    print()
        if unique:
            print(f"Unique entries: {len(unique_set)}")

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] == '-h':
        print("Usage: python dataset_show.py <input_csv_file> [filter1] [filter2] [filter3] [--d delimiter] [--unique]\n")
        print("Script processes .csv connections file and displays on output in formated style if non-csv output.")
        print("1 filter - filters only given attribute in file.")
        print("2 filters - filters only connections containing value of filter2 in attribute given in filter1")
        print("3 filters - same as 2 filters, but displays only attributes given in filter3.")
        print("Delimiter can be set with --d argument, default is ';'")
        print("Use --unique to display number of unique entries.")
        exit(1)
    
    delimiter = ';'
    if '--d' in sys.argv:
        d_index = sys.argv.index('--d')
        if d_index + 1 < len(sys.argv):
            delimiter = sys.argv[d_index + 1]
            del sys.argv[d_index:d_index+2]

    unique = False
    if '--unique' in sys.argv:
        unique_index = sys.argv.index('--unique')
        sys.argv.pop(unique_index)
        unique = True
    
    file = sys.argv[1]
    col_filter = sys.argv[2] if len(sys.argv) > 2 else ""
    col_filter_val = sys.argv[3] if len(sys.argv) > 3 else ""
    display_col = sys.argv[4] if len(sys.argv) > 4 else ""
    
    if (is_stdout_csv()):
        if len(sys.argv) == 2:
            with open(sys.argv[1], 'r') as f:
                print(f.read())
        else:
            apply_filters(file, col_filter, col_filter_val, display_col, delimiter)
    else:
        dataset_show(file, col_filter, col_filter_val, display_col, unique=unique, delimiter=delimiter)