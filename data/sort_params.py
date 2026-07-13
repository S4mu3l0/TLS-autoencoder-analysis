"""
file: data/sort_params.py
author: Samuel Kudla <xkudlas00@stud.fit.vutbr.cz>
date: 12.05.2026
"""
import sys

def sort_params(input, output, fil = None):
    """
    @param input: input file to process, can be either raw text or .csv
    @param output: output file to save sorted data
    @param fil: (optional) if provided, only the parameter with this name will be processed and written to output (without parameter name, only values)
    
    Function sorting lists of multiple numeric parameters that are observed in a file
    """
    
    observed = ['Client Extensions', 'Client Supported Groups', 'Client CipherSuite']
    if fil: fil = fil.strip()
    
    with open(input, 'r') as inpt, open(output, 'w') as out:
        if not input.endswith('.csv'):
            for line in inpt:
                line_raw = line.strip()
                if not line_raw:
                    if not fil: out.write('\n') # if filter is on, dont write empty lane
                    continue
                    
                params = line.split(':', 1)
                if len(params) < 2:
                    if not fil: out.write(line) 
                    continue
                
                name = params[0]
                value = params[1].strip()
                stripped_name = name.strip()
                
                # filter
                if fil and stripped_name != fil:
                    continue
                
                if stripped_name in observed:
                    if stripped_name.endswith('Groups'):
                        # hex values
                        items = [int(x.strip(), 16) for x in value.split(',') if x.strip()]
                        items.sort()
                        new_value = ','.join([f"0x{x:04x}" for x in items])
                    else:
                        items = [int(x.strip()) for x in value.split('-') if x.strip()]
                        items.sort()
                        new_value = '-'.join([str(x) for x in items])
                    
                    if not fil:
                        out.write(f"{name}: {new_value}\n")
                    else:
                        out.write(f"{new_value}\n")
                elif not fil:
                    out.write(line)
                    
        else: #.csv raw file
            # parse header
            header = inpt.readline()
            out.write(header)
            header_list = [h.strip() for h in header.split(';')]
            
            observed_indexes = {}
            for item in observed:
                observed_indexes[item] = header_list.index(item)
                
            # parse body
            for line in inpt:
                row = line.strip().split(';')
                
                # sort all observed parameters
                for o_name, index in observed_indexes.items():
                    # check for invalid rows
                    if index < len(row):
                        values_string = row[index].strip()
                        if not values_string:
                            continue
                        try:
                            if o_name.endswith('Groups'):
                                # hex values
                                items = [int(x.strip(), 16) for x in values_string.split(',') if x.strip()]
                                items.sort()
                                row[index] = ','.join([f"0x{x:04x}" for x in items])
                            else:
                                items = [int(x.strip()) for x in values_string.split('-') if x.strip()]
                                items.sort()
                                row[index] = '-'.join([str(x) for x in items])
                        except ValueError:
                            pass
                        
                # write result
                out.write(';'.join(row) + '\n')

if __name__ == '__main__':
    if (len(sys.argv) < 2 or '-h' in sys.argv or '--help' in sys.argv):
        print("Usage: python sort_params.py <input_file> [output_filename] [--f filter]")
        print("Sort numeric parameters in a file.")
        print("\nArguments:")
        print("  input_file         Path to input file (CSV or text format)")
        print("  output_filename    Optional output filename (default: ordered_data)")
        print("\nOptions:")
        print("  --f <filter>       Filter to only one parameter:")
        print("                     'Client Extensions', 'Client Supported Groups', or 'Client CipherSuite'")
        print("  -h, --help         Show this help message")
        exit(1)

    input = sys.argv[1]
    fil = None
    if '--f' in sys.argv:
        fil_index = sys.argv.index('--f')
        if fil_index + 1 >= len(sys.argv):
            print("Error: --f option requires a parameter name")
            exit(1)
        fil = sys.argv[fil_index + 1]
        sys.argv.pop(fil_index)  # remove --f
        sys.argv.pop(fil_index)  # remove filter value
        
    output = sys.argv[2] if len(sys.argv) > 2 else 'ordered_data'
    sort_params(input, output, fil)
