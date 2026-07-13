"""
file: data/apps_show.py
author: Samuel Kudla <xkudlas00@stud.fit.vutbr.cz>
date: 12.05.2026
"""
import sys
import csv

def show_apps(filename: str, print_unique: bool = False):
  """
  Function reading file with TLS connections and printing number of connections per app. 
  If print_unique is True, also prints number of unique parameters (SNI, SrcPort, Client CipherSuite, Client Extensions, Client Supported Groups) for each app.
  """
  apps = {}
  app_unique_params = {}
  param_types = ['SNI', 'SrcPort', 'Client CipherSuite', 'Client Extensions', 'Client Supported Groups']
  total = 0
  with open(filename, 'r') as f:
    if filename.endswith('.csv'):
      reader = csv.DictReader(f, delimiter=';')
      for row in reader:
        app = row.get('AppName')
        if app:
          total += 1
          if app not in apps:
              apps[app] = 1
              app_unique_params[app] = {p: set() for p in param_types}
          else:
              apps[app] += 1
          if print_unique:
            # Track unique parameters for this app
            for param_type in param_types:
              val = row.get(param_type, '')
              if val:
                app_unique_params[app][param_type].add(val)
    else: 
      # Parse block-based format (key:value pairs blocks separated by blank lines)
      block = {}
      line = f.readline()
      while line:
        line = line.strip()
        
        if not line:
          # End of block
          app = block.get('AppName', '')
          if app:
            total += 1
            if app not in apps:
              apps[app] = 1
              app_unique_params[app] = {p: set() for p in param_types}
            else:
              apps[app] += 1
            if print_unique:
              # Track unique parameters for this app
              for param_type in param_types:
                val = block.get(param_type, '')
                if val:
                  app_unique_params[app][param_type].add(val)
          block = {}
        elif ':' in line:
          # Parse key:value pair
          key, val = line.split(':', 1)
          key = key.strip()
          val = val.strip()
          block[key] = val
            
        line = f.readline()
      
      # Handle last block if file doesn't end with blank line
      app = block.get('AppName', '')
      if app:
        total += 1
        if app not in apps:
          apps[app] = 1
          app_unique_params[app] = {p: set() for p in param_types}
        else:
          apps[app] += 1
        if print_unique:
          for param_type in param_types:
            val = block.get(param_type, '')
            if val:
              app_unique_params[app][param_type].add(val)
  
  print("Apps (",len(apps),"): ")
  total = 0
  for key, value in apps.items():
    print(key,":", value)
    if print_unique and key in app_unique_params:
      for param_type in param_types:
        count = len(app_unique_params[key][param_type])
        if count > 0:
          print(f"  {param_type}: {count} unique")
    
    total += value
  print()
  print("Total connections:", total)
  print()

if __name__ == "__main__":
  if len(sys.argv) < 2 or '-h' in sys.argv or '--help' in sys.argv:
    print("Usage: python3 apps_show.py <app_file_name> [--unique]")
    print("Display number of connections per app from a connections file.")
    print("\nArguments:")
    print("  app_file_name    Path to connections file (CSV or text format)")
    print("\nOptions:")
    print("  --unique         Show number of unique parameters per app")
    print("  -h, --help       Show this help message")
    exit(1)
  show_apps(sys.argv[1], print_unique='--unique' in sys.argv)