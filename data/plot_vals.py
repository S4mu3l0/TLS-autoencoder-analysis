"""
file: data/plot_vals.py
author: Samuel Kudla <xkudlas00@stud.fit.vutbr.cz>
date: 12.05.2026
"""
import sys
import matplotlib.pyplot as plt
import numpy as np

def parse_block(block, valsX, valsY, valsY2, attr1, attr2, attr3):
  # 2 params
  if not attr3:
    if block.get(attr1) and block.get(attr2):
      valsX.append(block[attr1])
      valsY.append(block[attr2])
  # 3 params
  else:
    if block.get(attr1) and block.get(attr2) and block.get(attr3):
      valsX.append(block[attr1])
      valsY.append(block[attr2])
      valsY2.append(block[attr3])
  
def get_values(file: str, attr1: str, attr2: str, attr3: str = None):
  valsX = []
  valsY = []
  valsY2 = []
  block = {}
  if file.endswith('.csv'):
    import pandas as pd
    df = pd.read_csv(file)
    
    df.columns = df.columns.str.strip()
    valsX = df[attr1].tolist()
    valsY = df[attr2].astype(float).tolist()
    
    if attr3 and attr3 in df.columns:
      valsY2 = df[attr3].astype(float).tolist()
  else:  
    with open(file, 'r') as f:
      for line in f:
        if line == '\n':
          parse_block(block, valsX, valsY, valsY2, attr1, attr2, attr3)
          block = {}
          continue
        
        if not (line.startswith('|')):
          continue
        
        # data line
        line = line.strip('|').strip()
        if ':' not in line:
            continue
        name, value = line.split(':')[0].strip(), line.split(':')[1].strip()
        # get rid of units
        if ' ' in value:
          value = value.split(' ')[0]
        # get rid of percentage
        if value.endswith('%'):
          value = value[:-1]
          
        block[name] = value
  
  return valsX,valsY,valsY2
    
def plot_vals(args):
  output_file = None
  save = False

  descending = False
  if '--descending' in args:
    desc_index = args.index('--descending')
    args.pop(desc_index)
    descending = True

  if ('--save' in args):
    save = True
    index = args.index('--save')
    args.pop(index)
    if index < len(args):
      output_file = args[index]
      args.pop(index)
  if not output_file:
    output_file = 'output_graph.png'

  if ('--title' in args):
    title_index = args.index('--title')
    args.pop(title_index)
    if title_index < len(args):
      title = args[title_index]
      args.pop(title_index)
  else:
    title = None

  no_title = False
  if '--no_title' in args:
    title_index = args.index('--no_title')
    args.pop(title_index)
    no_title = True

  attrib_labels = []
  if '--attribs' in args:
    attrib_index = args.index('--attribs')
    args.pop(attrib_index)
    if attrib_index < len(args):
      raw_attribs = args[attrib_index]
      args.pop(attrib_index)
      attrib_labels = [label.strip() for label in raw_attribs.split(';') if label.strip()]

  one_axis = False
  if '--one_axis' in args:
    one_axis_index = args.index('--one_axis')
    args.pop(one_axis_index)
    one_axis = True

  log_scale = False
  if '--log' in args:
    log_index = args.index('--log')
    args.pop(log_index)
    log_scale = True
    
  file = args[1]
  attrX = args[2]
  attrY = args[3]
  attrY2_name = args[4] if len(args) > 4 else None
  
  valsX, valsY, valsY2 = get_values(file, attrX, attrY, attrY2_name)
  valsY = [float(y) for y in valsY]
  valsY2 = [float(y) for y in valsY2]

  numeric_x = True
  try:
    valsX = [float(x) for x in valsX]
  except (ValueError, TypeError):
    numeric_x = False
  
  if numeric_x:
    # 3 params
    if attrY2_name and len(valsY2) > 0:
        combined = sorted(zip(valsX, valsY, valsY2), reverse=descending)
        valsX, valsY, valsY2 = zip(*combined)
    # 2 params
    else:
        combined = sorted(zip(valsX, valsY), reverse=descending)
        valsX, valsY = zip(*combined)
    
    valsX, valsY = list(valsX), list(valsY)
    if attrY2_name and len(valsY2) > 0:
        valsY2 = list(valsY2)
  
  indices_x = np.arange(len(valsX))

  fig, ax1 = plt.subplots(figsize=(10, 8))
  fig.canvas.manager.set_window_title(file)

  y1_label = attrib_labels[0] if len(attrib_labels) > 0 else attrY
  y2_label = attrib_labels[1] if len(attrib_labels) > 1 else attrY2_name

  ax1.set_xticks(indices_x)
  ax1.set_xticklabels(valsX, rotation=45, ha='right')
  
  ax1.plot(indices_x, valsY, marker='o', linestyle='-', color='b', label=y1_label)
  ax1.set_xlabel("")
  ax1.set_ylabel("", color='b')
  tick_color = 'k' if one_axis else 'b'
  ax1.tick_params(axis='y', labelcolor=tick_color)
  if log_scale:
    ax1.set_yscale('log')
  ax1.grid(True, linestyle='--', alpha=0.7)

  if attrY2_name:
      if one_axis:
          ax1.plot(indices_x, valsY2, marker='s', linestyle='--', color='r', label=y2_label)
          ax1.legend(loc='upper left')
      else:
          ax2 = ax1.twinx()
          ax2.plot(indices_x, valsY2, marker='s', linestyle='--', color='r', label=y2_label)
          ax2.set_ylabel("", color='r')
          ax2.tick_params(axis='y', labelcolor='r')
          if log_scale:
            ax2.set_yscale('log')
          
          lines, labels = ax1.get_legend_handles_labels()
          lines2, labels2 = ax2.get_legend_handles_labels()
          ax2.legend(lines + lines2, labels + labels2, loc='upper left')
  else:
      ax1.legend()

  if not numeric_x:
    indices = range(len(valsX))
    ax1.set_xticks(indices)
    ax1.set_xticklabels(valsX, rotation=45, ha='right', fontsize=9)
  if no_title:
    plt.title("")
  else:
    if title:
      plt.title(title)
    else:
      plt.title(f"Dependency of {attrY} {'and ' + attrY2_name if attrY2_name else ''} to {attrX}")
  plt.tight_layout()
  print("Creating graph...")
  if (save):
    plt.savefig(output_file)
  else:
    plt.show()
  
if __name__ == '__main__':
  if (len(sys.argv) < 4 or sys.argv[1] == '-h' or '--help' in sys.argv):
    print("Usage: python plot_vals.py <values_filename> <attr1> <attr2> [attr3] [options]")
    print("\nPositional Arguments:")
    print("  values_filename      Path to values file (.csv or text format)")
    print("  attr1               X-axis attribute name")
    print("  attr2               Y-axis attribute name")
    print("  attr3               Optional second Y-axis attribute name (for dual-axis plot)")
    print("\nOptions:")
    print("  --title <text>      Plot title")
    print("  --no_title          Do not display title")
    print("  --attribs <names>   Custom Y-axis labels separated by semicolon")
    print("  --one_axis          Use single Y-axis for dual parameters")
    print("  --log               Use logarithmic scale")
    print("  --descending        Sort values in descending order")
    print("  --save [filename]   Save plot to file (default: output_graph.png)")
    print("  -h, --help          Show this help message")
    print("\nNote: Output file expected to have values starting with '| ', with structure 'name:value' or .csv format")
    exit(1)
    
  plot_vals(sys.argv)