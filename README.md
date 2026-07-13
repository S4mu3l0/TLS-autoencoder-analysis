# Identification of TLS application connections via autoencoders
This project contains the implementation of the system described in the bachelor thesis: "Use of autoencoders for application identification in network traffic". The development of the implementation scripts was supported by GitHub Copilot, which was utilized as an AI-powered programming assistant to streamline the coding process and optimize development efficiency.  

This file serves as manual for manipulating with the implemented scripts. All script can be run with ``-h`` argument, which displays usage form of script. Arguments in help message enclosed with "<>" are mandatory and arguments enclosed with "[]" are optional. Project consists of 3 parts:
- Data manipulation: directory ``data/``
- TLS parameters embedding: directory ``embeddings/``
- Autoencoder models training and evaluating: directory ``model/``

## Project structure:

```
.
├── README.md
├── config.json
├── requirements.txt
├── create_archive.sh
│
├── data/
│   ├── apps_show.py
│   ├── dataset_show.py
│   ├── join_docs.py
│   ├── plot_vals.py
│   ├── shuffle_csv.py
│   ├── sort_params.py
│   ├── desktop.csv
│   │
│   └── graph_data/
│
├── embeddings/
│   ├── ConnectionEmbedder.py
│   ├── compare.py
│   │
│   ├── ciphersuite/
│   │
│   ├── extensions/
│   │
│   ├── ports/
│   │   └── PortEmbedder.py
│   │
│   ├── SNI/
│   │   └── SNIEmbedder.py
│   │
│   ├── supportedgroup/
│   │
│   └── w2vec/
│       ├── EmbeddingDistanceTester.py
│       ├── EmbeddingViewer.py
│       ├── SkipGramTrainer.py
│       └── TableCreator.py
│
├── model/
│   ├── Classifier.py
│   ├── Undercomplete.py
│   ├── evaluate.py
│   ├── inspect_model.py
│   ├── make_test_data.py
│   ├── set_model_param.py
│   ├── show_loss.py
│   ├── test_model_directory.py
│   ├── train_model_variants.py
│   ├── thresholds.json
│   │
│   └── trained/
```

## Prerequisites
All prerequisites that are listed in ``requirements.txt`` file:
- Python: version >= 3.10
- numpy
- pandas
- matplotlib
- scikit-learn
- torch
- pytorch-ignite

## Data manipulation
Folder contains parsed reference MDA (mobile desktop application) dataset available at [https://github.com/matousp/tls-fingerprinting/tree/main](https://github.com/matousp/tls-fingerprinting/tree/main). This dataset contains TLS connections of applications in csv format divided by ";" delimiter. Each connection (one line) contains multiple TLS and TCP parameters. For this project, parameters "AppName", "SrcPort", "SNI", "Client Extensions", "Client Ciphersuite" and "Client Supported Groups" are necessary.

#### apps_show.py
Usage:
```
python3 apps_show.py <file> [--unique]
```     

Description:
- This script counts number of applications in given TLS connections file, alternatively also count number of unique parameters (only ones this work uses) for each application and prints the results to ``stdout``. This script expects AppName to be included in data file.

Parameters:
- ``<file>``: Path to the input file with TLS connections
- ``--unique``: Script also prints number of unique parameters for each application.

Examples:
```
# show all aplication count in TLS connections file
python3 data/apps_show.py data/desktop.csv

# show all aplication count in TLS connections file and include also count for each unique TLS parameter 
python3 data/apps_show.py data/desktop.csv --unique
--- result ---
AdmntMessenger : 1180
  SNI: 23 unique
  SrcPort: 115 unique
  Client CipherSuite: 1 unique
  Client Extensions: 1180 unique
  Client Supported Groups: 16 unique
AirDroidAirDroid : 1774
  SNI: 43 unique
  SrcPort: 149 unique
  Client CipherSuite: 2 unique
  Client Extensions: 1667 unique
  Client Supported Groups: 17 unique
...
```

#### dataset_show.py
Usage:
```
python3 data/dataset_show.py <input_csv_file> [filter1] [filter2] [filter3] [--d delimiter] [--unique]
```

Description:
- Parses TLS connections data in given csv file and prints result to ``stdout``. To save result, redirect output of script using '>'.

The script supports both formatted text output (default terminal output) and CSV-compatible output (when redirected into a ``.csv`` file).

Parameters:
- ``<input_csv_file>``: Path to the input CSV file with connections.
- ``[filter1]``: Column name used as the primary filter.
	- If only ``filter1`` is provided, script prints all values from this column.
- ``[filter2]``: Filter value for ``filter1`` column.
	- If provided together with ``filter1``, script prints only rows where ``filter1 == filter2``.
- ``[filter3]``: Display column name.
	- If provided with ``filter1`` and ``filter2``, script prints only this selected attribute from matching rows.
- ``--d <delimiter>``: Sets CSV delimiter (default is ``;``).
- ``--unique``: Prints count of unique entries in the current output selection.


Behavior summary:
- No filters: prints all rows in readable ``name: value`` format.
- ``filter1`` only: prints all values from selected column.
- ``filter1 + filter2``: prints full matching rows.
- ``filter1 + filter2 + filter3``: prints only selected attribute from matching rows.

Examples:
```
# Show all rows in readable format
python3 data/dataset_show.py data/desktop.csv

# Show all values from AppName column
python3 data/dataset_show.py data/desktop.csv AppName

# Show rows where AppName is AirDroid
python3 data/dataset_show.py data/desktop.csv AppName AirDroid

# Show only SNI for rows where AppName is AirDroid
python3 data/dataset_show.py data/desktop.csv AppName AirDroid SNI

# Use comma as delimiter and show unique values
python3 data/dataset_show.py data/desktop.csv AppName --d , --unique
```

#### join_docs.py
Usage:
```
python3 data/join_docs.py <--files file1.csv,file2.csv,...> --attrib <join_attribute> [output_name] [--loaded]
```

Description:
- Joins multiple CSV files on the specified attribute key (inner join). Non-key columns names are prefixed with the source file name to keep them distinct from matching attribute names.

Parameters:
- `--files`: comma-separated list of input CSV files (required).
- `--attrib`: column name used as the join key (required).
- `output_name`: optional output filename (default: `joined.csv`).
- `--loaded`: if provided, rows where the column `New` is True are filtered out before joining. This option was used for filtering pre-training data from fine-tuning data. 

Example:
```
# Join training_results.csv and test_results.csv on same Model attribute value and save to models_merged.csv
python3 data/join_docs.py --files training_results.csv,test_results.csv --attrib Model models_merged.csv
```

#### plot_vals.py
Usage:
```
python3 data/plot_vals.py <file> <attrX> <attrY> [attrY2] [--title "text"] [--no_title] [--attribs nameX;nameY] [--one_axis] [--log] [--descending] [--save [filename]]
```

Description:
- Creates a graph from values extracted from a CSV or the ``dataset_show.py`` text format. Supports two Y-series (dual-axis) when `attrY2` is provided.

Important flags:
- `--title <text>`: set the plot title. `--no_title` disables title.
- `--attribs nameX;nameY`: custom axis labels (semicolon-separated).
- `--one_axis`: force single Y-axis when two Y-series present.
- `--log`: use logarithmic Y-scale.
- `--descending`: sort points by X descending before plotting. Attribute X must by numeric.
- `--save [filename]`: save output image (default `output_graph.png`).

Examples:
```
# Simple plot from CSV columns
python3 data/plot_vals.py data/graph_data/JAirDroid.csv Model Accuracy

# Dual-series and save to file
python3 data/plot_vals.py data/graph_data/JAirDroid.csv Model Accuracy MCC --save droid.png --attribs "acc;mcc" --descending
```

Folder `graph_data` contains data that was used for plotting graphs in reference thesis.

#### shuffle_csv.py
Usage:
```
python3 data/shuffle_csv.py <csv_file> [output_file]
```
Description:
- Randomly shuffles rows of a semicolon-delimited CSV and writes result to `output_file` (default `shuffled.csv`). Uses a fixed random seed for reproducibility.

Example:
```
python3 data/shuffle_csv.py data/desktop.csv shuffled_desktop.csv
```

#### sort_params.py
Usage:
```
python3 data/sort_params.py <input_file> [output_filename] [--f "<parameter_name>"]
```
Description:
- Sorts numeric parameter lists found in either the project's text format or CSV files. For text format it will rewrite parameter lines with sorted values; for CSV it will sort items inside observed parameter columns and write a new CSV.
Parameters:
- `--f <parameter_name>`: restrict output to a single parameter (writes only values, not the parameter name). Supported names: `Client Extensions`, `Client Supported Groups`, `Client CipherSuite`.
- `output_filename`: optional output path (default `ordered_data` for text or header-based default for CSV).

Examples:
```
# Sort all parameters and output to file
python3 data/sort_params.py raw_params.txt sorted_params.txt

# Extract only sorted supported groups values
python3 data/sort_params.py raw_params.txt out.txt --f "Client Supported Groups"
```

## TLS parameters embedding
Folder contains embedding modules and all necessary files for embedding values from reference TLS connections dataset.

#### SNI
This directory contains the `SNIEmbedder` class for embedding SNI (Server Name Indication) TLS parameter. The embedder creates SNI vectors using a `HashingVectorizer` with character n-grams and augments them with two normalized features:
- **Entropy**: Shannon entropy of the domain name characters (normalized by max possible entropy of 5.25)
- **Length**: Normalized domain length (divided by 64.0)

The final embedding has 130 features (128 from hashing + entropy + length).

Usage:
```
python3 embeddings/SNI/SNIEmbedder.py <sni_file/SNI> [--inline]
```

Parameters:
- `<sni_file/SNI>`: Path to a file containing SNI values (one per line), or a single SNI string when using `--inline`
- `--inline`: If provided, treats the argument as a single SNI string and prints its embedding to stdout

Examples:
```
# Embed all SNI values from a file
python3 embeddings/SNI/SNIEmbedder.py sni_list.txt

# Embed a single SNI domain
python3 embeddings/SNI/SNIEmbedder.py "www.example.com" --inline
```

Output format:
```
SNI: test.com, Embedding: [0.0, -0.40824830532073975, 0.0, 0.0, ..., 0.0, 0.523809552192688, 0.125]
```

#### ports
This directory contains the `PortEmbedder` class for embedding TCP port numbers into a normalized scalar value. The embedding is built from the decimal digits of the port number, where digits with higher value have more impact of the final embedding. The final value is normalized by the maximum possible port embedding, so the output is in the range `(0, 1)`.

Usage:
```
python3 embeddings/ports/PortEmbedder.py <port_number>
```

Parameters:
- `<port_number>`: Port number to embed, from `0` to `65535`

Behavior:
- Computes a weighted embedding value from the decimal digits of the port number
- Normalizes the result against the maximum embedding value for port `65535`
- Prints the normalized scalar embedding to stdout

Example:
```
# Embed port 443
python3 embeddings/ports/PortEmbedder.py 443
```

Output format:
```
0.0024236010472482392
```

#### w2vec
This directory contains the Word2Vec-based helpers used to build, inspect, and compare embeddings for TLS parameter values.

##### TableCreator\.py
This script creates the embedding table for a feature directory. It reads `existing_vals` and `train_vals`, merges the values, creates `mapping.json`, and stores the initial PyTorch embedding model in `embedding.pt`.

Usage:
```
python3 embeddings/w2vec/TableCreator.py <delimiter_in_train_vals> <feature_dir_name>
```

Parameters:
- `<delimiter_in_train_vals>`: delimiter used to split values in `train_vals`
- `<feature_dir_name>`: feature directory name containing all necessary files to embed feature. 
Possible values: `extensions`, `ciphersuite`, or `supportedgroup`

Example:
```
python3 embeddings/w2vec/TableCreator.py - extensions
```

##### SkipGramTrainer\.py
This script trains the embedding table for a feature directory with a skip-gram-like objective. It loads `mapping.json` and `train_vals`, builds context pairs using a configurable window size, and updates `embedding.pt`.

Usage:
```
python3 embeddings/w2vec/SkipGramTrainer.py --path <feature_dir_name> [--delim delimiter] [--window N] [--epochs N] [--lr value]
```

Parameters:
- `--path <feature_dir_name>`: feature directory to train with possible values: `extensions`, `ciphersuite`, or `supportedgroup`
- `--delim <delimiter>`: delimiter used in `train_vals` entries (default `-`)
- `--window <N>`: context window size (default `2`)
- `--epochs <N>`: training epochs (default `10`)
- `--lr <value>`: learning rate (default `0.001`)

Example:
```
python3 embeddings/w2vec/SkipGramTrainer.py --path extensions --delim - --window 2 --epochs 5
```

##### EmbeddingViewer\.py
This module loads an embedding model and mapping from a feature directory and returns average embeddings for one or more values. It is used by `ConnectionEmbedder.py` for the `Client Extensions`, `Client CipherSuite`, and `Client Supported Groups` parameters.

Usage:
```
python3 embeddings/w2vec/EmbeddingViewer.py <value> <delimiter> <dir_of_model>
```

Parameters:
- `<value>`: one value or a delimiter-separated list of values
- `<delimiter>`: delimiter used to split the input values
- `<dir_of_model>`: feature directory containing `mapping.json` and `embedding.pt`

Example:
```
python3 embeddings/w2vec/EmbeddingViewer.py "0-5-10-11" - extensions
```

Output format:
```
tensor([ 0.3217, -0.0772,  0.2854, -0.3146, -0.1408, -0.1559,  0.3387, -0.1895,
        -0.3325,  0.2306, -0.3287, -0.0333, -0.1840,  0.0825,  0.2640,  0.3635,
         0.1333])
```

##### EmbeddingDistanceTester\.py
This script calculates the Euclidean distance between two embeddings from the same feature directory. It can compare single values directly (without normalization) or single and multi-value inputs when a delimiter is provided (with normalization).

Usage:
```
python3 embeddings/w2vec/EmbeddingDistanceTester.py <value1> <value2> <directory_of_model> [delimiter]
```

Parameters:
- `<value1>`: first value to compare
- `<value2>`: second value to compare
- `<directory_of_model>`: feature directory containing `mapping.json` and `embedding.pt`
- `[delimiter]`: optional delimiter for multi-value inputs

Example:
```
python3 embeddings/w2vec/EmbeddingDistanceTester.py 5 47 extensions 

python3 embeddings/w2vec/EmbeddingDistanceTester.py "0-5-10" "0-5-11" extensions -
```

Output format:
```
Distance between 0-5-10 and 0-5-11: 0.0459906980
```

##### ConnectionEmbedder\.py
This script combines all parameter embedders classes (`SNIEmbedder`, `PortEmbedder`, `EmbeddingViewer`) and creates one final connection embedding vector by merging all embeddings from these classes.

Usage:
```
python3 embeddings/ConnectionEmbedder.py <filename> [--one-line]
```

Parameters:
- `<filename>`: path to connections file (CSV or text format from `dataset_show.py`)
- `--one-line`: prints full parsed result as one dictionary line

Behavior:
- Parses all connections from input file
- Builds per-parameter embeddings
- Concatenates them in fixed order: ``['SNI', 'SrcPort', 'Client CipherSuite', 'Client Extensions', 'Client Supported Groups']``
- Prints embedding slices by parameter

Example:
```
python3 embeddings/ConnectionEmbedder.py embeddings/test
```

##### compare\.py
Utility script that compares first embedded connection from two files element-by-element and reports indices where values differ.

Usage:
```
python3 embeddings/compare.py <file1> <file2>
```

Parameters:
- `<file1>`: first connection file
- `<file2>`: second connection file

Example:
```
python3 embeddings/compare.py embeddings/AirDroid embeddings/Deezer
```


## Autoencoder models training and evaluating
This directory contains scripts for training, inspecting, testing, and evaluating autoencoder models.

#### Undercomplete\.py
Core autoencoder implementation and training entrypoint. Loads config from `config.json`, trains model on embedded tensor data (made by `make_test_data.py`), and updates threshold values. Dimensions of model is specified only for encoder in parameter `model_dimensions` in config. Early stopping can be used when enabling `enabled` to 1 (True) in `early_stopping`. This will result in training with early stopping parameters defined here. If multiple  Running script will also result in creating `.pth` file named after value set in `train_model_name` parameter in config that represents trained autoencoder model. This script expects the training file to have shuffled data, which is divided based on value in `train_split` parameter in config. Model is then trained with learing rate defined in `lr` parameter in config. Skript also saves threshold, but cannot be interupted.

Usage:
```
python3 model/Undercomplete.py <datafile.pt> [epochs] [csv_out_file] [--print_epochs]
```

Parameters:
- `<datafile.pt>`: input tensor file with embedded connections
- `[epochs]`: max epochs (default `100`)
- `[csv_out_file]`: optional CSV output for training summary
- `--print_epochs`: prints per-epoch loss

#### make_test_data\.py
Converts raw connection files (CSV or text format) into stacked embedding tensors for model training.

Usage:
```
python3 model/make_test_data.py <raw_data_file> [output_file_name]
```

Parameters:
- `<raw_data_file>`: raw connection file
- `[output_file_name]`: optional `.pt` output name (default `embedded_vectors.pt`)

Example:
```
python3 model/make_test_data.py data/desktop.csv all_embeddings.pt
```

#### show_loss\.py
Shows reconstruction loss for one file, all connections in a file, or batch files in `test_data`. Model that this script uses is specified in `train_model_name` parameter in config.

Usage:
```
python3 model/show_loss.py [connections_file] [--one|--stats]
```

Parameters:
- `[connections_file]`: optional input file; if omitted, script iterates `test_data`
- `--one`: evaluate only first connection in input file
- `--stats`: print summary statistics over all losses

#### evaluate\.py
Classifies connections with `Classifier` (binary/multiclass) and computes evaluation metrics. Supports writing summary rows to CSV. Evaluation is very sensitive to changes in TLS parameter embeddings, so it is necessary to make sure the trained models have the right embeddings in folders `embeddings/extensions`, `embeddings/supportedgroup`, `embeddings/ciphersuite`.

Usage:
```
python3 model/evaluate.py <connections_file> [out_csv_file]
```

Parameters:
- `<connections_file>`: raw connection file
- `[out_csv_file]`: optional CSV path for appending evaluation results

#### Classifier\.py
Classifier class for assigning connections to nearest model using reconstruction loss and per-model thresholds. Thresholds are firstly loaded from `thresholds.json` file, where the threshold can be manually modified. If unsuccessful, threshold is loaded from model property.

Usage:
```
python3 model/Classifier.py <connection_file> [--multiple]
```

Parameters:
- `<connection_file>`: file containing one or more connections
- `--multiple`: classify all connections and print per-connection results

Behavior:
- Loads models listed in `config.json` (`test_models_name`)
- Loads thresholds from `model/thresholds.json` or model attribute
- Predicts nearest model among candidates under threshold

#### test_model_directory\.py
Runs evaluation (`evaluate.py`) over all `.pth` models in a directory by temporarily updating config and writing result to one output file.

Usage:
```
python3 model/test_model_directory.py <directory_of_models> <reference_raw_data_file> [output_filename] [[dest_app_name]]
```

Parameters:
- `<directory_of_models>`: folder containing `.pth` models
- `<reference_raw_data_file>`: test file to evaluate each model on
- `[output_filename]`: output base name (default `results`)
- `[[dest_app_name]]`: optional override for destination app names

#### train_model_variants\.py
Automates training of many model topologies based on config (`proportional_scaling` or custom layer combinations). For proportional scaling, parameter `max_layers` defines the maximum number of layers of final model that the script will train, starting with the smallest. The range of sizes of encoder and reverse decoder can be defined in `input_size` and `latent_size`. For linear creation (progressively adding each layer, starting from the largest) of defined layers in `mid_layers_to_test` parameter, parameter `combine_dimensions` needs to be set to 0 (False). Otherwise all combinations of layers in `mid_layers_to_test` are created, while satisfying the descending tendency of layer size in encoder structure (decoder is always mirrored image of encoder).
Training uses `early_stopping` configuration, especially parameters `starting_delta` and `all_apps_fine_tune`. These parameters define starting values for fine tuning dynamic delta and pretraining dynamic delta. Dynamic delta is used to lower `min_delta` with increasing sizes of models, since the problem of vanishing gradient grows with number of model layers.

Usage:
```
python3 model/train_model_variants.py <AppFile.pt> [AllDataFile.pt]
```

Parameters:
- `<AppFile.pt>`: destination-app embeddings file for fine-tuning
- `[AllDataFile.pt]`: optional all-app embeddings for pretraining

#### set_model_param\.py
Sets one model attribute (for example threshold) and saves the model back.

Usage:
```
python3 model/set_model_param.py <model_file.pth> <model_param> <value>
```

Example:
```
python3 model/set_model_param.py model/trained/AirDroid/182-140-98-56-14.pth threshold 0.002
```

#### inspect_model\.py
Prints detailed model information: encoder/decoder structure, parameter shapes, optimizer settings, threshold, and early-stopping config.

Usage:
```
python3 model/inspect_model.py <model_path>
```

Parameters:
- `<model_path>`: path to `.pth` model file

### Reference models
Folder `trained` contains reference models that was trained (works only with) currently set embedding. These models were used in study and are source for some data in folder `graph_data`. 
