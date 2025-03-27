# NBA Data Format Converter

## Overview

This Python script converts detailed NBA game data from the `nba_report.csv` format into a simplified format suitable for machine learning models. The converter transforms comprehensive game statistics into a standardized format focusing on the Four Factors metrics and team abbreviations.

## Features

- Converts detailed NBA game reports into ML-ready data format
- Transforms full team names to standard three-letter abbreviations (e.g., "Boston Celtics" → "BOS")
- Maintains critical Four Factors statistics:
  - Effective Field Goal Percentage (eFG%)
  - Turnover Percentage (TOV%)
  - Offensive Rebounding Percentage (ORB%)
  - Free Throw Rate (FT Rate)
- Preserves game pace and offensive rating metrics
- Includes binary win/loss indicator for model training

## Requirements

- Python 3.6 or higher
- pandas
- argparse (for command-line operation)

## Installation

```bash
# Clone the repository or download the script
git clone <repository-url>

# Navigate to the directory
cd <repository-directory>

# Install required packages
pip install pandas
```

## Usage

### Command Line Interface

The script can be run from the command line with the following arguments:

```bash
python convert_nba_data.py --input <input-file> --output <output-file>
```

#### Arguments:

- `--input`, `-i` (required): Path to the input CSV file (nba_report.csv format)
- `--output`, `-o` (optional): Output file path (default: `nba_games_stats.csv`)

### Example

```bash
python convert_to_ml_format.py --input combined_nba_report.csv --output converted_report.csv
```

This is the typical usage pattern for converting the combined NBA report data to the ML-ready format.

### As a Module

You can also import and use the functionality in your own Python scripts:

```python
from convert_nba_data import convert_to_ml_format

# Convert data
data = convert_to_ml_format(
    input_file='./data/nba_report.csv',
    output_file='./data/nba_games_stats.csv'
)

# Now you can work with the converted data
print(f"Converted {len(data)} games")
```

## Output Format

The resulting CSV file includes the following columns:

- **date**: Game date in ISO format
- **visitor_team**: Away team abbreviation (3 letters)
- **visitor_score**: Away team score
- **home_team**: Home team abbreviation (3 letters)
- **home_score**: Home team score
- **visitor_pace** and **home_pace**: Possessions per 48 minutes (same value for both)
- **Four Factors metrics** for each team:
  - **visitor_efg**/**home_efg**: Effective Field Goal Percentage
  - **visitor_tov_pct**/**home_tov_pct**: Turnover Percentage
  - **visitor_orb_pct**/**home_orb_pct**: Offensive Rebounding Percentage
  - **visitor_ft_rate**/**home_ft_rate**: Free Throw Rate
- **visitor_ortg**/**home_ortg**: Offensive Rating
- **home_win**: Binary indicator (1 if home team won, 0 if away team won)

## Team Abbreviations

The script maps full team names to standard three-letter abbreviations:

| Full Team Name | Abbreviation |
|----------------|--------------|
| Atlanta Hawks | ATL |
| Boston Celtics | BOS |
| Brooklyn Nets | BKN |
| Charlotte Hornets | CHO |
| Chicago Bulls | CHI |
| ... | ... |

Full mapping is included in the code's `team_name_to_abbr` function.

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
