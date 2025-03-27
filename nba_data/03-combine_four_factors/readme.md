# NBA Four Factors Data Processor

## Overview

This Python script combines NBA "Four Factors" statistics from multiple CSV files with NBA schedule data to create a comprehensive reporting dataset. The four factors, as identified by basketball statistician Dean Oliver, are the key elements that determine basketball success:

1. Shooting (measured by effective field goal percentage)
2. Turnovers (turnover percentage)
3. Rebounding (offensive rebounding percentage)
4. Free throws (free throw rate)

The script reorganizes the data so each game appears as a single row with both teams' stats, making it easier to analyze game outcomes in relation to these key performance metrics.

## Features

- Combines multiple four factors CSV files into a single dataset
- Merges four factors data with NBA schedule information
- Computes important derived metrics:
  - Point differential
  - Four factors differentials between home and away teams
  - Binary win/loss indicators
- Organizes data in a logical, analysis-friendly format

## Requirements

- Python 3.6 or higher
- pandas
- os
- glob
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
python process_nba_data.py --four-factors-dir <directory> --schedule <schedule-file> --output <output-file>
```

#### Arguments:

- `--four-factors-dir`, `-f` (required): Directory containing the four factors CSV files
- `--schedule`, `-s` (required): Path to the NBA schedule CSV file
- `--output`, `-o` (optional): Output file path (default: `nba_report.csv`)

### Example

```bash
python process_nba_data.py --four-factors-dir ./data/four_factors --schedule ./data/nba_schedule.csv --output ./reports/nba_analysis.csv
```

### As a Module

You can also import and use the functionality in your own Python scripts:

```python
from process_nba_data import combine_four_factors_data

# Combine data
data = combine_four_factors_data(
    four_factors_dir='./data/four_factors',
    schedule_file='./data/nba_schedule.csv',
    output_file='./reports/nba_analysis.csv'
)

# Now you can work with the data
print(f"Processed {len(data)} games")
```

## Output Format

The resulting CSV file includes the following types of columns:

- **Game information**: Date, time, arena, attendance, etc.
- **Away team stats**: Team name, score, shooting, turnovers, rebounding, free throws, offensive rating
- **Home team stats**: Team name, score, shooting, turnovers, rebounding, free throws, offensive rating  
- **Pace**: Possessions per 48 minutes
- **Calculated metrics**: Point differential, four factors differentials
- **Source information**: Box score URL, notes

## Notes

- Each game appears exactly once in the output dataset
- The script automatically handles missing columns in the input data
- Four factors differentials are calculated as home team minus away team values (except for turnovers, which are reversed since lower is better)
- The script provides progress updates during processing

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
