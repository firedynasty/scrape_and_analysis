#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import MinMaxScaler
import itertools
from datetime import datetime
import os
import sys

# --- DATA LOADING AND PREPARATION FUNCTIONS ---

def load_data(csv_file):
    """Load NBA game data from CSV file"""
    try:
        df = pd.read_csv(csv_file)
        return df
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")
        return None

def harmonize_column_order(historical_df, new_df):
    """Harmonize column order between datasets to ensure consistency"""
    # Get the column order of the historical dataset
    historical_columns = historical_df.columns.tolist()
    new_columns = new_df.columns.tolist()
    
    if historical_columns != new_columns:
        print("Column order difference detected! Reordering columns.")
        
        # Ensure all columns exist in both datasets
        common_columns = [col for col in historical_columns if col in new_columns]
        missing_in_new = [col for col in historical_columns if col not in new_columns]
        extra_in_new = [col for col in new_columns if col not in historical_columns]
        
        if missing_in_new:
            print(f"Warning: These columns are missing in the new dataset: {missing_in_new}")
        if extra_in_new:
            print(f"Warning: These columns are only in the new dataset: {extra_in_new}")
        
        # Add missing columns with placeholder data
        for col in missing_in_new:
            if pd.api.types.is_numeric_dtype(historical_df[col]):
                new_df[col] = 0
            else:
                new_df[col] = ""
                
        # Reorder to match historical_columns
        reordered_new_df = new_df[historical_columns]
        return historical_df, reordered_new_df
    
    return historical_df, new_df

def preprocess_data(df):
    """Preprocess data into team-centric format"""
    # Sort by date
    df = df.sort_values("date")
    df = df.reset_index(drop=True)
    
    # Create a unified dataframe with team-based entries
    all_team_games = []
    
    # For each game, create two rows (one for each team)
    for _, row in df.iterrows():
        # Skip rows with missing essential data
        if pd.isnull(row['date']) or pd.isnull(row['visitor_team']) or pd.isnull(row['home_team']):
            continue
        
        try:
            # Home team entry
            home_entry = {
                'date': row['date'],
                'team': row['home_team'],
                'opponent': row['visitor_team'],
                'is_home': 1,
                'won': row['home_win'],
                'pace': row['home_pace'],
                'efg': row['home_efg'],
                'tov_pct': row['home_tov_pct'],
                'orb_pct': row['home_orb_pct'],
                'ft_rate': row['home_ft_rate'],
                'ortg': row['home_ortg'],
                'opp_pace': row['visitor_pace'],
                'opp_efg': row['visitor_efg'],
                'opp_tov_pct': row['visitor_tov_pct'],
                'opp_orb_pct': row['visitor_orb_pct'],
                'opp_ft_rate': row['visitor_ft_rate'],
                'opp_ortg': row['visitor_ortg'],
                'score': row['home_score'],
                'opp_score': row['visitor_score'],
                'score_diff': row['home_score'] - row['visitor_score']
            }
            
            # Away team entry
            away_entry = {
                'date': row['date'],
                'team': row['visitor_team'],
                'opponent': row['home_team'],
                'is_home': 0,
                'won': 1 if row['home_win'] == 0 else 0,  # Inverse of home_win
                'pace': row['visitor_pace'],
                'efg': row['visitor_efg'],
                'tov_pct': row['visitor_tov_pct'],
                'orb_pct': row['visitor_orb_pct'],
                'ft_rate': row['visitor_ft_rate'],
                'ortg': row['visitor_ortg'],
                'opp_pace': row['home_pace'],
                'opp_efg': row['home_efg'],
                'opp_tov_pct': row['home_tov_pct'],
                'opp_orb_pct': row['home_orb_pct'],
                'opp_ft_rate': row['home_ft_rate'],
                'opp_ortg': row['home_ortg'],
                'score': row['visitor_score'],
                'opp_score': row['home_score'],
                'score_diff': row['visitor_score'] - row['home_score']
            }
            
            all_team_games.append(home_entry)
            all_team_games.append(away_entry)
        except Exception as e:
            print(f"Error processing row: {e}")
            continue
    
    # Convert to DataFrame
    team_df = pd.DataFrame(all_team_games)
    
    # Extract year from date for grouping
    team_df['year'] = pd.to_datetime(team_df['date']).dt.year
    
    return team_df

def add_target(df):
    """Add target variable (will team win next game?)"""
    # For each team, add target variable (did they win their next game?)
    def process_team(group):
        group = group.sort_values('date')
        group['target'] = group['won'].shift(-1)
        return group
    
    df = df.groupby('team', group_keys=False).apply(process_team)
    
    # Handle NaN values in target
    df['target'] = df['target'].fillna(2)
    df['target'] = df['target'].astype(int)
    
    return df

def add_rolling_stats(df, window=10):
    """Calculate rolling statistics for each team"""
    # Features to calculate rolling averages for
    rolling_features = [
        'won', 'pace', 'efg', 'tov_pct', 'orb_pct', 'ft_rate', 'ortg',
        'opp_pace', 'opp_efg', 'opp_tov_pct', 'opp_orb_pct', 'opp_ft_rate', 'opp_ortg',
        'score_diff'
    ]
    
    try:
        # Calculate rolling stats for each team within each year
        def calculate_rolling(team_year_group):
            team_year_group = team_year_group.sort_values('date')
            
            # Calculate rolling stats
            for feature in rolling_features:
                team_year_group[f'{feature}_rolling'] = team_year_group[feature].rolling(window=window, min_periods=1).mean()
                
            return team_year_group
        
        df = df.groupby(['team', 'year'], group_keys=False).apply(calculate_rolling)
    except Exception as e:
        print(f"Error in rolling stats calculation: {e}")
        # Try a more robust approach if the above fails
        teams = df['team'].unique()
        years = df['year'].unique()
        
        processed_dfs = []
        for team in teams:
            for year in years:
                subset = df[(df['team'] == team) & (df['year'] == year)].copy()
                if not subset.empty:
                    subset = subset.sort_values('date')
                    for feature in rolling_features:
                        subset[f'{feature}_rolling'] = subset[feature].rolling(window=window, min_periods=1).mean()
                    processed_dfs.append(subset)
        
        if processed_dfs:
            df = pd.concat(processed_dfs)
    
    return df

def prepare_matchup_features(df):
    """Create features for upcoming matchup"""
    # Add next game info
    def add_next_game_info(team_group):
        team_group = team_group.sort_values('date')
        team_group['opponent_next'] = team_group['opponent'].shift(-1)
        team_group['is_home_next'] = team_group['is_home'].shift(-1)
        team_group['date_next'] = team_group['date'].shift(-1)
        return team_group
    
    df = df.groupby('team', group_keys=False).apply(add_next_game_info)
    
    # Remove rows where next game info is missing
    df = df.dropna(subset=['opponent_next', 'date_next'])
    
    # Prepare to merge with opponent's rolling stats
    matchup_df = df.copy()
    
    # Rename columns for clarity after merge
    rolling_cols = [col for col in matchup_df.columns if '_rolling' in col]
    opponent_cols = {col: f'opp_{col}' for col in rolling_cols}
    
    # Create a version of the dataframe with opponent stats
    opponent_df = matchup_df[['team', 'date'] + rolling_cols].copy()
    opponent_df = opponent_df.rename(columns=opponent_cols)
    
    # Merge team stats with their next opponent's stats
    merged_df = pd.merge(
        matchup_df,
        opponent_df,
        left_on=['opponent_next', 'date_next'],
        right_on=['team', 'date'],
        suffixes=('', '_opp')
    )
    
    return merged_df

def create_matchup_prediction(matchup_data, teams=['GSW', 'ATL'], prediction_date=None):
    """Create a simulated matchup based on recent data"""
    if prediction_date is None:
        prediction_date = datetime.now().strftime('%Y-%m-%d')
        
    # Filter to only include specified teams most recent games
    team1_games = matchup_data[matchup_data['team'] == teams[0]].sort_values('date')
    team2_games = matchup_data[matchup_data['team'] == teams[1]].sort_values('date')
    
    if team1_games.empty or team2_games.empty:
        return pd.DataFrame()  # Return empty dataframe if data is missing
    
    team1_latest = team1_games.iloc[-1].copy()
    team2_latest = team2_games.iloc[-1].copy()
    
    # Get the rolling columns
    rolling_cols = [col for col in matchup_data.columns if '_rolling' in col]
    
    # Create matchups in both directions
    prediction_rows = []
    
    # Team1 home vs Team2
    team1_home = {
        'date': team1_latest['date'],
        'team': teams[0],
        'opponent': teams[1],
        'is_home': 1,
        'opponent_next': teams[1],
        'is_home_next': 1,  # Team1 at home
        'date_next': prediction_date
    }
    
    # Team2 home vs Team1
    team2_home = {
        'date': team2_latest['date'],
        'team': teams[1],
        'opponent': teams[0],
        'is_home': 1,
        'opponent_next': teams[0],
        'is_home_next': 1,  # Team2 at home
        'date_next': prediction_date + ' alt'  # Different date to distinguish
    }
    
    # Add team's rolling stats
    for col in rolling_cols:
        team1_home[col] = team1_latest[col]
        team2_home[col] = team2_latest[col]
    
    # Add opponent stats
    for col in rolling_cols:
        opp_col = f'opp_{col}'
        team1_home[opp_col] = team2_latest[col]
        team2_home[opp_col] = team1_latest[col]
    
    prediction_rows.append(team1_home)
    prediction_rows.append(team2_home)
    
    # Convert to DataFrame
    prediction_df = pd.DataFrame(prediction_rows)
    
    return prediction_df

def build_model_and_predict(historical_data, matchup_prediction):
    """Build model and predict matchup outcome"""
    if matchup_prediction.empty:
        return pd.DataFrame(), None, None
    
    # Define features and target
    feature_cols = [col for col in historical_data.columns if '_rolling' in col]
    feature_cols.append('is_home_next')
    
    # Prepare training data
    train_df = historical_data.dropna(subset=feature_cols + ['target'])
    
    # Ensure prediction data has all required features
    pred_df = matchup_prediction.copy()
    
    # Check if any features are missing
    missing_cols = [col for col in feature_cols if col not in pred_df.columns]
    if missing_cols:
        for col in missing_cols:
            pred_df[col] = 0.0
    
    # Ensure the order of columns is the same for both dataframes
    train_features_df = train_df[feature_cols].copy()
    pred_features_df = pred_df[feature_cols].copy()
    
    # Scale features
    scaler = MinMaxScaler()
    train_features = scaler.fit_transform(train_features_df)
    pred_features = scaler.transform(pred_features_df)
    
    # Train model
    model = RidgeClassifier(alpha=1.0)
    model.fit(train_features, train_df['target'])
    
    # Make predictions
    pred_df['prediction'] = model.predict(pred_features)
    
    # Calculate feature importance
    if hasattr(model, 'coef_'):
        importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.coef_[0]
        })
        importance['Abs_Importance'] = abs(importance['Importance'])
        importance = importance.sort_values('Abs_Importance', ascending=False)
    else:
        importance = None
    
    return pred_df, model, importance

def analyze_team_stats(team_data, team1, team2, num_games=10):
    """Analyze recent performance stats for both teams"""
    team1_stats = team_data[team_data['team'] == team1].sort_values('date').tail(num_games)
    team2_stats = team_data[team_data['team'] == team2].sort_values('date').tail(num_games)
    
    if team1_stats.empty or team2_stats.empty:
        return None, None, None
    
    # Calculate key metric averages
    metrics = [
        ('Win Rate', 'won', True),
        ('Effective FG%', 'efg', True),
        ('Turnover Rate', 'tov_pct', False),
        ('Offensive Rebound %', 'orb_pct', True),
        ('Free Throw Rate', 'ft_rate', True),
        ('Offensive Rating', 'ortg', True),
        ('Opp. Effective FG%', 'opp_efg', False),
        ('Opp. Turnover Rate', 'opp_tov_pct', True),
        ('Opp. Offensive Rebound %', 'opp_orb_pct', False),
        ('Opp. Free Throw Rate', 'opp_ft_rate', False),
        ('Defensive Rating', 'opp_ortg', False),
        ('Score Differential', 'score_diff', True)
    ]
    
    comparison = []
    for name, feature, higher_is_better in metrics:
        team1_val = team1_stats[feature].mean()
        team2_val = team2_stats[feature].mean()
        
        if higher_is_better:
            advantage = team1 if team1_val > team2_val else team2
        else:
            advantage = team1 if team1_val < team2_val else team2
        
        margin = abs(team1_val - team2_val)
        
        comparison.append({
            'Metric': name,
            team1: team1_val,
            team2: team2_val,
            'Advantage': advantage,
            'Margin': margin
        })
    
    comparison_df = pd.DataFrame(comparison)
    
    # Count advantages
    team1_advantages = (comparison_df['Advantage'] == team1).sum()
    team2_advantages = (comparison_df['Advantage'] == team2).sum()
    
    return comparison_df, team1_advantages, team2_advantages

def main():
    """Main execution function"""
    print("NBA Matchup Prediction Generator")
    print("================================")
    print("This script generates predictions for all possible team matchups")
    print("and saves them to a CSV file.")
    
    # Check if the required CSV files exist
    historical_path = 'nba_games_stats.csv'
    current_path = 'converted_report.csv'
    
    if not os.path.exists(historical_path):
        print(f"Error: {historical_path} not found in the current directory.")
        return 1
        
    if not os.path.exists(current_path):
        print(f"Error: {current_path} not found in the current directory.")
        return 1
    
    print("\nLoading and processing data...")
    # Load and process data
    historical_data = load_data(historical_path)
    current_data = load_data(current_path)
    
    if historical_data is None or current_data is None:
        print("Error: Failed to load one or both data files.")
        return 1
    
    # Get the latest date from current_data
    current_data['date_temp'] = pd.to_datetime(current_data['date'])
    latest_date = current_data['date_temp'].max().strftime('%Y-%m-%d')
    print(f"Using latest date from data: {latest_date}")
    current_data = current_data.drop('date_temp', axis=1)
    
    # Harmonize column order 
    historical_data, current_data = harmonize_column_order(historical_data, current_data)
    
    # Combine datasets
    combined_data = pd.concat([historical_data, current_data])
    
    # Process data
    print("Preprocessing data...")
    team_data = preprocess_data(combined_data)
    
    print("Adding target variables...")
    team_data = add_target(team_data)
    
    print("Calculating rolling statistics...")
    team_data = add_rolling_stats(team_data, window=10)
    
    print("Preparing matchup features...")
    matchup_data = prepare_matchup_features(team_data)
    
    # Get all team names for matchups
    team_list = sorted(team_data['team'].unique())
    print(f"Found {len(team_list)} teams.")
    
    # Prepare to store all matchup results
    all_matchups = []
    current_date = latest_date  # Use the latest date from the data instead of datetime.now()
    
    print("\nGenerating all possible matchups...")
    # Generate all possible team combinations
    total_combinations = len(list(itertools.combinations(team_list, 2)))
    print(f"Total matchups to generate: {total_combinations}")
    
    for idx, (team1, team2) in enumerate(itertools.combinations(team_list, 2)):
        if (idx + 1) % 10 == 0 or (idx + 1) == total_combinations:
            print(f"Processing matchup {idx + 1}/{total_combinations}: {team1} vs {team2}")
        
        # Create prediction for this matchup
        prediction_matchups = create_matchup_prediction(matchup_data, teams=[team1, team2], prediction_date=current_date)
        predictions, model, importance = build_model_and_predict(matchup_data, prediction_matchups)
        
        # Skip if prediction couldn't be generated
        if predictions.empty:
            print(f"  Warning: Could not generate prediction for {team1} vs {team2}")
            continue
        
        # Get team stats
        team1_stats = team_data[team_data['team'] == team1].tail(10)
        team2_stats = team_data[team_data['team'] == team2].tail(10)
        
        # Skip if stats are missing
        if team1_stats.empty or team2_stats.empty:
            print(f"  Warning: Missing recent stats for {team1} or {team2}")
            continue
        
        # Get prediction results
        team1_home = predictions[predictions['team'] == team1]
        team2_home = predictions[predictions['team'] == team2]
        
        if len(team1_home) == 0 or len(team2_home) == 0:
            print(f"  Warning: Missing prediction results for {team1} or {team2}")
            continue
        
        team1_home_pred = team1_home['prediction'].values[0]
        team2_home_pred = team2_home['prediction'].values[0]
        
        # Analyze team advantages
        comparison_df, team1_adv, team2_adv = analyze_team_stats(team_data, team1, team2)
        
        if comparison_df is None:
            print(f"  Warning: Could not analyze advantages for {team1} vs {team2}")
            continue
        
        # Create a summary of this matchup
        matchup_summary = {
            'team1': team1,
            'team2': team2,
            'team1_home_prediction': int(team1_home_pred),
            'team2_home_prediction': int(team2_home_pred),
            'team1_advantages': int(team1_adv),
            'team2_advantages': int(team2_adv),
            
            # Team 1 stats
            'team1_wins': int(team1_stats['won'].sum()),
            'team1_games': len(team1_stats),
            'team1_ortg': float(team1_stats['ortg'].mean()),
            'team1_drtg': float(team1_stats['opp_ortg'].mean()),
            'team1_efg': float(team1_stats['efg'].mean()),
            'team1_tov_pct': float(team1_stats['tov_pct'].mean()),
            'team1_orb_pct': float(team1_stats['orb_pct'].mean()),
            'team1_ft_rate': float(team1_stats['ft_rate'].mean()),
            
            # Team 2 stats
            'team2_wins': int(team2_stats['won'].sum()),
            'team2_games': len(team2_stats),
            'team2_ortg': float(team2_stats['ortg'].mean()),
            'team2_drtg': float(team2_stats['opp_ortg'].mean()),
            'team2_efg': float(team2_stats['efg'].mean()),
            'team2_tov_pct': float(team2_stats['tov_pct'].mean()),
            'team2_orb_pct': float(team2_stats['orb_pct'].mean()),
            'team2_ft_rate': float(team2_stats['ft_rate'].mean()),
            
            # Prediction date
            'prediction_date': current_date
        }
        
        all_matchups.append(matchup_summary)
    
    # Check if any matchups were generated
    if not all_matchups:
        print("Error: No matchup predictions could be generated.")
        return 1
    
    # Save all matchups to CSV
    matchups_df = pd.DataFrame(all_matchups)
    output_file = 'all_matchup_predictions.csv'
    matchups_df.to_csv(output_file, index=False)
    print(f"\nSuccessfully saved {len(all_matchups)} matchup predictions to {output_file}")
    print(f"Prediction date: {current_date}")
    
    # Summary of matchup outcomes
    home_wins = sum(1 for m in all_matchups if m['team1_home_prediction'] == 1)
    home_win_pct = home_wins / len(all_matchups) if all_matchups else 0
    print(f"\nMatchup Statistics:")
    print(f"  Home teams win: {home_wins}/{len(all_matchups)} ({home_win_pct:.1%})")
    
    # Team with most advantages
    team_advantage_counts = {}
    for m in all_matchups:
        team1, team2 = m['team1'], m['team2']
        adv1, adv2 = m['team1_advantages'], m['team2_advantages']
        
        team_advantage_counts[team1] = team_advantage_counts.get(team1, 0) + (1 if adv1 > adv2 else 0)
        team_advantage_counts[team2] = team_advantage_counts.get(team2, 0) + (1 if adv2 > adv1 else 0)
    
    if team_advantage_counts:
        best_team = max(team_advantage_counts.items(), key=lambda x: x[1])
        print(f"  Team with most statistical advantages: {best_team[0]} ({best_team[1]} matchups)")
    
    print("\nDone! You can now use the generated CSV file with an application to serve the NBA prediction tool.")
    print("Output file: all_matchup_predictions.csv")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
