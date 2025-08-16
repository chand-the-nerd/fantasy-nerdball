# Fantasy Nerdball - FPL Squad Optimisation Tool

A comprehensive Fantasy Premier League (FPL) squad optimisation tool that uses mathematical modelling to select optimal squads based on player form, historical performance, fixture difficulty, and reliability metrics.

## Features

- **Intelligent Player Scoring**: Combines current form, historical performance, and fixture difficulty
- **Transfer Value Analysis**: Evaluates whether transfers provide sufficient value
- **Reliability Metrics**: Accounts for player rotation and injury risks
- **Forced Selections**: Ability to force specific players into your squad
- **Results Tracking**: Compares projected vs actual points from previous gameweeks
- **Substitute vs Transfer Analysis**: Recommends whether to use substitutes or make transfers
- **Unavailable Player Handling**: Automatically sets injured/suspended players to 0 projected points

## Installation

1. Clone or download this repository
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Configure your settings** in `config.py` (see Configuration section below)
2. **Run the optimisation**:
   ```bash
   python main.py
   ```

The tool will:
1. Fetch current player data from the FPL API
2. Analyse historical performance data
3. Calculate fixture difficulty for upcoming gameweeks
4. Score all players based on your configured weights
5. Optimise squad selection using linear programming
6. Evaluate transfer strategies
7. Display optimal starting XI and bench
8. Save results to CSV files

## Output Files

Results are saved in the `squads/gw{n}/` directory:
- `full_squad.csv`: Complete squad with all metrics
- `full_squad_simple.csv`: Simplified squad overview
- `full_squad_results.csv`: Comparison of projected vs actual points (previous GW)
- `summary.csv`: Performance summary and accuracy metrics

## Configuration (`config.py`)

### Basic Settings
```python
GAMEWEEK = 2                # Current gameweek you're planning for
BUDGET = 100.0             # Total budget in millions
FREE_TRANSFERS = 1         # Number of free transfers available
WILDCARD = False           # Set to True if using wildcard
```

### Transfer Efficiency Settings
```python
MIN_TRANSFER_VALUE = 0.3          # Minimum FPL score improvement per transfer required
CONSERVATIVE_MODE = False         # If True, be more cautious about making transfers
TRANSFER_ROLLOVER_VALUE = 0.3     # Value of keeping a transfer for next week
```
- **MIN_TRANSFER_VALUE**: How much improvement needed to justify a transfer
- **CONSERVATIVE_MODE**: Makes the system more reluctant to use transfers
- **TRANSFER_ROLLOVER_VALUE**: Assigns value to saving transfers for future gameweeks

### Points Projection Settings
```python
BASELINE_POINTS_PER_GAME = {
    "GK": 4.0,   # Average points for a decent goalkeeper
    "DEF": 4.5,  # Average points for a decent defender
    "MID": 5.0,  # Average points for a decent midfielder
    "FWD": 5.5,  # Average points for a decent forward
}
FPL_SCORE_TO_POINTS_MULTIPLIER = 1.5  # How much 1 FPL score unit translates to points
```
- **BASELINE_POINTS_PER_GAME**: Expected baseline points for average players by position
- **FPL_SCORE_TO_POINTS_MULTIPLIER**: Converts internal scoring to projected points

### Historical Data Settings
```python
PAST_SEASONS = ["2024-25", "2023-24"]     # Historic seasons to consider
HISTORIC_SEASON_WEIGHTS = [0.7, 0.3]      # Relative weights (should match seasons)
FIRST_N_GAMEWEEKS = 5                     # How many upcoming fixtures to consider
```
- **PAST_SEASONS**: Which previous seasons to include in analysis
- **HISTORIC_SEASON_WEIGHTS**: How much weight to give each season (recent = higher)
- **FIRST_N_GAMEWEEKS**: Number of upcoming fixtures to include in difficulty calculation

### Scoring Weights (Must Total 1.0)
```python
FORM_WEIGHT = 0.3          # Importance of current season form
HISTORIC_WEIGHT = 0.4      # Importance of historical performance
DIFFICULTY_WEIGHT = 0.3    # Importance of upcoming fixture difficulty
```
- **FORM_WEIGHT**: How much current season form matters (recent games)
- **HISTORIC_WEIGHT**: How much historical performance matters (past seasons)
- **DIFFICULTY_WEIGHT**: How much upcoming fixture difficulty matters

### Squad Composition
```python
SQUAD_SIZE = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}  # Required squad composition
MAX_PER_TEAM = 3                                        # Maximum players from one team
```

### Team Adjustments
```python
PROMOTED_TEAMS = ["Burnley", "Sunderland", "Leeds"]    # Newly promoted teams (get penalty)

TEAM_MODIFIERS = {
    "Man City": 1.05,      # Overperforming teams (boost players)
    "Man Utd": 1.1,        # Underperforming teams (boost players)
    "West Ham": 0.8,       # Overperforming teams (reduce players)
    # ... etc
}
```
- **PROMOTED_TEAMS**: Newly promoted teams get a small penalty (-0.3) for adjustment period
- **TEAM_MODIFIERS**: Adjust individual player scores based on team over/under-performance

### Player Selections
```python
FORCED_SELECTIONS = {
    "GK": ["dúbravka"],    # Force specific players (use lowercase names)
    "DEF": [], 
    "MID": ["baleba","m.salah"], 
    "FWD": []
}

BLACKLIST_PLAYERS = ["isak"]  # Players to exclude from consideration
```
- **FORCED_SELECTIONS**: Players you must include in your squad
- **BLACKLIST_PLAYERS**: Players to completely exclude from selection

## Algorithm Overview

1. **Data Collection**: Fetches current player data and historical performance from FPL API
2. **Scoring**: Calculates composite scores using:
   - Current form (weighted by FORM_WEIGHT)
   - Historical performance (weighted by HISTORIC_WEIGHT) 
   - Fixture difficulty (weighted by DIFFICULTY_WEIGHT)
   - Reliability adjustments (starts per gameweek)
3. **Unavailable Player Handling**: Sets FPL scores and projected points to 0 for injured/suspended players
4. **Optimisation**: Uses integer linear programming to select optimal squad within constraints
5. **Transfer Analysis**: Evaluates if proposed transfers provide sufficient value vs keeping current squad
6. **Results**: Outputs optimal starting XI, bench (ordered by projected points), and detailed analysis

## Key Metrics Explained

### Reliability
- **Current Season**: `starts / gameweeks_completed`
- **Historical**: Weighted average of `(games_played / 30)` across past seasons
- **Impact**: Regular starters get +1.0 FPL score boost, rotation risks get -0.7 penalty

### Minutes Per Game (minspg)
- **Calculation**: `total_minutes / gameweeks_completed`
- **Purpose**: Shows average involvement per gameweek

### Projected Points
- **Formula**: `baseline_points + (base_quality * multiplier)`
- **Baseline**: Position-specific expected points for average players
- **Base Quality**: Composite score from form, history, and fixtures

## Transfer Strategy

The system evaluates transfers by comparing:
- **Current squad score** (no transfers, including unavailable players at 0 points)
- **Proposed squad score** (with transfers)
- **Transfer cost** (opportunity cost of using free transfers)

Transfers are recommended when:
- Value per transfer > MIN_TRANSFER_VALUE
- Net benefit after rollover cost > 0
- Or when forced (unavailable players, transfer cap reached)

## Notes

- Requires internet connection to fetch FPL API data
- Historical data sourced from vaastav's Fantasy-Premier-League repository
- Uses PuLP library for linear programming optimisation
- Designed for experienced FPL managers who want data-driven decisions
- All calculations use British English terminology

## Troubleshooting

- **No fixtures showing**: Check that GAMEWEEK is set correctly in config.py
- **Poor reliability scores**: Early in season, players need time to establish reliability
- **Unexpected transfers**: Adjust MIN_TRANSFER_VALUE or enable CONSERVATIVE_MODE
- **Missing players**: Check BLACKLIST_PLAYERS doesn't include wanted players

## Support

For issues or questions, review the configuration options in `config.py` first. The tool is highly customisable to match your FPL strategy and preferences.