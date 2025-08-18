# Fantasy Nerdball - FPL Squad Optimisation Tool

A comprehensive Fantasy Premier League (FPL) squad optimisation tool that uses mathematical modelling to select optimal squads based on player form, historical performance, fixture difficulty, and reliability metrics.

## Features

- **Intelligent Player Scoring**: Combines current form, historical performance, and fixture difficulty
- **Transfer Penalty Mode**: Optional system to consider transfers beyond free limit with 4-point penalties
- **Transfer Value Analysis**: Evaluates whether transfers provide sufficient projected points improvement
- **Theoretical Best Squad**: Shows optimal squad ignoring transfer constraints for comparison
- **Captain Multiplier**: Automatically applies captain (2x) and vice-captain logic to projections
- **Form-Based Starting XI**: Only selects players with form > 0 for starting lineup
- **Reliability Metrics**: Accounts for player rotation and injury risks
- **Forced Selections**: Ability to force specific players into your squad
- **Starting XI Focus**: Results tracking focuses on players who actually contributed points
- **Substitute vs Transfer Analysis**: Recommends whether to use substitutes or make transfers
- **Availability Control**: Option to include/exclude unavailable players from optimization
- **Player History Tracking**: Updates and tracks player performance across gameweeks
- **Enhanced Transfer Display**: Shows next 3 fixtures for players being transferred

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
1. Prompt for player history updates from previous gameweek
2. Create results analysis for previous gameweek (Starting XI focus)
3. Fetch current player data from the FPL API
4. Show theoretical best squad for comparison
5. Analyse historical performance data
6. Calculate fixture difficulty for upcoming gameweeks
7. Score all players based on your configured weights
8. Evaluate unavailable players and substitution options
9. Test multiple transfer scenarios (including penalty analysis)
10. Optimise squad selection using linear programming
11. Optimize Starting XI with form constraints and captain selection
12. Display comprehensive squad comparison vs theoretical optimum
13. Save all results to CSV files

## New Transfer Penalty System

When `ACCEPT_TRANSFER_PENALTY = True`, the system:
- Tests all transfer scenarios (0 to free_transfers + 3)
- Applies 4-point penalty for each transfer beyond free limit
- Evaluates net projected points (gross points - penalties)
- Respects `MIN_TRANSFER_VALUE` threshold before recommending transfers
- Shows detailed analysis with player names and next fixtures

Example output:
```
🔄 Evaluating all transfer scenarios up to 4 transfers...
  🏠 Scenario 0: 0 transfers, 0 extra, penalty: -0, gross: 60.5, net: 60.5
  ✅ Scenario 1: 1 transfers, 0 extra, penalty: -0, gross: 61.4, net: 61.4
     (🔄 OUT: Bowen → IN: Wood)
  💰 Scenario 2: 2 transfers, 1 extra, penalty: -4, gross: 62.5, net: 58.5
     (🔄 OUT: Bowen, Baleba → IN: Wood, Semenyo)
```

## Output Files

Results are saved in the `squads/gw{n}/` directory:
- `full_squad.csv`: Complete squad with all metrics
- `full_squad_simple.csv`: Simplified squad overview
- `starting_xi_results.csv`: Starting XI projected vs actual points comparison
- `summary.csv`: Starting XI performance summary and accuracy metrics

## Configuration (`config.py`)

### Basic Settings
```python
GAMEWEEK = 2                    # Current gameweek you're planning for
BUDGET = 100.0                 # Total budget in millions
FREE_TRANSFERS = 1             # Number of free transfers available
ACCEPT_TRANSFER_PENALTY = True # Allow transfers beyond free limit (4 pts penalty each)
WILDCARD = False               # Set to True if using wildcard
```

### Availability Settings
```python
EXCLUDE_UNAVAILABLE = True     # Set to False to include unavailable players in optimization
```
- **True**: Only considers available players (realistic squads)
- **False**: Includes injured/suspended players (theoretical analysis)

### Transfer Settings
```python
MIN_TRANSFER_VALUE = 5.0       # Minimum projected points improvement needed to make transfers
```
- **Simple threshold**: Any transfer scenario must improve projected points by at least this amount
- **Example**: With 5.0, transfers must improve your Starting XI by 5+ points to be recommended

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

### Historical Data Settings
```python
PAST_SEASONS = ["2024-25", "2023-24"]     # Historic seasons to consider
HISTORIC_SEASON_WEIGHTS = [0.7, 0.3]      # Relative weights (should match seasons)
FIRST_N_GAMEWEEKS = 5                     # How many upcoming fixtures to consider
```

### Scoring Weights (Must Total 1.0)
```python
FORM_WEIGHT = 0.3          # Importance of current season form
HISTORIC_WEIGHT = 0.4      # Importance of historical performance
DIFFICULTY_WEIGHT = 0.3    # Importance of upcoming fixture difficulty
```

### Squad Composition
```python
SQUAD_SIZE = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}  # Required squad composition
MAX_PER_TEAM = 3                                        # Maximum players from one team
```

### Team Adjustments
```python
PROMOTED_TEAMS = ["Burnley", "Sunderland", "Leeds"]    # Newly promoted teams (get penalty)

TEAM_MODIFIERS = {
    "Man City": 1.05,      # Boost players from overperforming teams
    "West Ham": 0.95,      # Reduce players from underperforming teams
    # ... etc
}
```

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

## Algorithm Overview

### Phase 1: Data Collection & Analysis
1. **Player History Update**: Optional update of previous gameweek performance
2. **Results Analysis**: Creates Starting XI focused performance analysis
3. **Data Fetching**: Current player data and historical performance from FPL API
4. **Theoretical Best**: Shows optimal squad ignoring all constraints

### Phase 2: Player Scoring
1. **Scoring**: Calculates composite scores using form, history, and fixtures
2. **Availability Filter**: Optionally excludes unavailable players (configurable)
3. **Reliability Adjustments**: Accounts for rotation risks and playing time

### Phase 3: Transfer Analysis
1. **Substitute Analysis**: Evaluates bench coverage for unavailable players
2. **Multi-Scenario Testing**: Tests 0 to free_transfers+3 scenarios
3. **Penalty Calculation**: Applies 4-point penalties for extra transfers
4. **Value Threshold**: Checks improvements against MIN_TRANSFER_VALUE
5. **Optimal Selection**: Chooses best scenario respecting all constraints

### Phase 4: Starting XI Optimization
1. **Squad-to-Starting XI**: Optimizes final Starting XI from selected 15 players
2. **Form Constraint**: Only players with form > 0 can start
3. **Captain Selection**: Auto-selects captain (highest projected points) with 2x multiplier
4. **Position Constraints**: Ensures valid formation (1 GK, 3+ DEF, 3+ MID, 1+ FWD)

### Phase 5: Results & Comparison
1. **Squad Display**: Shows full squad, Starting XI, and bench with all metrics
2. **Captain Multiplier**: Displays captain points as "6.9 (x2)" format
3. **Theoretical Comparison**: Compares your squad vs theoretical optimum
4. **Gap Analysis**: Shows points gap and improvement suggestions

## Key Metrics Explained

### Reliability
- **Current Season**: `starts / gameweeks_completed`
- **Historical**: Weighted average across past seasons
- **Impact**: Regular starters get boost, rotation risks get penalty

### Projected Points (with Captain)
- **Base Formula**: `baseline_points + (base_quality * multiplier)`
- **Captain Bonus**: Highest projected player gets 2x multiplier
- **Display**: Shows as "6.9 (x2)" for captain, includes doubled points in totals

### Form Constraint
- **Starting XI Rule**: Only players with form > 0 can be selected to start
- **Bench Exception**: Players with form ≤ 0 can be on bench
- **Fallback**: If optimization fails, uses simple projected points ranking

## Transfer Strategy Modes

### Standard Mode (`ACCEPT_TRANSFER_PENALTY = False`)
- Strict adherence to free transfer limits
- Traditional transfer value analysis
- Conservative approach focused on transfer efficiency

### Penalty Mode (`ACCEPT_TRANSFER_PENALTY = True`)
- Tests unlimited transfer scenarios
- 4-point penalty per extra transfer
- Optimal mathematical approach
- Still respects MIN_TRANSFER_VALUE threshold

## Starting XI vs Full Squad Analysis

The tool now distinguishes between:
- **Squad Selection**: Choose best 15 players (transfer decisions)
- **Starting XI**: Choose best 11 from your 15 (weekly decisions)
- **Captain Choice**: Auto-select highest projected points (2x multiplier)

This mirrors real FPL decision-making where you:
1. Build squad with transfers
2. Pick Starting XI each week
3. Choose captain for double points

## Results Analysis Features

### Starting XI Focus
- **Performance tracking**: Only analyzes players who actually played
- **Captain-adjusted scoring**: Accounts for doubled captain points
- **Meaningful metrics**: Focuses on decisions that affected your score
- **Accuracy measurement**: How well projections matched actual Starting XI performance

### Historical Tracking
- **Player performance**: Track individual player results across gameweeks
- **Projection accuracy**: Measure how well the model predicts outcomes
- **Decision analysis**: Evaluate transfer and captain decisions

## Troubleshooting

- **No fixtures showing**: Check that GAMEWEEK is set correctly in config.py
- **Poor reliability scores**: Early in season, players need time to establish patterns
- **Unexpected transfers**: Adjust MIN_TRANSFER_VALUE or set ACCEPT_TRANSFER_PENALTY = False
- **Missing players**: Check BLACKLIST_PLAYERS and EXCLUDE_UNAVAILABLE settings
- **Form constraint issues**: Some players with form ≤ 0 cannot start (by design)
- **Captain not doubling**: Check that proj_pts_display column shows "(x2)" format

## Advanced Usage

### Research Mode
Set `EXCLUDE_UNAVAILABLE = False` to:
- Include all players regardless of injury status
- See theoretical maximum squad potential
- Plan for when injured players return
- Understand cost of player unavailability

### Conservative Mode
Set `ACCEPT_TRANSFER_PENALTY = False` and increase `MIN_TRANSFER_VALUE` to:
- Only make transfers when high confidence of improvement
- Preserve transfer flexibility for future gameweeks
- Reduce risk of unnecessary changes

### Aggressive Mode
Set `ACCEPT_TRANSFER_PENALTY = True` and lower `MIN_TRANSFER_VALUE` to:
- Explore all mathematically optimal solutions
- Accept penalties when improvement justifies cost
- Maximize theoretical points regardless of transfer usage

## Support

For issues or questions, review the configuration options in `config.py` first. The tool is highly customisable to match your FPL strategy and risk tolerance. The new penalty system allows both conservative and aggressive approaches to transfer strategy.