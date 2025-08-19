# Fantasy Nerdball - FPL Squad Optimisation Tool

Fantasy Nerdball is an advanced Fantasy Premier League (FPL) squad optimisation tool that uses data science and mathematical optimisation to help you select the best possible team each gameweek.

## How It Works

### Player Selection Criteria

The tool evaluates every player using a sophisticated scoring system that combines multiple data sources with **user-determined custom weighting**:

**Core Components:**
- **Current Form**: How well the player is performing this season - their average points per game
- **Historical Performance**: How consistently they've performed over the past 2-3 seasons  
- **Fixture Difficulty**: How easy or hard their upcoming fixtures are

*The relative importance of these three components is fully customizable in your config file (default: 40% form, 40% historical, 20% fixtures).*

**Smart Adjustments:**
- **Performance Analysis**: Uses expected output data to identify players who are likely to improve or decline based on whether they're over/under-performing their underlying statistics. This considers xGI (Expected Goal Involvements) for attacking output and xGC (Expected Goals Conceded) for defensive performance, weighted by position
- **Reliability Bonus**: Rewards players who start games consistently (reduces rotation risk)
- **Team Modifiers**: User-determined adjustments for teams that are over/under-performing expectations - helps provide additional context that the model is unlikely to know from data alone
- **Promotion Penalty**: Accounts for newly promoted teams typically struggling

### Squad Selection Process

The tool uses **Integer Linear Programming** (a mathematical optimisation technique) to find the perfect 15-player squad that:

1. **Maximises total FPL score** while respecting all FPL rules
2. **Stays within budget** (£100m or your current squad value)
3. **Respects position limits** (2 GK, 5 DEF, 5 MID, 3 FWD)
4. **Limits players per team** (maximum 3 from any single team)
5. **Considers transfer constraints** (how many free transfers you have)

The **FPL score** used for squad selection includes reliability adjustments to minimize rotation risk.

### Starting XI Selection

Once the 15-player squad is chosen, the tool optimises the starting XI by:

1. **Selecting exactly 11 players** who can start this gameweek
2. **Meeting formation requirements** (1 GK, 3-5 DEF, 3-5 MID, 1-3 FWD)
3. **Excluding players with zero form** (haven't performed recently)
4. **Maximising projected points** for this specific gameweek
5. **Auto-selecting captain and vice-captain** (highest projected scorers)

The **projected points** used for starting XI selection focuses purely on expected output when the player actually plays.

### Example Player Calculation

Let's see how **Bruno Fernandes (MID)** might be evaluated:

**Step 1: Core Components**
- Current Form: 6.2 points per game → Z-score: +1.5
- Historical Performance: 5.8 points per game → Z-score: +1.2  
- Fixture Difficulty: 3.2 average difficulty → Z-score: +0.8

**Step 2: Weighted Core Score** (40% + 40% + 20%)
- Core Score = (1.5 × 0.4) + (1.2 × 0.4) + (0.8 × 0.2) = 1.24

**Step 3: Smart Adjustments**
- Performance Analysis: 1.15 (slightly outperforming expected output)
- Reliability: +0.3 (starts 90% of games)
- Team Modifier: 1.0 (Man United performing as expected)

**Step 4: Final Scores**
- **FPL Score** (for squad selection): 1.24 + 0.3 = 1.54 × 1.15 = 1.77
- **Projected Points** (for starting XI): 4.5 baseline + 1.24 = 5.74 × 1.15 = 6.6 points

### Transfer Strategy

The tool has three modes for handling transfers:

**Standard Mode**: 
- Compares your current squad (with no transfers) vs. the optimal squad (with transfers)
- Only recommends transfers if the improvement exceeds a minimum threshold (configurable)
- Helps you avoid "sideways moves" that waste transfers

**Transfer Penalty Mode**:
- Tests multiple transfer scenarios (e.g., 1 transfer, 2 transfers, 3 transfers)
- Factors in the -4 point penalty for each extra transfer
- Recommends the strategy with the highest net points after penalties

**Wildcard Mode**:
- No transfer limits - optimises freely across all players

### The Nerdball XI

This is the **best team that fits the model's parameters** for the gameweek - what the perfect squad would look like if you had unlimited transfers and no budget constraints. It serves as a benchmark to see how close your actual team gets to the mathematical optimum based on the tool's scoring system.

## Understanding the Output Tables

### Main Columns Explained

| Column | What It Means |
|--------|---------------|
| **name** | Player's name (with (C) for captain, (V) for vice-captain) |
| **pos** | Position (GK/DEF/MID/FWD) |
| **team** | Player's club |
| **cost** | Current price in millions |
| **form** | Current season average points per game |
| **his_ppg** | Historical average points per game (weighted across past seasons) |
| **fix_diff** | Average fixture difficulty over next N gameweeks (default: 5 games, lower = easier) |
| **start_pct** | Reliability - percentage of games they've started |
| **hist_xOP** | Historical expected output performance ratio (>1.0 = tends to overperform expected output) |
| **cur_xOP** | Current season expected output performance ratio |
| **xMod** | Final expected output consistency modifier applied to projections |
| **minspg** | Minutes per game this season |
| **proj_pts** | Projected points for this gameweek |
| **next_fix** | Next opponent |

### Expected Output Performance Columns Explained

- **hist_xOP & cur_xOP**: Compare actual goals/assists vs. expected goals/assists. Values >1.0 mean they're scoring more than expected, <1.0 means less than expected
- **xMod**: The final modifier applied - accounts for whether current performance is sustainable or due for regression

## Getting Started

### 1. Copy the Config Template

Create a new file called `config.py` in the main directory and copy this template:

```python
"""
Configuration settings for Fantasy Nerdball FPL optimisation tool.
Edit this file to customise your optimisation preferences.
"""

class Config:
    """Configuration class containing all settings for the FPL optimisation."""
    
    # === BASIC SETTINGS ===
    GAMEWEEK = 1  # Current gameweek number
    BUDGET = 100.0  # Will be overridden by value of squad from prev gameweek
    FREE_TRANSFERS = 1  # How many free transfers you have
    ACCEPT_TRANSFER_PENALTY = False  # Set to True to consider extra transfers
    EXCLUDE_UNAVAILABLE = True  # Set to False to include injured players
    WILDCARD = False  # Set to True if playing wildcard

    # === TRANSFER EFFICIENCY SETTINGS ===
    MIN_TRANSFER_VALUE = 2.0  # Minimum point improvement needed per transfer

    # === SCORING WEIGHTS ===
    # These should total 1.0
    FORM_WEIGHT = 0.4      # Importance of current season average
    HISTORIC_WEIGHT = 0.4   # Importance of historic seasons' average
    DIFFICULTY_WEIGHT = 0.2 # Importance of upcoming fixture difficulty

    # === SQUAD COMPOSITION ===
    SQUAD_SIZE = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    MAX_PER_TEAM = 3

    # === TEAM ADJUSTMENTS ===
    # Team performance modifiers (adjust for over/under-performing teams)
    # Teams that have overperformed should be under 1.0 and vice versa
    TEAM_MODIFIERS = {
        "Arsenal": 1.0,
        "Man City": 1.0,
        # ... add all teams with your assessments
    }

    # === PLAYER SELECTIONS ===
    # Force specific players to be selected (use exact names as they appear in FPL)
    FORCED_SELECTIONS = {
        "GK": [], 
        "DEF": [], 
        "MID": [], 
        "FWD": []
    }

    # Players to exclude (use exact names as they appear in FPL)
    BLACKLIST_PLAYERS = []
```

### 2. Configure Your Settings

**Essential Settings to Update:**

- **GAMEWEEK**: Set this to the current gameweek number
- **FREE_TRANSFERS**: How many free transfers you currently have
- **ACCEPT_TRANSFER_PENALTY**: 
  - `False` = Only use free transfers
  - `True` = Consider extra transfers with -4 point penalties
- **WILDCARD**: Set to `True` if you're playing your wildcard this week

**Scoring Weights (Must Total 1.0):**
- **FORM_WEIGHT**: How much to weight current season performance (default: 0.4)
- **HISTORIC_WEIGHT**: How much to weight past seasons (default: 0.4)  
- **DIFFICULTY_WEIGHT**: How much to weight fixture difficulty (default: 0.2)

**Optional Settings:**

- **MIN_TRANSFER_VALUE**: How many points improvement you need per transfer (default: 2.0 is reasonable)
- **TEAM_MODIFIERS**: Adjust teams up/down based on your assessment (1.0 = neutral, >1.0 = boost, <1.0 = penalty)
- **FORCED_SELECTIONS**: Add player names if you want to force certain players into your squad
- **BLACKLIST_PLAYERS**: Add player names you never want selected

**Example Configuration:**
```python
GAMEWEEK = 15
FREE_TRANSFERS = 2
ACCEPT_TRANSFER_PENALTY = True  # Will consider making 3+ transfers if worth it

# Custom weights - more emphasis on current form
FORM_WEIGHT = 0.5
HISTORIC_WEIGHT = 0.3  
DIFFICULTY_WEIGHT = 0.2

# Team adjustments based on your analysis
TEAM_MODIFIERS = {
    "Arsenal": 0.95,  # Slightly overperforming
    "Brighton": 1.1,  # Underperforming their quality
    # ... etc
}

FORCED_SELECTIONS = {
    "FWD": ["Haaland"]  # Always include Haaland
}
BLACKLIST_PLAYERS = ["Martial"]  # Never select Martial
```

### 3. Run the Tool

Simply run:
```bash
python main.py
```

The tool will automatically fetch the latest data and provide you with optimised squad recommendations!

## Tips for Best Results

1. **Update weekly**: Run after each gameweek for the most accurate projections
2. **Customize the weights**: Adjust FORM_WEIGHT, HISTORIC_WEIGHT, and DIFFICULTY_WEIGHT based on your FPL philosophy
3. **Set team modifiers thoughtfully**: Use your football knowledge to adjust for teams the data might not capture
4. **Consider your risk tolerance**: Conservative managers might prefer `ACCEPT_TRANSFER_PENALTY = False`
5. **Use forced selections sparingly**: Let the algorithm do its work, but force key players if needed
6. **Check the transfer value analysis**: Don't make transfers unless they're clearly worthwhile
7. **Compare to Nerdball XI**: See how close you can get to the theoretical optimum

The tool combines the best of data science, mathematical optimisation, and FPL strategy to give you a competitive edge!