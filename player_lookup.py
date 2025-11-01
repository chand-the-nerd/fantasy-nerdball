#!/usr/bin/env python3
"""
Player Lookup Tool - Quick analysis of individual players
Displays detailed scoring breakdown and predictions without running full
optimisation.
"""

import sys
import pandas as pd
from config import Config
from src.api.fpl_client import FPLClient
from src.data.player_processor import PlayerProcessor
from src.data.fixture_manager import FixtureManager
from src.data.historical_data import HistoricalDataManager
from src.data.player_history_tracker import PlayerHistoryTracker
from src.analysis.scoring_engine import ScoringEngine
from src.analysis.points_calculator import PointsCalculator

def safe_float(value, default=0.0):
    """Safely convert a value to float."""
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value, default=0):
    """Safely convert a value to int."""
    if pd.isna(value):
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def normalise_name(name):
    """Normalise name for matching."""
    return name.strip().lower()


def find_player(players_df, search_term):
    """
    Find player by name (fuzzy matching).
    
    Args:
        players_df: DataFrame with all players
        search_term: Name to search for
        
    Returns:
        DataFrame row(s) matching the search
    """
    search_lower = normalise_name(search_term)
    
    # Try exact match first
    exact_matches = players_df[
        players_df['display_name'].str.lower() == search_lower
    ]
    if not exact_matches.empty:
        return exact_matches
    
    # Try partial match
    partial_matches = players_df[
        players_df['display_name'].str.lower().str.contains(
            search_lower, na=False
        )
    ]
    if not partial_matches.empty:
        return partial_matches
    
    # Try last name only
    last_name_matches = players_df[
        players_df['display_name'].str.lower().str.split().str[-1] 
        == search_lower
    ]
    if not last_name_matches.empty:
        return last_name_matches
    
    return pd.DataFrame()


def display_player_header(player):
    """Display player basic info header."""
    print("\n" + "=" * 79)
    print(f"PLAYER ANALYSIS: {player['display_name']}")
    print("=" * 79)
    print(f"Position: {player['position']}")
    print(f"Team: {player['team']}")
    print(f"Cost: £{player['now_cost_m']:.1f}m")
    
    # Handle ownership as string or float
    ownership = player.get('selected_by_percent', 0)
    try:
        ownership = float(ownership)
    except (ValueError, TypeError):
        ownership = 0.0
    print(f"Ownership: {ownership:.1f}%")
    
    print(f"Status: {player.get('status', 'a')}")
    
    chance_playing = player.get('chance_of_playing_next_round')
    if chance_playing is not None:
        try:
            chance_playing = int(chance_playing)
            print(f"Chance of playing: {chance_playing}%")
        except (ValueError, TypeError):
            pass
    print("=" * 79)


def display_current_season_stats(player, config):
    """Display current season statistics."""
    print("\n📊 CURRENT SEASON STATS")
    print("-" * 79)
    
    gameweeks_completed = max(1, config.GAMEWEEK - 1)
    
    # Safely convert numeric fields
    total_points = safe_int(player.get('total_points', 0))
    form = safe_float(player.get('form', 0))
    minutes = safe_int(player.get('minutes', 0))
    starts = safe_int(player.get('starts', 0))
    goals = safe_int(player.get('goals_scored', 0))
    assists = safe_int(player.get('assists', 0))
    clean_sheets = safe_int(player.get('clean_sheets', 0))
    bonus = safe_int(player.get('bonus', 0))
    
    mins_per_gw = minutes / gameweeks_completed
    starts_pct = (starts / gameweeks_completed * 100)
    
    print(f"Total Points: {total_points}")
    print(f"Form (avg last 5): {form:.1f}")
    print(f"Minutes: {minutes} ({mins_per_gw:.0f} per GW)")
    print(f"Starts: {starts}/{gameweeks_completed} ({starts_pct:.0f}%)")
    print(f"Goals: {goals}")
    print(f"Assists: {assists}")
    print(f"Clean Sheets: {clean_sheets}")
    print(f"Bonus Points: {bonus}")
    
    # xG stats if available
    expected_goals = safe_float(player.get('expected_goals', 0))
    expected_assists = safe_float(player.get('expected_assists', 0))
    expected_gi = safe_float(player.get('expected_goal_involvements', 0))
    expected_gc = safe_float(player.get('expected_goals_conceded', 0))
    
    has_xg = expected_goals > 0 or expected_assists > 0
    
    if has_xg:
        print(f"\nExpected Stats:")
        print(f"  xG: {expected_goals:.2f}")
        print(f"  xA: {expected_assists:.2f}")
        print(f"  xGI: {expected_gi:.2f}")
        if player['position'] in ['GK', 'DEF', 'MID']:
            print(f"  xGC: {expected_gc:.2f}")


def display_historical_stats(player):
    """Display historical performance."""
    print("\n📚 HISTORICAL PERFORMANCE")
    print("-" * 79)
    
    if player.get('avg_ppg_past2', 0) > 0:
        print(f"Historical PPG (weighted): {player['avg_ppg_past2']:.2f}")
        reliability_pct = player.get('avg_reliability', 0) * 100
        print(f"Historical Reliability: {reliability_pct:.0f}%")
        print(f"Historical xOP: {player.get('historical_xOP', 1.0):.2f}x")
    else:
        print("No historical data (new player or first season)")


def display_recent_form(player, config):
    """Display recent gameweek-by-gameweek performance."""
    print("\n📈 RECENT FORM (Last 8 Gameweeks)")
    print("-" * 79)
    
    tracker = PlayerHistoryTracker(config)
    
    try:
        player_history = tracker.get_player_history(
            player['web_name'], 
            player['team']
        )
        
        if not player_history.empty:
            recent = player_history.head(8)
            
            # Create display table
            print(f"{'GW':<4} {'Points':<8} {'Minutes':<8} {'Goals':<7} "
                  f"{'Assists':<8} {'CS':<4}")
            print("-" * 79)
            
            for _, gw in recent.iterrows():
                print(f"{gw['round']:<4} {gw['total_points']:<8} "
                      f"{gw.get('minutes', 0):<8} "
                      f"{gw.get('goals_scored', 0):<7} "
                      f"{gw.get('assists', 0):<8} "
                      f"{gw.get('clean_sheets', 0):<4}")
            
            # Calculate consistency metrics
            points = recent['total_points'].values
            if len(points) >= 3:
                mean_points = points.mean()
                std_points = points.std()
                cv = std_points / mean_points if mean_points > 0 else 0
                blank_freq = (points < 2).sum() / len(points)
                
                cv_label = '(volatile)' if cv > 1.0 else '(consistent)'
                
                print(f"\nConsistency Metrics:")
                print(f"  Average: {mean_points:.2f} points")
                print(f"  Std Dev: {std_points:.2f}")
                print(f"  CV: {cv:.2f} {cv_label}")
                print(f"  Blank Rate: {blank_freq * 100:.0f}% (< 2 points)")
                
                form_consistency = player.get('form_consistency', 1.0)
                print(f"  Form Consistency Modifier: "
                      f"{form_consistency:.2f}x")
        else:
            print("No recent gameweek history available")
            
    except Exception as e:
        print(f"Could not load gameweek history: {e}")


def display_upcoming_fixtures(player, config, fixture_manager):
    """Display upcoming fixture difficulty."""
    print(f"\n📅 UPCOMING FIXTURES (Next {config.FIRST_N_GAMEWEEKS} GWs)")
    print("-" * 79)
    
    # Get fixtures
    fixtures = fixture_manager.fpl_client.get_fixtures()
    fixtures_df = pd.DataFrame(fixtures)
    
    # Get team names
    teams_data = fixture_manager.fpl_client.get_bootstrap_static()
    teams_df = pd.DataFrame(teams_data["teams"])[["id", "name"]]
    teams_dict = dict(zip(teams_df["id"], teams_df["name"]))
    
    team_id = player['team_id']
    
    print(f"{'GW':<4} {'Opponent':<20} {'Venue':<6} {'Difficulty':<10}")
    print("-" * 79)
    
    for gw in range(config.GAMEWEEK, 
                   config.GAMEWEEK + config.FIRST_N_GAMEWEEKS):
        gw_fixtures = fixtures_df[fixtures_df["event"] == gw]
        
        # Find fixtures for this team
        team_fixtures = gw_fixtures[
            (gw_fixtures["team_h"] == team_id) | 
            (gw_fixtures["team_a"] == team_id)
        ]
        
        if len(team_fixtures) == 0:
            print(f"{gw:<4} {'BLANK GAMEWEEK':<20} {'-':<6} {'-':<10}")
        else:
            for _, fixture in team_fixtures.iterrows():
                if fixture["team_h"] == team_id:
                    opponent = teams_dict.get(
                        fixture["team_a"], "Unknown"
                    )
                    venue = "Home"
                    diff = fixture["team_h_difficulty"]
                else:
                    opponent = teams_dict.get(
                        fixture["team_h"], "Unknown"
                    )
                    venue = "Away"
                    diff = fixture["team_a_difficulty"]
                
                diff_label = f"{6 - diff:.1f}/5"
                print(f"{gw:<4} {opponent:<20} {venue:<6} {diff_label:<10}")
    
    fixture_diff = player.get('fixture_diff', 0)
    print(f"\nWeighted Fixture Difficulty: {fixture_diff:.2f}/5")


def display_scoring_breakdown(player, config):
    """Display detailed scoring breakdown."""
    print("\n🎯 SCORING BREAKDOWN")
    print("-" * 79)
    
    # Get position weights
    position = player['position']
    position_weights = config.POSITION_SCORING_WEIGHTS.get(
        position, 
        config.POSITION_SCORING_WEIGHTS.get("MID")
    )
    
    is_new_player = player.get('avg_ppg_past2', 0) <= 0
    
    if is_new_player:
        form_weight = 1.0 - position_weights["difficulty"]
        hist_weight = 0.0
        fix_weight = position_weights["difficulty"]
    else:
        form_weight = position_weights["form"]
        hist_weight = position_weights["historic"]
        fix_weight = position_weights["difficulty"]
    
    print(f"Position: {position}")
    print(f"Weighting: Form {form_weight:.1%}, History {hist_weight:.1%}, "
          f"Fixtures {fix_weight:.1%}")
    
    if is_new_player:
        print("(New player - no historical weighting)")
    
    print("\nComponent Scores:")
    print(f"  Form: {player.get('form', 0):.2f}")
    
    form_adj = player.get('form_adjusted', player.get('form', 0))
    print(f"  Form (adjusted): {form_adj:.2f}")
    print(f"  Historic PPG: {player.get('historic_ppg', 0):.2f}")
    
    fixture_diff = player.get('fixture_diff', 0)
    print(f"  Fixture Difficulty: {fixture_diff:.2f}/5")
    print(f"  Fixture Bonus: {player.get('fixture_bonus', 0):.2f}")
    
    print("\nModifiers:")
    print(f"  Team Modifier: {player.get('team_modifier', 1.0):.2f}x")
    print(f"  xG Consistency: {player.get('xConsistency', 1.0):.2f}x")
    
    form_consistency = player.get('form_consistency', 1.0)
    print(f"  Form Consistency: {form_consistency:.2f}x")
    
    current_rel = player.get('current_reliability', 0)
    print(f"  Current Reliability: {current_rel * 100:.0f}%")
    
    rel_bonus = ((current_rel * 1.5 + 
                 player.get('avg_reliability', 0) * 0.3) - 0.75)
    print(f"  Reliability Bonus: {rel_bonus:.2f}")
    
    if player.get('promoted_penalty', 0) != 0:
        print(f"  Promoted Team Penalty: "
              f"{player.get('promoted_penalty', 0):.2f}")
    
    print("\nFinal Scores:")
    print(f"  Base Quality: {player.get('base_quality', 0):.3f}")
    print(f"  FPL Score (with reliability): "
          f"{player.get('fpl_score', 0):.3f}")
    
    baseline = config.BASELINE_POINTS_PER_GAME.get(position, 4.5)
    print(f"  Baseline Points: {baseline:.1f}")
    print(f"  Points Adjustment: {player.get('base_quality', 0):.3f}")
    
    # Check for DGW/BGW
    fixture_multiplier = player.get('fixture_multiplier', 1.0)
    if fixture_multiplier != 1.0:
        if fixture_multiplier == 0.0:
            print(f"  Fixture Multiplier: {fixture_multiplier:.1f}x "
                  f"(BLANK GW)")
        elif fixture_multiplier == 2.0:
            print(f"  Fixture Multiplier: {fixture_multiplier:.1f}x "
                  f"(DOUBLE GW)")
        else:
            print(f"  Fixture Multiplier: {fixture_multiplier:.1f}x")


def display_prediction(player, config):
    """Display next gameweek prediction."""
    print("\n🔮 PREDICTION")
    print("-" * 79)
    
    projected = player.get('proj_pts', player.get('projected_points', 0))
    
    print(f"Next Gameweek Projection: {projected:.1f} points")
    
    # Show range based on consistency
    consistency = player.get('form_consistency', 1.0)
    if consistency > 1.05:
        print(f"Confidence: HIGH (consistent performer)")
    elif consistency < 0.95:
        print(f"Confidence: LOW (volatile performer)")
    else:
        print(f"Confidence: MEDIUM")
    
    # Show next opponent
    next_opp = player.get('next_opponent', 'Unknown')
    venue = player.get('venue', 'N/A')
    
    print(f"Next Fixture: {next_opp} ({venue})")
    
    # Captain potential
    if projected >= 8.0:
        captain_double = projected * 2
        print(f"\n⭐ CAPTAIN POTENTIAL: Strong candidate "
              f"(2x = {captain_double:.1f} points)")
    elif projected >= 6.5:
        captain_double = projected * 2
        print(f"\n✓ CAPTAIN POTENTIAL: Decent option "
              f"(2x = {captain_double:.1f} points)")


def display_comparison_context(player, all_players):
    """Display how this player ranks vs others."""
    print("\n📊 COMPARISON vs ALL PLAYERS")
    print("-" * 79)
    
    position = player['position']
    position_players = all_players[all_players['position'] == position]
    
    # Rank by projected points
    position_players_sorted = position_players.sort_values(
        'projected_points', ascending=False
    )
    player_rank = (
        position_players_sorted['id'] == player['id']
    ).idxmax() + 1
    
    print(f"Rank in {position}: #{player_rank} of {len(position_players)}")
    
    # Show percentile
    percentile = (1 - (player_rank / len(position_players))) * 100
    print(f"Percentile: Top {percentile:.0f}%")
    
    # Show top 5 in position for context
    print(f"\nTop 5 {position}s by Projected Points:")
    top_5 = position_players_sorted.head(5)
    for i, (_, p) in enumerate(top_5.iterrows(), 1):
        marker = "👉 " if p['id'] == player['id'] else "   "
        print(f"{marker}{i}. {p['display_name']:<20} "
              f"£{p['now_cost_m']:.1f}m  {p['projected_points']:.1f} pts")


def main():
    """Main function for player lookup tool."""
    print("=" * 79)
    print("FANTASY NERDBALL - PLAYER LOOKUP TOOL")
    print("=" * 79)
    
    # Initialise config and components
    config = Config()
    
    print(f"\nGameweek: {config.GAMEWEEK}")
    print("Loading player data...")
    
    # Initialise components
    fpl_client = FPLClient()
    player_processor = PlayerProcessor(config)
    fixture_manager = FixtureManager(config)
    historical_manager = HistoricalDataManager(config)
    scoring_engine = ScoringEngine(config)
    points_calculator = PointsCalculator(config)
    
    # Fetch and process data
    players = player_processor.fetch_current_players()
    players = historical_manager.merge_past_seasons(players)
    
    fixture_scores = fixture_manager.fetch_player_fixture_difficulty(
        1, players, config.GAMEWEEK
    )
    
    scored, _ = scoring_engine.build_scores(players, fixture_scores)
    scored = points_calculator.add_points_analysis_to_display(scored)
    
    # Add next fixture info
    scored = fixture_manager.add_next_fixture(scored, config.GAMEWEEK)
    
    print(f"✓ Loaded {len(scored)} players")
    
    # Interactive loop
    while True:
        print("\n" + "=" * 79)
        player_name = input(
            "\nEnter player name (or 'quit' to exit): "
        ).strip()
        
        if player_name.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not player_name:
            continue
        
        # Find player
        matches = find_player(scored, player_name)
        
        if matches.empty:
            print(f"❌ No players found matching '{player_name}'")
            continue
        
        if len(matches) > 1:
            print(f"\n⚠️  Multiple players found:")
            for i, (_, p) in enumerate(matches.iterrows(), 1):
                print(f"{i}. {p['display_name']} "
                      f"({p['position']}, {p['team']})")
            
            try:
                choice = int(input("\nSelect number: ")) - 1
                if 0 <= choice < len(matches):
                    player = matches.iloc[choice]
                else:
                    print("Invalid selection")
                    continue
            except ValueError:
                print("Invalid input")
                continue
        else:
            player = matches.iloc[0]
        
        # Display all information
        display_player_header(player)
        display_current_season_stats(player, config)
        display_historical_stats(player)
        display_recent_form(player, config)
        display_upcoming_fixtures(player, config, fixture_manager)
        display_scoring_breakdown(player, config)
        display_prediction(player, config)
        display_comparison_context(player, scored)
        
        print("\n" + "=" * 79)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)