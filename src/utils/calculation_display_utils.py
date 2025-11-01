"""Utility functions for displaying detailed score calculations."""

import pandas as pd


class CalculationDisplayUtils:
    """Handles detailed calculation breakdowns for transparency."""
    
    def __init__(self, config):
        self.config = config
    
    def display_squad_calculations(self, starting_df: pd.DataFrame, 
                                 bench_df: pd.DataFrame):
        """
        Display detailed score calculations for each player in the squad.
        
        Args:
            starting_df (pd.DataFrame): Starting XI dataframe
            bench_df (pd.DataFrame): Bench dataframe
        """
        if not self.config.GRANULAR_OUTPUT or not self.config.DETAILED_CALCULATION:
            return
        
        print(f"\n" + "=" * 79)
        print(f"DETAILED SCORE CALCULATIONS FOR GW{self.config.GAMEWEEK}")
        print(f"=" * 79)
        
        # Show calculation methodology
        self._display_calculation_methodology()
        
        # Display starting XI calculations
        print(f"\n{'='*25} STARTING XI CALCULATIONS {'='*25}")
        self._display_players_calculations(starting_df, "Starting XI")
        
        # Display bench calculations
        print(f"\n{'='*25} BENCH CALCULATIONS {'='*29}")
        self._display_players_calculations(bench_df, "Bench")
        
        print(f"\n" + "=" * 79)
    
    def _display_calculation_methodology(self):
        """Display the calculation methodology overview."""
        gameweeks_completed = max(1, self.config.GAMEWEEK - 1)
        early_season_penalty = self._calculate_early_season_penalty_display(
            gameweeks_completed
        )
        
        print(f"\nMETHODOLOGY:")
        penalty_end_gw = 2 + self.config.EARLY_SEASON_PENALTY_GAMEWEEKS
        print(f"  Early Season Penalty: ÷{early_season_penalty:.3f} " +
            f"(GW{self.config.GAMEWEEK}, penalty applied GW2-GW{penalty_end_gw-1})")
        
        # Use MID as example for position-specific weights (since it's most common)
        mid_weights = self.config.POSITION_SCORING_WEIGHTS.get("MID", {
            "form": 0.5, "historic": 0.3, "difficulty": 0.2
        })
        
        print(f"  Scoring Weights (players with history, MID example): " +
            f"Form {mid_weights['form']:.1%}, " +
            f"History {mid_weights['historic']:.1%}, " +
            f"Fixtures {mid_weights['difficulty']:.1%}")
        print(f"  Scoring Weights (new players): " +
            f"Form {1.0 - mid_weights['difficulty']:.1%}, " +
            f"History 0%, " +
            f"Fixtures {mid_weights['difficulty']:.1%}")
        print(f"  Baseline Points: GK {self.config.BASELINE_POINTS_PER_GAME['GK']}, " +
            f"DEF {self.config.BASELINE_POINTS_PER_GAME['DEF']}, " +
            f"MID {self.config.BASELINE_POINTS_PER_GAME['MID']}, " +
            f"FWD {self.config.BASELINE_POINTS_PER_GAME['FWD']}")
    
    def _calculate_early_season_penalty_display(self, 
                                              gameweeks_completed: int) -> float:
        """Calculate early season penalty divisor for display purposes."""
        current_gameweek = self.config.GAMEWEEK
        
        if current_gameweek <= 1:
            return 1.0
        
        # Calculate when penalty should end
        penalty_end_gw = 2 + self.config.EARLY_SEASON_PENALTY_GAMEWEEKS
        if current_gameweek >= penalty_end_gw:
            return 1.0
        
        # Use config values
        initial_divisor = self.config.EARLY_SEASON_PENALTY_INITIAL
        decay_factor = self.config.EARLY_SEASON_DECAY_FACTOR
        penalty_steps = current_gameweek - 2
        current_divisor = initial_divisor * (decay_factor ** penalty_steps)
        
        return max(1.0, current_divisor)
    
    def _display_players_calculations(self, df: pd.DataFrame, squad_type: str):
        """Display calculations for a group of players."""
        if df.empty:
            print(f"  No {squad_type.lower()} players to display")
            return
        
        for idx, player in df.iterrows():
            self._display_single_player_calculation(player, squad_type)
    
    def _display_single_player_calculation(self, player: pd.Series, 
                                        squad_type: str):
        """Display detailed calculation for a single player."""
        name = player.get('display_name', 'Unknown')
        pos = player.get('position', 'UNK')
        team = player.get('team', 'Unknown')
        
        print(f"\n{name} ({pos}, {team})")
        print(f"  " + "-" * (len(name) + len(pos) + len(team) + 6))
        
        # Raw inputs
        form_raw = player.get('form', 0)
        hist_ppg = player.get('avg_ppg_past2', 0)
        fixture_bonus = player.get('fixture_bonus', 0)
        
        # Check if new player
        is_new_player = hist_ppg <= 0
        
        # Early season penalty
        gameweeks_completed = max(1, self.config.GAMEWEEK - 1)
        early_penalty = self._calculate_early_season_penalty_display(
            gameweeks_completed
        )
        form_adjusted = form_raw / early_penalty
        
        print(f"  RAW INPUTS:")
        print(f"    Form (raw): {form_raw:.2f}")
        print(f"    Early season penalty: ÷{early_penalty:.3f}")
        print(f"    Form (adjusted): {form_adjusted:.2f}")
        print(f"    Historical PPG: {hist_ppg:.2f} " + 
            ("(NEW PLAYER)" if is_new_player else ""))
        print(f"    Fixture bonus: {fixture_bonus:.2f}")
        
        # Get position-specific weights
        position_weights = self.config.POSITION_SCORING_WEIGHTS.get(
            pos, self.config.POSITION_SCORING_WEIGHTS.get("MID", {
                "form": 0.5, "historic": 0.3, "difficulty": 0.2
            })
        )
        
        print(f"  WEIGHTS APPLIED:")
        if is_new_player:
            form_weight = 1.0 - position_weights["difficulty"]
            hist_weight = 0.0
            fix_weight = position_weights["difficulty"]
            print(f"    New player weighting: Form {form_weight:.1%}, " +
                f"History {hist_weight:.1%}, Fixtures {fix_weight:.1%}")
        else:
            form_weight = position_weights["form"]
            hist_weight = position_weights["historic"]
            fix_weight = position_weights["difficulty"]
            print(f"    Standard weighting: Form {form_weight:.1%}, " +
                f"History {hist_weight:.1%}, Fixtures {fix_weight:.1%}")
        
        # Adjustments
        promoted_penalty = player.get('promoted_penalty', 0)
        team_modifier = player.get('team_modifier', 1.0)
        x_consistency = player.get('xConsistency', 1.0)
        
        print(f"  ADJUSTMENTS:")
        print(f"    Promoted team penalty: {promoted_penalty:+.2f}")
        print(f"    Team modifier: ×{team_modifier:.2f}")
        print(f"    xG consistency: ×{x_consistency:.2f}")
        
        # NEW: Show form consistency if not neutral
        form_consistency = player.get('form_consistency', 1.0)
        if form_consistency != 1.0:
            consistency_label = "consistent" if form_consistency > 1.0 else "volatile"
            print(f"    Form consistency: ×{form_consistency:.2f} ({consistency_label})")
        
        # Reliability
        current_reliability = player.get('current_reliability', 0)
        avg_reliability = player.get('avg_reliability', 0)
        reliability_bonus = (current_reliability * 1.5 + avg_reliability * 0.3) - 0.75
        unreliable_penalty = player.get('historically_unreliable_penalty', 0)
        
        print(f"  RELIABILITY:")
        print(f"    Current reliability: {current_reliability:.2%}")
        print(f"    Historical reliability: {avg_reliability:.2%}")
        print(f"    Reliability bonus: {reliability_bonus:+.2f}")
        print(f"    Unreliable penalty: {unreliable_penalty:+.2f}")
        
        # Final scores
        base_quality = player.get('base_quality', 0)
        fpl_score = player.get('fpl_score', 0)
        baseline_points = self.config.BASELINE_POINTS_PER_GAME.get(pos, 4.0)
        projected_points = player.get('projected_points', 0)
        
        print(f"  FINAL CALCULATIONS:")
        print(f"    Base quality score: {base_quality:.3f}")
        print(f"    FPL score (base + reliability): {fpl_score:.3f}")
        print(f"    Baseline points ({pos}): {baseline_points:.1f}")
        print(f"    Points adjustment: {base_quality:.3f}")
        print(f"    Projected points: {projected_points:.1f}")
        
        # Fixture multiplier if applicable
        fixture_multiplier = player.get('fixture_multiplier', 1.0)
        if fixture_multiplier != 1.0:
            if fixture_multiplier == 0.0:
                print(f"    Fixture multiplier: {fixture_multiplier:.1f} (BLANK GW)")
            elif fixture_multiplier == 2.0:
                print(f"    Fixture multiplier: {fixture_multiplier:.1f} (DOUBLE GW)")
            else:
                print(f"    Fixture multiplier: {fixture_multiplier:.1f}")
        
        # Captain multiplier if applicable
        if "(C)" in name:
            captain_multiplier = 3 if self.config.TRIPLE_CAPTAIN else 2
            captain_label = "Triple Captain" if self.config.TRIPLE_CAPTAIN else "Captain"
            print(f"    {captain_label} multiplier: ×{captain_multiplier}")
        elif "(V)" in name:
            print(f"    Vice-captain (no multiplier)")
    
    def get_calculation_summary_for_player(self, player: pd.Series) -> dict:
        """
        Get a summary of calculations for a single player.
        
        Args:
            player (pd.Series): Player data
            
        Returns:
            dict: Summary of calculation components
        """
        # Extract key values
        form_raw = player.get('form', 0)
        hist_ppg = player.get('avg_ppg_past2', 0)
        fixture_bonus = player.get('fixture_bonus', 0)
        is_new_player = hist_ppg <= 0
        
        # Calculate early season penalty
        gameweeks_completed = max(1, self.config.GAMEWEEK - 1)
        early_penalty = self._calculate_early_season_penalty_display(
            gameweeks_completed
        )
        
        # Get weights
        if is_new_player:
            form_weight = 1.0 - self.config.DIFFICULTY_WEIGHT
            hist_weight = 0.0
            fix_weight = self.config.DIFFICULTY_WEIGHT
        else:
            form_weight = self.config.FORM_WEIGHT
            hist_weight = self.config.HISTORIC_WEIGHT
            fix_weight = self.config.DIFFICULTY_WEIGHT
        
        # Get final scores
        base_quality = player.get('base_quality', 0)
        fpl_score = player.get('fpl_score', 0)
        projected_points = player.get('projected_points', 0)
        
        return {
            'form_raw': form_raw,
            'early_penalty': early_penalty,
            'form_adjusted': form_raw / early_penalty,
            'hist_ppg': hist_ppg,
            'fixture_bonus': fixture_bonus,
            'is_new_player': is_new_player,
            'form_weight': form_weight,
            'hist_weight': hist_weight,
            'fix_weight': fix_weight,
            'base_quality': base_quality,
            'fpl_score': fpl_score,
            'projected_points': projected_points,
            'team_modifier': player.get('team_modifier', 1.0),
            'x_consistency': player.get('xConsistency', 1.0),
            'fixture_multiplier': player.get('fixture_multiplier', 1.0)
        }