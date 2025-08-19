"""Utility functions for squad display and formatting."""

import pandas as pd


class SquadDisplayUtils:
    """Handles consistent squad display formatting and utilities."""
    
    def __init__(self, config):
        self.config = config
        self.position_order = ["GK", "DEF", "MID", "FWD"]
    
    def sort_and_format_starting_xi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort starting XI by position and apply consistent formatting.
        
        Args:
            df (pd.DataFrame): Starting XI dataframe
            
        Returns:
            pd.DataFrame: Sorted and formatted dataframe
        """
        if df.empty:
            return df
        
        df["position"] = pd.Categorical(
            df["position"], categories=self.position_order, ordered=True
        )
        return df.sort_values("position")
    
    def sort_and_format_bench(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort bench players: GK first, then by descending projected points.
        
        Args:
            df (pd.DataFrame): Bench dataframe
            
        Returns:
            pd.DataFrame: Sorted bench dataframe
        """
        if df.empty:
            return df
        
        gk_bench = df[df["position"] == "GK"].copy()
        non_gk_bench = df[df["position"] != "GK"].copy()
        non_gk_bench = non_gk_bench.sort_values(
            "projected_points", ascending=False
        )
        return pd.concat([gk_bench, non_gk_bench], ignore_index=True)
    
    def apply_captain_and_vice(
            self,
            starting_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply captain and vice-captain designations to starting XI.
        
        Args:
            starting_df (pd.DataFrame): Starting XI dataframe
            
        Returns:
            pd.DataFrame: Dataframe with captain and vice-captain marked
        """
        if starting_df.empty:
            return starting_df
        
        df = starting_df.copy()
        
        # Find top two players by projected points
        top_two_idx = df["proj_pts"].nlargest(2).index
        
        if len(top_two_idx) > 0:
            captain_idx = top_two_idx[0]
            df.loc[captain_idx, "display_name"] += " (C)"
            
            # Create display column for projected points with (c)) multiplier
            captain_original_points = df.loc[captain_idx, "proj_pts"]
            df.loc[captain_idx, "proj_pts_display"] = (
                f"{captain_original_points:.1f} (x2)"
            )
            # Update actual projected_points for total calculation
            df.loc[captain_idx, "projected_points"] = (
                captain_original_points * 2
            )
            
        if len(top_two_idx) > 1:
            df.loc[top_two_idx[1], "display_name"] += " (V)"

        # Create display column for non-captain players
        if "proj_pts_display" not in df.columns:
            df["proj_pts_display"] = df["proj_pts"].round(1).astype(str)
        else:
            # Fill in non-captain players
            mask = df["proj_pts_display"].isna()
            df.loc[mask, "proj_pts_display"] = (
                df.loc[mask, "proj_pts"].round(1).astype(str)
            )
        
        return df
    
    def create_display_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create proj_pts_display column for consistent display.
        
        Args:
            df (pd.DataFrame): Squad dataframe
            
        Returns:
            pd.DataFrame: Dataframe with display columns
        """
        df = df.copy()
        if "proj_pts_display" not in df.columns:
            df["proj_pts_display"] = df["proj_pts"].round(1).astype(str)
        return df
    
    def print_squad_table(self, df: pd.DataFrame, rename_map: dict):
        """
        Print squad table with renamed headers and filtered columns.
        
        Args:
            df (pd.DataFrame): Squad dataframe to print
            rename_map (dict): Column renaming mapping
        """
        available_cols = [c for c in rename_map.keys() if c in df.columns]
        renamed = df.rename(columns=rename_map)
        display_cols = [rename_map[c] for c in available_cols]
        print(renamed[display_cols])
    
    def select_starting_xi_fallback(
            self,
            squad_players: pd.DataFrame
            ) -> pd.DataFrame:
        """
        Fallback function to select starting XI when optimisation fails.
        Uses simple greedy selection based on projected points.
        
        Args:
            squad_players (pd.DataFrame): Available squad players
            
        Returns:
            pd.DataFrame: Selected starting XI
        """
        starting_xi = []
        remaining_players = squad_players.copy()
        
        # Required positions for starting XI
        required_positions = {
            "GK": 1,
            "DEF": 3,  # minimum
            "MID": 3,  # minimum  
            "FWD": 1   # minimum
        }
        
        # First, ensure minimum requirements
        for pos, min_count in required_positions.items():
            pos_players = remaining_players[
                remaining_players["position"] == pos
            ].copy()
            pos_players = pos_players.sort_values(
                "projected_points", ascending=False
            )
            
            selected_count = min(min_count, len(pos_players))
            for i in range(selected_count):
                starting_xi.append(pos_players.iloc[i])
                remaining_players = remaining_players[
                    remaining_players["id"] != pos_players.iloc[i]["id"]
                ]
        
        # Fill remaining spots (up to 11 total) with highest projected points
        # Respect maximum constraints: max 5 DEF, max 5 MID, max 3 FWD
        max_positions = {"DEF": 5, "MID": 5, "FWD": 3, "GK": 1}
        current_counts = {"GK": 1, "DEF": 3, "MID": 3, "FWD": 1}
        
        remaining_players = remaining_players.sort_values(
            "projected_points", ascending=False
        )
        
        for _, player in remaining_players.iterrows():
            if len(starting_xi) >= 11:
                break
                
            pos = player["position"]
            if current_counts.get(pos, 0) < max_positions.get(pos, 0):
                starting_xi.append(player)
                current_counts[pos] = current_counts.get(pos, 0) + 1
        
        return pd.DataFrame(starting_xi)
    
    def get_enhanced_display_columns(self) -> list:
        """
        Get the standard set of enhanced display columns.
        
        Returns:
            list: List of column names for enhanced display
        """
        return [
            "display_name",
            "position",
            "team", 
            "now_cost_m",
            "form",
            "historic_ppg",
            "fixture_diff",
            "reliability",
            "historical_xOP",  # Historical Expected Overperformance
            "current_xOP",     # Current Expected Overperformance
            "xConsistency",    # Final Expected modifier
            "minspg",
            "proj_pts_display",
            "next_opponent"
        ]