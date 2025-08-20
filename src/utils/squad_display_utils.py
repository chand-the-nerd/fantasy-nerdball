"""Utility functions for squad display and formatting."""

import pandas as pd


class SquadDisplayUtils:
    """Handles consistent squad display formatting and utilities."""
    
    def __init__(self, config):
        self.config = config
        self.position_order = ["GK", "DEF", "MID", "FWD"]
    
    def sort_and_format_starting_xi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort starting XI by position and apply consistent formatting,
        including DGW asterisk notation.
        
        Args:
            df (pd.DataFrame): Starting XI dataframe
            
        Returns:
            pd.DataFrame: Sorted and formatted dataframe
        """
        if df.empty:
            return df
        
        # Add DGW asterisk to player names
        df = self._add_dgw_asterisk(df)
        
        df["position"] = pd.Categorical(
            df["position"], categories=self.position_order, ordered=True
        )
        return df.sort_values("position")
    
    def sort_and_format_bench(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort bench players: GK first, then by descending projected points,
        including DGW asterisk notation.
        
        Args:
            df (pd.DataFrame): Bench dataframe
            
        Returns:
            pd.DataFrame: Sorted bench dataframe
        """
        if df.empty:
            return df
        
        # Add DGW asterisk to player names
        df = self._add_dgw_asterisk(df)
        
        gk_bench = df[df["position"] == "GK"].copy()
        non_gk_bench = df[df["position"] != "GK"].copy()
        non_gk_bench = non_gk_bench.sort_values(
            "projected_points", ascending=False
        )
        return pd.concat([gk_bench, non_gk_bench], ignore_index=True)
    
    def _add_dgw_asterisk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add asterisk to player names who have double gameweeks.
        Uses the reliable fixture multiplier and DGW flags.
        
        Args:
            df (pd.DataFrame): Player dataframe
            
        Returns:
            pd.DataFrame: Dataframe with DGW asterisks added
        """
        df = df.copy()
        
        # Initialize DGW mask as False for all players
        dgw_mask = pd.Series([False] * len(df), index=df.index)
        
        # Method 1: Check has_dgw_next column (most reliable for current GW)
        if "has_dgw_next" in df.columns:
            dgw_mask = df["has_dgw_next"] == True
        
        # Method 2: Check for multiple opponents in next_opponent string
        elif "next_opponent" in df.columns:
            dgw_mask = df["next_opponent"].str.contains(" & ", na=False)
        
        # Method 3: Check fixture_multiplier > 1 (from fixture analysis)
        elif "fixture_multiplier" in df.columns:
            dgw_mask = df["fixture_multiplier"] > 1.0
        
        # Method 4: Check has_dgw column (from fixture analysis)
        elif "has_dgw" in df.columns:
            dgw_mask = df["has_dgw"] == True
        
        # Only add asterisk if we found actual DGW players
        if dgw_mask.any():
            df.loc[dgw_mask, "display_name"] = (
                df.loc[dgw_mask, "display_name"].astype(str) + "*"
            )
        
        return df
    
    def apply_captain_and_vice(self, starting_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply captain and vice-captain designations to starting XI.
        Enhanced to support Triple Captain chip and DGW handling.
        
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
            
            # Check for Triple Captain chip
            captain_multiplier = 3 if self.config.TRIPLE_CAPTAIN else 2
            captain_display = "x3" if self.config.TRIPLE_CAPTAIN else "x2"
            
            # Create display column for projected points with captain 
            # multiplier
            captain_original_points = df.loc[captain_idx, "proj_pts"]
            df.loc[captain_idx, "proj_pts_display"] = (
                f"{captain_original_points:.1f} ({captain_display})"
            )
            # Update actual projected_points for total calculation
            df.loc[captain_idx, "projected_points"] = (
                captain_original_points * captain_multiplier
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
        Print squad table with renamed headers and filtered columns,
        including DGW notation explanation if asterisks are present.
        
        Args:
            df (pd.DataFrame): Squad dataframe to print
            rename_map (dict): Column renaming mapping
        """
        available_cols = [c for c in rename_map.keys() if c in df.columns]
        renamed = df.rename(columns=rename_map)
        display_cols = [rename_map[c] for c in available_cols]
        print(renamed[display_cols])
        
        # Check if any players have asterisks (DGW indicator)
        if "name" in display_cols:
            has_asterisk = renamed["name"].astype(str).str.contains(
                r"\*", na=False
            ).any()
            if has_asterisk:
                print("\n*Double gameweek")
    
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