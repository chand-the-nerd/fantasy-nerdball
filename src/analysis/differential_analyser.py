"""Module for analysing differential picks with low ownership but high 
expected performance."""

import pandas as pd


class DifferentialAnalyser:
    """Handles analysis and display of differential player picks."""
    
    def __init__(self, config):
        self.config = config
        self.ownership_threshold = 5.0  # Players with <5% ownership
    
    def add_multi_fixture_info(self, df: pd.DataFrame, starting_gameweek: int,
                             num_gameweeks: int, fixture_manager) -> pd.DataFrame:
        """
        Add multi-fixture information showing upcoming opponents across
        multiple gameweeks with individual GW columns.
        
        Args:
            df (pd.DataFrame): Player dataframe
            starting_gameweek (int): Starting gameweek number
            num_gameweeks (int): Number of gameweeks to look ahead
            fixture_manager: FixtureManager instance
            
        Returns:
            pd.DataFrame: Dataframe with individual GW columns added
        """
        df = df.copy()
        
        # Get fixture data for multiple gameweeks
        fixtures = fixture_manager.fpl_client.get_fixtures()
        fixtures_df = pd.DataFrame(fixtures)
        
        # Get team names
        teams_data = fixture_manager.fpl_client.get_bootstrap_static()
        teams_df = pd.DataFrame(teams_data["teams"])[["id", "name"]]
        teams_dict = dict(zip(teams_df["id"], teams_df["name"]))
        
        # Build fixture info for each player
        fixture_info = []
        
        for _, player in df.iterrows():
            team_id = player["team_id"]
            player_fixture_data = {"name_key": player["name_key"]}
            
            # Add individual gameweek columns
            for gw in range(starting_gameweek, 
                          starting_gameweek + num_gameweeks):
                gw_fixtures = fixtures_df[fixtures_df["event"] == gw]
                
                # Find fixtures for this team in this gameweek
                team_fixtures = gw_fixtures[
                    (gw_fixtures["team_h"] == team_id) | 
                    (gw_fixtures["team_a"] == team_id)
                ]
                
                gw_opponents = []
                for _, fixture in team_fixtures.iterrows():
                    if fixture["team_h"] == team_id:
                        # Home game
                        opponent_id = fixture["team_a"]
                    else:
                        # Away game  
                        opponent_id = fixture["team_h"]
                    
                    opponent_name = teams_dict.get(opponent_id, "Unknown")
                    gw_opponents.append(opponent_name)
                
                # Create column name and value
                gw_col = f"gw{gw}"
                if len(gw_opponents) == 0:
                    player_fixture_data[gw_col] = "-"
                elif len(gw_opponents) == 1:
                    player_fixture_data[gw_col] = gw_opponents[0]
                else:
                    # Multiple fixtures (DGW)
                    player_fixture_data[gw_col] = " & ".join(gw_opponents)
            
            fixture_info.append(player_fixture_data)
        
        # Merge back to main dataframe
        fixture_df = pd.DataFrame(fixture_info)
        df = df.merge(fixture_df, on="name_key", how="left")
        
        # Fill any missing values
        gw_columns = [f"gw{gw}" for gw in range(starting_gameweek, 
                                               starting_gameweek + num_gameweeks)]
        for col in gw_columns:
            if col in df.columns:
                df[col] = df[col].fillna("-")
        
        return df
    
    def get_differential_suggestions(self, scored_players: pd.DataFrame,
                                   starting_gameweek: int = None,
                                   num_gameweeks: int = None) -> dict:
        """
        Get top 3 differential suggestions for each position based on FPL 
        score and low ownership.
        
        Args:
            scored_players (pd.DataFrame): Player data with FPL scores and 
                                         ownership
            starting_gameweek (int): Starting gameweek number
            num_gameweeks (int): Number of gameweeks being analysed
            
        Returns:
            dict: Dictionary with position keys and top 3 differentials for 
                  each
        """
        differentials = {}
        
        # Check if ownership column exists
        if 'selected_by_percent' not in scored_players.columns:
            print("Warning: selected_by_percent column not found. "
                  "Cannot generate differential suggestions.")
            return differentials
        
        # Create a working copy and convert ownership to float
        df_work = scored_players.copy()
        df_work['ownership_numeric'] = pd.to_numeric(
            df_work['selected_by_percent'], errors='coerce'
        ).fillna(100.0)  # If conversion fails, assume high ownership
        
        # Remove duplicates based on player ID or name to avoid duplicates
        if 'id' in df_work.columns:
            df_work = df_work.drop_duplicates(subset=['id'])
        else:
            df_work = df_work.drop_duplicates(subset=['display_name', 'team'])
        
        positions = ["GK", "DEF", "MID", "FWD"]
        
        for position in positions:
            # Filter players by position and low ownership
            position_players = df_work[
                (df_work["position"] == position) &
                (df_work["ownership_numeric"] < self.ownership_threshold) &
                (df_work["fpl_score"] > 0)  # Only positive scores
            ].copy()
            
            if position_players.empty:
                differentials[position] = []
                continue
            
            # Sort by FPL score (descending) and take top 3
            top_differentials = position_players.nlargest(3, "fpl_score")
            
            differentials[position] = self._format_differential_data(
                top_differentials, starting_gameweek or self.config.GAMEWEEK,
                num_gameweeks or self.config.FIRST_N_GAMEWEEKS)
        
        return differentials
    
    def _format_differential_data(self, players: pd.DataFrame, 
                                starting_gameweek: int, 
                                num_gameweeks: int) -> list:
        """
        Format differential player data for display.
        
        Args:
            players (pd.DataFrame): Top differential players
            starting_gameweek (int): Starting gameweek number
            num_gameweeks (int): Number of gameweeks being analysed
            
        Returns:
            list: List of formatted player dictionaries
        """
        formatted_players = []
        
        for _, player in players.iterrows():
            # Use the numeric ownership we created, or fallback to original
            ownership_val = player.get('ownership_numeric', 
                                     player.get('selected_by_percent', 0))
            
            # Calculate points per gameweek
            total_proj_pts = player.get('projected_points', 0)
            proj_pts_pgw = total_proj_pts / num_gameweeks if num_gameweeks > 0 else 0
            
            formatted_player = {
                'name': player['display_name'],
                'team': player['team'],
                'cost': player['now_cost_m'],
                'ownership': ownership_val,
                'form': player.get('form', 0),
                'proj_pts_total': total_proj_pts,
                'proj_pts_pgw': proj_pts_pgw
            }
            
            # Add individual gameweek columns
            for gw in range(starting_gameweek, 
                          starting_gameweek + num_gameweeks):
                gw_col = f"gw{gw}"
                formatted_player[gw_col] = player.get(gw_col, "-")
            
            formatted_players.append(formatted_player)
        
        return formatted_players
    
    def print_differential_suggestions(self, differentials: dict,
                                     starting_gameweek: int = None,
                                     num_gameweeks: int = None):
        """
        Print differential suggestions in a formatted table.
        
        Args:
            differentials (dict): Dictionary of differential suggestions by 
                                position
            starting_gameweek (int): Starting gameweek number
            num_gameweeks (int): Number of gameweeks being analysed
        """
        if not differentials or not any(differentials.values()):
            print(f"\n=== 💎 DIFFERENTIAL SUGGESTIONS ===")
            print("No suitable differentials found (players <5% ownership "
                  "with positive FPL scores)")
            return
        
        # Use config defaults if not provided
        start_gw = starting_gameweek or self.config.GAMEWEEK
        num_gw = num_gameweeks or self.config.FIRST_N_GAMEWEEKS
        
        print(f"\n=== 💎 DIFFERENTIAL SUGGESTIONS ===")
        print(f"Low ownership (<5%) players with high expected performance "
              f"(GW{start_gw}-{start_gw + num_gw - 1})")
        
        for position in ["GK", "DEF", "MID", "FWD"]:
            if position not in differentials or not differentials[position]:
                continue
            
            print(f"\n{position}:")
            
            # Create DataFrame for nice formatting
            df_data = []
            for player in differentials[position]:
                player_data = {
                    'name': player['name'],
                    'team': player['team'],
                    'cost': player['cost'],
                    'ownership': f"{player['ownership']:.1f}%",
                    'form': player['form'],
                    'proj_pts_total': f"{player['proj_pts_total']:.1f}",
                    'proj_pts_pgw': f"{player['proj_pts_pgw']:.1f}"
                }
                
                # Add individual gameweek columns
                for gw in range(start_gw, start_gw + num_gw):
                    gw_col = f"gw{gw}"
                    player_data[gw_col] = player.get(gw_col, "-")
                
                df_data.append(player_data)
            
            if df_data:
                df = pd.DataFrame(df_data)
                print(df.to_string(index=False))
    
    def get_differential_summary_stats(self, differentials: dict) -> dict:
        """
        Get summary statistics about differential suggestions.
        
        Args:
            differentials (dict): Dictionary of differential suggestions
            
        Returns:
            dict: Summary statistics
        """
        total_found = sum(len(players) for players in differentials.values())
        positions_with_differentials = len([pos for pos, players in 
                                          differentials.items() if players])
        
        avg_ownership = 0
        avg_proj_pts = 0
        count = 0
        
        for players in differentials.values():
            for player in players:
                avg_ownership += player['ownership']
                avg_proj_pts += player['proj_pts']
                count += 1
        
        if count > 0:
            avg_ownership /= count
            avg_proj_pts /= count
        
        return {
            'total_found': total_found,
            'positions_covered': positions_with_differentials,
            'avg_ownership': avg_ownership,
            'avg_projected_points': avg_proj_pts
        }