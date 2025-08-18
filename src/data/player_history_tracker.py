"""Module for tracking individual player performance history."""

import os
import pandas as pd
from datetime import datetime
from ..api.fpl_client import FPLClient
from ..utils.text_utils import normalize_name


class PlayerHistoryTracker:
    """Handles individual player performance tracking and CSV storage."""
    
    def __init__(self, config):
        self.config = config
        self.fpl_client = FPLClient()
        self.base_dir = "data/players"
    
    def update_all_players(self, force_overwrite=False):
        """
        Update all players' CSV files with the previous gameweek's data.
        Only fetches data for the previous gameweek to avoid large API calls.
        
        Args:
            force_overwrite (bool): If True, overwrites existing data for the gameweek
        """
        target_gameweek = self.config.GAMEWEEK - 1
        
        if target_gameweek < 1:
            print("No previous gameweek to update (current gameweek is 1)")
            return
        
        print(f"\n=== Updating Player History for GW{target_gameweek} ===")
        if force_overwrite:
            print("🔄 Force overwrite mode enabled - existing data will be replaced")
        
        # Get current player list to know who to track
        bootstrap_data = self.fpl_client.get_bootstrap_static()
        players_df = pd.DataFrame(bootstrap_data["elements"])
        teams_df = pd.DataFrame(bootstrap_data["teams"])
        
        # Create team name mapping
        team_map = dict(zip(teams_df["id"], teams_df["name"]))
        
        successful_updates = 0
        failed_updates = 0
        skipped_updates = 0
        
        for _, player in players_df.iterrows():
            try:
                team_name = team_map.get(player["team"], "unknown")
                result = self._update_player_history(player, team_name, target_gameweek, force_overwrite)
                
                if result == "updated":
                    successful_updates += 1
                elif result == "skipped":
                    skipped_updates += 1
                
                # Progress indicator
                if (successful_updates + skipped_updates) % 50 == 0:
                    print(f"  Processed {successful_updates + skipped_updates} players...")
                    
            except Exception as e:
                print(f"  Error updating {player.get('web_name', 'Unknown')}: {e}")
                failed_updates += 1
        
        print(f"\n✅ Update complete: {successful_updates} updated, {skipped_updates} skipped, {failed_updates} failed")
    
    def _update_player_history(self, player, team_name, target_gameweek, force_overwrite=False):
        """
        Update a single player's CSV file with data from the target gameweek.
        
        Args:
            player (dict): Player data from bootstrap API
            team_name (str): Team name for directory structure
            target_gameweek (int): Gameweek to fetch data for
            force_overwrite (bool): If True, overwrites existing data for the gameweek
            
        Returns:
            str: "updated", "skipped", or "error"
        """
        player_id = player["id"]
        player_name = self._clean_filename(player["web_name"])
        team_dir_name = self._clean_filename(team_name)
        
        # Create directory structure
        player_dir = os.path.join(self.base_dir, team_dir_name)
        os.makedirs(player_dir, exist_ok=True)
        
        csv_path = os.path.join(player_dir, f"{player_name}.csv")
        
        # Check if we already have this gameweek's data (only skip if not forcing overwrite)
        if not force_overwrite and self._gameweek_already_exists(csv_path, target_gameweek):
            return "skipped"  # Skip if already updated and not forcing overwrite
        
        # Fetch player's detailed data
        player_data = self.fpl_client.get_player_summary(player_id)
        
        # Find the target gameweek's data
        gameweek_data = None
        for gw_record in player_data.get("history", []):
            if gw_record["round"] == target_gameweek:
                gameweek_data = gw_record
                break
        
        if gameweek_data is None:
            # Player didn't play this gameweek - create zero record
            gameweek_data = self._create_zero_record(target_gameweek, player)
        
        # Add additional player info
        gameweek_data.update({
            "player_name": player["web_name"],
            "team": team_name,
            "position": self._get_position_name(player["element_type"]),
            "price": player["now_cost"] / 10.0,
            "updated_at": datetime.now().isoformat()
        })
        
        # Convert to DataFrame
        new_row = pd.DataFrame([gameweek_data])
        
        # Load existing data or create new
        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            # Always remove any existing data for this gameweek (enables overwriting)
            existing_df = existing_df[existing_df["round"] != target_gameweek]
            # Append new data
            updated_df = pd.concat([existing_df, new_row], ignore_index=True)
        else:
            updated_df = new_row
        
        # Sort by gameweek
        updated_df = updated_df.sort_values("round")
        
        # Save to CSV
        updated_df.to_csv(csv_path, index=False)
        
        return "updated"
    
    def update_specific_gameweek(self, gameweek, force_overwrite=True):
        """
        Update all players' CSV files with data from a specific gameweek.
        
        Args:
            gameweek (int): The gameweek to update
            force_overwrite (bool): If True, overwrites existing data for the gameweek
        """
        if gameweek < 1 or gameweek > 38:
            print(f"Invalid gameweek: {gameweek}. Must be between 1 and 38.")
            return
        
        print(f"\n=== Updating Player History for GW{gameweek} ===")
        if force_overwrite:
            print("🔄 Force overwrite mode enabled - existing data will be replaced")
        
        # Get current player list to know who to track
        bootstrap_data = self.fpl_client.get_bootstrap_static()
        players_df = pd.DataFrame(bootstrap_data["elements"])
        teams_df = pd.DataFrame(bootstrap_data["teams"])
        
        # Create team name mapping
        team_map = dict(zip(teams_df["id"], teams_df["name"]))
        
        successful_updates = 0
        failed_updates = 0
        skipped_updates = 0
        
        for _, player in players_df.iterrows():
            try:
                team_name = team_map.get(player["team"], "unknown")
                result = self._update_player_history(player, team_name, gameweek, force_overwrite)
                
                if result == "updated":
                    successful_updates += 1
                elif result == "skipped":
                    skipped_updates += 1
                
                # Progress indicator
                if (successful_updates + skipped_updates) % 50 == 0:
                    print(f"  Processed {successful_updates + skipped_updates} players...")
                    
            except Exception as e:
                print(f"  Error updating {player.get('web_name', 'Unknown')}: {e}")
                failed_updates += 1
        
        print(f"\n✅ Update complete: {successful_updates} updated, {skipped_updates} skipped, {failed_updates} failed")
    
    def _gameweek_already_exists(self, csv_path, gameweek):
        """Check if a gameweek already exists in the CSV file."""
        if not os.path.exists(csv_path):
            return False
        
        try:
            df = pd.read_csv(csv_path)
            return gameweek in df["round"].values
        except:
            return False
    
    def _create_zero_record(self, gameweek, player):
        """Create a zero-stats record for players who didn't play."""
        return {
            "round": gameweek,
            "total_points": 0,
            "minutes": 0,
            "goals_scored": 0,
            "assists": 0,
            "clean_sheets": 0,
            "goals_conceded": 0,
            "own_goals": 0,
            "penalties_saved": 0,
            "penalties_missed": 0,
            "yellow_cards": 0,
            "red_cards": 0,
            "saves": 0,
            "bonus": 0,
            "bps": 0,
            "influence": "0.0",
            "creativity": "0.0",
            "threat": "0.0",
            "ict_index": "0.0",
            "starts": 0,
            "expected_goals": "0.0",
            "expected_assists": "0.0",
            "expected_goal_involvements": "0.0",
            "expected_goals_conceded": "0.0",
            "value": player["now_cost"],
            "transfers_balance": 0,
            "selected": 0,
            "transfers_in": 0,
            "transfers_out": 0
        }
    
    def _get_position_name(self, element_type):
        """Convert element_type number to position name."""
        position_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
        return position_map.get(element_type, "Unknown")
    
    def _clean_filename(self, name):
        """Clean a name to be safe for use as a filename."""
        # Remove or replace characters that aren't safe for filenames
        clean_name = name.replace(" ", "_")
        clean_name = clean_name.replace("'", "")
        clean_name = clean_name.replace(".", "")
        clean_name = clean_name.replace("/", "_")
        clean_name = clean_name.replace("\\", "_")
        clean_name = clean_name.replace(":", "_")
        clean_name = clean_name.replace("*", "_")
        clean_name = clean_name.replace("?", "_")
        clean_name = clean_name.replace('"', "_")
        clean_name = clean_name.replace("<", "_")
        clean_name = clean_name.replace(">", "_")
        clean_name = clean_name.replace("|", "_")
        return clean_name.lower()
    
    def get_player_history(self, player_name, team_name, gameweeks=None):
        """
        Retrieve a player's historical data from their CSV file.
        
        Args:
            player_name (str): Player's web name
            team_name (str): Team name
            gameweeks (list, optional): Specific gameweeks to retrieve
            
        Returns:
            pd.DataFrame: Player's historical data
        """
        player_filename = self._clean_filename(player_name)
        team_dir_name = self._clean_filename(team_name)
        csv_path = os.path.join(self.base_dir, team_dir_name, f"{player_filename}.csv")
        
        if not os.path.exists(csv_path):
            return pd.DataFrame()
        
        df = pd.read_csv(csv_path)
        
        if gameweeks is not None:
            df = df[df["round"].isin(gameweeks)]
        
        return df
    
    def get_team_history(self, team_name, gameweeks=None):
        """
        Retrieve all players' data for a specific team.
        
        Args:
            team_name (str): Team name
            gameweeks (list, optional): Specific gameweeks to retrieve
            
        Returns:
            pd.DataFrame: All team players' historical data
        """
        team_dir_name = self._clean_filename(team_name)
        team_dir = os.path.join(self.base_dir, team_dir_name)
        
        if not os.path.exists(team_dir):
            return pd.DataFrame()
        
        all_data = []
        
        for csv_file in os.listdir(team_dir):
            if csv_file.endswith('.csv'):
                csv_path = os.path.join(team_dir, csv_file)
                try:
                    df = pd.read_csv(csv_path)
                    if gameweeks is not None:
                        df = df[df["round"].isin(gameweeks)]
                    all_data.append(df)
                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def cleanup_old_data(self, keep_gameweeks=38):
        """
        Remove old gameweek data to keep file sizes manageable.
        
        Args:
            keep_gameweeks (int): Number of recent gameweeks to keep
        """
        current_gw = self.config.GAMEWEEK
        min_gameweek = max(1, current_gw - keep_gameweeks)
        
        print(f"Cleaning up data older than GW{min_gameweek}...")
        
        cleaned_files = 0
        
        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith('.csv'):
                    csv_path = os.path.join(root, file)
                    try:
                        df = pd.read_csv(csv_path)
                        original_len = len(df)
                        
                        # Keep only recent gameweeks
                        df = df[df["round"] >= min_gameweek]
                        
                        if len(df) < original_len:
                            df.to_csv(csv_path, index=False)
                            cleaned_files += 1
                            
                    except Exception as e:
                        print(f"Error cleaning {csv_path}: {e}")
        
        print(f"✅ Cleaned {cleaned_files} files")
    
    def get_summary_stats(self):
        """Get summary statistics about the stored data."""
        total_files = 0
        total_records = 0
        teams = []
        
        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith('.csv'):
                    total_files += 1
                    csv_path = os.path.join(root, file)
                    try:
                        df = pd.read_csv(csv_path)
                        total_records += len(df)
                    except:
                        pass
            
            # Count teams (directories)
            if dirs:
                teams.extend(dirs)
        
        return {
            "total_players": total_files,
            "total_records": total_records,
            "total_teams": len(teams),
            "average_records_per_player": total_records / max(1, total_files)
        }