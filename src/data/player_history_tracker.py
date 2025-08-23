"""Player history tracking and management module using CSV files."""

import os
import pandas as pd
from datetime import datetime
from ..api.fpl_client import FPLClient
import re


class PlayerHistoryTracker:
    """Tracks and manages historical player performance data using CSV files."""
    
    def __init__(self, config):
        """
        Initialise the player history tracker.
        
        Args:
            config: Configuration object containing settings
        """
        self.config = config
        self.fpl_client = FPLClient()
        self.base_path = "data/players"
        self._ensure_directory_structure()
    
    def _ensure_directory_structure(self):
        """Create base directory structure if it doesn't exist."""
        os.makedirs(self.base_path, exist_ok=True)
    
    def _sanitize_name(self, name):
        """
        Sanitize a name for use as a filename/directory.
        
        Args:
            name (str): Name to sanitize
            
        Returns:
            str: Sanitized name safe for filesystem
        """
        # Convert to lowercase and replace spaces/special chars with underscores
        sanitized = re.sub(r'[^\w\s-]', '', name.lower())
        sanitized = re.sub(r'[-\s]+', '_', sanitized)
        return sanitized.strip('_')
    
    def _get_player_csv_path(self, team_name, player_name):
        """
        Get the CSV file path for a specific player.
        
        Args:
            team_name (str): Team name
            player_name (str): Player name
            
        Returns:
            str: Path to player's CSV file
        """
        team_dir = os.path.join(self.base_path, self._sanitize_name(team_name))
        os.makedirs(team_dir, exist_ok=True)
        
        player_file = f"{self._sanitize_name(player_name)}.csv"
        return os.path.join(team_dir, player_file)
    
    def _load_player_history(self, csv_path):
        """
        Load existing player history from CSV.
        
        Args:
            csv_path (str): Path to player's CSV file
            
        Returns:
            pd.DataFrame: Existing history data
        """
        if os.path.exists(csv_path):
            try:
                return pd.read_csv(csv_path)
            except Exception as e:
                print(f"Warning: Could not load {csv_path}: {e}")
                return pd.DataFrame()
        return pd.DataFrame()
    
    def _save_player_history(self, csv_path, history_df):
        """
        Save player history to CSV.
        
        Args:
            csv_path (str): Path to player's CSV file
            history_df (pd.DataFrame): History data to save
        """
        try:
            # Sort by round descending (most recent first)
            history_df = history_df.sort_values('round', ascending=False)
            history_df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"Error saving to {csv_path}: {e}")
    
    def update_all_players(self, force_overwrite=False):
        """
        Update player history data for the previous gameweek.
        
        Args:
            force_overwrite (bool): Whether to overwrite existing data
        """
        if self.config.GAMEWEEK <= 1:
            print("Cannot update history for gameweek 1 or earlier")
            return
        
        prev_gw = self.config.GAMEWEEK - 1
        updated_count = 0
        error_count = 0
        skipped_count = 0
        
        print(f"Updating player histories for GW{prev_gw}...")
        
        # Get current players
        try:
            bootstrap_data = self.fpl_client.get_bootstrap_static()
            players = pd.DataFrame(bootstrap_data["elements"])
            teams = pd.DataFrame(bootstrap_data["teams"])
            team_map = dict(zip(teams["id"], teams["name"]))
        except Exception as e:
            print(f"Error fetching player data: {e}")
            return
        
        total_players = len(players)
        
        for idx, (_, player) in enumerate(players.iterrows()):
            try:
                player_id = player["id"]
                player_name = player["web_name"]
                team_name = team_map.get(player["team"], "Unknown")
                
                # Progress indicator every 50 players
                if (idx + 1) % 50 == 0:
                    print(f"Processing player {idx + 1}/{total_players}...")
                
                # Get CSV path for this player
                csv_path = self._get_player_csv_path(team_name, player_name)
                
                # Load existing history
                existing_history = self._load_player_history(csv_path)
                
                # Check if data already exists for this gameweek
                if not force_overwrite and not existing_history.empty:
                    if prev_gw in existing_history['round'].values:
                        skipped_count += 1
                        continue
                
                # Get player's gameweek data
                history_data = self._get_player_gameweek_data(player_id, prev_gw)
                
                if history_data:
                    # Create new row
                    new_row = {
                        'player_id': player_id,
                        'player_name': player_name,
                        'team_name': team_name,
                        'round': prev_gw,
                        'total_points': history_data.get("total_points", 0),
                        'minutes': history_data.get("minutes", 0),
                        'goals_scored': history_data.get("goals_scored", 0),
                        'assists': history_data.get("assists", 0),
                        'clean_sheets': history_data.get("clean_sheets", 0),
                        'goals_conceded': history_data.get("goals_conceded", 0),
                        'saves': history_data.get("saves", 0),
                        'yellow_cards': history_data.get("yellow_cards", 0),
                        'red_cards': history_data.get("red_cards", 0),
                        'bps': history_data.get("bps", 0),
                        'updated_at': datetime.now().isoformat()
                    }
                    
                    # Update existing history
                    if not existing_history.empty:
                        # Remove existing data for this gameweek if it exists
                        existing_history = existing_history[existing_history['round'] != prev_gw]
                        # Add new row
                        updated_history = pd.concat([
                            existing_history, 
                            pd.DataFrame([new_row])
                        ], ignore_index=True)
                    else:
                        updated_history = pd.DataFrame([new_row])
                    
                    # Save updated history
                    self._save_player_history(csv_path, updated_history)
                    updated_count += 1
                
            except Exception as e:
                error_count += 1
                if error_count <= 5:  # Only print first few errors
                    print(f"Error updating {player_name}: {e}")
        
        print(f"Update completed!")
        print(f"  Updated: {updated_count} players")
        print(f"  Skipped: {skipped_count} players (data already exists)")
        if error_count > 0:
            print(f"  Errors: {error_count} players")
    
    def _get_player_gameweek_data(self, player_id, gameweek):
        """
        Get player data for a specific gameweek.
        
        Args:
            player_id (int): FPL player ID
            gameweek (int): Gameweek number
            
        Returns:
            dict: Player's gameweek data or None if not found
        """
        try:
            data = self.fpl_client.get_player_summary(player_id)
            
            for history in data.get("history", []):
                if history["round"] == gameweek:
                    return history
            
            return None
            
        except Exception:
            return None
    
    def get_summary_stats(self):
        """
        Get summary statistics about tracked data.
        
        Returns:
            dict: Summary statistics
        """
        total_records = 0
        total_players = 0
        total_teams = 0
        teams_found = set()
        
        try:
            # Walk through all team directories
            for team_dir in os.listdir(self.base_path):
                team_path = os.path.join(self.base_path, team_dir)
                if os.path.isdir(team_path):
                    teams_found.add(team_dir)
                    
                    # Count CSV files (players) in this team
                    for filename in os.listdir(team_path):
                        if filename.endswith('.csv'):
                            total_players += 1
                            csv_path = os.path.join(team_path, filename)
                            
                            try:
                                df = pd.read_csv(csv_path)
                                total_records += len(df)
                            except Exception:
                                pass
            
            total_teams = len(teams_found)
            avg_records = total_records / total_players if total_players > 0 else 0
            
            return {
                "total_records": total_records,
                "total_players": total_players,
                "total_teams": total_teams,
                "average_records_per_player": avg_records
            }
            
        except Exception as e:
            print(f"Error getting summary stats: {e}")
            return {
                "total_records": 0,
                "total_players": 0,
                "total_teams": 0,
                "average_records_per_player": 0
            }
    
    def cleanup_old_data(self, keep_weeks=38):
        """
        Remove old player history data.
        
        Args:
            keep_weeks (int): Number of recent weeks to keep
        """
        cutoff_gameweek = max(1, self.config.GAMEWEEK - keep_weeks)
        deleted_records = 0
        processed_files = 0
        
        try:
            # Walk through all team directories
            for team_dir in os.listdir(self.base_path):
                team_path = os.path.join(self.base_path, team_dir)
                if os.path.isdir(team_path):
                    
                    # Process CSV files in this team
                    for filename in os.listdir(team_path):
                        if filename.endswith('.csv'):
                            csv_path = os.path.join(team_path, filename)
                            
                            try:
                                df = pd.read_csv(csv_path)
                                original_len = len(df)
                                
                                # Keep only recent gameweeks
                                df_filtered = df[df['round'] >= cutoff_gameweek]
                                
                                deleted_from_file = original_len - len(df_filtered)
                                deleted_records += deleted_from_file
                                
                                if deleted_from_file > 0:
                                    # Save filtered data
                                    self._save_player_history(csv_path, df_filtered)
                                
                                processed_files += 1
                                
                            except Exception as e:
                                print(f"Error processing {csv_path}: {e}")
        
        except Exception as e:
            print(f"Error during cleanup: {e}")
        
        print(f"Cleanup completed!")
        print(f"  Processed {processed_files} player files")
        print(f"  Deleted {deleted_records} old records (kept GW{cutoff_gameweek}+)")
    
    def get_player_history(self, player_name, team_name):
        """
        Get history for a specific player.
        
        Args:
            player_name (str): Player's name
            team_name (str): Team name
            
        Returns:
            pd.DataFrame: Player's historical data
        """
        # Try exact match first
        csv_path = self._get_player_csv_path(team_name, player_name)
        df = self._load_player_history(csv_path)
        
        if not df.empty:
            return df.sort_values('round', ascending=False)
        
        # If no exact match, search for similar names
        team_sanitized = self._sanitize_name(team_name)
        team_path = os.path.join(self.base_path, team_sanitized)
        
        if os.path.exists(team_path):
            player_sanitized = self._sanitize_name(player_name)
            
            # Look for files that contain the player name
            for filename in os.listdir(team_path):
                if (filename.endswith('.csv') and 
                    player_sanitized in filename):
                    csv_path = os.path.join(team_path, filename)
                    df = self._load_player_history(csv_path)
                    if not df.empty:
                        return df.sort_values('round', ascending=False)
        
        return pd.DataFrame()
    
    def get_team_history(self, team_name, gameweeks=None):
        """
        Get history for all players in a team.
        
        Args:
            team_name (str): Team name
            gameweeks (list): Specific gameweeks to query
            
        Returns:
            pd.DataFrame: Team's historical data
        """
        team_sanitized = self._sanitize_name(team_name)
        team_path = os.path.join(self.base_path, team_sanitized)
        
        if not os.path.exists(team_path):
            return pd.DataFrame()
        
        all_data = []
        
        # Load all player files from this team
        for filename in os.listdir(team_path):
            if filename.endswith('.csv'):
                csv_path = os.path.join(team_path, filename)
                df = self._load_player_history(csv_path)
                
                if not df.empty:
                    if gameweeks is not None:
                        df = df[df['round'].isin(gameweeks)]
                    
                    all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            return combined_df.sort_values(['round', 'total_points'], 
                                         ascending=[False, False])
        
        return pd.DataFrame()
    
    def get_all_players_gameweek(self, gameweek):
        """
        Get all players' data for a specific gameweek.
        
        Args:
            gameweek (int): Gameweek number
            
        Returns:
            pd.DataFrame: All players' data for the gameweek
        """
        all_data = []
        
        try:
            # Walk through all team directories
            for team_dir in os.listdir(self.base_path):
                team_path = os.path.join(self.base_path, team_dir)
                if os.path.isdir(team_path):
                    
                    # Process CSV files in this team
                    for filename in os.listdir(team_path):
                        if filename.endswith('.csv'):
                            csv_path = os.path.join(team_path, filename)
                            df = self._load_player_history(csv_path)
                            
                            if not df.empty:
                                gw_data = df[df['round'] == gameweek]
                                if not gw_data.empty:
                                    all_data.append(gw_data)
        
        except Exception as e:
            print(f"Error loading gameweek data: {e}")
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        
        return pd.DataFrame()