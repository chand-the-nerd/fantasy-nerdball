"""Player history tracking and management module."""

import os
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from ..api.fpl_client import FPLClient


class PlayerHistoryTracker:
    """Tracks and manages historical player performance data."""
    
    def __init__(self, config):
        """
        Initialise the player history tracker.
        
        Args:
            config: Configuration object containing settings
        """
        self.config = config
        self.fpl_client = FPLClient()
        self.db_path = "data/player_history.db"
        self._ensure_database_exists()
    
    def _ensure_database_exists(self):
        """Create database and tables if they don't exist."""
        os.makedirs("data", exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS player_history (
                    player_id INTEGER,
                    player_name TEXT,
                    team_name TEXT,
                    round INTEGER,
                    total_points INTEGER,
                    minutes INTEGER,
                    goals_scored INTEGER,
                    assists INTEGER,
                    clean_sheets INTEGER,
                    goals_conceded INTEGER,
                    saves INTEGER,
                    yellow_cards INTEGER,
                    red_cards INTEGER,
                    bps INTEGER,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (player_id, round)
                )
            """)
            conn.commit()
    
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
        
        # Get current players
        try:
            bootstrap_data = self.fpl_client.get_bootstrap_static()
            players = pd.DataFrame(bootstrap_data["elements"])
            teams = pd.DataFrame(bootstrap_data["teams"])
            team_map = dict(zip(teams["id"], teams["name"]))
        except Exception as e:
            print(f"Error fetching player data: {e}")
            return
        
        with sqlite3.connect(self.db_path) as conn:
            for _, player in players.iterrows():
                try:
                    player_id = player["id"]
                    player_name = player["web_name"]
                    team_name = team_map.get(player["team"], "Unknown")
                    
                    # Check if data already exists
                    if not force_overwrite:
                        existing = conn.execute(
                            "SELECT COUNT(*) FROM player_history "
                            "WHERE player_id = ? AND round = ?",
                            (player_id, prev_gw)
                        ).fetchone()[0]
                        
                        if existing > 0:
                            continue
                    
                    # Get player's gameweek data
                    history_data = self._get_player_gameweek_data(
                        player_id, prev_gw
                    )
                    
                    if history_data:
                        # Insert or replace data
                        conn.execute("""
                            INSERT OR REPLACE INTO player_history
                            (player_id, player_name, team_name, round,
                             total_points, minutes, goals_scored, assists,
                             clean_sheets, goals_conceded, saves,
                             yellow_cards, red_cards, bps)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            player_id, player_name, team_name, prev_gw,
                            history_data.get("total_points", 0),
                            history_data.get("minutes", 0),
                            history_data.get("goals_scored", 0),
                            history_data.get("assists", 0),
                            history_data.get("clean_sheets", 0),
                            history_data.get("goals_conceded", 0),
                            history_data.get("saves", 0),
                            history_data.get("yellow_cards", 0),
                            history_data.get("red_cards", 0),
                            history_data.get("bps", 0)
                        ))
                        updated_count += 1
                    
                except Exception as e:
                    error_count += 1
                    if error_count <= 5:  # Only print first few errors
                        print(f"Error updating {player_name}: {e}")
            
            conn.commit()
        
        print(f"Updated {updated_count} player records for GW{prev_gw}")
        if error_count > 0:
            print(f"Encountered {error_count} errors during update")
    
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
        with sqlite3.connect(self.db_path) as conn:
            total_records = conn.execute(
                "SELECT COUNT(*) FROM player_history"
            ).fetchone()[0]
            
            total_players = conn.execute(
                "SELECT COUNT(DISTINCT player_id) FROM player_history"
            ).fetchone()[0]
            
            total_teams = conn.execute(
                "SELECT COUNT(DISTINCT team_name) FROM player_history"
            ).fetchone()[0]
            
            avg_records = (total_records / total_players 
                          if total_players > 0 else 0)
            
            return {
                "total_records": total_records,
                "total_players": total_players,
                "total_teams": total_teams,
                "average_records_per_player": avg_records
            }
    
    def cleanup_old_data(self, keep_weeks=38):
        """
        Remove old player history data.
        
        Args:
            keep_weeks (int): Number of recent weeks to keep
        """
        cutoff_gameweek = max(1, self.config.GAMEWEEK - keep_weeks)
        
        with sqlite3.connect(self.db_path) as conn:
            deleted = conn.execute(
                "DELETE FROM player_history WHERE round < ?",
                (cutoff_gameweek,)
            ).rowcount
            conn.commit()
        
        print(f"Deleted {deleted} old records (kept GW{cutoff_gameweek}+)")
    
    def get_player_history(self, player_name, team_name):
        """
        Get history for a specific player.
        
        Args:
            player_name (str): Player's name
            team_name (str): Team name
            
        Returns:
            pd.DataFrame: Player's historical data
        """
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT * FROM player_history
                WHERE LOWER(player_name) LIKE LOWER(?)
                AND LOWER(team_name) LIKE LOWER(?)
                ORDER BY round DESC
            """
            
            df = pd.read_sql_query(
                query,
                conn,
                params=(f"%{player_name}%", f"%{team_name}%")
            )
            
            return df
    
    def get_team_history(self, team_name, gameweeks=None):
        """
        Get history for all players in a team.
        
        Args:
            team_name (str): Team name
            gameweeks (list): Specific gameweeks to query
            
        Returns:
            pd.DataFrame: Team's historical data
        """
        with sqlite3.connect(self.db_path) as conn:
            if gameweeks:
                placeholders = ",".join("?" * len(gameweeks))
                query = f"""
                    SELECT * FROM player_history
                    WHERE LOWER(team_name) LIKE LOWER(?)
                    AND round IN ({placeholders})
                    ORDER BY round DESC, total_points DESC
                """
                params = [f"%{team_name}%"] + gameweeks
            else:
                query = """
                    SELECT * FROM player_history
                    WHERE LOWER(team_name) LIKE LOWER(?)
                    ORDER BY round DESC, total_points DESC
                """
                params = [f"%{team_name}%"]
            
            df = pd.read_sql_query(query, conn, params=params)
            return df