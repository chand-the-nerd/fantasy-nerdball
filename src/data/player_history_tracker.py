"""Player history tracking and management module using CSV files."""

import os
import re
from datetime import datetime

import pandas as pd

from ..api.fpl_client import FPLClient


class PlayerHistoryTracker:
    """Tracks and manages historical player performance using CSVs."""

    def __init__(self, config):
        """
        Initialise the player history tracker.

        Args:
            config: Configuration object containing settings
        """
        self.config = config
        self.fpl_client = FPLClient()
        self.base_path = "data/players"
        self._id_index = None
        self._ensure_directory_structure()

    def _ensure_directory_structure(self):
        """Create the base directory structure if it does not exist."""
        os.makedirs(self.base_path, exist_ok=True)

    def _sanitize_name(self, name):
        """
        Sanitise a name for use as a filename or directory.

        Args:
            name (str): Name to sanitise

        Returns:
            str: Name safe for the filesystem
        """
        sanitized = re.sub(r'[^\w\s-]', '', str(name).lower())
        sanitized = re.sub(r'[-\s]+', '_', sanitized)
        return sanitized.strip('_')

    def _get_player_csv_path(self, team_name, player_name):
        """
        Get the CSV file path for a specific player.

        Args:
            team_name (str): Team name
            player_name (str): Player name

        Returns:
            str: Path to the player's CSV file
        """
        team_dir = os.path.join(
            self.base_path, self._sanitize_name(team_name)
        )
        os.makedirs(team_dir, exist_ok=True)

        player_file = f"{self._sanitize_name(player_name)}.csv"
        return os.path.join(team_dir, player_file)

    def _all_history_files(self):
        """Yield every player history CSV path under the base path."""
        if not os.path.exists(self.base_path):
            return

        for team_dir in os.listdir(self.base_path):
            team_path = os.path.join(self.base_path, team_dir)
            if not os.path.isdir(team_path):
                continue

            for filename in os.listdir(team_path):
                if filename.endswith('.csv'):
                    yield os.path.join(team_path, filename)

    def _load_player_history(self, csv_path):
        """
        Load existing player history from CSV.

        Args:
            csv_path (str): Path to the player's CSV file

        Returns:
            pd.DataFrame: Existing history data
        """
        if os.path.exists(csv_path):
            try:
                return pd.read_csv(csv_path)
            except Exception as error:
                print(f"Warning: could not load {csv_path}: {error}")
                return pd.DataFrame()
        return pd.DataFrame()

    def _save_player_history(self, csv_path, history_df):
        """
        Save player history to CSV.

        Args:
            csv_path (str): Path to the player's CSV file
            history_df (pd.DataFrame): History data to save
        """
        try:
            history_df = history_df.sort_values(
                'round', ascending=False
            )
            history_df.to_csv(csv_path, index=False)
        except Exception as error:
            print(f"Error saving to {csv_path}: {error}")

        # Any write invalidates the cached id index.
        self._id_index = None

    def _build_id_index(self):
        """
        Build a map from FPL element id to the files holding that
        player's history.

        A player who moves club mid-season has rows written under both
        clubs' directories, and two players who share a surname within
        one club would otherwise resolve to the same file. Indexing on
        the element id avoids both problems.
        """
        index = {}

        for csv_path in self._all_history_files():
            df = self._load_player_history(csv_path)
            if df.empty or 'player_id' not in df.columns:
                continue

            for player_id in df['player_id'].dropna().unique():
                index.setdefault(int(player_id), []).append(csv_path)

        self._id_index = index
        return index

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

        try:
            bootstrap_data = self.fpl_client.get_bootstrap_static()
            players = pd.DataFrame(bootstrap_data["elements"])
            teams = pd.DataFrame(bootstrap_data["teams"])
            team_map = dict(zip(teams["id"], teams["name"]))
        except Exception as error:
            print(f"Error fetching player data: {error}")
            return

        total_players = len(players)
        player_name = "unknown"

        for idx, (_, player) in enumerate(players.iterrows()):
            try:
                player_id = player["id"]
                player_code = player.get("code")
                player_name = player["web_name"]
                team_name = team_map.get(player["team"], "Unknown")

                if (idx + 1) % 50 == 0:
                    print(f"Processing player {idx + 1}/"
                          f"{total_players}...")

                csv_path = self._get_player_csv_path(
                    team_name, player_name
                )

                existing_history = self._load_player_history(csv_path)

                if not force_overwrite and not existing_history.empty:
                    same_gw = existing_history['round'] == prev_gw
                    if 'player_id' in existing_history.columns:
                        same_gw = same_gw & (
                            existing_history['player_id'] == player_id
                        )
                    if same_gw.any():
                        skipped_count += 1
                        continue

                history_data = self._get_player_gameweek_data(
                    player_id, prev_gw
                )

                if history_data:
                    new_row = {
                        'player_id': player_id,
                        'player_code': player_code,
                        'player_name': player_name,
                        'team_name': team_name,
                        'round': prev_gw,
                        'total_points': history_data.get(
                            "total_points", 0),
                        'minutes': history_data.get("minutes", 0),
                        'goals_scored': history_data.get(
                            "goals_scored", 0),
                        'assists': history_data.get("assists", 0),
                        'clean_sheets': history_data.get(
                            "clean_sheets", 0),
                        'goals_conceded': history_data.get(
                            "goals_conceded", 0),
                        'saves': history_data.get("saves", 0),
                        'yellow_cards': history_data.get(
                            "yellow_cards", 0),
                        'red_cards': history_data.get("red_cards", 0),
                        'bps': history_data.get("bps", 0),
                        'updated_at': datetime.now().isoformat()
                    }

                    if not existing_history.empty:
                        stale = existing_history['round'] == prev_gw
                        if 'player_id' in existing_history.columns:
                            stale = stale & (
                                existing_history['player_id']
                                == player_id
                            )
                        existing_history = existing_history[~stale]

                        updated_history = pd.concat([
                            existing_history,
                            pd.DataFrame([new_row])
                        ], ignore_index=True)
                    else:
                        updated_history = pd.DataFrame([new_row])

                    self._save_player_history(
                        csv_path, updated_history
                    )
                    updated_count += 1

            except Exception as error:
                error_count += 1
                if error_count <= 5:
                    print(f"Error updating {player_name}: {error}")

        print("Update completed.")
        print(f"  Updated: {updated_count} players")
        print(f"  Skipped: {skipped_count} players (already present)")
        if error_count > 0:
            print(f"  Errors: {error_count} players")

    def _get_player_gameweek_data(self, player_id, gameweek):
        """
        Get player data for a specific gameweek.

        Args:
            player_id (int): FPL player ID
            gameweek (int): Gameweek number

        Returns:
            dict: The player's gameweek data, or None if not found
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
        Get summary statistics about the tracked data.

        Returns:
            dict: Summary statistics
        """
        total_records = 0
        total_players = 0
        teams_found = set()

        try:
            for csv_path in self._all_history_files():
                teams_found.add(os.path.basename(
                    os.path.dirname(csv_path)
                ))
                total_players += 1

                try:
                    df = pd.read_csv(csv_path)
                    total_records += len(df)
                except Exception:
                    pass

            avg_records = (
                total_records / total_players
                if total_players > 0 else 0
            )

            return {
                "total_records": total_records,
                "total_players": total_players,
                "total_teams": len(teams_found),
                "average_records_per_player": avg_records
            }

        except Exception as error:
            print(f"Error getting summary stats: {error}")
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
            for csv_path in list(self._all_history_files()):
                try:
                    df = pd.read_csv(csv_path)
                    original_len = len(df)

                    df_filtered = df[df['round'] >= cutoff_gameweek]

                    deleted_from_file = original_len - len(df_filtered)
                    deleted_records += deleted_from_file

                    if deleted_from_file > 0:
                        self._save_player_history(
                            csv_path, df_filtered
                        )

                    processed_files += 1

                except Exception as error:
                    print(f"Error processing {csv_path}: {error}")

        except Exception as error:
            print(f"Error during cleanup: {error}")

        print("Cleanup completed.")
        print(f"  Processed {processed_files} player files")
        print(f"  Deleted {deleted_records} old records "
              f"(kept GW{cutoff_gameweek}+)")

    def get_player_history(self, player_name, team_name,
                           player_id=None):
        """
        Get the history for a specific player.

        Args:
            player_name (str): Player's name
            team_name (str): Team name
            player_id (int, optional): FPL element id. Preferred, since
                it is unambiguous and survives a mid-season club change.

        Returns:
            pd.DataFrame: The player's historical data
        """
        if player_id is not None and pd.notna(player_id):
            history = self._history_by_id(int(player_id))
            if not history.empty:
                return history

        csv_path = self._get_player_csv_path(team_name, player_name)
        df = self._load_player_history(csv_path)

        if not df.empty:
            return df.sort_values('round', ascending=False)

        # Fall back to a name search within the club's directory only.
        # A search across every club would happily return a different
        # player who shares the surname.
        team_sanitized = self._sanitize_name(team_name)
        team_path = os.path.join(self.base_path, team_sanitized)

        if os.path.exists(team_path):
            player_sanitized = self._sanitize_name(player_name)

            for filename in os.listdir(team_path):
                if (filename.endswith('.csv') and
                        player_sanitized in filename):
                    csv_path = os.path.join(team_path, filename)
                    df = self._load_player_history(csv_path)
                    if not df.empty:
                        return df.sort_values(
                            'round', ascending=False
                        )

        return pd.DataFrame()

    def _history_by_id(self, player_id: int) -> pd.DataFrame:
        """
        Collect a player's rows from every file that holds them.

        Rows are gathered across clubs so that a player who moved
        mid-season keeps a continuous record.
        """
        if self._id_index is None:
            self._build_id_index()

        paths = self._id_index.get(player_id, [])
        if not paths:
            return pd.DataFrame()

        frames = []
        for csv_path in paths:
            df = self._load_player_history(csv_path)
            if df.empty or 'player_id' not in df.columns:
                continue
            frames.append(df[df['player_id'] == player_id])

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=['round'])
        return combined.sort_values('round', ascending=False)

    def get_team_history(self, team_name, gameweeks=None):
        """
        Get the history for all players in a team.

        Args:
            team_name (str): Team name
            gameweeks (list): Specific gameweeks to query

        Returns:
            pd.DataFrame: The team's historical data
        """
        team_sanitized = self._sanitize_name(team_name)
        team_path = os.path.join(self.base_path, team_sanitized)

        if not os.path.exists(team_path):
            return pd.DataFrame()

        all_data = []

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
            return combined_df.sort_values(
                ['round', 'total_points'], ascending=[False, False]
            )

        return pd.DataFrame()

    def get_all_players_gameweek(self, gameweek):
        """
        Get every player's data for a specific gameweek.

        Args:
            gameweek (int): Gameweek number

        Returns:
            pd.DataFrame: All players' data for the gameweek
        """
        all_data = []

        try:
            for csv_path in self._all_history_files():
                df = self._load_player_history(csv_path)

                if not df.empty:
                    gw_data = df[df['round'] == gameweek]
                    if not gw_data.empty:
                        all_data.append(gw_data)

        except Exception as error:
            print(f"Error loading gameweek data: {error}")

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            if 'player_id' in combined.columns:
                combined = combined.drop_duplicates(
                    subset=['player_id', 'round']
                )
            return combined

        return pd.DataFrame()