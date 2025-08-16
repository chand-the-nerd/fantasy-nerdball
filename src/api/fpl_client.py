"""FPL API client for fetching data from the Fantasy Premier League API."""

import requests


class FPLClient:
    """Client for interacting with the Fantasy Premier League API."""
    
    BASE_URL = "https://fantasy.premierleague.com/api"
    
    def get_json(self, url: str) -> dict:
        """
        Fetch JSON data from a URL.

        Args:
            url (str): The URL to fetch JSON data from.

        Returns:
            dict: The JSON response as a dictionary.
        """
        return requests.get(url).json()
    
    def get_bootstrap_static(self) -> dict:
        """Get the main bootstrap data containing players, teams, and gameweeks."""
        return self.get_json(f"{self.BASE_URL}/bootstrap-static/")
    
    def get_fixtures(self) -> dict:
        """Get fixture data."""
        return self.get_json(f"{self.BASE_URL}/fixtures/")
    
    def get_player_summary(self, player_id: int) -> dict:
        """Get detailed summary for a specific player."""
        return self.get_json(f"{self.BASE_URL}/element-summary/{player_id}/")