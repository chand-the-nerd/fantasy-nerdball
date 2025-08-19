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

        Raises:
            requests.RequestException: If the request fails.
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching data from {url}: {e}")
            raise
    
    def get_bootstrap_static(self) -> dict:
        """
        Get the main bootstrap data containing players, teams, and gameweeks.
        
        Returns:
            dict: Bootstrap static data from FPL API.
        """
        return self.get_json(f"{self.BASE_URL}/bootstrap-static/")
    
    def get_fixtures(self) -> dict:
        """
        Get fixture data.
        
        Returns:
            dict: Fixture data from FPL API.
        """
        return self.get_json(f"{self.BASE_URL}/fixtures/")
    
    def get_player_summary(self, player_id: int) -> dict:
        """
        Get detailed summary for a specific player.
        
        Args:
            player_id (int): The FPL player ID.
            
        Returns:
            dict: Player summary data from FPL API.
        """
        return self.get_json(f"{self.BASE_URL}/element-summary/{player_id}/")