"""FPL API client for fetching data from the Fantasy Premier League API.

Responses are cached in-process and shared by every caller. A single run asks
for ``bootstrap-static`` from four or five different places — the player
processor, the fixture manager twice, the history tracker, the differential
analyser — and that payload is several megabytes each time. With more than
one manager optimising, the same download was being repeated for each of
them, so the cache is worth more the busier things get.

Set ``FPL_CACHE_SECONDS=0`` to switch it off.
"""

import os
import threading
import time

import requests


class FPLClient:
    """Client for interacting with the Fantasy Premier League API."""

    BASE_URL = "https://fantasy.premierleague.com/api"

    # Shared across instances on purpose: components each build their own
    # client, and they are all asking for the same public data.
    _cache: dict = {}
    _lock = threading.Lock()
    _session = None

    @classmethod
    def cache_seconds(cls) -> int:
        try:
            return int(os.getenv("FPL_CACHE_SECONDS", "600"))
        except ValueError:
            return 600

    @classmethod
    def clear_cache(cls) -> None:
        """Drop everything held. Prices change overnight; nothing else does."""
        with cls._lock:
            cls._cache.clear()

    @classmethod
    def _http(cls):
        # One session, so connections are reused rather than renegotiated.
        if cls._session is None:
            cls._session = requests.Session()
        return cls._session

    def get_json(self, url: str) -> dict:
        """
        Fetch JSON data from a URL, reusing a recent response when there is one.

        Args:
            url (str): The URL to fetch JSON data from.

        Returns:
            dict: The JSON response as a dictionary.

        Raises:
            requests.RequestException: If the request fails.
        """
        ttl = self.cache_seconds()
        if ttl > 0:
            with self._lock:
                entry = self._cache.get(url)
            if entry is not None and (time.monotonic() - entry[0]) < ttl:
                return entry[1]

        try:
            response = self._http().get(url, timeout=30)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as e:
            print(f"Error fetching data from {url}: {e}")
            raise

        if ttl > 0:
            with self._lock:
                self._cache[url] = (time.monotonic(), payload)
        return payload

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