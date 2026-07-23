#!/usr/bin/env python3
"""
Cost Calculator - Calculate current value of historical squads

This script takes a gameweek number and calculates what the squad selected
in that gameweek would cost at current prices, showing price changes and
total value gained/lost.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from src.api.fpl_client import FPLClient


class SquadCostCalculator:
    """Calculate current cost of historical squads."""
    
    def __init__(self, config):
        self.config = config
        self.fpl_client = FPLClient()
        self.current_prices = {}
        self.team_names = {}
        
    def load_current_prices(self):
        """Load current player prices from FPL API."""
        try:
            print("Fetching current player prices from FPL API...")
            bootstrap_data = self.fpl_client.get_bootstrap_static()
            
            players = bootstrap_data['elements']
            teams = bootstrap_data['teams']
            
            # Create team name mapping
            self.team_names = {team['id']: team['name'] for team in teams}
            
            # Store current prices by player ID
            for player in players:
                self.current_prices[player['id']] = {
                    'name': player['web_name'],
                    'team': self.team_names.get(player['team'], 'Unknown'),
                    'position': self._get_position_name(player['element_type']),
                    'current_cost': player['now_cost'] / 10.0,  # Convert to millions
                }
            
            print(f"✓ Loaded prices for {len(self.current_prices)} players")
            return True
            
        except Exception as e:
            print(f"❌ Error loading current prices: {e}")
            return False
    
    def _get_position_name(self, position_id):
        """Convert position ID to name."""
        position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        return position_map.get(position_id, 'UNK')
    
    def load_historical_squad(self, gameweek):
        """
        Load squad from a specific gameweek.
        
        Args:
            gameweek (int): Gameweek number to load
            
        Returns:
            pd.DataFrame or None: Squad data if found
        """
        squad_file = f"squads/gw{gameweek}/full_squad.csv"
        
        if not os.path.exists(squad_file):
            print(f"❌ Squad file not found: {squad_file}")
            return None
        
        try:
            squad = pd.read_csv(squad_file)
            print(f"✓ Loaded squad from GW{gameweek} ({len(squad)} players)")
            return squad
        except Exception as e:
            print(f"❌ Error loading squad: {e}")
            return None
    
    def calculate_squad_costs(self, squad_df, gameweek):
        """
        Calculate costs for each player in the squad.
        
        Args:
            squad_df (pd.DataFrame): Historical squad data
            gameweek (int): Gameweek number
            
        Returns:
            pd.DataFrame: Squad with cost analysis
        """
        results = []
        
        for _, player in squad_df.iterrows():
            player_id = player.get('id')
            original_cost = player.get('now_cost_m', 0)
            player_name = player.get('display_name', 'Unknown')
            
            if pd.isna(player_id):
                # No player ID - can't look up current price
                results.append({
                    'name': player_name,
                    'position': player.get('position', 'UNK'),
                    'team': player.get('team', 'Unknown'),
                    'original_cost': original_cost,
                    'current_cost': None,
                    'price_change': None,
                    'squad_role': player.get('squad_role', 'Unknown'),
                    'status': 'No ID'
                })
                continue
            
            # Look up current price
            current_info = self.current_prices.get(int(player_id))
            
            if current_info is None:
                # Player no longer in FPL (transferred, retired, etc.)
                results.append({
                    'name': player_name,
                    'position': player.get('position', 'UNK'),
                    'team': player.get('team', 'Unknown'),
                    'original_cost': original_cost,
                    'current_cost': None,
                    'price_change': None,
                    'squad_role': player.get('squad_role', 'Unknown'),
                    'status': 'Not in FPL'
                })
            else:
                # Calculate price change
                current_cost = current_info['current_cost']
                price_change = current_cost - original_cost
                
                results.append({
                    'name': player_name,
                    'position': player.get('position', 'UNK'),
                    'team': player.get('team', 'Unknown'),
                    'original_cost': original_cost,
                    'current_cost': current_cost,
                    'price_change': price_change,
                    'squad_role': player.get('squad_role', 'Unknown'),
                    'status': 'Active'
                })
        
        return pd.DataFrame(results)
    
    def print_cost_analysis(self, results_df, gameweek):
        """
        Print formatted cost analysis.
        
        Args:
            results_df (pd.DataFrame): Cost analysis results
            gameweek (int): Gameweek number
        """
        print("\n" + "=" * 100)
        print(f"SQUAD COST ANALYSIS - GW{gameweek} vs CURRENT")
        print("=" * 100)
        
        # Calculate totals
        original_total = results_df['original_cost'].sum()
        
        # Only include active players in current total
        active_players = results_df[results_df['status'] == 'Active']
        current_total = active_players['current_cost'].sum()
        total_change = current_total - original_total
        
        print(f"\nOriginal Squad Cost (GW{gameweek}): £{original_total:.1f}m")
        print(f"Current Squad Cost: £{current_total:.1f}m")
        
        if total_change > 0:
            print(f"Total Value Gained: +£{total_change:.1f}m 📈")
        elif total_change < 0:
            print(f"Total Value Lost: £{total_change:.1f}m 📉")
        else:
            print(f"Total Value Change: £{total_change:.1f}m ➡️")
        
        # Show unavailable players
        unavailable = results_df[results_df['status'] != 'Active']
        if not unavailable.empty:
            print(f"\n⚠️  {len(unavailable)} player(s) no longer in FPL or missing data")
        
        # Separate starting XI and bench
        starting_xi = results_df[results_df['squad_role'] == 'Starting XI']
        bench = results_df[results_df['squad_role'] == 'Bench']
        
        # Print Starting XI
        print("\n" + "-" * 100)
        print("STARTING XI")
        print("-" * 100)
        self._print_player_table(starting_xi)
        
        # Print Bench
        if not bench.empty:
            print("\n" + "-" * 100)
            print("BENCH")
            print("-" * 100)
            self._print_player_table(bench)
        
        # Print summary by position
        print("\n" + "-" * 100)
        print("SUMMARY BY POSITION")
        print("-" * 100)
        self._print_position_summary(results_df)
        
        # Print biggest gainers/losers
        print("\n" + "-" * 100)
        print("BIGGEST MOVERS")
        print("-" * 100)
        self._print_biggest_movers(results_df)
        
        print("\n" + "=" * 100)
    
    def _print_player_table(self, df):
        """Print formatted player table."""
        if df.empty:
            print("No players in this category")
            return
        
        print(f"{'Name':<20} {'Pos':<5} {'Team':<15} {'Original':<10} {'Current':<10} {'Change':<10} {'Status':<12}")
        print("-" * 100)
        
        for _, player in df.iterrows():
            name = player['name'][:19]  # Truncate long names
            pos = player['position']
            team = player['team'][:14]  # Truncate long team names
            original = f"£{player['original_cost']:.1f}m"
            
            if pd.isna(player['current_cost']):
                current = "N/A"
                change = "N/A"
            else:
                current = f"£{player['current_cost']:.1f}m"
                change_val = player['price_change']
                if change_val > 0:
                    change = f"+£{change_val:.1f}m 📈"
                elif change_val < 0:
                    change = f"£{change_val:.1f}m 📉"
                else:
                    change = f"£{change_val:.1f}m ➡️"
            
            status = player['status']
            
            print(f"{name:<20} {pos:<5} {team:<15} {original:<10} {current:<10} {change:<10} {status:<12}")
    
    def _print_position_summary(self, df):
        """Print summary statistics by position."""
        active_players = df[df['status'] == 'Active']
        
        print(f"{'Position':<10} {'Players':<10} {'Original Cost':<15} {'Current Cost':<15} {'Change':<15}")
        print("-" * 100)
        
        for position in ['GK', 'DEF', 'MID', 'FWD']:
            pos_players = active_players[active_players['position'] == position]
            
            if pos_players.empty:
                continue
            
            count = len(pos_players)
            original = pos_players['original_cost'].sum()
            current = pos_players['current_cost'].sum()
            change = current - original
            
            change_str = f"+£{change:.1f}m" if change > 0 else f"£{change:.1f}m"
            
            print(f"{position:<10} {count:<10} £{original:.1f}m{'':<9} £{current:.1f}m{'':<9} {change_str:<15}")
    
    def _print_biggest_movers(self, df):
        """Print biggest price gainers and losers."""
        active_players = df[df['status'] == 'Active'].copy()
        
        if active_players.empty:
            print("No active players to analyze")
            return
        
        # Top 5 gainers
        gainers = active_players.nlargest(5, 'price_change')
        print("\nTop 5 Price Gains:")
        for i, (_, player) in enumerate(gainers.iterrows(), 1):
            if player['price_change'] <= 0:
                break
            print(f"  {i}. {player['name']} ({player['position']}, {player['team']}): "
                  f"+£{player['price_change']:.1f}m "
                  f"(£{player['original_cost']:.1f}m → £{player['current_cost']:.1f}m)")
        
        # Top 5 losers
        losers = active_players.nsmallest(5, 'price_change')
        print("\nTop 5 Price Drops:")
        for i, (_, player) in enumerate(losers.iterrows(), 1):
            if player['price_change'] >= 0:
                break
            print(f"  {i}. {player['name']} ({player['position']}, {player['team']}): "
                  f"£{player['price_change']:.1f}m "
                  f"(£{player['original_cost']:.1f}m → £{player['current_cost']:.1f}m)")
    
    def run(self):
        """Main execution flow."""
        print("=" * 100)
        print("FANTASY NERDBALL - SQUAD COST CALCULATOR")
        print("=" * 100)
        
        # Load current prices first
        if not self.load_current_prices():
            return False
        
        # Interactive loop
        while True:
            print("\n" + "=" * 100)
            
            # Prompt for gameweek
            gameweek_input = input("\nEnter gameweek number (or 'quit' to exit): ").strip()
            
            if gameweek_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            try:
                gameweek = int(gameweek_input)
            except ValueError:
                print("❌ Invalid gameweek number. Please enter a number.")
                continue
            
            if gameweek < 1 or gameweek > 38:
                print("❌ Invalid gameweek. Please enter a number between 1 and 38.")
                continue
            
            # Load historical squad
            squad = self.load_historical_squad(gameweek)
            
            if squad is None:
                continue
            
            # Calculate costs
            results = self.calculate_squad_costs(squad, gameweek)
            
            # Print analysis
            self.print_cost_analysis(results, gameweek)
        
        return True


def main():
    """Main entry point."""
    try:
        config = Config()
        calculator = SquadCostCalculator(config)
        calculator.run()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()