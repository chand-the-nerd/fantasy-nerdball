#!/usr/bin/env python3
"""
Standalone script to update player history data.
Run this after each gameweek to capture the previous gameweek's data.
"""

import sys
import logging
import os
from datetime import datetime
from config import Config
from src.data.player_history_tracker import PlayerHistoryTracker


def main():
    """Update player history data for the previous gameweek."""
    
    config = Config()
    tracker = PlayerHistoryTracker(config)
    
    print(f"Fantasy Nerdball - Player History Updater")
    print(f"Current gameweek: {config.GAMEWEEK}")
    print(f"Timestamp: {datetime.now()}")
    
    # Add some debug info
    logging.info(f"Starting update - Current GW: {config.GAMEWEEK}")
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "update":
            handle_update_command(tracker, config)
        elif command == "update-safe":
            handle_update_safe_command(tracker, config)
        elif command == "debug":
            handle_debug_command(tracker, config)
        elif command == "force":
            handle_force_command(tracker, config)
        elif command == "stats":
            handle_stats_command(tracker)
        elif command == "cleanup":
            handle_cleanup_command(tracker, sys.argv)
        elif command == "player":
            handle_player_command(tracker, sys.argv)
        elif command == "team":
            handle_team_command(tracker, sys.argv, config)
        elif command == "verify":
            handle_verify_command(tracker, config)
        elif command == "list-teams":
            handle_list_teams_command(tracker)
        else:
            print("Unknown command. See usage below.")
            show_usage()
    else:
        show_usage()


def handle_update_command(tracker, config):
    """Handle the 'update' command."""
    print(f"Updating all players with GW{config.GAMEWEEK - 1} data "
          f"(with overwrite)...")
    logging.info(f"About to update all players for GW{config.GAMEWEEK - 1} "
                 f"with overwrite")
    
    try:
        # Update all players with previous gameweek data - FORCE OVERWRITE
        tracker.update_all_players(force_overwrite=True)
        logging.info(f"Update completed with overwrite")
        
        # Show some stats after update
        stats = tracker.get_summary_stats()
        print(f"Final totals:")
        print(f"  Players tracked: {stats.get('total_players', 'Unknown')}")
        print(f"  Total gameweek records: {stats.get('total_records', 'Unknown')}")
        print(f"  Teams: {stats.get('total_teams', 'Unknown')}")
        
    except Exception as e:
        logging.error(f"Error during update: {str(e)}", exc_info=True)
        print(f"Error during update: {str(e)}")


def handle_update_safe_command(tracker, config):
    """Handle the 'update-safe' command."""
    print(f"Updating all players with GW{config.GAMEWEEK - 1} data "
          f"(safe mode - no overwrite)...")
    logging.info(f"About to update all players for GW{config.GAMEWEEK - 1} "
                 f"without overwrite")
    
    try:
        # Update all players with previous gameweek data - NO OVERWRITE
        tracker.update_all_players(force_overwrite=False)
        logging.info(f"Safe update completed")
        
        # Show some stats after update
        stats = tracker.get_summary_stats()
        print(f"Final totals:")
        print(f"  Players tracked: {stats.get('total_players', 'Unknown')}")
        print(f"  Total gameweek records: {stats.get('total_records', 'Unknown')}")
        
    except Exception as e:
        logging.error(f"Error during safe update: {str(e)}", exc_info=True)
        print(f"Error during safe update: {str(e)}")


def handle_debug_command(tracker, config):
    """Handle the 'debug' command."""
    print(f"\n=== Debug Information ===")
    try:
        stats = tracker.get_summary_stats()
        print(f"Current statistics: {stats}")
        
        # Check if data exists for previous gameweek
        prev_gw = config.GAMEWEEK - 1
        print(f"\nChecking for existing GW{prev_gw} data...")
        
        # Try to load some sample data for the previous gameweek
        gw_data = tracker.get_all_players_gameweek(prev_gw)
        if not gw_data.empty:
            print(f"Found {len(gw_data)} player records for GW{prev_gw}")
            print(f"Sample players:")
            for _, row in gw_data.head(5).iterrows():
                print(f"  {row['player_name']} ({row['team_name']}): "
                      f"{row['total_points']} pts")
        else:
            print(f"No GW{prev_gw} data found")
        
        logging.info(f"Debug info - Stats: {stats}")
        
    except Exception as e:
        logging.error(f"Error in debug: {str(e)}", exc_info=True)
        print(f"Debug error: {str(e)}")


def handle_force_command(tracker, config):
    """Handle the 'force' command."""
    print(f"Force updating GW{config.GAMEWEEK - 1} data...")
    logging.info(f"Force update initiated for GW{config.GAMEWEEK - 1}")
    
    try:
        tracker.update_all_players(force_overwrite=True)
        logging.info(f"Force update completed")
        print(f"Force update completed!")
        
    except Exception as e:
        logging.error(f"Error during force update: {str(e)}", exc_info=True)
        print(f"Error during force update: {str(e)}")


def handle_stats_command(tracker):
    """Handle the 'stats' command."""
    try:
        stats = tracker.get_summary_stats()
        print(f"\n=== Player History Statistics ===")
        print(f"Total players tracked: {stats['total_players']}")
        print(f"Total gameweek records: {stats['total_records']}")
        print(f"Teams covered: {stats['total_teams']}")
        print(f"Average records per player: "
              f"{stats['average_records_per_player']:.1f}")
            
    except Exception as e:
        logging.error(f"Error getting stats: {str(e)}", exc_info=True)
        print(f"Error getting stats: {str(e)}")


def handle_cleanup_command(tracker, argv):
    """Handle the 'cleanup' command."""
    keep_weeks = 38
    if len(argv) > 2:
        try:
            keep_weeks = int(argv[2])
        except ValueError:
            print("Invalid number for cleanup. Using default 38 gameweeks.")
    
    try:
        tracker.cleanup_old_data(keep_weeks)
        
    except Exception as e:
        logging.error(f"Error during cleanup: {str(e)}", exc_info=True)
        print(f"Error during cleanup: {str(e)}")


def handle_player_command(tracker, argv):
    """Handle the 'player' command."""
    if len(argv) < 4:
        print("Usage: python update_player_history.py player "
              "<player_name> <team_name>")
        print("Example: python update_player_history.py player cunha wolves")
        return
    
    player_name = argv[2]
    team_name = argv[3]
    
    try:
        history = tracker.get_player_history(player_name, team_name)
        if history.empty:
            print(f"No data found for {player_name} at {team_name}")
            print(f"Try: python update_player_history.py list-teams")
        else:
            print(f"\n=== {player_name} ({team_name}) History ===")
            display_cols = ['round', 'total_points', 'minutes', 
                          'goals_scored', 'assists']
            print(history[display_cols].to_string(index=False))
            
    except Exception as e:
        logging.error(f"Error getting player data: {str(e)}", exc_info=True)
        print(f"Error getting player data: {str(e)}")


def handle_team_command(tracker, argv, config):
    """Handle the 'team' command."""
    if len(argv) < 3:
        print("Usage: python update_player_history.py team <team_name>")
        print("Example: python update_player_history.py team wolves")
        print("Try: python update_player_history.py list-teams")
        return
    
    team_name = argv[2]
    last_gw = config.GAMEWEEK - 1
    
    try:
        team_data = tracker.get_team_history(team_name, gameweeks=[last_gw])
        if team_data.empty:
            print(f"No data found for {team_name} in GW{last_gw}")
            print(f"Try: python update_player_history.py list-teams")
        else:
            print(f"\n=== {team_name} GW{last_gw} Performance ===")
            summary_cols = ['player_name', 'total_points', 'minutes', 
                          'goals_scored', 'assists']
            summary = team_data[summary_cols].sort_values(
                'total_points', ascending=False
            )
            print(summary.to_string(index=False))
            
    except Exception as e:
        logging.error(f"Error getting team data: {str(e)}", exc_info=True)
        print(f"Error getting team data: {str(e)}")


def handle_verify_command(tracker, config):
    """Handle the 'verify' command to check CSV data."""
    prev_gw = config.GAMEWEEK - 1
    
    try:
        print(f"Verifying GW{prev_gw} data in CSV files...")
        
        # Get all data for previous gameweek
        gw_data = tracker.get_all_players_gameweek(prev_gw)
        
        if not gw_data.empty:
            print(f"Found {len(gw_data)} player records for GW{prev_gw}")
            
            # Show top performers
            top_performers = gw_data.nlargest(10, 'total_points')
            print(f"\nTop 10 performers in GW{prev_gw}:")
            for _, row in top_performers.iterrows():
                print(f"  {row['player_name']} ({row['team_name']}): "
                      f"{row['total_points']} pts, {row['minutes']} mins")
                      
            # Show some team statistics
            team_counts = gw_data['team_name'].value_counts()
            print(f"\nData coverage by team (top 5):")
            for team, count in team_counts.head(5).items():
                print(f"  {team}: {count} players")
                
        else:
            print(f"No data found for GW{prev_gw}")
            
    except Exception as e:
        print(f"Error verifying data: {str(e)}")


def handle_list_teams_command(tracker):
    """Handle the 'list-teams' command."""
    try:
        base_path = tracker.base_path
        if os.path.exists(base_path):
            teams = [d for d in os.listdir(base_path) 
                    if os.path.isdir(os.path.join(base_path, d))]
            
            if teams:
                print(f"\n=== Available Teams ===")
                for team in sorted(teams):
                    team_path = os.path.join(base_path, team)
                    player_count = len([f for f in os.listdir(team_path) 
                                      if f.endswith('.csv')])
                    print(f"  {team} ({player_count} players)")
            else:
                print("No team data found. Run 'update' first.")
        else:
            print("No data directory found. Run 'update' first.")
            
    except Exception as e:
        print(f"Error listing teams: {str(e)}")


def show_usage():
    """Show usage instructions."""
    print(f"\nUsage:")
    print(f"  python update_player_history.py update          # Update all "
          f"players with previous GW data (with overwrite)")
    print(f"  python update_player_history.py update-safe     # Update all "
          f"players but skip existing data")
    print(f"  python update_player_history.py debug           # Show debug "
          f"information")
    print(f"  python update_player_history.py force           # Force update "
          f"even if data exists (same as 'update')")
    print(f"  python update_player_history.py stats           # Show summary "
          f"statistics")
    print(f"  python update_player_history.py verify          # Verify data "
          f"for previous gameweek")
    print(f"  python update_player_history.py list-teams      # List all "
          f"available teams")
    print(f"  python update_player_history.py cleanup [weeks] # Clean up old "
          f"data (default: keep 38 weeks)")
    print(f"  python update_player_history.py player <name> <team>  # Show "
          f"player history")
    print(f"  python update_player_history.py team <team>     # Show team's "
          f"last gameweek performance")
    print(f"\nExamples:")
    print(f"  python update_player_history.py update")
    print(f"  python update_player_history.py verify")
    print(f"  python update_player_history.py list-teams")
    print(f"  python update_player_history.py player cunha wolves")
    print(f"  python update_player_history.py team arsenal")
    print(f"  python update_player_history.py cleanup 20")


if __name__ == "__main__":
    main()