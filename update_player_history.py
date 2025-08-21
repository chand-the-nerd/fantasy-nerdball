#!/usr/bin/env python3
"""
Standalone script to update player history data.
Run this after each gameweek to capture the previous gameweek's data.
"""

import sys
import logging
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
    
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "update":
            print(f"Updating all players with GW{config.GAMEWEEK - 1} data "
                  "(with overwrite)...")
            logging.info(f"About to update all players for "
                         f"GW{config.GAMEWEEK - 1} with overwrite")
            
            try:
                # FORCE OVERWRITE
                tracker.update_all_players(force_overwrite=True)
                logging.info(f"Update completed with overwrite")
                print(f"Update completed!")
                
                # Show some stats after update
                stats = tracker.get_summary_stats()
                print(f"Total records now: "
                      f"{stats.get('total_records', 'Unknown')}")
                
            except Exception as e:
                logging.error(f"Error during update: {str(e)}", exc_info=True)
                print(f"Error during update: {str(e)}")
                
        elif command == "update-safe":
            # New command for non-overwrite update
            print(f"Updating all players with GW{config.GAMEWEEK - 1} data "
                  "(safe mode - no overwrite)...")
            logging.info(f"About to update all players for "
                         f"GW{config.GAMEWEEK - 1} without overwrite")
            
            try:
                # Update all players with previous gameweek data - NO OVERWRITE
                tracker.update_all_players(force_overwrite=False)
                logging.info(f"Safe update completed")
                print(f"Safe update completed!")
                
                # Show some stats after update
                stats = tracker.get_summary_stats()
                print(f"Total records now: "
                      f"{stats.get('total_records', 'Unknown')}")
                
            except Exception as e:
                logging.error(f"Error during safe update: "
                              f"{str(e)}", exc_info=True)
                print(f"Error during safe update: {str(e)}")
                
        elif command == "debug":
            # Debug command to check current state
            print(f"\n=== Debug Information ===")
            try:
                stats = tracker.get_summary_stats()
                print(f"Current statistics: {stats}")
                
                # Check if data exists for previous gameweek
                prev_gw = config.GAMEWEEK - 1
                print(f"Checking for existing GW{prev_gw} data...")
                
                logging.info(f"Debug info - Stats: {stats}")
                
            except Exception as e:
                logging.error(f"Error in debug: {str(e)}", exc_info=True)
                print(f"Debug error: {str(e)}")
            
        elif command == "force":
            # Force update even if data exists (same as 'update' now)
            print(f"Force updating GW{config.GAMEWEEK - 1} data...")
            logging.info(f"Force update initiated for GW{config.GAMEWEEK - 1}")
            
            try:
                tracker.update_all_players(force_overwrite=True)
                logging.info(f"Force update completed")
                print(f"Force update completed!")
                
            except Exception as e:
                logging.error(f"Error during force update: "
                              f"{str(e)}", exc_info=True)
                print(f"Error during force update: {str(e)}")
                
        elif command == "stats":
            # Show summary statistics
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
            
        elif command == "cleanup":
            # Clean up old data (keep last 38 gameweeks)
            keep_weeks = 38
            if len(sys.argv) > 2:
                try:
                    keep_weeks = int(sys.argv[2])
                except ValueError:
                    print("Invalid number for cleanup. Using default 38 "
                          "gameweeks.")
            
            try:
                tracker.cleanup_old_data(keep_weeks)
                print(f"Cleanup completed - kept last {keep_weeks} gameweeks")
                
            except Exception as e:
                logging.error(f"Error during cleanup: {str(e)}", exc_info=True)
                print(f"Error during cleanup: {str(e)}")
            
        elif command == "player":
            # Show specific player data
            if len(sys.argv) < 4:
                print("Usage: python update_player_history.py player "
                      "<player_name> <team_name>")
                print("Example: python update_player_history.py player "
                      "cunha wolves")
                return
            
            player_name = sys.argv[2]
            team_name = sys.argv[3]
            
            try:
                history = tracker.get_player_history(player_name, team_name)
                if history.empty:
                    print(f"No data found for {player_name} at {team_name}")
                else:
                    print(f"\n=== {player_name} ({team_name}) History ===")
                    print(history[[
                        'round',
                        'total_points',
                        'minutes',
                        'goals_scored',
                        'assists'
                        ]].to_string(index=False))
                    
            except Exception as e:
                logging.error(f"Error getting player data: "
                              f"{str(e)}", exc_info=True)
                print(f"Error getting player data: {str(e)}")
                
        elif command == "team":
            # Show team data
            if len(sys.argv) < 3:
                print("Usage: python update_player_history.py team "
                      "<team_name>")
                print("Example: python update_player_history.py team wolves")
                return
            
            team_name = sys.argv[2]
            last_gw = config.GAMEWEEK - 1
            
            try:
                team_data = tracker.get_team_history(
                    team_name, gameweeks=[last_gw])
                if team_data.empty:
                    print(f"No data found for {team_name} in GW{last_gw}")
                else:
                    print(f"\n=== {team_name} GW{last_gw} Performance ===")
                    summary = team_data[[
                        'player_name',
                        'total_points',
                        'minutes',
                        'goals_scored',
                        'assists']].sort_values(
                            'total_points',
                            ascending=False
                            )
                    print(summary.to_string(index=False))
                    
            except Exception as e:
                logging.error(f"Error getting team data: "
                              f"{str(e)}", exc_info=True)
                print(f"Error getting team data: {str(e)}")
        else:
            print("Unknown command. See usage below.")
            show_usage()
    else:
        show_usage()


def show_usage():
    """Show usage instructions."""
    print(f"\nUsage:")
    print(f"  python update_player_history.py update          "
          "# Update all players with previous GW data (with overwrite)")
    print(f"  python update_player_history.py update-safe     "
          "# Update all players but skip existing data")
    print(f"  python update_player_history.py debug           "
          "# Show debug information")
    print(f"  python update_player_history.py force           "
          "# Force update even if data exists (same as 'update')")
    print(f"  python update_player_history.py stats           "
          "# Show summary statistics")
    print(f"  python update_player_history.py cleanup [weeks] "
          "# Clean up old data (default: keep 38 weeks)")
    print(f"  python update_player_history.py player <name> <team>  "
          "# Show player history")
    print(f"  python update_player_history.py team <team>     "
          "# Show team's last gameweek performance")
    print(f"\nExamples:")
    print(f"  python update_player_history.py update")
    print(f"  python update_player_history.py update-safe")
    print(f"  python update_player_history.py debug")
    print(f"  python update_player_history.py force")
    print(f"  python update_player_history.py player cunha wolves")
    print(f"  python update_player_history.py team 'man utd'")
    print(f"  python update_player_history.py cleanup 20")


if __name__ == "__main__":
    main()