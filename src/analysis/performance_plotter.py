"""Module for creating model performance visualisations.

matplotlib is imported inside the plotting call rather than at module
load. main.py imports this module, so every optimiser run paid for
pyplot - a couple of seconds on a small container - to draw a chart the
web app never asks for.
"""

import os
import pandas as pd
from datetime import datetime


def _pyplot():
    """Import matplotlib on first use and return pyplot."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


class PerformancePlotter:
    """Handles creation of model performance visualisations."""
    
    def __init__(self, config):
        self.config = config
        self.output_dir = "model_performance"
        self._ensure_output_dir()
    
    def _ensure_output_dir(self):
        """Create model performance directory if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_model_performance_data(self, current_gameweek):
        """
        Extract model performance from historical summary files.
        Returns cumulative points by gameweek.
        
        Args:
            current_gameweek (int): Current gameweek number
            
        Returns:
            pd.DataFrame: Model performance by gameweek (cumulative)
        """
        model_data = []
        cumulative_points = 0
        
        # Start at gameweek 0 with 0 points
        model_data.append({
            'gameweek': 0,
            'model_points': 0
        })
        
        for gw in range(1, current_gameweek):
            summary_file = f"squads/gw{gw}/summary.csv"
            
            if os.path.exists(summary_file):
                try:
                    df = pd.read_csv(summary_file)
                    
                    # Find total points metric
                    total_metric = df[
                        df['metric'] == 'Starting XI Total Points'
                    ]
                    
                    if len(total_metric) > 0:
                        actual_points = total_metric.iloc[0]['actual']
                        cumulative_points += int(actual_points)
                        model_data.append({
                            'gameweek': gw,
                            'model_points': cumulative_points
                        })
                        
                except Exception as e:
                    if self.config.GRANULAR_OUTPUT:
                        print(f"Warning: Could not read GW{gw} "
                              f"summary: {e}")
        
        return pd.DataFrame(model_data)
    
    def create_performance_plot(self, standings_df, model_df, 
                               current_gameweek):
        """
        Create a line plot comparing model performance to global 
        rankings.
        All lines start at 0 points at gameweek 0.
        
        Args:
            standings_df (pd.DataFrame): Global standings data by 
                                        gameweek (cumulative)
            model_df (pd.DataFrame): Model performance data by gameweek 
                                    (cumulative)
            current_gameweek (int): Current gameweek number
        """
        plt = _pyplot()

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot global ranking cutoffs - ONLY ACTUAL DATA
        if not standings_df.empty:
            # Plot top 10k - only actual data points
            if 'top_10k' in standings_df.columns:
                actual_10k = standings_df[
                    standings_df['data_quality'] == 'actual'
                ]
                
                if not actual_10k.empty:
                    ax.plot(
                        actual_10k['gameweek'], 
                        actual_10k['top_10k'], 
                        'o-', 
                        color='#2ecc71', 
                        linewidth=2, 
                        markersize=6,
                        label='Top 10k', 
                        zorder=3
                    )
            
            # Plot top 100k - only actual data points
            if 'top_100k' in standings_df.columns:
                actual_100k = standings_df[
                    standings_df['data_quality'] == 'actual'
                ]
                
                if not actual_100k.empty:
                    ax.plot(
                        actual_100k['gameweek'], 
                        actual_100k['top_100k'], 
                        'o-', 
                        color='#e74c3c', 
                        linewidth=2, 
                        markersize=6,
                        label='Top 100k', 
                        zorder=3
                    )
            
            # Plot percentiles - grey dotted lines
            percentiles = [
                ('p01', 'Top 1%'),
                ('p10', 'Top 10%'),
                ('p25', 'Top 25%'),
                ('p50', 'Top 50%'),
                ('p75', 'Top 75%')
            ]
            
            for col, label in percentiles:
                if col in standings_df.columns:
                    actual_percentile = standings_df[
                        standings_df['data_quality'] == 'actual'
                    ]
                    if not actual_percentile.empty:
                        ax.plot(
                            actual_percentile['gameweek'], 
                            actual_percentile[col], 
                            'o:', 
                            color='#7f8c8d',  # Grey colour
                            linewidth=1.5, 
                            markersize=4,
                            label=label, 
                            zorder=2, 
                            alpha=0.6
                        )
        
        # Plot model performance
        if not model_df.empty:
            ax.plot(
                model_df['gameweek'], 
                model_df['model_points'], 
                'o-', 
                color='#9b59b6', 
                linewidth=2.5, 
                markersize=7,
                label='Your Model', 
                zorder=4
            )
        
        # Styling
        ax.set_xlabel('Gameweek', fontsize=12, fontweight='bold')
        ax.set_ylabel(
            'Total Points (Cumulative)', 
            fontsize=12, 
            fontweight='bold'
        )
        ax.set_title(
            'Model Performance vs Global Rankings', 
            fontsize=14, 
            fontweight='bold', 
            pad=20
        )
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Legend
        ax.legend(loc='upper left', framealpha=0.9, fontsize=10)
        
        # X-axis: start at 0 and show all gameweeks up to current
        ax.set_xticks(range(0, current_gameweek))
        ax.set_xlim(-0.5, current_gameweek - 0.5)
        
        # Y-axis: start at 0
        ax.set_ylim(bottom=0)
        
        # Tight layout
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(
            self.output_dir, f'model_plot_gw{current_gameweek}.jpeg')
        
        # Check if file exists and prompt user (always prompt)
        if os.path.exists(output_path):
            print(f"\n⚠️  Plot already exists at {output_path}")
            response = input(
                "Overwrite existing plot? (y/n): "
            ).lower().strip()
            
            if response not in ['y', 'yes']:
                print("Plot generation cancelled.")
                plt.close()
                return None
        
        plt.savefig(
            output_path, 
            dpi=150, 
            bbox_inches='tight', 
            facecolor='white', 
            edgecolor='none'
        )
        plt.close()
        
        if self.config.GRANULAR_OUTPUT:
            print(f"✅ Performance plot saved to {output_path}")
        
        return output_path
    
    def generate_performance_report(self, standings_df, model_df, 
                                   current_gameweek):
        """
        Generate a complete performance report with plot and summary 
        stats.
        
        Args:
            standings_df (pd.DataFrame): Global standings data
            model_df (pd.DataFrame): Model performance data
            current_gameweek (int): Current gameweek number
            
        Returns:
            dict: Performance statistics
        """
        # Create the plot
        plot_path = self.create_performance_plot(
            standings_df, model_df, current_gameweek
        )
        
        # If plot generation was cancelled, return empty stats
        if plot_path is None:
            return {'plot_cancelled': True}
        
        # Calculate performance statistics
        stats = {}
        
        if not model_df.empty:
            latest_gw = model_df['gameweek'].max()
            latest_points = model_df[
                model_df['gameweek'] == latest_gw
            ]['model_points'].iloc[0]
            
            stats['latest_gameweek'] = latest_gw
            stats['model_points'] = latest_points
            
            # Compare to rankings
            if not standings_df.empty:
                latest_standings = standings_df[
                    standings_df['gameweek'] == latest_gw
                ]
                
                if not latest_standings.empty:
                    top_10k = latest_standings['top_10k'].iloc[0]
                    top_50k = latest_standings.get(
                        'top_50k', 
                        pd.Series([None])
                    ).iloc[0]
                    top_100k = latest_standings['top_100k'].iloc[0]
                    
                    stats['top_10k_cutoff'] = top_10k
                    if top_50k is not None:
                        stats['top_50k_cutoff'] = top_50k
                    stats['top_100k_cutoff'] = top_100k
                    
                    # Calculate gaps
                    stats['gap_to_10k'] = latest_points - top_10k
                    if top_50k is not None:
                        stats['gap_to_50k'] = latest_points - top_50k
                    stats['gap_to_100k'] = latest_points - top_100k
                    
                    # Determine ranking tier
                    if latest_points >= top_10k:
                        stats['ranking_tier'] = 'Top 10k pace'
                    elif (top_50k is not None and 
                          latest_points >= top_50k):
                        stats['ranking_tier'] = 'Top 50k pace'
                    elif latest_points >= top_100k:
                        stats['ranking_tier'] = 'Top 100k pace'
                    else:
                        stats['ranking_tier'] = 'Below Top 100k pace'
        
        stats['plot_path'] = plot_path
        
        return stats
    
    def print_performance_summary(self, stats):
        """
        Print performance summary to console.
        
        Args:
            stats (dict): Performance statistics
        """
        if not self.config.GRANULAR_OUTPUT:
            return
        
        # Check if plot was cancelled
        if stats.get('plot_cancelled'):
            return
        
        print(f"\n=== MODEL PERFORMANCE SUMMARY ===")
        
        if 'latest_gameweek' in stats:
            print(f"\nLatest Data: GW{stats['latest_gameweek']}")
            print(f"Your Model: {stats['model_points']} pts")
            
            if 'ranking_tier' in stats:
                print(f"\nCurrent Pace: {stats['ranking_tier']}")
                
                print(f"\nGaps to Cutoffs:")
                if 'gap_to_10k' in stats:
                    gap = stats['gap_to_10k']
                    print(f"  Top 10k:  {gap:+.0f} pts (cutoff: "
                          f"{stats['top_10k_cutoff']:.0f})")
                if 'gap_to_100k' in stats:
                    gap = stats['gap_to_100k']
                    print(f"  Top 100k: {gap:+.0f} pts (cutoff: "
                          f"{stats['top_100k_cutoff']:.0f})")