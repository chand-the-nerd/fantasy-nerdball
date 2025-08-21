"""Clean display utilities for GRANULAR_OUTPUT = False mode."""

import pandas as pd
from datetime import datetime

class CleanSquadDisplayUtils:
    """Handles clean, dashboard-style output formatting."""
    
    # Use exactly 80 characters to match terminal width
    OUTER_WIDTH = 80
    INNER_WIDTH = 76          # 80 - 4 for borders "│ " and " │"
    IBOX_WIDTH = 74           # Inner box total width
    IBOX_CONTENT = 70         # Content area within inner box

    def __init__(self, config):
        self.config = config
        self._token_manager = None

    def set_token_manager(self, token_manager):
        """Set the token manager for chip detection."""
        self._token_manager = token_manager

    def pad_to_width(self, text, width=80):
        """Ensure text is exactly `width` characters."""
        text = str(text)
        if len(text) > width:
            return text[:width]
        elif len(text) < width:
            return text + " " * (width - len(text))
        return text

    def format_line(self, content, align='left'):
        """Format a line to exactly 80 characters with outer borders."""
        content = str(content).strip()
        
        # Truncate content if too long
        if len(content) > self.INNER_WIDTH:
            content = content[:self.INNER_WIDTH]
        
        if align == 'center':
            padding_total = self.INNER_WIDTH - len(content)
            left_pad = padding_total // 2
            right_pad = padding_total - left_pad
            line = f"│ {' ' * left_pad}{content}{' ' * right_pad} │"
        else:
            padding = self.INNER_WIDTH - len(content)
            line = f"│ {content}{' ' * padding} │"

        return self.pad_to_width(line, self.OUTER_WIDTH)
    
    def format_empty_line(self):
        return self.pad_to_width(
            "│" + " " * (self.OUTER_WIDTH - 2) + "│", self.OUTER_WIDTH)
    
    def format_border_top(self):
        return self.pad_to_width(
            "╭" + "─" * (self.OUTER_WIDTH - 2) + "╮", self.OUTER_WIDTH)
    
    def format_border_bottom(self):
        return self.pad_to_width(
            "╰" + "─" * (self.OUTER_WIDTH - 2) + "╯", self.OUTER_WIDTH)
    
    def format_separator(self):
        return self.pad_to_width(
            "├" + "─" * (self.OUTER_WIDTH - 2) + "┤", self.OUTER_WIDTH)

    # ---------- inner mini-box methods ----------
    def _ibox_top(self, title=""):
        """Build inner box top line that's exactly IBOX_WIDTH characters."""
        if title:
            title_str = str(title)
            prefix = "┌─ "
            suffix = " ┐"
            

            space_for_dashes = (
                self.IBOX_WIDTH
                - len(prefix)
                - len(title_str)
                - 1
                - len(suffix)
                )
            
            # Ensure we have at least some dashes
            if space_for_dashes < 1:
                # Title too long, truncate it
                max_title_length = (self.IBOX_WIDTH
                                    - len(prefix)
                                    - 1
                                    - 1
                                    - len(suffix)
                                    )  # Leave space for 1 dash
                title_str = title_str[:max_title_length]
                space_for_dashes = 1
            
            line = prefix + title_str + " " + "─" * space_for_dashes + suffix
        else:
            line = "┌" + "─" * (self.IBOX_WIDTH - 2) + "┐"
        
        return self.pad_to_width(line, self.IBOX_WIDTH)

    def _ibox_bottom(self):
        line = "└" + "─" * (self.IBOX_WIDTH - 2) + "┘"
        return self.pad_to_width(line, self.IBOX_WIDTH)

    def _ibox_inner(self, content):
        content_str = str(content).strip()
        
        # Ensure content fits
        if len(content_str) > self.IBOX_CONTENT:
            content_str = content_str[:self.IBOX_CONTENT]
        
        # Pad content to exact width
        padded_content = content_str + " " * (
            self.IBOX_CONTENT - len(content_str)
            )
        
        line = "│ " + padded_content + " │"
        return self.pad_to_width(line, self.IBOX_WIDTH)

    def format_box_line(self, content, box_type='inner'):
        """Return an outer-framed line containing inner box."""
        if box_type == 'top':
            inner_line = self._ibox_top(str(content))
        elif box_type == 'bottom':
            inner_line = self._ibox_bottom()
        else:
            inner_line = self._ibox_inner(str(content))

        # Center the inner box within the outer frame
        padding_needed = self.INNER_WIDTH - len(inner_line)
        left_pad = padding_needed // 2
        right_pad = padding_needed - left_pad
        
        centered_content = " " * left_pad + inner_line + " " * right_pad
        line = "│ " + centered_content + " │"
        return self.pad_to_width(line, self.OUTER_WIDTH)

    # ---------- action box methods ----------
    def _action_top(self):
        prefix = "┌─ ACTIONS "
        dashes_needed = self.OUTER_WIDTH - len(prefix) - 1  # -1 for ┐
        line = prefix + "─" * dashes_needed + "┐"
        return self.pad_to_width(line, self.OUTER_WIDTH)

    def _action_bottom(self):
        line = "└" + "─" * (self.OUTER_WIDTH - 2) + "┘"
        return self.pad_to_width(line, self.OUTER_WIDTH)

    def format_action_line(self, content):
        content_str = str(content).strip()
        max_content = self.OUTER_WIDTH - 4  # -4 for "│ " and " │"
        
        if len(content_str) > max_content:
            content_str = content_str[:max_content]
        
        padded_content = content_str + " " * (max_content - len(content_str))
        line = "│ " + padded_content + " │"
        return self.pad_to_width(line, self.OUTER_WIDTH)

    # ---------- main printer ----------
    def print_clean_dashboard(self, nerdball_starting, nerdball_bench, 
                              nerdball_points, your_starting, your_bench, 
                              your_points, captain_name, captain_points,
                              action_data):
        lines = []
        
        # Header
        lines.append(self.format_border_top())
        header = f"NERDBALL                     GW{self.config.GAMEWEEK} | "
        header += datetime.now().strftime('%b %d, %Y')
        lines.append(self.format_line(header))
        lines.append(self.format_separator())
        lines.append(self.format_empty_line())
        
        # Nerdball XI section
        lines.append(self.format_box_line("NERDBALL XI", 'top'))
        content = f"{nerdball_points:.1f} pts    Theoretical maximum squad"
        lines.append(self.format_box_line(content))
        lines.append(self.format_box_line("", 'bottom'))
        lines.append(self.format_empty_line())
        
        # Formation
        if nerdball_starting is not None and not nerdball_starting.empty:
            formation_lines = self._format_formation(nerdball_starting)
            for line in formation_lines:
                lines.append(self.format_line(line, align='center'))
        else:
            lines.append(
                self.format_line("No theoretical squad data", align='center')
                )
        
        lines.append(self.format_empty_line())
        
        # Nerdball subs
        if nerdball_bench is not None and not nerdball_bench.empty:
            subs_line = self._format_subs_line(nerdball_bench)
            lines.append(self.format_line(subs_line))
        else:
            lines.append(self.format_line("SUBS: No data available"))
        
        lines.append(self.format_empty_line())
        
        # Your squad section
        lines.append(self.format_box_line("YOUR SQUAD", 'top'))
        
        efficiency = ((
            your_points / nerdball_points * 100
            ) if nerdball_points > 0 else 0)
        gap = nerdball_points - your_points
        
        # Captain line (always first)
        if self._token_manager and self._token_manager.config.TRIPLE_CAPTAIN:
            captain_display_points = captain_points * 3
            lines.append(
                self.format_box_line(f"{captain_name}      "
                                     f"{captain_display_points:.1f} "
                                     "pts triple captain"))
        else:
            captain_display_points = captain_points * 2
            lines.append(self.format_box_line(f"{captain_name}      "
                                              f"{captain_display_points:.1f} "
                                              "pts captain"))

        # Bench boost line (if active)
        if self._token_manager and self._token_manager.config.BENCH_BOOST:
            bench_points = sum(
                player.get(
                'projected_points', 0) for _, player in your_bench.iterrows())
            lines.append(self.format_box_line(f"Bench Boost    + "
                                              f"{bench_points:.1f} pts"))

        # Standard lines
        lines.append(self.format_box_line(f"{your_points:.1f} pts    "
                                          "Current optimisation"))
        lines.append(self.format_box_line(f"{efficiency:.1f}%       "
                                          "Efficiency vs Nerdball XI"))
        lines.append(self.format_box_line(f"{gap:.1f}         "
                                          "Points gap to Nerdball XI"))
        
        lines.append(self.format_box_line("", 'bottom'))
        lines.append(self.format_empty_line())
        
        # Your starting XI
        lines.append(self.format_line("STARTING XI"))
        lines.append(self.format_line("───────────"))
        
        for _, player in your_starting.iterrows():
            player_line = self._format_player_line(player)
            lines.append(self.format_line(player_line))
        
        lines.append(self.format_empty_line())
        
        # Your subs
        lines.append(self.format_line("SUBS"))
        lines.append(self.format_line("────"))
        
        for _, player in your_bench.iterrows():
            player_line = self._format_player_line(player, is_sub=True)
            lines.append(self.format_line(player_line))
        
        lines.append(self.format_empty_line())
        lines.append(self.format_border_bottom())
        lines.append("")
        
        # Actions box - remove emojis from action data
        lines.append(self._action_top())
        for action in action_data:
            # Replace emojis with dashes
            clean_action = action.replace(
                '✅', '-').replace('💡', '-').replace('⚠️', '-')
            lines.append(self.format_action_line(clean_action))
        lines.append(self._action_bottom())
        
        # Print all lines
        for line in lines:
            print(line)

    # ---------- content formatting helpers ----------
    def _format_formation(self, starting_df):
        """Format formation layout centered between outer borders."""
        formation = {'GK': [], 'DEF': [], 'MID': [], 'FWD': []}
        
        for _, player in starting_df.iterrows():
            pos = player['position']
            name = player['display_name']
            points = player.get('proj_pts', player.get('projected_points', 0))
            formation[pos].append(f"{name} ({points:.1f})")
        
        lines = []
        # Use consistent 4-character prefix for all positions
        if formation['GK']:
            lines.append(f"GK:  {formation['GK'][0]}")
        # DEF
        if formation['DEF']:
            def_line = "DEF: " + "  ".join(formation['DEF'])
            if len(def_line) > 70:
                lines.append("DEF: " + "  ".join(formation['DEF'][:2]))
                if len(formation['DEF']) > 2:
                    lines.append("     " + "  ".join(formation['DEF'][2:]))
            else:
                lines.append(def_line)
        # MID
        if formation['MID']:
            mid_line = "MID: " + "  ".join(formation['MID'])
            if len(mid_line) > 70:
                lines.append("MID: " + "  ".join(formation['MID'][:2]))
                if len(formation['MID']) > 2:
                    lines.append("     " + "  ".join(formation['MID'][2:]))
            else:
                lines.append(mid_line)
        # FWD
        if formation['FWD']:
            lines.append("FWD: " + "  ".join(formation['FWD']))
        
        return lines
    
    def _format_subs_line(self, bench_df):
        subs = []
        for _, player in bench_df.iterrows():
            name = player['display_name']
            points = player.get('proj_pts', player.get('projected_points', 0))
            subs.append(f"{name} ({points:.1f})")
        
        subs_text = "SUBS: " + " ".join(subs)
        if len(subs_text) > 70:
            subs_text = subs_text[:67] + "..."
        return subs_text
    
    def _format_player_line(self, player, is_sub=False):
        pos_abbrev = {
            'GK': 'GK', 'DEF': 'DEF', 'MID': 'MID', 'FWD': 'FWD'
        }.get(player['position'], 'UNK')
        
        name = str(player['display_name'])
        # Truncate name if too long
        if len(name) > 18:
            name = name[:18]
        
        # Pad name to consistent width
        name_field = name + " " * (18 - len(name))
        
        points = player.get('proj_pts', player.get('projected_points', 0))
        opponent = player.get('next_opponent', 'Unknown')
        
        # Check if unavailable - only based on status, not points
        unavailable = ""
        if ('status' in player and player['status'] != 'a'):
            unavailable = " [UNAVAIL]"
        
        points_str = f"{points:>4.1f}"
        line = (f"{pos_abbrev} {name_field} {points_str}    "
        f"vs {opponent}{unavailable}")
        
        if len(line) > 70:
            line = line[:67] + "..."
        
        return line
    
    def get_captain_info(self, starting_df):
        captain_mask = starting_df[
            'display_name'].str.contains(r'\(C\)', na=False)
        if captain_mask.any():
            captain = starting_df[captain_mask].iloc[0]
            captain_name = captain['display_name'].replace(' (C)', '')
            captain_points = captain.get(
                'proj_pts', captain.get('projected_points', 0))
            return captain_name, captain_points
        return "Unknown", 0.0