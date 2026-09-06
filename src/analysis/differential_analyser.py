"""Module for analysing differential picks with low ownership but high
expected performance."""

import pandas as pd


class DifferentialAnalyser:
    """Handles analysis and display of differential player picks."""

    def __init__(self, config):
        self.config = config
        self.ownership_threshold = 5.0  # Players below 5% ownership

    def add_multi_fixture_info(self, df: pd.DataFrame,
                               starting_gameweek: int,
                               num_gameweeks: int,
                               fixture_manager) -> pd.DataFrame:
        """
        Add multi-fixture information showing upcoming opponents across
        several gameweeks, one column per gameweek.

        Args:
            df (pd.DataFrame): Player dataframe
            starting_gameweek (int): Starting gameweek number
            num_gameweeks (int): Number of gameweeks to look ahead
            fixture_manager: FixtureManager instance

        Returns:
            pd.DataFrame: Dataframe with per-gameweek columns added
        """
        df = df.copy()

        if "player_code" not in df.columns:
            raise ValueError(
                "Player data has no 'player_code' column - fixture "
                "columns cannot be joined reliably."
            )

        fixtures_df = pd.DataFrame(fixture_manager.fpl_client
                                   .get_fixtures())

        teams_data = fixture_manager.fpl_client.get_bootstrap_static()
        teams_df = pd.DataFrame(teams_data["teams"])[["id", "name"]]
        teams_dict = dict(zip(teams_df["id"], teams_df["name"]))

        gameweeks = list(range(
            starting_gameweek, starting_gameweek + num_gameweeks
        ))

        # Precompute each team's opponents per gameweek once, rather
        # than filtering the fixture list inside a per-player loop.
        team_opponents = {gw: {} for gw in gameweeks}

        for gw in gameweeks:
            gw_fixtures = fixtures_df[fixtures_df["event"] == gw]

            for _, fixture in gw_fixtures.iterrows():
                home_id = fixture["team_h"]
                away_id = fixture["team_a"]

                team_opponents[gw].setdefault(home_id, []).append(
                    teams_dict.get(away_id, "Unknown")
                )
                team_opponents[gw].setdefault(away_id, []).append(
                    teams_dict.get(home_id, "Unknown")
                )

        fixture_info = []

        for _, player in df.iterrows():
            team_id = player["team_id"]
            player_fixture_data = {
                "player_code": player["player_code"]
            }

            for gw in gameweeks:
                opponents = team_opponents[gw].get(team_id, [])

                gw_col = f"gw{gw}"
                if not opponents:
                    player_fixture_data[gw_col] = "-"
                elif len(opponents) == 1:
                    player_fixture_data[gw_col] = opponents[0]
                else:
                    player_fixture_data[gw_col] = " & ".join(opponents)

            fixture_info.append(player_fixture_data)

        fixture_df = pd.DataFrame(fixture_info)
        fixture_df = fixture_df.drop_duplicates(subset=["player_code"])

        before = len(df)
        df = df.merge(
            fixture_df, on="player_code", how="left", validate="1:1"
        )

        if len(df) != before:
            raise ValueError(
                f"Multi-fixture merge changed row count from {before} "
                f"to {len(df)}."
            )

        for gw in gameweeks:
            col = f"gw{gw}"
            if col in df.columns:
                df[col] = df[col].fillna("-")

        return df

    def get_differential_suggestions(self, scored_players: pd.DataFrame,
                                     starting_gameweek: int = None,
                                     num_gameweeks: int = None) -> dict:
        """
        Get the top three differential suggestions for each position,
        based on FPL score and low ownership.

        Args:
            scored_players (pd.DataFrame): Player data with scores and
                                           ownership
            starting_gameweek (int): Starting gameweek number
            num_gameweeks (int): Number of gameweeks being analysed

        Returns:
            dict: Position keys with the top three differentials each
        """
        differentials = {}

        if 'selected_by_percent' not in scored_players.columns:
            print("Warning: selected_by_percent column not found. "
                  "Cannot generate differential suggestions.")
            return differentials

        df_work = scored_players.copy()
        df_work['ownership_numeric'] = pd.to_numeric(
            df_work['selected_by_percent'], errors='coerce'
        ).fillna(100.0)  # If conversion fails, assume high ownership

        if 'id' in df_work.columns:
            df_work = df_work.drop_duplicates(subset=['id'])
        elif 'player_code' in df_work.columns:
            df_work = df_work.drop_duplicates(subset=['player_code'])

        positions = ["GK", "DEF", "MID", "FWD"]

        for position in positions:
            position_players = df_work[
                (df_work["position"] == position) &
                (df_work["ownership_numeric"] <
                 self.ownership_threshold) &
                (df_work["fpl_score"] > 0)
            ].copy()

            if position_players.empty:
                differentials[position] = []
                continue

            top_differentials = position_players.nlargest(
                3, "fpl_score"
            )

            differentials[position] = self._format_differential_data(
                top_differentials,
                starting_gameweek or self.config.GAMEWEEK,
                num_gameweeks or self.config.FIRST_N_GAMEWEEKS
            )

        return differentials

    def _format_differential_data(self, players: pd.DataFrame,
                                  starting_gameweek: int,
                                  num_gameweeks: int) -> list:
        """
        Format differential player data for display.

        Args:
            players (pd.DataFrame): Top differential players
            starting_gameweek (int): Starting gameweek number
            num_gameweeks (int): Number of gameweeks being analysed

        Returns:
            list: Formatted player dictionaries
        """
        formatted_players = []

        for _, player in players.iterrows():
            ownership_val = player.get(
                'ownership_numeric',
                player.get('selected_by_percent', 0)
            )

            # projected_points is a single gameweek's projection, so it
            # is already the per-gameweek figure.
            proj_pts = player.get('projected_points', 0)

            formatted_player = {
                'name': player['display_name'],
                'team': player['team'],
                'cost': player['now_cost_m'],
                'ownership': ownership_val,
                'form': player.get('form', 0),
                'changed_club': bool(player.get('changed_club', False)),
                'proj_pts_total': proj_pts,
                'proj_pts_pgw': proj_pts
            }

            for gw in range(starting_gameweek,
                            starting_gameweek + num_gameweeks):
                gw_col = f"gw{gw}"
                formatted_player[gw_col] = player.get(gw_col, "-")

            formatted_players.append(formatted_player)

        return formatted_players

    def print_differential_suggestions(self, differentials: dict,
                                       starting_gameweek: int = None,
                                       num_gameweeks: int = None):
        """
        Print differential suggestions in a formatted table.

        Args:
            differentials (dict): Differential suggestions by position
            starting_gameweek (int): Starting gameweek number
            num_gameweeks (int): Number of gameweeks being analysed
        """
        if not differentials or not any(differentials.values()):
            print("\n=== 💎 DIFFERENTIAL SUGGESTIONS ===")
            print("No suitable differentials found (players below 5% "
                  "ownership with positive FPL scores)")
            return

        start_gw = starting_gameweek or self.config.GAMEWEEK
        num_gw = num_gameweeks or self.config.FIRST_N_GAMEWEEKS

        print("\n=== 💎 DIFFERENTIAL SUGGESTIONS ===")
        print(f"Low ownership (<5%) players with high expected "
              f"performance (GW{start_gw}-{start_gw + num_gw - 1})")

        for position in ["GK", "DEF", "MID", "FWD"]:
            if (position not in differentials
                    or not differentials[position]):
                continue

            print(f"\n{position}:")

            df_data = []
            for player in differentials[position]:
                player_data = {
                    'name': player['name'],
                    'team': player['team'],
                    'cost': player['cost'],
                    'ownership': f"{player['ownership']:.1f}%",
                    'form': player['form'],
                    'new_club': "Y" if player['changed_club'] else "",
                    'proj_pts': f"{player['proj_pts_pgw']:.1f}",
                }

                for gw in range(start_gw, start_gw + num_gw):
                    gw_col = f"gw{gw}"
                    player_data[gw_col] = player.get(gw_col, "-")

                df_data.append(player_data)

            if df_data:
                print(pd.DataFrame(df_data).to_string(index=False))

    def get_differential_summary_stats(self,
                                       differentials: dict) -> dict:
        """
        Get summary statistics about the differential suggestions.

        Args:
            differentials (dict): Differential suggestions

        Returns:
            dict: Summary statistics
        """
        total_found = sum(
            len(players) for players in differentials.values()
        )
        positions_with_differentials = len([
            pos for pos, players in differentials.items() if players
        ])

        avg_ownership = 0
        avg_proj_pts = 0
        count = 0

        for players in differentials.values():
            for player in players:
                avg_ownership += player['ownership']
                # The formatted key is proj_pts_pgw; the previous
                # 'proj_pts' lookup raised a KeyError whenever this
                # method was called.
                avg_proj_pts += player['proj_pts_pgw']
                count += 1

        if count > 0:
            avg_ownership /= count
            avg_proj_pts /= count

        return {
            'total_found': total_found,
            'positions_covered': positions_with_differentials,
            'avg_ownership': avg_ownership,
            'avg_projected_points': avg_proj_pts
        }