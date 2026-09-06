"""Module for managing fixture data and difficulty calculations with
decaying weights."""

import os
import pandas as pd
from ..api.fpl_client import FPLClient


class EnhancedFixtureDifficultyCalculator:
    """
    Enhanced fixture difficulty calculator that combines:
    1. FPL official difficulty ratings (baseline)
    2. Team attacking ratings (actual goal-scoring performance)
    3. Team defensive ratings (actual defensive performance)
    4. Position-specific adjustments
    """

    def __init__(self, config):
        self.config = config
        self.team_ratings = {}
        self.load_team_ratings()

    def load_team_ratings(self):
        """Load team ratings from CSV file."""
        ratings_file = 'data/team_ratings.csv'

        if not os.path.exists(ratings_file):
            if self.config.GRANULAR_OUTPUT:
                print(f"Note: {ratings_file} not found. Using FPL "
                      "difficulty only.")
            self._use_neutral_ratings()
            return

        try:
            ratings_df = pd.read_csv(ratings_file, index_col='team')
            self.team_ratings = ratings_df[
                ['attacking_rating', 'defensive_rating']
            ].to_dict('index')
            if self.config.GRANULAR_OUTPUT:
                print(f"Enhanced fixture ratings loaded for "
                      f"{len(self.team_ratings)} teams")
        except Exception as error:
            if self.config.GRANULAR_OUTPUT:
                print(f"Error loading team ratings: {error}")
                print("Using FPL difficulty only.")
            self._use_neutral_ratings()

    def _use_neutral_ratings(self):
        """Fall back to neutral ratings for every configured team."""
        for team in self.config.TEAM_MODIFIERS.keys():
            self.team_ratings[team] = {
                'attacking_rating': 1.0,
                'defensive_rating': 1.0
            }

    def get_attacking_rating(self, team_name):
        """Get the attacking rating for a team."""
        if team_name in self.team_ratings:
            return self.team_ratings[team_name]['attacking_rating']
        return 1.0

    def get_defensive_rating(self, team_name):
        """Get the defensive rating for a team."""
        if team_name in self.team_ratings:
            return self.team_ratings[team_name]['defensive_rating']
        return 1.0

    def calculate_enhanced_difficulty(self, opponent_name,
                                      player_position, fpl_difficulty,
                                      is_home=True):
        """
        Calculate enhanced fixture difficulty for a player against an
        opponent.

        Args:
            opponent_name (str): Name of the opponent team
            player_position (str): Player's position (GK, DEF, MID, FWD)
            fpl_difficulty (int): FPL's official difficulty rating (1-5)
            is_home (bool): Whether playing at home

        Returns:
            float: Enhanced difficulty (1.0 = easiest, 5.0 = hardest)
        """
        base_difficulty = float(fpl_difficulty)
        opponent_attack = self.get_attacking_rating(opponent_name)
        opponent_defence = self.get_defensive_rating(opponent_name)

        if player_position == 'GK':
            difficulty_adjustment = (opponent_attack - 1.0) * 1.5
        elif player_position == 'DEF':
            difficulty_adjustment = (opponent_attack - 1.0) * 1.5
        elif player_position == 'FWD':
            difficulty_adjustment = (opponent_defence - 1.0) * 1.5
        elif player_position == 'MID':
            attack_component = (opponent_attack - 1.0) * 0.6
            defence_component = (opponent_defence - 1.0) * 0.9
            difficulty_adjustment = attack_component + defence_component
        else:
            difficulty_adjustment = 0.0

        if is_home:
            difficulty_adjustment -= 0.3

        enhanced_difficulty = base_difficulty + difficulty_adjustment
        return max(1.0, min(5.0, enhanced_difficulty))


class FixtureManager:
    """Handles fixture data and difficulty calculations."""

    def __init__(self, config):
        self.config = config
        self.fpl_client = FPLClient()
        self.enhanced_calculator = EnhancedFixtureDifficultyCalculator(
            config
        )

    def _drop_started_fixtures(self, fixtures: pd.DataFrame,
                               label: str) -> tuple:
        """
        Remove fixtures that have already kicked off from the window
        being optimised for.

        Running midway through a gameweek, the clubs that have already
        played cannot be bought into for that match - those points are
        gone. Leaving their fixture in credits them with a full
        projection they can no longer deliver, on top of whatever
        cumulative-stat skew they carry.

        Returns:
            tuple: (remaining_fixtures, set of team ids already played)
        """
        if fixtures.empty or "started" not in fixtures.columns:
            return fixtures, set()

        started = fixtures["started"].fillna(False).astype(bool)

        if not started.any():
            return fixtures, set()

        played = fixtures[started]
        played_teams = set(played["team_h"].tolist()) | set(
            played["team_a"].tolist()
        )

        if not getattr(self.config, "EXCLUDE_STARTED_FIXTURES", True):
            print(f"Note: {int(started.sum())} fixture(s) in {label} "
                  "have already kicked off but are still being scored "
                  "(EXCLUDE_STARTED_FIXTURES is off).")
            return fixtures, played_teams

        print(f"Excluding {int(started.sum())} fixture(s) in {label} "
              "that have already kicked off. Those clubs are scored as "
              "blanks, since their points can no longer be bought.")

        return fixtures[~started], played_teams

    def _calculate_decay_weights(self, num_gameweeks: int) -> list:
        """
        Calculate decaying weights for fixtures.

        Args:
            num_gameweeks (int): Number of gameweeks to weight

        Returns:
            list: Weights that decay exponentially, summing to 1.0
        """
        if num_gameweeks <= 0:
            return []

        if num_gameweeks == 1:
            return [1.0]

        decay_factor = getattr(
            self.config, 'FIXTURE_DECAY_FACTOR', 0.6
        )

        raw_weights = [decay_factor ** i for i in range(num_gameweeks)]

        total_weight = sum(raw_weights)
        return [w / total_weight for w in raw_weights]

    def fetch_player_fixture_difficulty(self, first_n_gws: int,
                                        players: pd.DataFrame,
                                        starting_gameweek: int
                                        ) -> pd.DataFrame:
        """
        Calculate fixture difficulty for each player over the next N
        gameweeks, with DGW and BGW detection and decaying weights.

        Args:
            first_n_gws (int): Number of gameweeks to consider.
            players (pd.DataFrame): Player data with team_id and
                                    player_code.
            starting_gameweek (int): The gameweek to start from.

        Returns:
            pd.DataFrame: player_code, fixture info and DGW/BGW flags.
        """
        if "player_code" not in players.columns:
            raise ValueError(
                "Player data has no 'player_code' column. Fixture "
                "difficulty must be keyed on the permanent player code."
            )

        fixtures = pd.DataFrame(self.fpl_client.get_fixtures())

        teams_data = self.fpl_client.get_bootstrap_static()
        teams_df = pd.DataFrame(
            teams_data["teams"])[["id", "name", "short_name"]]

        self._save_fixture_data(fixtures, teams_df)

        end_gameweek = starting_gameweek + first_n_gws - 1
        events = pd.to_numeric(fixtures["event"], errors="coerce")
        fixtures = fixtures[
            (events >= starting_gameweek) & (events <= end_gameweek)
        ]

        fixtures, _ = self._drop_started_fixtures(
            fixtures, f"GW{starting_gameweek}-{end_gameweek}"
        )

        return self._calculate_player_difficulties(
            fixtures, players, teams_df, starting_gameweek,
            end_gameweek, first_n_gws
        )

    def _save_fixture_data(self, fixtures: pd.DataFrame,
                           teams_df: pd.DataFrame):
        """Process and save fixture data to CSV."""
        fixtures_for_csv = fixtures.copy()

        fixtures_for_csv = fixtures_for_csv.merge(
            teams_df.rename(
                columns={
                    "id": "team_h",
                    "name": "home_team",
                    "short_name": "home_team_short",
                }
            ),
            on="team_h",
            how="left",
        )
        fixtures_for_csv = fixtures_for_csv.merge(
            teams_df.rename(
                columns={
                    "id": "team_a",
                    "name": "away_team",
                    "short_name": "away_team_short",
                }
            ),
            on="team_a",
            how="left",
        )

        fixtures_csv = fixtures_for_csv[
            [
                "id",
                "event",
                "kickoff_time",
                "home_team",
                "away_team",
                "home_team_short",
                "away_team_short",
                "team_h_difficulty",
                "team_a_difficulty",
                "team_h_score",
                "team_a_score",
                "finished",
            ]
        ].copy()

        fixtures_csv = fixtures_csv.rename(
            columns={
                "id": "fixture_id",
                "event": "gameweek",
                "team_h_difficulty": "home_difficulty",
                "team_a_difficulty": "away_difficulty",
                "team_h_score": "home_score",
                "team_a_score": "away_score",
            }
        )

        fixtures_csv = fixtures_csv.sort_values(
            ["gameweek", "kickoff_time"]
        )

        os.makedirs("data", exist_ok=True)
        fixtures_csv.to_csv("data/fixtures.csv", index=False)

    def _calculate_player_difficulties(
            self, fixtures: pd.DataFrame, players: pd.DataFrame,
            teams_df: pd.DataFrame, starting_gameweek: int,
            end_gameweek: int, num_gameweeks: int) -> pd.DataFrame:
        """
        Calculate per-player fixture difficulty with DGW and BGW
        detection and decaying weights across the window.
        """
        decay_weights = self._calculate_decay_weights(num_gameweeks)

        fixture_counts = self._count_team_fixtures(
            fixtures, teams_df, starting_gameweek, end_gameweek
        )

        team_names = dict(zip(teams_df["id"], teams_df["name"]))

        player_diffs = []

        for gw_offset, weight in enumerate(decay_weights):
            current_gw = starting_gameweek + gw_offset
            gw_fixtures = fixtures[fixtures["event"] == current_gw]

            for _, fixture_row in gw_fixtures.iterrows():
                for home_away, team_col, diff_col, opponent_col in [
                    ("home", "team_h", "team_h_difficulty", "team_a"),
                    ("away", "team_a", "team_a_difficulty", "team_h"),
                ]:
                    team_id = fixture_row[team_col]
                    opponent_id = fixture_row[opponent_col]
                    fpl_diff = fixture_row[diff_col]
                    is_home = (home_away == "home")

                    opponent_name = team_names.get(opponent_id, "Unknown")

                    team_players = players[
                        players["team_id"] == team_id
                    ]
                    for _, player_row in team_players.iterrows():
                        enhanced_diff = (
                            self.enhanced_calculator
                            .calculate_enhanced_difficulty(
                                opponent_name, player_row["position"],
                                fpl_diff, is_home
                            )
                        )

                        player_diffs.append({
                            "player_code": player_row["player_code"],
                            "team_id": team_id,
                            "gameweek": current_gw,
                            "diff": enhanced_diff,
                            "weight": weight
                        })

        if not player_diffs:
            print(f"Warning: no fixtures found for gameweeks "
                  f"{starting_gameweek} to {end_gameweek}")
            return pd.DataFrame(columns=[
                "player_code", "diff", "fixture_bonus", "has_dgw",
                "has_bgw", "fixture_multiplier"
            ])

        df = pd.DataFrame(player_diffs)

        weighted = df.groupby("player_code").apply(
            lambda group: pd.Series({
                "diff": (
                    (group["diff"] * group["weight"]).sum()
                    / group["weight"].sum()
                ),
                "team_id": group["team_id"].iloc[0]
            }),
            include_groups=False
        ).reset_index()

        weighted = weighted.merge(
            fixture_counts, on="team_id", how="left"
        )
        weighted["has_dgw"] = weighted["has_dgw"].fillna(False)
        weighted["has_bgw"] = weighted["has_bgw"].fillna(False)

        # fixture_multiplier is the number of fixtures in the gameweek
        # being picked for: 0 for a blank, 1 normally, 2 for a double. It
        # scales that gameweek's projected points.
        weighted["fixture_multiplier"] = weighted[
            "fixtures_in_target_gw"
        ].fillna(1.0)

        # The fixture bonus reflects the whole window, so it uses the
        # mean fixtures per gameweek rather than the sum. Summing across
        # the window gave a normal team a multiplier equal to the window
        # length, inflating both the bonus and projected points.
        weighted["avg_fixtures_per_gw"] = weighted[
            "avg_fixtures_per_gw"
        ].fillna(1.0)

        weighted["fixture_bonus"] = (
            (6 - weighted["diff"]) * weighted["avg_fixtures_per_gw"]
        )

        # Every player must appear exactly once, including those whose
        # club has a blank gameweek and therefore no fixture rows.
        all_players = players[["player_code"]].drop_duplicates()
        weighted = all_players.merge(
            weighted, on="player_code", how="left", validate="1:1"
        )
        weighted["diff"] = weighted["diff"].fillna(5.0)
        weighted["fixture_bonus"] = weighted["fixture_bonus"].fillna(0.0)
        weighted["fixture_multiplier"] = weighted[
            "fixture_multiplier"
        ].fillna(0.0)
        weighted["has_dgw"] = weighted["has_dgw"].fillna(False)
        weighted["has_bgw"] = weighted["has_bgw"].fillna(True)

        return weighted[[
            "player_code", "diff", "fixture_bonus", "has_dgw",
            "has_bgw", "fixture_multiplier"
        ]]

    def _count_team_fixtures(self, fixtures: pd.DataFrame,
                             teams_df: pd.DataFrame,
                             starting_gameweek: int,
                             end_gameweek: int) -> pd.DataFrame:
        """
        Count each team's fixtures per gameweek across the window.

        Returns:
            pd.DataFrame: team_id, has_dgw, has_bgw,
                          fixtures_in_target_gw, avg_fixtures_per_gw
        """
        gameweeks = list(range(starting_gameweek, end_gameweek + 1))
        records = []

        for gw in gameweeks:
            gw_fixtures = fixtures[fixtures["event"] == gw]

            appearances = pd.Series(
                gw_fixtures["team_h"].tolist()
                + gw_fixtures["team_a"].tolist()
            ).value_counts()

            for team_id in teams_df["id"]:
                count = int(appearances.get(team_id, 0))

                if count > 2:
                    print(f"Warning: team {team_id} has {count} "
                          f"fixtures in GW{gw} (expected 0, 1 or 2)")

                records.append({
                    "team_id": team_id,
                    "gameweek": gw,
                    "fixtures": count,
                })

        if not records:
            return pd.DataFrame(columns=[
                "team_id", "has_dgw", "has_bgw",
                "fixtures_in_target_gw", "avg_fixtures_per_gw"
            ])

        analysis = pd.DataFrame(records)

        summary = analysis.groupby("team_id").agg(
            has_dgw=("fixtures", lambda s: bool((s >= 2).any())),
            has_bgw=("fixtures", lambda s: bool((s == 0).any())),
            avg_fixtures_per_gw=("fixtures", "mean"),
        ).reset_index()

        target = analysis[
            analysis["gameweek"] == starting_gameweek
        ][["team_id", "fixtures"]].rename(
            columns={"fixtures": "fixtures_in_target_gw"}
        )

        summary = summary.merge(target, on="team_id", how="left")
        summary["fixtures_in_target_gw"] = summary[
            "fixtures_in_target_gw"
        ].fillna(0).astype(float)

        self._report_dgw_bgw(summary, teams_df, starting_gameweek,
                             end_gameweek)

        return summary

    def _report_dgw_bgw(self, summary: pd.DataFrame,
                        teams_df: pd.DataFrame, starting_gameweek: int,
                        end_gameweek: int):
        """Print any doubles or blanks detected in the window."""
        dgw_teams = summary[summary["has_dgw"]]
        bgw_teams = summary[summary["has_bgw"]]

        if len(dgw_teams) > 0:
            names = teams_df[
                teams_df["id"].isin(dgw_teams["team_id"])
            ]["name"].tolist()
            print(f"DGW teams in GW{starting_gameweek}-{end_gameweek}: "
                  f"{', '.join(names)}")

        if len(bgw_teams) > 0:
            names = teams_df[
                teams_df["id"].isin(bgw_teams["team_id"])
            ]["name"].tolist()
            print(f"BGW teams in GW{starting_gameweek}-{end_gameweek}: "
                  f"{', '.join(names)}")

    def add_next_fixture(self, df: pd.DataFrame,
                         target_gameweek: int) -> pd.DataFrame:
        """
        Add fixture information for a specific gameweek for each player,
        handling doubles and blanks.

        Args:
            df (pd.DataFrame): Player data to add fixture info to.
            target_gameweek (int): The gameweek to get fixtures for.

        Returns:
            pd.DataFrame: Player data with opponent, venue and fixture
                          difficulty columns for the target gameweek.
        """
        fixtures = pd.DataFrame(self.fpl_client.get_fixtures())
        teams_data = self.fpl_client.get_bootstrap_static()
        teams_df = pd.DataFrame(teams_data["teams"])

        gw_fixtures = fixtures[fixtures["event"] == target_gameweek]

        if gw_fixtures.empty:
            print(f"Warning: no fixtures found for gameweek "
                  f"{target_gameweek}")
            return self._add_empty_fixture_info(df)

        gw_fixtures, played_teams = self._drop_started_fixtures(
            gw_fixtures, f"GW{target_gameweek}"
        )

        if gw_fixtures.empty:
            print(f"Every GW{target_gameweek} fixture has already been "
                  "played. Set GAMEWEEK to the next gameweek to pick "
                  "for it.")
            return self._add_empty_fixture_info(df)

        fixture_counts = self._count_team_fixtures(
            gw_fixtures, teams_df, target_gameweek, target_gameweek
        )

        next_fixtures = self._process_gameweek_fixtures(
            df, gw_fixtures, fixture_counts, played_teams
        )

        return self._add_team_names_to_fixtures(
            df, next_fixtures, teams_df
        )

    def _add_empty_fixture_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add empty fixture information when no fixtures are found."""
        df["next_opponent"] = "No fixture"
        df["venue"] = "N/A"
        df["fixture_difficulty"] = None
        df["has_dgw_next"] = False
        df["has_bgw_next"] = True
        df["already_played"] = False
        return df

    def _process_gameweek_fixtures(self, df: pd.DataFrame,
                                   gw_fixtures: pd.DataFrame,
                                   fixture_counts: pd.DataFrame,
                                   played_teams: set = None) -> list:
        """Process fixtures for each player in the target gameweek."""
        played_teams = played_teams or set()
        next_fixtures = []

        counts = fixture_counts.set_index("team_id")

        for _, player_row in df.iterrows():
            team_id = player_row["team_id"]

            if team_id in counts.index:
                has_dgw = bool(counts.loc[team_id, "has_dgw"])
                has_bgw = bool(counts.loc[team_id, "has_bgw"])
            else:
                has_dgw = False
                has_bgw = False

            team_fixtures = gw_fixtures[
                (gw_fixtures["team_h"] == team_id) |
                (gw_fixtures["team_a"] == team_id)
            ]

            if len(team_fixtures) == 0 or has_bgw:
                next_fixtures.append({
                    "player_code": player_row["player_code"],
                    "next_opponent_ids": [],
                    "venues": [],
                    "fixture_difficulties": [],
                    "has_dgw_next": False,
                    "has_bgw_next": True,
                    "already_played": team_id in played_teams,
                })
                continue

            opponent_ids = []
            venues = []
            difficulties = []

            for _, fixture in team_fixtures.iterrows():
                if fixture["team_h"] == team_id:
                    opponent_ids.append(fixture["team_a"])
                    venues.append("Home")
                    difficulties.append(fixture["team_h_difficulty"])
                else:
                    opponent_ids.append(fixture["team_h"])
                    venues.append("Away")
                    difficulties.append(fixture["team_a_difficulty"])

            next_fixtures.append({
                "player_code": player_row["player_code"],
                "next_opponent_ids": opponent_ids,
                "venues": venues,
                "fixture_difficulties": difficulties,
                "has_dgw_next": has_dgw,
                "has_bgw_next": False,
                "already_played": False,
            })

        return next_fixtures

    def _add_team_names_to_fixtures(self, df: pd.DataFrame,
                                    next_fixtures: list,
                                    teams_df: pd.DataFrame
                                    ) -> pd.DataFrame:
        """Add team names to fixture information."""
        nf_df = pd.DataFrame(next_fixtures)
        team_names = dict(zip(teams_df["id"], teams_df["name"]))

        for idx, row in nf_df.iterrows():
            if (not row["next_opponent_ids"]
                    or row.get("has_bgw_next", False)):
                nf_df.at[idx, "next_opponent"] = (
                    "Already played"
                    if row.get("already_played", False) else "Blank GW"
                )
                nf_df.at[idx, "venue"] = "N/A"
                nf_df.at[idx, "fixture_difficulty"] = None
            elif len(row["next_opponent_ids"]) == 1:
                opponent_id = row["next_opponent_ids"][0]
                nf_df.at[idx, "next_opponent"] = team_names.get(
                    opponent_id, "Unknown"
                )
                nf_df.at[idx, "venue"] = row["venues"][0]
                nf_df.at[idx, "fixture_difficulty"] = row[
                    "fixture_difficulties"
                ][0]
            else:
                opponent_names = [
                    team_names.get(opponent_id, "Unknown")
                    for opponent_id in row["next_opponent_ids"]
                ]

                nf_df.at[idx, "next_opponent"] = " & ".join(
                    opponent_names
                )
                nf_df.at[idx, "venue"] = " & ".join(row["venues"])
                nf_df.at[idx, "fixture_difficulty"] = sum(
                    row["fixture_difficulties"]
                ) / len(row["fixture_difficulties"])

        nf_df = nf_df.drop_duplicates(subset=["player_code"])

        before = len(df)
        df = df.merge(
            nf_df[["player_code", "next_opponent", "venue",
                   "fixture_difficulty", "has_dgw_next",
                   "has_bgw_next", "already_played"]],
            on="player_code",
            how="left",
            validate="1:1",
        )

        if len(df) != before:
            raise ValueError(
                f"Next-fixture merge changed row count from {before} to "
                f"{len(df)}."
            )

        df["next_opponent"] = df["next_opponent"].fillna("No fixture")
        df["venue"] = df["venue"].fillna("N/A")
        df["has_dgw_next"] = df["has_dgw_next"].fillna(False)
        df["has_bgw_next"] = df["has_bgw_next"].fillna(False)
        df["already_played"] = df["already_played"].fillna(False)

        return df