"""Request and response shapes."""

from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    name: str
    avatar_url: str
    is_admin: bool
    fpl_entry_id: int | None = None


class SettingsOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    budget: float
    free_transfers: int
    accept_transfer_penalty: bool
    exclude_unavailable: bool
    wildcard: bool
    free_hit: bool = False
    free_hit_prev_gw: bool
    bench_boost: bool
    triple_captain: bool
    theme: str = "legacy"
    tutorial_seen: bool = False
    use_ml_weights: bool
    first_n_gameweeks: int
    min_transfer_value: float
    transfer_horizon_gws: int
    overrides: dict[str, Any] = Field(default_factory=dict)
    team_modifiers: dict[str, float] = Field(default_factory=dict)
    forced_selections: dict[str, list[str]] = Field(default_factory=dict)
    blacklist_players: list[str] = Field(default_factory=list)


class SettingsIn(BaseModel):
    budget: float | None = Field(default=None, ge=50, le=200)
    free_transfers: int | None = Field(default=None, ge=0, le=15)
    accept_transfer_penalty: bool | None = None
    exclude_unavailable: bool | None = None
    wildcard: bool | None = None
    free_hit: bool | None = None
    free_hit_prev_gw: bool | None = None
    bench_boost: bool | None = None
    triple_captain: bool | None = None
    theme: str | None = None
    tutorial_seen: bool | None = None
    use_ml_weights: bool | None = None
    first_n_gameweeks: int | None = Field(default=None, ge=1, le=10)
    min_transfer_value: float | None = Field(default=None, ge=0, le=20)
    transfer_horizon_gws: int | None = Field(default=None, ge=1, le=15)
    overrides: dict[str, Any] | None = None
    team_modifiers: dict[str, float] | None = None
    forced_selections: dict[str, list[str]] | None = None
    blacklist_players: list[str] | None = None


class PlayerRefIn(BaseModel):
    """One player, as named by whichever table or panel the button sat in."""

    name: str = Field(min_length=1, max_length=80)
    # Only needed for forced picks, and only when the caller knows it. When
    # it's missing the server resolves it from the FPL pool.
    position: str | None = None


class EntryLinkIn(BaseModel):
    fpl_entry_id: int | None = Field(default=None, ge=1)


class ImportSquadIn(BaseModel):
    gameweek: int | None = Field(default=None, ge=1, le=38)
    apply_budget: bool = True
    apply_free_transfers: bool = True


class ManualSquadIn(BaseModel):
    """Fifteen players entered by hand, saved as a past gameweek's squad."""

    gameweek: int | None = Field(default=None, ge=1, le=38)
    player_ids: list[int] = Field(min_length=15, max_length=15)
    starting_ids: list[int] = Field(min_length=11, max_length=11)
    bank: float = Field(default=0.0, ge=0, le=100)
    apply_budget: bool = True


class PlanIn(BaseModel):
    """A request to run the optimiser forward over several gameweeks."""

    weeks: int = Field(default=5, ge=3, le=8)
    start_gameweek: int | None = Field(default=None, ge=1, le=38)
    season: str | None = None
    # {"7": "wildcard"} — gameweek number to chip name.
    chips: dict[str, str] = Field(default_factory=dict)


class PlanOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    season: str
    start_gameweek: int
    weeks: int
    chips: dict
    status: str
    progress: int
    log: str
    error: str
    payload: list
    created_at: dt.datetime
    finished_at: dt.datetime | None


class RunIn(BaseModel):
    gameweek: int | None = Field(default=None, ge=1, le=38)
    season: str | None = None


class RunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    gameweek: int
    season: str
    status: str
    log: str
    error: str
    squad_id: int | None
    result: dict[str, Any]


class SquadOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    gameweek: int
    season: str
    formation: str
    projected_points: float
    squad_value: float
    bank: float
    transfers_made: int
    penalty_points: int
    chip: str
    active_option: str = "option-1"
    payload: dict[str, Any]


class ActivateOptionIn(BaseModel):
    """Which of a run's squads to put in force."""

    option: str = Field(min_length=1, max_length=24)


class ResultIn(BaseModel):
    gameweek: int = Field(ge=1, le=38)
    actual_points: float | None = Field(default=None, ge=0, le=300)
    season: str | None = None


class InviteIn(BaseModel):
    email: str
