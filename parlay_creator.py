#!/usr/bin/env python3
"""
parlay_engine.py

Production-leaning Parlay Engine (single-file version)

Core design goals
-----------------
- Explicit data validation and normalization (esp. percentages vs decimals)
- Clear modeling pipeline:
    1. Load & validate props.
    2. Estimate minutes (hierarchical shrinkage).
    3. Build an event-level "model probability" from history & projections.
    4. Blend model probability with market implied probability.
    5. Enforce OVER/UNDER complementarity within each (Player, Game, Stat, Line).
    6. Build a correlation matrix across props (copula).
    7. Run Monte Carlo to estimate joint hit probabilities.
    8. Search for +EV parlays and rank by EV / Kelly.
- Optional backtest over historical sheets to assess calibration.

Major modeling features
-----------------------
- OVER/UNDER pairs are treated as a single latent event:
    For each (Team, Player, Game, StatKey, Line) we estimate P(OVER),
    then set P(UNDER) = 1 - P(OVER).
- Historical hit rates (Last 5/10/20, 2025 season) are converted into
  pseudo-counts and combined into a Beta-Binomial estimate of P(OVER).
- The Beta variance (via an effective sample size n_eff) is used directly
  to drive the Monte Carlo copula layer.
- Candidate legs are ranked by single-leg EV, with hard probability and
  edge filters to keep the pool focused.

NOTE: This is **still a modeling prototype**. It is cleaner, safer, and
more maintainable than the original, but YOU must:
- Fit / calibrate weights on real data.
- Add unit tests and continuous monitoring.
- Respect responsible gambling practices.

Usage
-----
    python3 parlay_engine.py props.xlsx --trials 100000 --max_legs 4
    python3 parlay_engine.py props.csv  --trials 50000  --max_legs 3

    # Backtest over a folder of historical prop sheets (each with Outcome column)
    python3 parlay_engine.py dummy.csv --backtest history_folder/ --trials 20000
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
import logging
import math
import os
import re
from dataclasses import dataclass
from itertools import combinations
from time import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import eigh, cholesky
from scipy.stats import beta as sp_beta
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EngineConfig:
    trials: int = 50_000
    batch_size: int = 20_000
    max_legs: int = 4
    pool_limit: int = 40                # how many candidate legs to consider
    top_k: int = 20                     # how many parlays to show per list
    confidence_in_model: float = 0.35   # blend between market and model
    rho_same_team: float = 0.30
    rho_opp_team: float = 0.10
    rho_same_player: float = 0.95
    kelly_cap: float = 0.10             # cap Kelly fraction
    avoid_same_player: bool = True
    random_seed: int = 42
    debug: bool = False


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def json_default(o):
    """
    Serializer for numpy / pandas scalar types so json.dump works.
    """
    if isinstance(o, (np.generic,)):
        return o.item()  # convert np.int64/np.float64 -> Python int/float
    return str(o)  # last-resort fallback (should rarely trigger)


def american_to_implied(odds: Any) -> float:
    """Convert American odds to implied probability in [0,1]."""
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if o < 0:
        return -o / (-o + 100.0)
    else:
        return 100.0 / (o + 100.0)


def implied_to_decimal(p: float) -> float:
    """Convert probability to decimal odds. p must be in (0,1]."""
    if p <= 0.0:
        return np.inf  # "broken" / degenerate
    return 1.0 / p


def decimal_to_american(d: float) -> str:
    """Convert decimal odds to American odds as a string."""
    if d <= 1.0:
        return "-INF"
    if d >= 2.0:
        return f"+{int(round((d - 1.0) * 100.0))}"
    return f"-{int(round(100.0 / (d - 1.0)))}"


def parse_side(proposition: str) -> Optional[str]:
    """Return 'OVER', 'UNDER', or None from a proposition string."""
    if not isinstance(proposition, str):
        return None
    s = proposition.lower()
    if "over" in s:
        return "OVER"
    if "under" in s:
        return "UNDER"
    m = re.search(r"\b(over|under)\b", s)
    if m:
        return m.group(1).upper()
    return None


def parse_stat_key(proposition: str) -> str:
    """
    Extract a canonical stat key from a proposition string.

    Examples:
        "Over 4.5 Assists"            -> "assists"
        "Under 21.5 Points + Assists" -> "points + assists"
        "Over 1.5 3-PT Made"          -> "3-pt made"
    """
    if not isinstance(proposition, str):
        return ""
    s = proposition.strip()
    # pattern: (over|under) <number> <rest>
    m = re.search(r"\b(over|under)\b\s+[0-9]+(?:\.[0-9]+)?\s*(.*)", s, flags=re.IGNORECASE)
    if m:
        stat = m.group(2).strip()
    else:
        # Fallback: strip side words and numeric tokens
        s = re.sub(r"\b(over|under)\b", "", s, flags=re.IGNORECASE)
        s = re.sub(r"[0-9]+(?:\.[0-9]+)?", "", s)
        stat = s.strip()
    stat = re.sub(r"\s+", " ", stat)
    return stat.lower()


def batch_iter(total: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    """Yield (start, end) index pairs for batching 0..total-1."""
    start = 0
    while start < total:
        end = min(total, start + batch_size)
        yield start, end
        start = end


def ensure_pd(mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Ensure matrix is symmetric positive-definite by clipping eigenvalues.
    Returns a new matrix that is SPD.
    """
    mat = (mat + mat.T) / 2.0
    vals, vecs = eigh(mat)
    vals_clipped = np.clip(vals, eps, None)
    pd_mat = (vecs * vals_clipped) @ vecs.T
    # Re-symmetrize numerically
    pd_mat = (pd_mat + pd_mat.T) / 2.0
    return pd_mat


def parse_probability_value(val: Any) -> float:
    """
    Robustly parse a probability-like field.

    Accepts:
      - 0.69  -> 0.69
      - 69    -> 0.69  (assumes percent)
      - '69%' -> 0.69
      - '0.69'-> 0.69
      - NaN   -> np.nan
    Raises ValueError if value cannot be parsed.
    """
    if pd.isna(val):
        return np.nan

    if isinstance(val, str):
        s = val.strip()
        if s.endswith("%"):
            s = s[:-1].strip()
            return float(s) / 100.0
        # plain string; might be 0.69 or 69
        f = float(s)
    else:
        f = float(val)

    # If > 1, assume it's a percent (e.g., 69)
    if f > 1.0:
        f = f / 100.0

    if f < 0.0 or f > 1.0:
        raise ValueError(f"Probability out of bounds after parsing: {val} -> {f}")
    return f


def normalize_probability_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    In-place normalization of probability-like columns.
    All values become floats in [0,1] or NaN. Raises on invalid entries.
    """
    for col in columns:
        if col not in df.columns:
            continue
        df[col] = df[col].apply(lambda v: parse_probability_value(v) if not pd.isna(v) else np.nan)
    return df


# ---------------------------------------------------------------------------
# Hierarchical minutes model (player <- team <- global)
# ---------------------------------------------------------------------------

def hierarchical_minutes_estimate(
    df: pd.DataFrame,
    player_col: str = "Player",
    team_col: str = "Team",
    last_minutes_col: str = "LastMinutes",
    default_minutes: float = 25.0,
    shrinkage_strength: float = 10.0,
) -> pd.DataFrame:
    """
    Estimate expected minutes per player using hierarchical shrinkage.

    - Player-level "LastMinutes" is shrunk toward team mean and global mean.
    - If LastMinutes is missing, optionally use ExpectedMinutes if present.
    - Otherwise default to team/global averages, then default_minutes.
    """
    df = df.copy()

    if last_minutes_col not in df.columns:
        df[last_minutes_col] = np.nan

    def parse_last_minutes(x: Any) -> float:
        try:
            if pd.isna(x):
                return np.nan
            if isinstance(x, str) and "," in x:
                parts = [float(p) for p in x.split(",") if p.strip()]
                return float(np.mean(parts)) if parts else np.nan
            return float(x)
        except Exception:
            return np.nan

    df["_LastMinNum"] = df[last_minutes_col].apply(parse_last_minutes)

    if "ExpectedMinutes" in df.columns:
        df["_ExpMin"] = pd.to_numeric(df["ExpectedMinutes"], errors="coerce")
    else:
        df["_ExpMin"] = np.nan

    # Global mean (from last minutes if available, else from expected)
    valid_last = df["_LastMinNum"].dropna()
    if len(valid_last) > 0:
        global_mean = float(np.nanmean(valid_last))
    else:
        valid_exp = df["_ExpMin"].dropna()
        if len(valid_exp) > 0:
            global_mean = float(np.nanmean(valid_exp))
        else:
            global_mean = default_minutes

    if math.isnan(global_mean):
        global_mean = default_minutes

    # Team means
    team_means: Dict[Any, float] = {}
    grouped = df.groupby(team_col)["_LastMinNum"]
    for team, series in grouped:
        vals = series.dropna().values
        if len(vals) > 0:
            team_means[team] = float(np.mean(vals))
        else:
            team_means[team] = global_mean

    est_minutes: List[float] = []
    for _, row in df.iterrows():
        pmin = row["_LastMinNum"]
        emin = row["_ExpMin"]
        t = row[team_col]
        team_mean = team_means.get(t, global_mean)

        if not math.isnan(pmin):
            w_player = 1.0
            w_team = shrinkage_strength
            w_global = 1.0
            numerator = w_player * pmin + w_team * team_mean + w_global * global_mean
            denom = w_player + w_team + w_global
            est = numerator / denom
        elif not math.isnan(emin):
            w_exp = 1.0
            w_team = shrinkage_strength
            numerator = w_exp * emin + w_team * team_mean
            denom = w_exp + w_team
            est = numerator / denom
        else:
            est = team_mean

        est_minutes.append(float(est))

    df["EstMinutes"] = est_minutes
    df.drop(columns=["_LastMinNum", "_ExpMin"], inplace=True, errors="ignore")
    return df


# ---------------------------------------------------------------------------
# Probability modeling (event-level, OVER/UNDER paired)
# ---------------------------------------------------------------------------

HISTORY_COLS = [
    "Last 5 Hit Rate",
    "Last 10 Hit Rate",
    "Last 20 Hit Rate",
    "2025 Hit Rate",
]

# (column, base_n, weight) for historical horizons
HORIZON_SPECS = [
    ("Last 5 Hit Rate", 5, 0.10),
    ("Last 10 Hit Rate", 10, 0.45),
    ("Last 20 Hit Rate", 20, 0.25),
    ("2025 Hit Rate", 30, 0.20),  # treat 2025 as ~30 pseudo-games
]

PROJECTION_PSEUDO_GAMES = 20.0  # how much "weight" to give to projection-based prob


def compute_model_probabilities(
    df: pd.DataFrame,
    confidence_in_model: float = 0.35,
) -> pd.DataFrame:
    """
    Compute 'final_prob' for each individual prop, with the following logic:

    1) Normalize / create 'Implied Probability' from odds.
    2) For each unique event (Team, Player, Game, StatKey, Line):
       - Choose a canonical row (prefer OVER if present).
       - Convert that row's historical hit rates into pseudo-counts for the
         OVER event using a Beta-Binomial model.
       - Optionally incorporate a projection-based probability for OVER
         (normal approximation around Projection vs Line) as additional
         pseudo-counts.
       - Combine historical & projection pseudo-counts into a model
         probability p_model_over and an effective sample size N_eff.
       - Extract a de-vigged market probability for OVER from the canonical
         row's implied probability (flipping if canonical is UNDER).
       - Blend model vs market:
            p_final_over = (1 - confidence_in_model) * p_market_over
                         + confidence_in_model * p_model_over
         with sensible fall-backs if either piece is missing.
    3) Assign row-level probabilities:
       - For OVER rows:
            final_prob = p_final_over
       - For UNDER rows:
            final_prob = 1 - p_final_over
       - hist_prob/proj_prob/model_prob follow the same flipping.
       - n_eff is shared across both sides in the event.
    4) Precompute economic metrics:
       - LegDecimal = 1 / Implied Probability
       - edge       = final_prob - Implied Probability
       - single_ev  = final_prob * LegDecimal - 1

    The resulting df has columns:
        hist_prob, proj_prob, model_prob, final_prob, n_eff,
        LegDecimal, edge, single_ev.
    """
    df = df.copy().reset_index(drop=True)
    n = len(df)

    # --- 1) Market implied probability for each row ---
    if "Implied Probability" not in df.columns:
        df["Implied Probability"] = df["Odds"].apply(american_to_implied)
    else:
        df["Implied Probability"] = df["Implied Probability"].apply(
            lambda v: parse_probability_value(v) if not pd.isna(v) else np.nan
        )
        missing = df["Implied Probability"].isna()
        df.loc[missing, "Implied Probability"] = df.loc[missing, "Odds"].apply(american_to_implied)
    df["Implied Probability"] = df["Implied Probability"].clip(0.001, 0.999)

    # --- 2) Side & StatKey for event grouping ---
    if "Side" not in df.columns:
        df["Side"] = df["Proposition"].apply(parse_side)
    df["Side"] = df["Side"].fillna("OVER")

    if "StatKey" not in df.columns:
        df["StatKey"] = df["Proposition"].apply(parse_stat_key)

    # Normalize historical hit rates
    for col in HISTORY_COLS:
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = df[col].apply(lambda v: parse_probability_value(v) if not pd.isna(v) else np.nan)

    # Prepare arrays for row-level outputs
    hist_prob_arr = np.full(n, np.nan, dtype=float)
    proj_prob_arr = np.full(n, np.nan, dtype=float)
    model_prob_arr = np.full(n, np.nan, dtype=float)
    final_prob_arr = np.full(n, np.nan, dtype=float)
    n_eff_arr = np.full(n, np.nan, dtype=float)

    event_cols = ["Team", "Player", "Game", "StatKey", "Line"]
    w_model = float(confidence_in_model)
    w_market = 1.0 - w_model

    for event_key, g in df.groupby(event_cols, sort=False):
        idxs = g.index.tolist()

        # Canonical row for the underlying event: prefer OVER if present.
        over_rows = g[g["Side"] == "OVER"]
        if len(over_rows) > 0:
            canon = over_rows.iloc[0]
        else:
            canon = g.iloc[0]
        canon_side = canon["Side"]

        # --- Historical Beta-Binomial for OVER event ---
        S_hist = 0.0
        N_hist = 0.0
        for col, base_n, weight in HORIZON_SPECS:
            val = canon.get(col, np.nan)
            if pd.isna(val):
                continue
            r = float(val)  # this is the canonical row's hit rate for its side
            # Convert to OVER orientation if canonical is UNDER
            if canon_side == "UNDER":
                r = 1.0 - r
            n_eff_i = base_n * weight
            S_hist += r * n_eff_i
            N_hist += n_eff_i

        alpha0 = 1.0
        beta0 = 1.0
        if N_hist > 0:
            alpha_hist = S_hist + alpha0
            beta_hist = (N_hist - S_hist) + beta0
            p_hist_over = alpha_hist / (alpha_hist + beta_hist)
            N_hist_eff = alpha_hist + beta_hist  # includes prior
        else:
            p_hist_over = np.nan
            N_hist_eff = 0.0

        # --- Projection-based probability for OVER event ---
        proj = canon.get("Projection", np.nan)
        line = canon.get("Line", np.nan)
        if not pd.isna(proj) and not pd.isna(line):
            try:
                proj_f = float(proj)
                line_f = float(line)
                std = canon.get("StdDev", np.nan)
                if pd.isna(std):
                    # Heuristic std if not provided
                    std = max(0.1, abs(proj_f) * 0.18)
                z = (proj_f - line_f) / float(std)
                p_proj_over = float(norm.cdf(z))
            except Exception:
                p_proj_over = np.nan
        else:
            p_proj_over = np.nan

        # --- Combine hist & projection pseudo-counts into model P(OVER) ---
        S_model = 0.0
        N_model = 0.0
        if not np.isnan(p_hist_over):
            S_model += p_hist_over * N_hist_eff
            N_model += N_hist_eff
        if not np.isnan(p_proj_over):
            S_model += p_proj_over * PROJECTION_PSEUDO_GAMES
            N_model += PROJECTION_PSEUDO_GAMES

        if N_model > 0:
            p_model_over = S_model / N_model
        else:
            p_model_over = np.nan

        # --- Market-implied P(OVER) (de-vigged) ---
        p_market_row = float(canon["Implied Probability"])
        if canon_side == "UNDER":
            p_market_over = 1.0 - p_market_row
        else:
            p_market_over = p_market_row

        # --- Final blended P(OVER) and effective sample size ---
        if np.isnan(p_model_over) and not np.isnan(p_market_over):
            p_final_over = p_market_over
            N_eff = 10.0  # we trust the market somewhat, but not infinitely
        elif not np.isnan(p_model_over) and np.isnan(p_market_over):
            p_final_over = p_model_over
            N_eff = N_model
        elif not np.isnan(p_model_over) and not np.isnan(p_market_over):
            p_final_over = w_market * p_market_over + w_model * p_model_over
            N_eff = N_model
        else:
            p_final_over = 0.5
            N_eff = 2.0

        # Clamp probabilities and cap N_eff to a reasonable range
        p_final_over = float(np.clip(p_final_over, 0.001, 0.999))
        if not np.isnan(p_model_over):
            p_model_over = float(np.clip(p_model_over, 0.001, 0.999))
        if not np.isnan(p_hist_over):
            p_hist_over = float(np.clip(p_hist_over, 0.001, 0.999))

        N_eff = float(max(2.0, min(N_eff, 80.0)))

        # --- Assign to individual rows, flipping for UNDER ---
        for idx in idxs:
            side = df.at[idx, "Side"]
            if side == "UNDER":
                hist_prob_arr[idx] = 1.0 - p_hist_over if not np.isnan(p_hist_over) else np.nan
                proj_prob_arr[idx] = 1.0 - p_proj_over if not np.isnan(p_proj_over) else np.nan
                model_prob_arr[idx] = 1.0 - p_model_over if not np.isnan(p_model_over) else np.nan
                final_prob_arr[idx] = 1.0 - p_final_over
            else:
                hist_prob_arr[idx] = p_hist_over
                proj_prob_arr[idx] = p_proj_over
                model_prob_arr[idx] = p_model_over
                final_prob_arr[idx] = p_final_over
            n_eff_arr[idx] = N_eff

    # Attach all columns
    df["hist_prob"] = hist_prob_arr
    df["proj_prob"] = proj_prob_arr
    df["model_prob"] = model_prob_arr
    df["final_prob"] = np.clip(final_prob_arr, 0.001, 0.999)
    df["n_eff"] = n_eff_arr

    # --- 4) Economic metrics for candidate selection ---
    df["LegDecimal"] = 1.0 / df["Implied Probability"].astype(float)
    df["edge"] = df["final_prob"] - df["Implied Probability"]
    df["single_ev"] = df["final_prob"] * df["LegDecimal"] - 1.0

    return df


# ---------------------------------------------------------------------------
# Copula correlation modeling
# ---------------------------------------------------------------------------

def scale_correlation_by_pace(
    base_corr: float,
    row_i: pd.Series,
    row_j: pd.Series,
    pace_col: str = "Pace",
) -> float:
    """
    Simple heuristic: scale correlation by average pace of the two teams.
    You should calibrate this on historical data.
    """
    try:
        pi = float(row_i.get(pace_col, np.nan))
        pj = float(row_j.get(pace_col, np.nan))
    except Exception:
        pi, pj = np.nan, np.nan

    mult = 1.0
    if not math.isnan(pi) and not math.isnan(pj):
        avg_pace = 0.5 * (pi + pj)
        baseline = 100.0  # NBA-ish
        mult += 0.004 * (avg_pace - baseline)
        mult = max(0.6, min(1.6, mult))
    return base_corr * mult


def get_position_corr(
    pos_i: str,
    pos_j: str,
    default_same_team: float = 0.30,
) -> float:
    """
    Position-based correlation heuristic for same-team props.
    Tune on real data if possible.
    """
    if not isinstance(pos_i, str) or not isinstance(pos_j, str):
        return default_same_team

    pi, pj = pos_i.upper(), pos_j.upper()
    if ("C" in pi and "C" in pj) or ("PF" in pi and "PF" in pj):
        return 0.45
    if ("PG" in pi and any(x in pj for x in ["SG", "SF", "PG"])) or \
       ("PG" in pj and any(x in pi for x in ["SG", "SF", "PG"])):
        return 0.40
    if ("PG" in pi and "C" in pj) or ("C" in pi and "PG" in pj):
        return 0.20
    if (any(x in pi for x in ["SF", "SG"]) and any(x in pj for x in ["PF", "C"])) or \
       (any(x in pj for x in ["SF", "SG"]) and any(x in pi for x in ["PF", "C"])):
        return 0.25
    return default_same_team


def build_conditional_corr_matrix(
    df: pd.DataFrame,
    rho_same_team: float = 0.30,
    rho_opp_team: float = 0.10,
    rho_same_player: float = 0.95,
    pace_col: str = "Pace",
) -> np.ndarray:
    """
    Build an N x N correlation matrix between props.

    Heuristics:
    - Same game & same team → base rho_same_team (modulated by positions & pace).
    - Same game & opposing teams → rho_opp_team (modulated by pace).
    - Same player (multiple props) → max(rho_same_player, team-based correlation).
    """
    n = len(df)
    corr = np.eye(n, dtype=float)

    games = df.get("Game", pd.Series([None] * n)).values
    teams = df.get("Team", pd.Series([None] * n)).values
    players = df.get("Player", pd.Series([None] * n)).values

    has_position = "Position" in df.columns

    for i in range(n):
        for j in range(i + 1, n):
            base_r = 0.0
            if games[i] == games[j]:
                if teams[i] == teams[j]:
                    if has_position:
                        base_r = get_position_corr(
                            str(df.iloc[i].get("Position", "")),
                            str(df.iloc[j].get("Position", "")),
                            default_same_team=rho_same_team,
                        )
                    else:
                        base_r = rho_same_team

                    if players[i] == players[j]:
                        base_r = max(base_r, rho_same_player)
                else:
                    base_r = rho_opp_team

            r = scale_correlation_by_pace(base_r, df.iloc[i], df.iloc[j], pace_col=pace_col)
            corr[i, j] = r
            corr[j, i] = r

    # Clip off-diagonals and enforce SPD
    for i in range(n):
        for j in range(n):
            if i != j:
                corr[i, j] = float(np.clip(corr[i, j], -0.999, 0.999))
    np.fill_diagonal(corr, 1.0)
    corr = ensure_pd(corr, eps=1e-8)
    return corr


# ---------------------------------------------------------------------------
# Monte Carlo engine (Gaussian copula + Beta-Bernoulli marginals)
# ---------------------------------------------------------------------------

def beta_params_from_mean_var(mean: float, var: float) -> Tuple[float, float]:
    """
    Compute Beta(alpha, beta) parameters from mean and variance.
    """
    mean = float(np.clip(mean, 1e-6, 1.0 - 1e-6))
    max_var = mean * (1.0 - mean) - 1e-8
    var = float(min(max_var, max(1e-8, var)))
    m = (mean * (1.0 - mean) / var) - 1.0
    alpha = max(1e-6, mean * m)
    beta = max(1e-6, (1.0 - mean) * m)
    return alpha, beta


def monte_carlo_copula(
    df: pd.DataFrame,
    corr_matrix: np.ndarray,
    trials: int,
    batch_size: int,
    seed: int = 0,
) -> np.ndarray:
    """
    Monte Carlo simulation of joint outcomes using:

    - Gaussian copula with correlation matrix.
    - For each prop i, a Beta-Bernoulli marginal:
        p_i ~ Beta(alpha_i, beta_i)
        outcome_i | p_i ~ Bernoulli(p_i)
    - 'final_prob' is the mean; variance is derived from n_eff.
    Returns:
        out: (n_props x trials) array of 0/1 hits.
    """
    rng = np.random.default_rng(seed)
    n = len(df)

    # Prepare Beta parameters for each leg
    alpha: List[float] = []
    beta: List[float] = []

    for _, row in df.iterrows():
        mean = float(row["final_prob"])
        n_eff = float(row.get("n_eff", 8.0))
        var = max(1e-6, mean * (1.0 - mean) / (n_eff + 1.0))
        a, b = beta_params_from_mean_var(mean, var)
        alpha.append(a)
        beta.append(b)

    corr_matrix = ensure_pd(corr_matrix, eps=1e-8)
    L = cholesky(corr_matrix, lower=True)

    out = np.zeros((n, trials), dtype=np.int8)

    for start, end in batch_iter(trials, batch_size):
        batch_len = end - start

        # Step 1: correlated normals → uniforms
        Z = rng.standard_normal(size=(batch_len, n))
        correlated = Z @ L.T
        U = norm.cdf(correlated)

        # Step 2: for each leg, sample p from Beta and generate Bernoulli outcomes
        block = np.zeros((n, batch_len), dtype=np.int8)
        for i in range(n):
            p_samples = sp_beta.rvs(alpha[i], beta[i], size=batch_len, random_state=rng)
            block[i, :] = (U[:, i] < p_samples).astype(np.int8)

        out[:, start:end] = block

    return out


# ---------------------------------------------------------------------------
# Parlay evaluation + search
# ---------------------------------------------------------------------------

def kelly_from_p_and_decimal(p: float, decimal: float, cap: float = 0.10) -> float:
    """
    Fractional Kelly bet size for a single event with win prob p and decimal odds.
    Returns capped fraction in [0, cap].
    """
    b = decimal - 1.0
    if b <= 0.0:
        return 0.0
    q = 1.0 - p
    f = (b * p - q) / b
    if f <= 0.0:
        return 0.0
    return float(min(f, cap))


def evaluate_parlay(
    indices: List[int],
    out_matrix: np.ndarray,
    df: pd.DataFrame,
    kelly_cap: float,
) -> Optional[Dict[str, Any]]:
    """
    Compute simulated hit probability, EV, and Kelly size for a given parlay.
    """
    joint = out_matrix[indices, :].all(axis=0)
    hit_prob = float(joint.mean())
    if hit_prob <= 0.0:
        return None

    legs = df.iloc[indices]
    implied_probs = legs["Implied Probability"].astype(float).values
    implied_decimals = np.prod([implied_to_decimal(p) for p in implied_probs])
    ev = hit_prob * implied_decimals - 1.0
    kelly = kelly_from_p_and_decimal(hit_prob, implied_decimals, cap=kelly_cap)

    return {
        "indices": indices,
        "legs": legs,
        "hit_prob": hit_prob,
        "implied_decimal": implied_decimals,
        "ev": ev,
        "kelly": kelly,
    }


def search_top_parlays(
    df: pd.DataFrame,
    out_matrix: np.ndarray,
    config: EngineConfig,
) -> List[Dict[str, Any]]:
    """
    Search over combinations of props to find +EV parlays.

    Candidate selection is now EV-driven:
      - Focus on legs with:
            0.45 <= final_prob <= 0.97
            single_ev > 0
      - Score = single_ev (EV of single leg)
      - Take top config.pool_limit legs by score.

    Returns:
        A list of ALL +EV parlays found, sorted by EV (descending).
        Truncation to top_k is now handled by the caller so we can
        reuse the full set for multiple rankings (EV vs balanced).
    """
    df = df.copy()

    # Make sure we have economic metrics
    if "LegDecimal" not in df.columns:
        df["LegDecimal"] = 1.0 / df["Implied Probability"].astype(float)
    if "single_ev" not in df.columns:
        df["single_ev"] = df["final_prob"] * df["LegDecimal"] - 1.0

    # Hard filters for candidate legs
    mask = (df["final_prob"] >= 0.45) & (df["final_prob"] <= 0.97) & (df["single_ev"] > 0.0)
    df_candidates = df[mask]
    if df_candidates.empty:
        df_candidates = df.sort_values("single_ev", ascending=False)
    else:
        df_candidates = df_candidates.sort_values("single_ev", ascending=False)

    candidates = df_candidates.head(config.pool_limit).index.tolist()
    logging.info("Candidate pool size: %d", len(candidates))

    results: List[Dict[str, Any]] = []

    for r in range(2, config.max_legs + 1):
        for combo in combinations(candidates, r):
            if config.avoid_same_player:
                players = df.loc[list(combo), "Player"]
                if len(players) != len(set(players)):
                    continue
            stats = evaluate_parlay(list(combo), out_matrix, df, kelly_cap=config.kelly_cap)
            if stats is not None and stats["ev"] > 0.0:
                results.append(stats)

    results_sorted = sorted(results, key=lambda x: x["ev"], reverse=True)
    return results_sorted   # <<< no truncation here


# ------------------------------
# Special parlay builders (+100 styles)
# ------------------------------

def compute_parlay_decimal_from_rows(legs: pd.DataFrame) -> float:
    """
    Compute combined decimal odds from Implied Probability (preferred) or Odds.
    """
    decimals = []
    for _, r in legs.iterrows():
        if not pd.isna(r.get("Implied Probability", np.nan)):
            p = float(r["Implied Probability"])
        else:
            p = american_to_implied(r["Odds"])
        decimals.append(implied_to_decimal(p))
    if not decimals:
        return 0.0
    return float(np.prod(decimals))


def build_parlay_json_entry(label: str, parlay: dict, df: pd.DataFrame) -> dict:
    """
    Turn a parlay stats dict (from evaluate_parlay) into a JSON-friendly object.
    """
    if parlay is None:
        return None

    indices = parlay["indices"]
    legs_df = df.loc[indices]

    legs_json = []
    for idx in indices:
        r = df.loc[idx]
        legs_json.append({
            "row_index": int(idx),
            "team": r.get("Team"),
            "player": r.get("Player"),
            "game": r.get("Game"),
            "proposition": r.get("Proposition"),
            "line": float(r.get("Line")) if not pd.isna(r.get("Line")) else None,
            "odds": r.get("Odds"),
            "implied_probability": float(r.get("Implied Probability")) if not pd.isna(r.get("Implied Probability")) else None,
            "model_probability": float(r.get("final_prob")) if not pd.isna(r.get("final_prob")) else None,
        })

    return {
        "label": label,
        "num_legs": len(indices),
        "ev": float(parlay["ev"]),
        "hit_probability": float(parlay["hit_prob"]),
        "implied_decimal": float(parlay["implied_decimal"]),
        "implied_american": decimal_to_american(float(parlay["implied_decimal"])),
        "kelly_fraction": float(parlay["kelly"]),
        "legs": legs_json,
    }


def search_plus100_parlay(
    df: pd.DataFrame,
    out_matrix: np.ndarray,
    prob_min: float,
    prob_max: float,
    min_legs: int,
    max_legs: int,
    kelly_cap: float,
    avoid_same_player: bool = True,
    pool_limit: int = 40,
    decimal_target: float = 2.0,
    decimal_tolerance: float = 0.25,
    objective: str = "hit_prob",
) -> dict | None:
    """
    Search for a parlay roughly +100 (decimal ~2.0) under leg-probability constraints.
    - prob_min / prob_max: per-leg true probability range using df['final_prob'].
    - min_legs / max_legs: number of legs allowed.
    - objective:
        'hit_prob' -> maximize hit probability (subject to +EV, +100-ish)
        'ev'       -> maximize EV
    Returns same shape dict as evaluate_parlay(...) or None.
    """
    df = df.copy()

    # Filter legs by probability band
    mask = (df["final_prob"] >= prob_min) & (df["final_prob"] <= prob_max)
    candidates_df = df[mask].copy()
    if candidates_df.empty:
        return None

    # Ensure economic metrics
    if "LegDecimal" not in candidates_df.columns:
        candidates_df["LegDecimal"] = 1.0 / candidates_df["Implied Probability"].astype(float)
    if "single_ev" not in candidates_df.columns:
        candidates_df["single_ev"] = candidates_df["final_prob"] * candidates_df["LegDecimal"] - 1.0

    # Only keep legs with positive single-leg EV
    candidates_df = candidates_df[candidates_df["single_ev"] > 0.0]
    if candidates_df.empty:
        return None

    candidates_df = candidates_df.sort_values("single_ev", ascending=False).head(pool_limit)
    candidate_indices = list(candidates_df.index)

    best_parlay = None
    best_score = None

    for r in range(min_legs, max_legs + 1):
        for combo in combinations(candidate_indices, r):
            # Avoid multi-leg same-player if requested
            if avoid_same_player:
                players = df.loc[list(combo), "Player"]
                if len(players) != len(set(players)):
                    continue

            stats = evaluate_parlay(list(combo), out_matrix, df, kelly_cap=kelly_cap)
            if not stats:
                continue

            dec = stats["implied_decimal"]
            # roughly +100: decimal ~2.0 within tolerance
            if not (decimal_target - decimal_tolerance <= dec <= decimal_target + decimal_tolerance):
                continue

            # Require positive EV
            if stats["ev"] <= 0:
                continue

            if objective == "hit_prob":
                score = (stats["hit_prob"], stats["ev"])
            else:  # 'ev'
                score = (stats["ev"], stats["hit_prob"])

            if (best_score is None) or (score > best_score):
                best_score = score
                best_parlay = stats

    return best_parlay


# ---------------------------------------------------------------------------
# Backtesting (simplified)
# ---------------------------------------------------------------------------

def backtest_on_history_folder(
    history_folder: str,
    engine_cfg: EngineConfig,
) -> Dict[str, Any]:
    """
    Very simple backtest:
    - For each historical sheet with 'Outcome' column (0/1):
        * compute final_prob,
        * for each single leg, see if EV > 0,
        * bet 1 unit on those and compute realized ROI from Outcome.
    - Aggregate ROI across sheets.

    This is intentionally conservative (singles only). You can extend to parlays.
    """
    files = [
        os.path.join(history_folder, f)
        for f in os.listdir(history_folder)
        if f.lower().endswith(".csv") or f.lower().endswith(".xlsx")
    ]
    if not files:
        raise ValueError(f"No history files found in folder: {history_folder}")

    total_bets = 0
    total_profit = 0.0

    for path in files:
        try:
            if path.lower().endswith(".csv"):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
        except Exception as e:
            logging.warning("Skipping %s (read error: %s)", path, e)
            continue

        if "Outcome" not in df.columns:
            logging.warning("Skipping %s (no Outcome column)", path)
            continue

        # Normalize and compute probabilities
        df = preprocess_props_df(df)
        df = hierarchical_minutes_estimate(df)
        df = compute_model_probabilities(df, confidence_in_model=engine_cfg.confidence_in_model)

        for _, row in df.iterrows():
            implied_p = float(row["Implied Probability"])
            decimal = implied_to_decimal(implied_p)
            p = float(row["final_prob"])
            ev = p * decimal - 1.0

            if ev <= 0.0:
                continue

            outcome = int(row["Outcome"])
            if outcome not in (0, 1):
                continue

            # 1 unit stake
            if outcome == 1:
                profit = decimal - 1.0
            else:
                profit = -1.0

            total_profit += profit
            total_bets += 1

    roi = (total_profit / total_bets) if total_bets > 0 else 0.0
    logging.info(
        "Backtest finished: total_bets=%d total_profit=%.3f ROI=%.3f",
        total_bets,
        total_profit,
        roi,
    )
    return {"total_bets": total_bets, "total_profit": total_profit, "roi": roi}


# ---------------------------------------------------------------------------
# Data loading & validation
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "Team",
    "Player",
    "Game",
    "Proposition",
    "Line",
    "Odds",
]

PROBABILITY_COLUMNS = [
    "Implied Probability",
    "Last 5 Hit Rate",
    "Last 10 Hit Rate",
    "Last 20 Hit Rate",
    "H2H Hit Rate",
    "2025 Hit Rate",
    "2024 Hit Rate",
]

def preprocess_props_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean the props dataframe:
    - Strip column names.
    - Check required columns exist.
    - Normalize probability-like columns to floats in [0,1] or NaN.
    - Coerce numeric fields where appropriate.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Normalize probability-like columns
    df = normalize_probability_columns(df, PROBABILITY_COLUMNS)

    # Coerce numeric columns
    for col in ["Line", "Odds"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if df["Odds"].isna().any():
        raise ValueError("Some rows have missing or invalid 'Odds' values.")

    return df


# ---------------------------------------------------------------------------
# Main engine pipeline
# ---------------------------------------------------------------------------

def run_parlay_engine(
    file_path: str,
    cfg: EngineConfig,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    End-to-end pipeline:
      1. Load & preprocess props sheet.
      2. Estimate minutes.
      3. Compute model & final probabilities (event-level, OVER/UNDER paired).
      4. Shrink to a candidate pool based on single-leg EV.
      5. Build correlation matrix.
      6. Run Monte Carlo copula.
      7. Search & return +EV parlays (full set).
      8. Build:
            - A/B/C +100-style parlays
            - Top-K EV parlays
            - Top-K "balanced" EV & hit-rate parlays
         and export all to JSON.
    """
    logging.info("Loading props sheet: %s", file_path)
    if file_path.lower().endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    df = preprocess_props_df(df)
    df = hierarchical_minutes_estimate(df)
    df = compute_model_probabilities(df, confidence_in_model=cfg.confidence_in_model)

    # ---------------------------------------------------
    # Shrink to a candidate pool BEFORE correlations
    # ---------------------------------------------------
    # Focus on legs with sensible final probabilities; sort by single-leg EV.
    mask = (df["final_prob"] >= 0.45) & (df["final_prob"] <= 0.97)
    df_candidates = df[mask].copy()
    if df_candidates.empty:
        df_candidates = df.copy()

    df_candidates = df_candidates.sort_values("single_ev", ascending=False)

    pool_for_corr = max(cfg.pool_limit * 4, 200)  # typically 200–300 candidates
    pool_for_corr = min(pool_for_corr, len(df_candidates))

    df = df_candidates.head(pool_for_corr).reset_index(drop=True)
    logging.info("Using %d legs for correlation / Monte Carlo", len(df))

    # ---------------------------------------------------
    # Build correlation matrix & run Monte Carlo
    # ---------------------------------------------------
    logging.info("Computing correlation matrix...")
    corr = build_conditional_corr_matrix(
        df,
        rho_same_team=cfg.rho_same_team,
        rho_opp_team=cfg.rho_opp_team,
        rho_same_player=cfg.rho_same_player,
        pace_col="Pace",
    )
    logging.info("Correlation matrix shape: %s", corr.shape)

    logging.info("Running Monte Carlo: trials=%d batch_size=%d", cfg.trials, cfg.batch_size)
    start = time()
    out = monte_carlo_copula(
        df,
        corr_matrix=corr,
        trials=cfg.trials,
        batch_size=cfg.batch_size,
        seed=cfg.random_seed,
    )

    elapsed = time() - start
    logging.info("Monte Carlo finished in %.2fs", elapsed)

    # ---------------------------------------------------
    # Search all +EV parlays
    # ---------------------------------------------------
    logging.info("Searching for +EV parlays...")
    all_parlays = search_top_parlays(df, out, cfg)
    logging.info("Found %d +EV parlays.", len(all_parlays))

    if not all_parlays:
        top_ev_parlays: List[Dict[str, Any]] = []
        balanced_parlays: List[Dict[str, Any]] = []
    else:
        # Top-K by EV
        top_ev_parlays = all_parlays[: cfg.top_k]

        # ---------------------------------------------------
        # Balanced EV & hit-rate ranking
        #   - focus on parlays with "middle" hit probs (10%–40%) if possible
        #   - then score = 0.5 * EV_norm + 0.5 * Prob_norm
        # ---------------------------------------------------
        balanced_candidates = [p for p in all_parlays if 0.10 <= p["hit_prob"] <= 0.40]
        if not balanced_candidates:
            balanced_candidates = all_parlays

        ev_max = max(p["ev"] for p in balanced_candidates) if balanced_candidates else 0.0
        prob_max = max(p["hit_prob"] for p in balanced_candidates) if balanced_candidates else 0.0

        def balanced_score(p: Dict[str, Any]) -> float:
            ev_component = (p["ev"] / ev_max) if ev_max > 0 else 0.0
            prob_component = (p["hit_prob"] / prob_max) if prob_max > 0 else 0.0
            return 0.5 * ev_component + 0.5 * prob_component

        balanced_parlays = sorted(balanced_candidates, key=balanced_score, reverse=True)[: cfg.top_k]

    # ---------------------------------------------------
    # A/B/C special +100-style parlays
    # ---------------------------------------------------
    print("\nBuilding special +100-style parlays...")

    ultra_safe_parlay = search_plus100_parlay(
        df,
        out_matrix=out,
        prob_min=0.70,
        prob_max=0.90,
        min_legs=2,
        max_legs=3,
        kelly_cap=cfg.kelly_cap,
        avoid_same_player=cfg.avoid_same_player,
        pool_limit=cfg.pool_limit,
        objective="hit_prob",
    )

    balanced_plus100_parlay = search_plus100_parlay(
        df,
        out_matrix=out,
        prob_min=0.60,
        prob_max=0.80,
        min_legs=3,
        max_legs=4,
        kelly_cap=cfg.kelly_cap,
        avoid_same_player=cfg.avoid_same_player,
        pool_limit=max(cfg.pool_limit, 50),
        objective="ev",
    )

    aggressive_parlay = search_plus100_parlay(
        df,
        out_matrix=out,
        prob_min=0.45,   # allow riskier legs but still not total lottery
        prob_max=0.80,
        min_legs=3,
        max_legs=cfg.max_legs,
        kelly_cap=cfg.kelly_cap,
        avoid_same_player=cfg.avoid_same_player,
        pool_limit=max(cfg.pool_limit, 60),
        objective="ev",
    )

    def _print_special(name: str, parlay_obj: Optional[Dict[str, Any]]) -> None:
        print(f"\n{name}:")
        if parlay_obj is None:
            print("  (no suitable parlay found under constraints)")
            return
        print(
            f"  EV={parlay_obj['ev']:.4f} | "
            f"SimProb={parlay_obj['hit_prob']:.2%} | "
            f"Decimal={parlay_obj['implied_decimal']:.3f} | "
            f"Kelly={parlay_obj['kelly']*100:.2f}%"
        )
        for idx in parlay_obj["indices"]:
            r = df.loc[idx]
            print(
                f"   - {r['Player']} | {r['Proposition']} {r['Line']} "
                f"| Odds {r['Odds']}, final_prob={r['final_prob']:.3f}"
            )

    _print_special("A) Ultra-safe +100 parlay", ultra_safe_parlay)
    _print_special("B) Balanced +100 parlay", balanced_plus100_parlay)
    _print_special("C) Aggressive +EV +100 parlay", aggressive_parlay)

    # ---------------------------------------------------
    # JSON export (top EV + balanced + A/B/C) for dashboard
    # ---------------------------------------------------
    parlay_entries: List[Dict[str, Any]] = []

    # Top EV parlays
    for rank, pstats in enumerate(top_ev_parlays, start=1):
        entry = build_parlay_json_entry(f"top_ev_{rank}", pstats, df)
        if entry is not None:
            parlay_entries.append(entry)

    # Balanced EV & hit-prob parlays
    for rank, pstats in enumerate(balanced_parlays, start=1):
        entry = build_parlay_json_entry(f"balanced_evprob_{rank}", pstats, df)
        if entry is not None:
            parlay_entries.append(entry)

    # Special labeled parlays
    if ultra_safe_parlay is not None:
        parlay_entries.append(build_parlay_json_entry("ultra_safe_plus100", ultra_safe_parlay, df))
    if balanced_plus100_parlay is not None:
        parlay_entries.append(build_parlay_json_entry("balanced_plus100", balanced_plus100_parlay, df))
    if aggressive_parlay is not None:
        parlay_entries.append(build_parlay_json_entry("aggressive_plus100", aggressive_parlay, df))

    try:
        engine_cfg = dict(cfg.__dict__)
    except Exception:
        engine_cfg = None

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": file_path,
        "engine_config": engine_cfg,
        "num_parlays": len(parlay_entries),
        "parlays": parlay_entries,
    }

    json_path = os.path.splitext(file_path)[0] + "_parlays.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=json_default)

    logging.info("JSON export written to: %s", json_path)

    return df, corr, out, top_ev_parlays, balanced_parlays


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Production-leaning parlay engine")
    parser.add_argument("file", help="Props input .xlsx or .csv")
    parser.add_argument("--trials", type=int, default=50_000, help="Monte Carlo trials")
    parser.add_argument("--batch", type=int, default=20_000, help="Batch size for simulation")
    parser.add_argument("--max_legs", type=int, default=4, help="Max legs in parlay search")
    parser.add_argument("--pool_limit", type=int, default=40, help="Candidate legs pool size")
    parser.add_argument("--top_k", type=int, default=20, help="How many parlays to print")
    parser.add_argument("--rho_same_team", type=float, default=0.30)
    parser.add_argument("--rho_opp_team", type=float, default=0.10)
    parser.add_argument("--rho_same_player", type=float, default=0.95)
    parser.add_argument("--confidence", type=float, default=0.35, help="Confidence in model vs market [0,1]")
    parser.add_argument("--kelly_cap", type=float, default=0.10, help="Max Kelly fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--backtest", type=str, default=None, help="Folder of historical sheets to backtest on")
    parser.add_argument("--config_json", type=str, default=None, help="Optional JSON file with config overrides")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> EngineConfig:
    cfg = EngineConfig(
        trials=args.trials,
        batch_size=args.batch,
        max_legs=args.max_legs,
        pool_limit=args.pool_limit,
        top_k=args.top_k,
        confidence_in_model=args.confidence,
        rho_same_team=args.rho_same_team,
        rho_opp_team=args.rho_opp_team,
        rho_same_player=args.rho_same_player,
        kelly_cap=args.kelly_cap,
        random_seed=args.seed,
        debug=args.debug,
    )

    if args.config_json:
        with open(args.config_json, "r") as f:
            overrides = json.load(f)
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    return cfg


def main() -> None:
    args = parse_args()
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    cfg = build_config_from_args(args)
    logging.info("EngineConfig: %s", cfg)

    # Optional backtest
    if args.backtest:
        logging.info("Running backtest on folder: %s", args.backtest)
        stats = backtest_on_history_folder(args.backtest, cfg)
        logging.info("Backtest summary: %s", stats)

    # Run the engine
    df, corr, out, ev_parlays, balanced_parlays = run_parlay_engine(args.file, cfg)

    # -------------------------------
    # Print EV-based parlays
    # -------------------------------
    print(f"\nTop {len(ev_parlays)} parlays by EV:")
    for i, p in enumerate(ev_parlays, start=1):
        print(
            f"\nParlay #{i}: "
            f"EV={p['ev']:.4f}, "
            f"SimProb={p['hit_prob']:.2%}, "
            f"Decimal={p['implied_decimal']:.3f}, "
            f"Kelly={p['kelly']*100:.2f}%"
        )
        for idx in p["indices"]:
            r = df.loc[idx]
            print(
                f"  - {r['Player']} | {r['Proposition']} {r['Line']} "
                f"| Odds {r['Odds']}, final_prob={r['final_prob']:.3f}"
            )

    # -------------------------------
    # Print balanced EV & hit-rate parlays
    # -------------------------------
    print(f"\nTop {len(balanced_parlays)} parlays by balanced EV & hit rate:")
    for i, p in enumerate(balanced_parlays, start=1):
        print(
            f"\nBalanced Parlay #{i}: "
            f"EV={p['ev']:.4f}, "
            f"SimProb={p['hit_prob']:.2%}, "
            f"Decimal={p['implied_decimal']:.3f}, "
            f"Kelly={p['kelly']*100:.2f}%"
        )
        for idx in p["indices"]:
            r = df.loc[idx]
            print(
                f"  - {r['Player']} | {r['Proposition']} {r['Line']} "
                f"| Odds {r['Odds']}, final_prob={r['final_prob']:.3f}"
            )


if __name__ == "__main__":
    main()