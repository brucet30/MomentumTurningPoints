"""
Momentum Turning Points — SPY Implementation
=============================================
Framework from Goulding, Harvey & Mazzoleni (JFE 2021).
Applied to SPY (S&P 500 ETF) monthly total returns via yfinance.

Signals
-------
  SLOW  : 12-month arithmetic average of monthly returns
  FAST  : 1-month return
  BLEND : alpha * FAST + (1-alpha) * SLOW  (alpha in [0, 1])
  DYN   : adapts alpha based on estimated market cycle

Market Cycles (classified from sign of SLOW & FAST)
----------------------------------------------------
  Bull       : SLOW > 0, FAST > 0
  Correction : SLOW > 0, FAST < 0
  Bear       : SLOW < 0, FAST < 0
  Rebound    : SLOW < 0, FAST > 0

Outputs (console + figures)
---------------------------
  1. Cycle frequency & conditional return statistics
  2. Performance by speed (SLOW, FAST, and 4 blended alphas)
  3. Cycle decomposition of returns by speed
  4. Cycle transition probability matrix
  5. Cycle time-series bar chart
  6. Equity curves


"""

import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================

TICKER      = "SPY"
START_DATE  = "1993-02-01"   # SPY launched Jan 1993; need Feb for first monthly return
END_DATE    = "2026-01-01"           # None → today

# Evaluation window for tables / charts
EVAL_START  = "1994-01-01"  # allow 12 months warm-up for SLOW
EVAL_END    = None           # None → end of data

OUTPUT_DIR  = r"C:\Users\Betil\OneDrive\momentum_turning_points"
SAVE_FIGS   = True
LONG_FLAT   = True       # True = long/flat (retail); False = long/short (paper-style)

# Blended strategy alphas to test (0 = pure SLOW, 1 = pure FAST)
BLEND_ALPHAS = [0.0, 0.25, 0.50, 0.75, 1.0]

# DYN training window
# The paper estimates aCo and aRe on data BEFORE the evaluation period.
# Set DYN_TRAIN_END to the last month of your training window (YYYY-MM).
# Everything from EVAL_START onward is out-of-sample for DYN.
# None → use all data before EVAL_START automatically.
DYN_TRAIN_END = "2014-12"   # e.g. "2004-12" to use 1993-2005 as training

# Alpha grid for DYN estimation (finer = slower but more precise)
DYN_ALPHA_GRID = np.arange(0.0, 1.01, 0.05)

# =============================================================================
# DATA
# =============================================================================

def get_spy_monthly(ticker: str, start: str, end=None) -> pd.Series:
    """
    Download daily adjusted close for *ticker* via yfinance and resample
    to end-of-month total returns (%).

    Handles both the legacy flat-column layout and the newer multi-level
    column layout introduced in yfinance >= 0.2.40.
    """
    df = yf.download(ticker, start=start, end=end,
                     auto_adjust=True, progress=False, multi_level_index=False)

    # --- robustly extract Close ---
    if isinstance(df.columns, pd.MultiIndex):
        # Multi-level: (field, ticker) — flatten
        df.columns = ["_".join(c).strip() for c in df.columns]
        # Try exact match first, then partial
        close_cols = [c for c in df.columns if c.lower().startswith("close")]
        if not close_cols:
            raise KeyError(f"Cannot find Close column. Available: {list(df.columns)}")
        raw = df[close_cols[0]]
    else:
        if "Close" in df.columns:
            raw = df["Close"]
        elif "Adj Close" in df.columns:
            raw = df["Adj Close"]
        else:
            # Last resort: try positional (some builds put Close last)
            close_cols = [c for c in df.columns if "close" in c.lower()]
            if not close_cols:
                raise KeyError(f"Cannot find Close column. Available: {list(df.columns)}")
            raw = df[close_cols[0]]

    raw = raw.squeeze()

    if raw.empty:
        raise ValueError(
            f"yfinance returned no data for {ticker}. "
            "Check ticker, dates, and internet connection."
        )

    monthly = raw.resample("ME").last()
    ret = monthly.pct_change().dropna() * 100   # in percent
    ret.index = ret.index.to_period("M")
    ret.name = ticker
    return ret


# =============================================================================
# SIGNALS
# =============================================================================

def build_signals(r: pd.Series):
    """
    SLOW = arithmetic average of the last 12 monthly returns (%).
    FAST = current month's return (%).
    Returns two Series aligned to r's index.
    """
    w_slow = r.rolling(12).mean()          # first valid at month 12
    w_fast = r.copy()
    return w_slow, w_fast


def classify_cycles(w_slow: pd.Series, w_fast: pd.Series) -> pd.Series:
    """
    Assign each month to Bull / Correction / Bear / Rebound.
    NaN where either signal is unavailable.
    """
    def _cycle(s, f):
        if pd.isna(s) or pd.isna(f):
            return np.nan
        if s > 0 and f > 0:
            return "Bull"
        if s > 0 and f < 0:
            return "Correction"
        if s < 0 and f < 0:
            return "Bear"
        if s < 0 and f > 0:
            return "Rebound"
        return np.nan   # edge case: exactly zero

    return pd.Series(
        [_cycle(s, f) for s, f in zip(w_slow, w_fast)],
        index=w_slow.index,
        name="cycle"
    )


def blend_signal(w_slow, w_fast, alpha):
    """Linear blend of SLOW and FAST."""
    return (1 - alpha) * w_slow + alpha * w_fast


def dyn_signal(w_slow, w_fast, cycle, a_co, a_re):
    """
    Cycle-adaptive blended signal using estimated (a_co, a_re).

    Alpha by cycle:
      Bull       : 0.0  (SLOW > 0 and FAST > 0 → always long regardless of blend)
      Correction : a_co (estimated from training data)
      Bear       : 0.0  (SLOW < 0 and FAST < 0 → always short/flat regardless)
      Rebound    : a_re (estimated from training data)
    """
    alpha_map = {"Bull": 0.0, "Correction": a_co, "Bear": 0.0, "Rebound": a_re}
    alpha = cycle.map(alpha_map).fillna(0.0)
    return (1 - alpha) * w_slow + alpha * w_fast


def estimate_dyn_alphas(r, w_slow, w_fast, cycle, train_end):
    """
    Estimate optimal aCo and aRe independently via grid search on the
    training window, matching the paper's Section 5 procedure.

    For each candidate alpha:
      - Build the blended signal using that alpha only in the target state
        (Correction or Rebound), with SLOW used in all other states.
      - Compute the Sharpe ratio of the resulting strategy over the training
        window.
    The alpha that maximises Sharpe is selected.

    Parameters
    ----------
    train_end : pd.Period
        Last month of the training window (inclusive).

    Returns
    -------
    a_co, a_re : float
        Estimated alphas for Correction and Rebound states.
    """
    train_mask = r.index <= train_end

    r_tr    = r[train_mask]
    cyc_tr  = cycle[train_mask]
    ws_tr   = w_slow[train_mask]
    wf_tr   = w_fast[train_mask]

    def sharpe_for_alpha(target_state, alpha):
        """Sharpe of the strategy that uses `alpha` only in target_state."""
        alpha_map = {"Bull": 0.0, "Bear": 0.0,
                     "Correction": 0.0, "Rebound": 0.0}
        alpha_map[target_state] = alpha
        a_series = cyc_tr.map(alpha_map).fillna(0.0)
        sig = (1 - a_series) * ws_tr + a_series * wf_tr
        sr  = strategy_returns(sig, r_tr, LONG_FLAT)
        r_dec = sr.dropna() / 100
        if r_dec.std() == 0 or len(r_dec) < 12:
            return -np.inf
        return r_dec.mean() / r_dec.std() * np.sqrt(12)

    best_co, best_sr_co = 0.0, -np.inf
    best_re, best_sr_re = 0.0, -np.inf

    for a in DYN_ALPHA_GRID:
        s = sharpe_for_alpha("Correction", a)
        if s > best_sr_co:
            best_sr_co, best_co = s, a

        s = sharpe_for_alpha("Rebound", a)
        if s > best_sr_re:
            best_sr_re, best_re = s, a

    return round(best_co, 4), round(best_re, 4)


# =============================================================================
# STRATEGY RETURNS
# =============================================================================

def strategy_returns(signal: pd.Series, r: pd.Series,
                     long_flat: bool = True) -> pd.Series:
    """
    Long when signal_{t-1} > 0, applied to r_t.

    long_flat=True  (default): flat (0%) when signal <= 0.
                               Realistic for a retail investor who will not short.
                               SLOW = step out of SPY when 12-month mean flips negative.
    long_flat=False           : short (-r_t) when signal <= 0  (paper-style L/S).
    """
    sig_lag = signal.shift(1)
    if long_flat:
        pos = (sig_lag > 0).astype(float)   # 1 long, 0 flat
    else:
        pos = np.sign(sig_lag)              # +1 / 0 / -1
    return (pos * r).rename(signal.name)


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

def perf_metrics(ret: pd.Series, freq: int = 12) -> dict:
    """
    Annualised performance metrics from a monthly return series (%).
    ret is in percent (e.g. 1.5 means 1.5%).
    """
    r = ret.dropna() / 100           # convert to decimal

    ann_ret   = r.mean() * freq
    ann_vol   = r.std() * np.sqrt(freq)
    sharpe    = ann_ret / ann_vol if ann_vol > 0 else np.nan

    downside  = r[r < 0].std() * np.sqrt(freq)
    sortino   = ann_ret / downside if downside > 0 else np.nan

    cum       = (1 + r).cumprod()
    roll_max  = cum.cummax()
    dd        = (cum - roll_max) / roll_max
    max_dd    = dd.min()
    calmar    = ann_ret / abs(max_dd) if max_dd < 0 else np.nan

    return {
        "CAGR (%)":        ann_ret * 100,
        "Volatility (%)":  ann_vol * 100,
        "Sharpe":          sharpe,
        "Sortino":         sortino,
        "Max DD (%)":      max_dd * 100,
        "Calmar":          calmar,
    }


# =============================================================================
# OUTPUT SECTIONS
# =============================================================================

def section_cycle_stats(r, cycle, eval_start, eval_end):
    """
    Section 1 — Cycle frequencies and conditional return statistics.
    """
    print("\n" + "=" * 65)
    print("SECTION 1 — CYCLE FREQUENCIES & CONDITIONAL RETURNS")
    print("=" * 65)

    mask = (r.index >= pd.Period(eval_start, "M"))
    if eval_end:
        mask &= (r.index <= pd.Period(eval_end, "M"))

    r_ev   = r[mask]
    cyc_ev = cycle[mask].dropna()
    common = r_ev.index.intersection(cyc_ev.index)
    r_ev   = r_ev[common]
    cyc_ev = cyc_ev[common]

    order   = ["Bull", "Correction", "Bear", "Rebound"]
    total_n = len(cyc_ev)

    rows = []
    for s in order:
        m  = cyc_ev == s
        rc = r_ev[m]
        rows.append({
            "Cycle":     s,
            "N":         m.sum(),
            "Freq (%)":  m.sum() / total_n * 100,
            "Mean (%)":  rc.mean(),
            "Std (%)":   rc.std(),
            "Min (%)":   rc.min(),
            "Max (%)":   rc.max(),
        })

    df = pd.DataFrame(rows).set_index("Cycle")
    print(df.round(2).to_string())
    print(f"\n  Total months evaluated: {total_n}")


def section_performance(r, w_slow, w_fast, cycle, eval_start, eval_end,
                        a_co=0.0, a_re=0.0):
    """
    Section 2 — Performance by speed (SLOW, blends, FAST, DYN).
    a_co, a_re: estimated DYN alphas for Correction and Rebound states.
    """
    mode_str = "Long/Flat (cash when signal ≤ 0)" if LONG_FLAT else "Long/Short"
    print("\n" + "=" * 65)
    print(f"SECTION 2 — PERFORMANCE BY SPEED  [{mode_str}]")
    print(f"  DYN alphas: Correction={a_co:.2f}  Rebound={a_re:.2f}")
    print("=" * 65)

    mask = (r.index >= pd.Period(eval_start, "M"))
    if eval_end:
        mask &= (r.index <= pd.Period(eval_end, "M"))

    rows = []
    strategy_rets = {}

    # Buy-and-hold SPY
    bh = r[mask]
    bh.name = "Buy & Hold SPY"
    strategy_rets["Buy & Hold SPY"] = bh
    m = perf_metrics(bh)
    m["Speed"] = "Buy & Hold SPY"
    rows.append(m)

    # SLOW (alpha=0)
    sig = blend_signal(w_slow, w_fast, 0.0)
    sig.name = "SLOW (α=0)"
    sr  = strategy_returns(sig, r, LONG_FLAT)[mask]
    strategy_rets["SLOW"] = sr
    m = perf_metrics(sr); m["Speed"] = "SLOW (α=0)"; rows.append(m)

    # Blended
    for a in [0.25, 0.50, 0.75]:
        sig = blend_signal(w_slow, w_fast, a)
        sig.name = f"Blend α={a}"
        sr  = strategy_returns(sig, r, LONG_FLAT)[mask]
        strategy_rets[f"α={a}"] = sr
        m = perf_metrics(sr); m["Speed"] = f"Blend (α={a})"; rows.append(m)

    # FAST (alpha=1)
    sig = blend_signal(w_slow, w_fast, 1.0)
    sig.name = "FAST (α=1)"
    sr  = strategy_returns(sig, r, LONG_FLAT)[mask]
    strategy_rets["FAST"] = sr
    m = perf_metrics(sr); m["Speed"] = "FAST (α=1)"; rows.append(m)

    # DYN
    dsig = dyn_signal(w_slow, w_fast, cycle, a_co, a_re)
    dsig.name = "DYN"
    sr   = strategy_returns(dsig, r, LONG_FLAT)[mask]
    strategy_rets["DYN"] = sr
    m = perf_metrics(sr); m["Speed"] = "DYN"; rows.append(m)

    df = pd.DataFrame(rows).set_index("Speed")
    print(df.round(3).to_string())

    return strategy_rets


def section_decomposition(r, w_slow, w_fast, cycle, eval_start, eval_end):
    """
    Section 3 — Cycle decomposition of returns by speed.
    Mean return * frequency = contribution to overall mean.
    """
    print("\n" + "=" * 65)
    print("SECTION 3 — CYCLE DECOMPOSITION OF RETURNS")
    print("=" * 65)

    mask = (r.index >= pd.Period(eval_start, "M"))
    if eval_end:
        mask &= (r.index <= pd.Period(eval_end, "M"))

    order  = ["Bull", "Correction", "Bear", "Rebound"]
    speeds = [0.0, 0.25, 0.50, 0.75, 1.0]
    labels = {a: f"α={a}" for a in speeds}
    labels[0.0] = "SLOW"; labels[1.0] = "FAST"

    cyc_ev = cycle[mask]
    total_n = cyc_ev.dropna().shape[0]

    # freq weights
    freq = {s: (cyc_ev == s).sum() / total_n for s in order}

    print(f"\n  {'Speed':>10s}", end="")
    for s in order:
        print(f"  {s:>12s}", end="")
    print(f"  {'Total':>10s}")
    print("  " + "-" * (10 + 12 * 4 + 12))

    for a in speeds:
        sig = blend_signal(w_slow, w_fast, a)
        sr  = strategy_returns(sig, r, LONG_FLAT)[mask]
        row = f"  {labels[a]:>10s}"
        total = 0.0
        for s in order:
            m    = cyc_ev == s
            cret = sr[m].mean() * freq[s] * 12   # annualised contribution
            row += f"  {cret:>12.3f}"
            total += cret
        row += f"  {total:>10.3f}"
        print(row)

    print("\n  Values = annualised contribution (% per year) from each cycle")


def section_transitions(cycle, eval_start, eval_end):
    """
    Section 4 — Cycle transition probability matrix.
    """
    print("\n" + "=" * 65)
    print("SECTION 4 — CYCLE TRANSITION PROBABILITIES (%)")
    print("=" * 65)

    mask = (cycle.index >= pd.Period(eval_start, "M"))
    if eval_end:
        mask &= (cycle.index <= pd.Period(eval_end, "M"))

    cyc_ev  = cycle[mask]
    cyc_nxt = cyc_ev.shift(-1)
    order   = ["Bull", "Correction", "Bear", "Rebound"]

    hdr = f"  {'':14s}" + "".join(f"{s:>12s}" for s in order) + f"  {'→Up':>7s}  {'→Dn':>7s}"
    print(hdr)
    print("  " + "-" * (14 + 12 * 4 + 18))

    for s in order:
        m     = cyc_ev == s
        total = m.sum()
        if total == 0:
            continue
        row = f"  {s:<14s}"
        for s2 in order:
            p = (m & (cyc_nxt == s2)).sum() / total * 100
            row += f"  {p:>10.1f}"
        up   = (m & cyc_nxt.isin(["Bull", "Rebound"])).sum() / total * 100
        down = (m & cyc_nxt.isin(["Correction", "Bear"])).sum() / total * 100
        row += f"  {up:>7.1f}  {down:>7.1f}"
        print(row)


# =============================================================================
# CHARTS
# =============================================================================

def chart_cycle_bar(cycle, eval_start, eval_end):
    """Horizontal bar chart of cycle states over time."""
    mask = (cycle.index >= pd.Period(eval_start, "M"))
    if eval_end:
        mask &= (cycle.index <= pd.Period(eval_end, "M"))
    cyc_ev = cycle[mask].dropna()

    color_map = {
        "Bull":       "#2ecc71",
        "Correction": "#3498db",
        "Bear":       "#e74c3c",
        "Rebound":    "#95a5a6",
    }

    fig, ax = plt.subplots(figsize=(14, 2.5))
    for t, (idx, s) in enumerate(cyc_ev.items()):
        ax.bar(t, 1, color=color_map.get(s, "white"), width=1.0, edgecolor="none")

    tick_pos    = [i for i, idx in enumerate(cyc_ev.index)
                   if idx.month == 1 and idx.year % 5 == 0]
    tick_labels = [str(cyc_ev.index[i].year) for i in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_yticks([])
    ax.set_title(f"SPY Market Cycle States ({eval_start[:4]}–present)", fontsize=11)

    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(facecolor=v, label=k) for k, v in color_map.items()],
        loc="upper right", fontsize=8, ncol=4, framealpha=0.9
    )
    plt.tight_layout()
    if SAVE_FIGS:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        p = os.path.join(OUTPUT_DIR, "cycle_bar.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"\n  Saved: {p}")
    plt.show()


def chart_equity_curves(strategy_rets: dict, eval_start):
    """
    Equity curves for all strategies + Buy & Hold SPY.
    """
    fig, ax = plt.subplots(figsize=(13, 6))

    colors = {
        "Buy & Hold SPY": "#333333",
        "SLOW":           "#2980b9",
        "α=0.25":         "#8e44ad",
        "α=0.5":          "#27ae60",
        "α=0.75":         "#e67e22",
        "FAST":           "#c0392b",
        "DYN":            "#f39c12",
    }
    lw_map = {"Buy & Hold SPY": 1.5, "DYN": 2.0}

    for label, ret in strategy_rets.items():
        r_dec = ret.dropna() / 100
        curve = (1 + r_dec).cumprod()
        # Convert PeriodIndex to DatetimeIndex for plotting
        curve.index = curve.index.to_timestamp()
        ax.plot(curve.index, curve.values,
                label=label,
                color=colors.get(label, "#555555"),
                linewidth=lw_map.get(label, 1.2),
                alpha=0.85)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.1f}x"))
    ax.set_title(f"Equity Curves — SPY Momentum Strategies ({eval_start[:4]}–present)",
                 fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.legend(fontsize=9, ncol=2, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()

    if SAVE_FIGS:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        p = os.path.join(OUTPUT_DIR, "equity_curves.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.show()


def chart_annual_returns(strategy_rets: dict):
    """
    Grouped bar chart of annual returns for selected strategies.
    """
    selected = ["Buy & Hold SPY", "SLOW", "DYN", "FAST"]
    colors   = ["#333333", "#2980b9", "#f39c12", "#c0392b"]

    annual = {}
    for label in selected:
        if label not in strategy_rets:
            continue
        ret = strategy_rets[label].dropna() / 100
        ret.index = ret.index.to_timestamp()
        annual[label] = ret.resample("YE").apply(lambda x: (1 + x).prod() - 1) * 100

    df = pd.DataFrame(annual)
    df.index = df.index.year

    fig, ax = plt.subplots(figsize=(14, 5))
    n  = len(df.columns)
    w  = 0.18
    x  = np.arange(len(df))

    for i, (col, color) in enumerate(zip(df.columns, colors)):
        ax.bar(x + i * w, df[col], width=w, label=col, color=color, alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x + w * (n - 1) / 2)
    ax.set_xticklabels(df.index.astype(str), rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.set_title("Annual Returns — Selected Strategies", fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()

    if SAVE_FIGS:
        p = os.path.join(OUTPUT_DIR, "annual_returns.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("MOMENTUM TURNING POINTS — SPY IMPLEMENTATION")
    print("Goulding, Harvey & Mazzoleni (JFE 2021) applied to SPY")
    print("=" * 65)

    eval_start = EVAL_START
    eval_end   = EVAL_END  # None → end of data

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    print(f"\nDownloading {TICKER} monthly returns...")
    r = get_spy_monthly(TICKER, START_DATE, END_DATE)
    print(f"  {len(r)} monthly observations: {r.index[0]} → {r.index[-1]}")

    if eval_end is None:
        eval_end_str = str(r.index[-1])
    else:
        eval_end_str = eval_end

    # ------------------------------------------------------------------
    # 2. Signals & Cycles
    # ------------------------------------------------------------------
    print("\nBuilding signals...")
    w_slow, w_fast = build_signals(r)
    cycle          = classify_cycles(w_slow, w_fast)
    first_valid    = w_slow.first_valid_index()
    print(f"  First valid SLOW: {first_valid}  |  Evaluation from: {eval_start}")

    # ------------------------------------------------------------------
    # 3. Estimate DYN alphas on training window
    # ------------------------------------------------------------------
    train_end_period = (
        pd.Period(DYN_TRAIN_END, "M") if DYN_TRAIN_END
        else pd.Period(eval_start, "M") - 1
    )
    print(f"\nEstimating DYN alphas on training window: "
          f"{r.index[0]} → {train_end_period}")
    a_co, a_re = estimate_dyn_alphas(r, w_slow, w_fast, cycle, train_end_period)
    print(f"  Estimated aCo (Correction): {a_co:.2f}  "
          f"aRe (Rebound): {a_re:.2f}")
    print(f"  Paper values (1969–2018 window): aCo=0.00  aRe=0.61")

    # ------------------------------------------------------------------
    # 4. Console Sections
    # ------------------------------------------------------------------
    section_cycle_stats(r, cycle, eval_start, eval_end_str)
    strategy_rets = section_performance(r, w_slow, w_fast, cycle,
                                        eval_start, eval_end_str, a_co, a_re)
    section_decomposition(r, w_slow, w_fast, cycle, eval_start, eval_end_str)
    section_transitions(cycle, eval_start, eval_end_str)

    # ------------------------------------------------------------------
    # 5. Charts
    # ------------------------------------------------------------------
    print("\nGenerating charts...")
    if SAVE_FIGS:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    chart_cycle_bar(cycle, eval_start, eval_end_str)
    chart_equity_curves(strategy_rets, eval_start)
    chart_annual_returns(strategy_rets)

    print("\nDone.")


if __name__ == "__main__":
    main()
