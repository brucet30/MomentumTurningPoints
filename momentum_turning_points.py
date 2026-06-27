"""
Momentum Turning Points - Full Replication
==========================================
Paper: Goulding, Harvey & Mazzoleni (2021)
       "Momentum Turning Points", Journal of Financial Economics

Replicates:
  Figure 1  – U.S. Stock Market Cycles (conditional return stats by cycle)
  Table 2   – Performance Summary by Speed
  Table 3   – Market-Cycle Decomposition of Returns (Panel A: means)
  Table 7   – Market Cycle Transition Probabilities

Data: Kenneth French Data Library  (Mkt-RF monthly excess returns, in %)
      Downloaded automatically via pandas-datareader.

Usage: Run with F5 in Spyder.  Results print to console; figures save to
       OUTPUT_DIR and display inline.
"""

# =============================================================================
# CONFIG  – edit these before running
# =============================================================================
EVAL_START  = '1969-01'     # paper evaluation period start (Period string)
EVAL_END    = '2018-12'     # paper evaluation period end
DATA_START  = '1926-01'     # start of French data download (for SLOW warm-up)
DATA_END    = '2019-01'     # one month past eval end so we have 2018-12 fully

OUTPUT_DIR  = r'C:\Users\Betil\OneDrive\Research\MomentumTurningPoints'
SAVE_FIGS   = True          # set False to skip saving (still displays inline)

# Speed blend parameters  a=0 → SLOW,  a=1 → FAST
SPEEDS  = [0.0, 0.25, 0.50, 0.75, 1.0]
LABELS  = {0.0: 'SLOW', 0.25: 'a=1/4', 0.50: 'MED', 0.75: 'a=3/4', 1.0: 'FAST'}

# Paper values for quick comparison (Table 2, Figure 1)
PAPER_TABLE2 = {
    'Market': dict(avg_ann=5.91,  vol_ann=15.64, sharpe=0.38, avg_pos=1.00,
                   beta=1.00,  alpha_ann=0.00,  alpha_tstat=0.00,
                   skewness=-0.55, max_dd=-54.36, avg_over_dd=0.11),
    'SLOW':   dict(avg_ann=6.46,  vol_ann=15.62, sharpe=0.41, avg_pos=0.46,
                   beta=0.15,  alpha_ann=5.58,  alpha_tstat=2.55,
                   skewness=-0.43, max_dd=-43.43, avg_over_dd=0.14),
    'a=1/4':  dict(avg_ann=6.17,  vol_ann=12.72, sharpe=0.48, avg_pos=0.39,
                   beta=0.05,  alpha_ann=5.85,  alpha_tstat=3.26,
                   skewness=-0.13, max_dd=-37.96, avg_over_dd=0.16),
    'MED':    dict(avg_ann=5.88,  vol_ann=11.60, sharpe=0.51, avg_pos=0.32,
                   beta=-0.04, alpha_ann=6.12,  alpha_tstat=3.73,
                   skewness=0.02,  max_dd=-34.43, avg_over_dd=0.17),
    'a=3/4':  dict(avg_ann=5.59,  vol_ann=12.74, sharpe=0.44, avg_pos=0.25,
                   beta=-0.13, alpha_ann=6.39,  alpha_tstat=3.59,
                   skewness=0.03,  max_dd=-34.07, avg_over_dd=0.17),
    'FAST':   dict(avg_ann=5.30,  vol_ann=15.66, sharpe=0.34, avg_pos=0.18,
                   beta=-0.23, alpha_ann=6.66,  alpha_tstat=3.09,
                   skewness=0.15,  max_dd=-44.53, avg_over_dd=0.12),
}
PAPER_FIG1 = {
    #             avg_ret   vol    skew  freq
    'Bull':       ( 9.5,   11.3,  -0.30, 48.3),
    'Correction': ( 6.5,   17.8,  -0.95, 24.5),
    'Bear':       (-7.7,   20.8,   0.05, 16.7),
    'Rebound':    ( 9.6,   17.3,  -0.25, 10.5),
}

# =============================================================================
# IMPORTS
# =============================================================================
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy import stats
import statsmodels.api as sm

warnings.filterwarnings('ignore')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# SECTION 1: DATA
# =============================================================================

def get_french_data(data_start=DATA_START, data_end=DATA_END):
    """
    Download monthly Fama-French factors from the Kenneth French Data Library.

    Returns pd.DataFrame indexed by pandas Period ('M'), returns in percent (%).
    Tries pandas-datareader first; falls back to a direct URL download.
    """
    try:
        import pandas_datareader.data as web
        print("Downloading French factors via pandas-datareader...")
        raw = web.DataReader('F-F_Research_Data_Factors', 'famafrench',
                             start=data_start, end=data_end)
        df = raw[0].copy()          # [0] = monthly table
        # pandas-datareader may return a PeriodIndex already — only convert if needed
        if not isinstance(df.index, pd.PeriodIndex):
            df.index = df.index.to_period('M')
        print(f"  {len(df)} monthly observations: {df.index[0]} to {df.index[-1]}")
        return df
    except Exception as e:
        print(f"  pandas-datareader failed ({type(e).__name__}).  Trying direct URL...")
        return _french_direct_download()


def _french_direct_download():
    """Fallback: fetch the ZIP directly from Dartmouth and parse it."""
    import urllib.request
    import zipfile
    import io

    # Note: French library now requires _CSV suffix and a browser-like User-Agent
    url = ('https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'
           'F-F_Research_Data_Factors_CSV.zip')
    print(f"  Fetching: {url}")
    req = urllib.request.Request(
        url,
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        zf = zipfile.ZipFile(io.BytesIO(resp.read()))

    with zf.open(zf.namelist()[0]) as fh:
        text = fh.read().decode('latin-1')

    # Monthly data appears before the "Annual Factors" section
    monthly_text = text.split('Annual Factors')[0]

    rows = []
    for line in monthly_text.splitlines():
        parts = line.split()
        if len(parts) == 5 and len(parts[0]) == 6:
            try:
                yyyymm = int(parts[0])
                rows.append([yyyymm] + [float(x) for x in parts[1:]])
            except ValueError:
                continue

    df = pd.DataFrame(rows, columns=['yyyymm', 'Mkt-RF', 'SMB', 'HML', 'RF'])
    df['date'] = df['yyyymm'].apply(
        lambda x: pd.Period(year=x // 100, month=x % 100, freq='M'))
    df = df.set_index('date').drop(columns='yyyymm')
    print(f"  {len(df)} monthly observations: {df.index[0]} to {df.index[-1]}")
    return df


# =============================================================================
# SECTION 2: SIGNALS & CYCLE CLASSIFICATION
# =============================================================================

def build_signals(r):
    """
    Construct SLOW and FAST signals at each month t.

    SLOW: sign of trailing 12-month arithmetic average monthly excess return
          (i.e., mean of r[t-11 : t] inclusive — no skip lag per paper §2.1)
    FAST: sign of trailing 1-month excess return (r[t])

    Position rule: nonneg → long (+1), negative → short (-1).

    Returns:
        w_slow, w_fast : pd.Series of {-1.0, +1.0} indexed same as r
    """
    slow_avg = r.rolling(12).mean()          # mean of r[t-11:t+1] at each t
    w_slow   = pd.Series(np.where(slow_avg >= 0, 1.0, -1.0),
                          index=r.index, name='w_slow')
    w_fast   = pd.Series(np.where(r >= 0, 1.0, -1.0),
                          index=r.index, name='w_fast')
    return w_slow, w_fast


def classify_cycles(w_slow, w_fast):
    """
    Map (w_slow, w_fast) pairs at each month t to one of four cycle states.

    Bull       : slow=+1, fast=+1  (agreement: uptrend)
    Correction : slow=+1, fast=-1  (disagreement: possible downtrend onset)
    Bear       : slow=-1, fast=-1  (agreement: downtrend)
    Rebound    : slow=-1, fast=+1  (disagreement: possible uptrend onset)

    Returns pd.Series of strings indexed same as w_slow.
    """
    cycle = pd.Series('', index=w_slow.index, name='cycle')
    cycle[(w_slow ==  1) & (w_fast ==  1)] = 'Bull'
    cycle[(w_slow ==  1) & (w_fast == -1)] = 'Correction'
    cycle[(w_slow == -1) & (w_fast == -1)] = 'Bear'
    cycle[(w_slow == -1) & (w_fast ==  1)] = 'Rebound'
    return cycle


# =============================================================================
# SECTION 3: STRATEGY RETURNS
# =============================================================================

def build_strategies(r, w_slow, w_fast, speeds=SPEEDS):
    """
    Build intermediate-speed strategy returns.

    w(a)[t]         = (1-a)*w_slow[t] + a*w_fast[t]
    r_strategy[t+1] = w(a)[t] * r[t+1]

    In pandas: strat = w.shift(1) * r   (shift makes signal at t align with r at t+1)

    Weights per cycle state:
      Bull       → w(a) =  1.0  for all a  (both agree long)
      Bear       → w(a) = -1.0  for all a  (both agree short)
      Correction → w(a) = 1 - 2a           (MED exits market at a=0.5)
      Rebound    → w(a) = 2a - 1           (MED exits market at a=0.5)

    Returns:
        strats  : dict {a: pd.Series of strategy monthly excess returns (%)}
        weights : dict {a: pd.Series of raw weights w(a)[t]}
    """
    strats  = {}
    weights = {}
    for a in speeds:
        w             = (1 - a) * w_slow + a * w_fast
        weights[a]    = w
        strats[a]     = w.shift(1) * r          # NO look-ahead: signal at t-1
        strats[a].name = LABELS[a]
    return strats, weights


# =============================================================================
# SECTION 4: PERFORMANCE METRICS
# =============================================================================

def compute_metrics(r_strat, r_market, w_strat_full):
    """
    Compute performance metrics matching Table 2.

    Args:
        r_strat      : pd.Series — strategy monthly excess returns (%) in eval period
        r_market     : pd.Series — market monthly excess returns (%) in eval period
        w_strat_full : pd.Series — strategy weights w(a)[t] on FULL index (for avg_pos)

    All returns in percent (e.g., 5.0 means 5%).
    Sharpe ratio and alpha are annualized (×√12 and ×12 respectively).
    """
    rs = r_strat.dropna()
    rm = r_market.reindex(rs.index)

    # Average position: weight IN EFFECT during month t is w[t-1]
    # Use the full weight series so shift(1) finds the prior-period value
    w_eff   = w_strat_full.shift(1).reindex(rs.index)
    avg_pos = w_eff.mean()

    # Return / risk
    avg_ann = rs.mean() * 12
    vol_ann = rs.std(ddof=1) * np.sqrt(12)
    sharpe  = avg_ann / vol_ann

    # CAGR (geometric mean, annualized)
    cum  = (1 + rs / 100).cumprod()
    n    = len(rs)
    cagr = (cum.iloc[-1] ** (12 / n) - 1) * 100   # in percent

    # Beta and Alpha via OLS: r_strat = alpha_monthly + beta * r_market + eps
    X           = sm.add_constant(rm, has_constant='add')
    res         = sm.OLS(rs, X).fit()
    beta        = res.params.iloc[1]
    alpha_ann   = res.params.iloc[0] * 12
    alpha_tstat = res.tvalues.iloc[0]

    # Tail behaviour
    skewness = stats.skew(rs)

    peak   = cum.cummax()
    dd     = (cum - peak) / peak * 100
    max_dd = dd.min()

    avg_over_dd = avg_ann / abs(max_dd)

    return dict(
        avg_ann     = avg_ann,
        cagr        = cagr,
        vol_ann     = vol_ann,
        sharpe      = sharpe,
        avg_pos     = avg_pos,
        beta        = beta,
        alpha_ann   = alpha_ann,
        alpha_tstat = alpha_tstat,
        skewness    = skewness,
        max_dd      = max_dd,
        avg_over_dd = avg_over_dd,
    )


# =============================================================================
# SECTION 5: FIGURE 1
# =============================================================================

def plot_figure1(r, cycle, eval_start=EVAL_START, eval_end=EVAL_END):
    """
    Replicate Figure 1: conditional return stats by market cycle.

    For each cycle state s at month t, collects the SUBSEQUENT month's
    market excess return (r[t+1]) and computes its conditional mean (anlzd.),
    volatility (anlzd.), and skewness.  Relative frequency counts all states
    in the evaluation period.
    """
    em       = (r.index >= eval_start) & (r.index <= eval_end)
    r_ev     = r[em]
    cyc_ev   = cycle[em]

    # Subsequent return: state[t] → r[t+1] (shift(-1) gives next value)
    r_next   = r_ev.shift(-1)

    order  = ['Bull', 'Correction', 'Bear', 'Rebound']
    colors = {'Bull': '#2ecc71', 'Correction': '#3498db',
              'Bear': '#e74c3c',  'Rebound':    '#95a5a6'}

    cond = {}
    for s in order:
        mask = cyc_ev == s
        rn   = r_next[mask].dropna()
        cond[s] = dict(
            avg  = rn.mean() * 12,
            vol  = rn.std(ddof=1) * np.sqrt(12),
            skew = stats.skew(rn),
            freq = mask.mean() * 100,
        )

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle(
        'Figure 1: U.S. Stock Market Cycles\n'
        f'Evaluation Period: {eval_start} to {eval_end}  '
        r'(data source: French Mkt-RF)',
        fontsize=11, fontweight='bold', y=1.01,
    )

    panels = [
        ('avg',  'Avg. Subsequent Return (anlzd., %)',  axes[0, 0]),
        ('vol',  'Subsequent Return Vol. (anlzd., %)',   axes[0, 1]),
        ('skew', 'Subsequent Return Skewness',           axes[1, 0]),
        ('freq', 'Relative Frequency (%)',               axes[1, 1]),
    ]

    for key, title, ax in panels:
        vals = [cond[s][key] for s in order]
        bars = ax.bar(order, vals,
                      color=[colors[s] for s in order],
                      edgecolor='black', linewidth=0.5)
        ax.axhline(0, color='black', linewidth=0.7, zorder=0)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_xlabel("Previous Month's State", fontsize=8)
        for bar, v in zip(bars, vals):
            va  = 'bottom' if v >= 0 else 'top'
            off = 0.02 * ax.get_ylim()[1] * (1 if v >= 0 else -1)
            ax.text(bar.get_x() + bar.get_width() / 2, v + off,
                    f'{v:.1f}' if abs(v) >= 1 else f'{v:.2f}',
                    ha='center', va=va, fontsize=8, fontweight='bold')

    plt.tight_layout()
    if SAVE_FIGS:
        path = os.path.join(OUTPUT_DIR, 'figure1_market_cycles.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.show()

    # --- Console comparison to paper ---
    print("\n  Figure 1 vs. Paper:")
    print(f"  {'State':<12} {'Avg Ret':>12} {'Vol':>12} {'Skew':>12} {'Freq':>12}")
    print("  " + "-" * 60)
    for s in order:
        p  = PAPER_FIG1[s]
        c  = cond[s]
        print(f"  {s:<12} "
              f"{c['avg']:>6.1f}({p[0]:>4.1f})  "
              f"{c['vol']:>6.1f}({p[1]:>4.1f})  "
              f"{c['skew']:>6.2f}({p[2]:>5.2f})  "
              f"{c['freq']:>6.1f}({p[3]:>4.1f})")
    print("  (format: computed(paper))")


# =============================================================================
# SECTION 6: TABLE 2
# =============================================================================

def print_table2(r, strats, weights, eval_start=EVAL_START, eval_end=EVAL_END):
    """
    Replicate Table 2: Performance Summary by Speed.
    All strategies evaluated over the same period as the market.
    """
    em = (r.index >= eval_start) & (r.index <= eval_end)
    r_mkt = r[em]

    # Market: buy-and-hold (weight always +1 on full index)
    w_mkt_full = pd.Series(1.0, index=r.index)
    mkt_m      = compute_metrics(r_mkt, r_mkt, w_mkt_full)

    all_m  = {'Market': mkt_m}
    cols   = ['Market']
    for a in SPEEDS:
        r_s         = strats[a][em]
        m           = compute_metrics(r_s, r_mkt, weights[a])
        all_m[LABELS[a]] = m
        cols.append(LABELS[a])

    ROWS = [
        ('Return and Risk',          None,          None),
        ('  Average (%) (anlzd.)',   'avg_ann',     '{:>8.2f}'),
        ('  CAGR (%) (anlzd.)',      'cagr',        '{:>8.2f}'),
        ('  Volatility (%) (anlzd.)','vol_ann',     '{:>8.2f}'),
        ('  Sharpe Ratio (anlzd.)',  'sharpe',      '{:>8.2f}'),
        ('Market Timing',            None,          None),
        ('  Average Position',       'avg_pos',     '{:>8.2f}'),
        ('  Market Beta',            'beta',        '{:>8.2f}'),
        ('  Alpha (%) (anlzd.)',     'alpha_ann',   '{:>8.2f}'),
        ('  Alpha t-statistic',      'alpha_tstat', '{:>8.2f}'),
        ('Tail Behavior',            None,          None),
        ('  Skewness',               'skewness',    '{:>8.2f}'),
        ('  Max. Drawdown (%)',       'max_dd',      '{:>8.2f}'),
        ('  Avg(anlzd.)/|Max.DD|',   'avg_over_dd', '{:>8.2f}'),
    ]

    w_col = 9  # column width
    header = f"{'':30s}" + "".join(f"{c:>{w_col}s}" for c in cols)

    print("\n" + "=" * (30 + w_col * len(cols)))
    print("TABLE 2: Performance Summary by Speed")
    print(f"Evaluation: {eval_start} to {eval_end}  (N={em.sum()} months)")
    print("=" * (30 + w_col * len(cols)))
    print(header)
    print("-" * (30 + w_col * len(cols)))

    for label, key, fmt in ROWS:
        if key is None:
            print(f"\n{label}")
            continue
        line = f"{label:<30s}"
        for col in cols:
            line += fmt.format(all_m[col][key])
        print(line)

    print("=" * (30 + w_col * len(cols)))

    # --- Comparison to paper ---
    print("\n  Table 2 vs. Paper (key metrics):")
    check_keys = [
        ('avg_ann',     'Avg Return (% anlzd.)'),
        ('vol_ann',     'Volatility (% anlzd.)'),
        ('sharpe',      'Sharpe Ratio'),
        ('avg_pos',     'Avg Position'),
        ('beta',        'Beta'),
        ('alpha_ann',   'Alpha (% anlzd.)'),
        ('alpha_tstat', 'Alpha t-stat'),
        ('skewness',    'Skewness'),
        ('max_dd',      'Max Drawdown (%)'),
    ]
    for col in ['Market', 'SLOW', 'MED', 'FAST']:
        if col not in PAPER_TABLE2:
            continue
        print(f"\n  ── {col} ──")
        for key, desc in check_keys:
            got  = all_m[col][key]
            exp  = PAPER_TABLE2[col][key]
            diff = got - exp
            flag = '✓' if abs(diff) < 0.10 else f'← diff={diff:+.2f}'
            print(f"    {desc:<28s} got={got:>7.2f}  paper={exp:>7.2f}  {flag}")

    return all_m


# =============================================================================
# SECTION 6b: TABLE 2 vs. BUY-AND-HOLD
# =============================================================================

def print_vs_buyandhold(all_metrics):
    """
    Print strategy metrics minus buy-and-hold (Market) for each speed.
    Positive = strategy beats market; negative = market wins.
    """
    mkt   = all_metrics['Market']
    cols  = [LABELS[a] for a in SPEEDS]
    w_col = 9

    DIFF_ROWS = [
        ('Return and Risk',                 None,          None),
        ('  Avg Return (%) (anlzd.)',        'avg_ann',     '{:>+8.2f}'),
        ('  CAGR (%) (anlzd.)',             'cagr',        '{:>+8.2f}'),
        ('  Volatility (%) (anlzd.)',        'vol_ann',     '{:>+8.2f}'),
        ('  Sharpe Ratio',                  'sharpe',      '{:>+8.2f}'),
        ('Market Timing',                   None,          None),
        ('  Alpha vs. Market (% anlzd.)',   'alpha_ann',   '{:>+8.2f}'),
        ('  Alpha t-statistic',             'alpha_tstat', '{:>+8.2f}'),
        ('Tail Behavior',                   None,          None),
        ('  Skewness',                      'skewness',    '{:>+8.2f}'),
        ('  Max. Drawdown (%)',             'max_dd',      '{:>+8.2f}'),
        ('  Avg(anlzd.)/|Max.DD|',          'avg_over_dd', '{:>+8.2f}'),
    ]

    width = 32 + w_col * len(cols)
    print("\n" + "=" * width)
    print("TABLE 2b: Strategy vs. Buy-and-Hold  (strategy minus market)")
    print("  Positive = strategy wins  |  Negative = market wins")
    print("=" * width)
    print(f"{'':32s}" + "".join(f"{c:>{w_col}s}" for c in cols))
    print("-" * width)

    for label, key, fmt in DIFF_ROWS:
        if key is None:
            print(f"\n{label}")
            continue
        line = f"{label:<32s}"
        for col in cols:
            # Alpha and t-stat are already relative to market; others are raw diffs
            if key in ('alpha_ann', 'alpha_tstat'):
                val = all_metrics[col][key]
            else:
                val = all_metrics[col][key] - mkt[key]
            line += fmt.format(val)
        print(line)

    print("=" * width)


# =============================================================================
# SECTION 7: TABLE 3
# =============================================================================

def print_table3(r, strats, cycle, eval_start=EVAL_START, eval_end=EVAL_END):
    """
    Replicate Table 3 Panel A: Cycle-Conditional Return Decomposition.

    Contribution of state s to strategy a's unconditional average return:
        E[r_strategy[t] * 1{cycle[t-1] = s}]  × 12  (annualized)

    Since strategy returns at t use the signal formed at t-1, the cycle
    STATE that generated the signal is cycle[t-1] = cycle.shift(1) at t.
    """
    em         = (r.index >= eval_start) & (r.index <= eval_end)
    cyc_lagged = cycle.shift(1)[em]   # cycle[t-1] aligned with return at t

    all_r  = {'Market': r[em]}
    for a in SPEEDS:
        all_r[LABELS[a]] = strats[a][em]

    order  = ['Bull', 'Correction', 'Bear', 'Rebound']
    cols   = ['Market'] + [LABELS[a] for a in SPEEDS]
    w_col  = 9

    print("\n" + "=" * (20 + w_col * len(cols)))
    print("TABLE 3 (Panel A): Avg Return Decomposition by Cycle (% anlzd.)")
    print(f"Evaluation: {eval_start} to {eval_end}")
    print("=" * (20 + w_col * len(cols)))
    print(f"{'':20s}" + "".join(f"{c:>{w_col}s}" for c in cols))
    print("-" * (20 + w_col * len(cols)))

    line = f"{'Unconditional':<20s}"
    for col in cols:
        line += f"{all_r[col].mean() * 12:>{w_col}.2f}"
    print(line)
    print()

    for s in order:
        mask = cyc_lagged == s
        line = f"{s:<20s}"
        for col in cols:
            contrib = (all_r[col] * mask).mean() * 12
            line += f"{contrib:>{w_col}.2f}"
        print(line)

    print()
    for pair, label in [(['Bull', 'Bear'],        'Bull + Bear'),
                        (['Correction','Rebound'], 'Corr + Rebound')]:
        mask = cyc_lagged.isin(pair)
        line = f"{label:<20s}"
        for col in cols:
            contrib = (all_r[col] * mask).mean() * 12
            line += f"{contrib:>{w_col}.2f}"
        print(line)

    print("=" * (20 + w_col * len(cols)))
    print("  Note: MED (a=0.5) contributions from Correction and Rebound "
          "should both be ≈ 0.00")
    print("        (MED exits the market in those states)")


# =============================================================================
# SECTION 8: TABLE 7
# =============================================================================

def print_table7(cycle, eval_start=EVAL_START, eval_end=EVAL_END):
    """
    Replicate Table 7: Market Cycle Transition Probabilities.

    P(state_{t+1} = s2 | state_t = s1)
    """
    em       = (cycle.index >= eval_start) & (cycle.index <= eval_end)
    cyc_ev   = cycle[em]
    cyc_next = cyc_ev.shift(-1)

    order  = ['Bull', 'Correction', 'Bear', 'Rebound']
    w_col  = 12

    print("\n" + "=" * (16 + w_col * len(order) + 16))
    print("TABLE 7: Market Cycle Transition Probabilities (%)")
    print(f"Evaluation: {eval_start} to {eval_end}")
    print("=" * (16 + w_col * len(order) + 16))
    header = (f"{'':16s}" +
              "".join(f"{s:>{w_col}s}" for s in order) +
              f"{'Up(t+1)':>9s}{'Down(t+1)':>10s}")
    print(header)
    print("-" * (16 + w_col * len(order) + 19))

    paper_t7 = {
        'Bull':       (63.0, 34.9,  1.7,  0.3, 63.3, 36.7),
        'Correction': (61.2, 29.9,  8.8,  0.0, 61.2, 38.8),
        'Bear':       ( 9.0,  0.0, 55.0, 36.0, 45.0, 55.0),
        'Rebound':    (14.3,  1.6, 42.9, 41.3, 55.6, 44.4),
    }

    for s in order:
        mask  = cyc_ev == s
        total = mask.sum()
        line  = f"{s:<16s}"
        for s2 in order:
            p = (mask & (cyc_next == s2)).sum() / total * 100
            line += f"{p:>{w_col}.1f}"
        up   = (mask & cyc_next.isin(['Bull', 'Rebound'])).sum() / total * 100
        down = (mask & cyc_next.isin(['Correction', 'Bear'])).sum() / total * 100
        line += f"{up:>9.1f}{down:>10.1f}"
        print(line)

    print("=" * (16 + w_col * len(order) + 19))
    print("\n  Paper targets (Table 7):")
    for s, vals in paper_t7.items():
        print(f"  {s:<14s}: {vals[0]:>5.1f}  {vals[1]:>5.1f}  {vals[2]:>5.1f}  "
              f"{vals[3]:>5.1f} | Up={vals[4]:.1f}  Down={vals[5]:.1f}")


# =============================================================================
# SECTION 9b: CUMULATIVE PnL PLOT — MED vs. BUY-AND-HOLD
# =============================================================================

def plot_cumulative_pnl(r, strats, r_dyn=None,
                        eval_start=EVAL_START, eval_end=EVAL_END):
    """
    Plot cumulative growth of $1 for MED strategy vs. buy-and-hold market,
    with drawdown panel beneath.
    """
    em     = (r.index >= eval_start) & (r.index <= eval_end)
    r_mkt  = r[em]
    r_med  = strats[0.5][em]

    # Cumulative growth of $1
    cum_mkt = (1 + r_mkt / 100).cumprod()
    cum_med = (1 + r_med / 100).cumprod()
    cum_dyn = (1 + r_dyn[em] / 100).cumprod() if r_dyn is not None else None

    # Drawdown series
    def drawdown(cum):
        peak = cum.cummax()
        return (cum - peak) / peak * 100

    dd_mkt = drawdown(cum_mkt)
    dd_med = drawdown(cum_med)
    dd_dyn = drawdown(cum_dyn) if cum_dyn is not None else None

    # x-axis: convert PeriodIndex to timestamps for matplotlib
    dates = r_mkt.index.to_timestamp()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7),
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True,
    )
    title_suffix = ' & DYN' if r_dyn is not None else ''
    fig.suptitle(
        f'Cumulative Growth of $1: MED{title_suffix} vs. Buy-and-Hold\n'
        f'{eval_start} to {eval_end}  (gross of transaction costs)',
        fontsize=11, fontweight='bold',
    )

    # ── Panel 1: Cumulative PnL ──────────────────────────────────────────────
    ax1.plot(dates, cum_mkt, color='#2c3e50', linewidth=1.4,
             label='Buy-and-Hold (Mkt-RF)')
    ax1.plot(dates, cum_med, color='#3498db', linewidth=1.4,
             label='MED  (a = 0.5 blend)')
    if cum_dyn is not None:
        ax1.plot(dates, cum_dyn, color='#e74c3c', linewidth=1.4,
                 label='DYN  (dynamic speed)')

    ax1.set_yscale('log')
    ax1.set_ylabel('Cumulative Value ($, log scale)', fontsize=9)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f'${y:.0f}' if y >= 1 else f'${y:.2f}'))
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.grid(axis='x', alpha=0.2, linestyle='--')

    # Annotate final values
    for cum, color in [(cum_mkt, '#2c3e50'), (cum_med, '#3498db')]:
        ax1.annotate(f'  ${cum.iloc[-1]:.1f}',
                     xy=(dates[-1], cum.iloc[-1]),
                     fontsize=8, color=color, va='center')
    if cum_dyn is not None:
        ax1.annotate(f'  ${cum_dyn.iloc[-1]:.1f}',
                     xy=(dates[-1], cum_dyn.iloc[-1]),
                     fontsize=8, color='#e74c3c', va='center')
    ax1.legend(fontsize=9, loc='upper left')

    # ── Panel 2: Drawdown ────────────────────────────────────────────────────
    ax2.fill_between(dates, dd_mkt, 0, alpha=0.25, color='#2c3e50',
                     label='Buy-and-Hold')
    ax2.fill_between(dates, dd_med, 0, alpha=0.35, color='#3498db',
                     label='MED')
    if dd_dyn is not None:
        ax2.fill_between(dates, dd_dyn, 0, alpha=0.35, color='#e74c3c',
                         label='DYN')

    ax2.set_ylabel('Drawdown (%)', fontsize=9)
    ax2.set_xlabel('Date', fontsize=9)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.legend(fontsize=8, loc='lower left')

    # Annotate worst drawdowns
    for dd, color in [(dd_mkt, '#2c3e50'), (dd_med, '#3498db')]:
        ax2.annotate(f'{dd.min():.1f}%',
                     xy=(dates[dd.argmin()], dd.min()),
                     xytext=(10, -12), textcoords='offset points',
                     fontsize=7, color=color,
                     arrowprops=dict(arrowstyle='->', color=color, lw=0.8))
    if dd_dyn is not None:
        ax2.annotate(f'{dd_dyn.min():.1f}%',
                     xy=(dates[dd_dyn.argmin()], dd_dyn.min()),
                     xytext=(10, 12), textcoords='offset points',
                     fontsize=7, color='#e74c3c',
                     arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=0.8))

    plt.tight_layout()
    if SAVE_FIGS:
        fname = 'cumulative_pnl_dyn_med_market.png' if r_dyn is not None \
                else 'cumulative_pnl_med_market.png'
        path  = os.path.join(OUTPUT_DIR, fname)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.show()

    # Console summary
    n      = em.sum()
    series = [('Market', cum_mkt, '#2c3e50'), ('MED', cum_med, '#3498db')]
    if cum_dyn is not None:
        series.append(('DYN', cum_dyn, '#e74c3c'))

    print(f"\n  Cumulative PnL Summary ({eval_start} to {eval_end}):")
    hdr = f"  {'':20s}" + "".join(f"{'  '+s[0]:>12s}" for s in series)
    print(hdr)
    print("  " + "-" * (20 + 12 * len(series)))

    finals = [c.iloc[-1] for _, c, _ in series]
    cagrs  = [(f**(12/n) - 1)*100 for f in finals]
    dds    = [drawdown(c).min() for _, c, _ in series]

    for label, vals in [('Final $1 →', [f'${v:.2f}' for v in finals]),
                        ('CAGR (%)',   [f'{v:.2f}'  for v in cagrs]),
                        ('Max DD (%)', [f'{v:.2f}'  for v in dds])]:
        print(f"  {label:<20s}" + "".join(f"{v:>12s}" for v in vals))


# =============================================================================
# SECTION 9c: DYNAMIC SPEED SELECTION (DYN)  — Paper Section 5 / Table 6 / Figure 7
# =============================================================================

def estimate_dyn_speeds(r, cycle, train_mask):
    """
    Estimate optimal state-conditional speeds aCo and aRe from training data.

    Implements Proposition 9, equations (32) and (33):

        aCo = 0.5 * (1 - K * E[r|Co] / E[r²|Co])
        aRe = 0.5 * (1 + K * E[r|Re] / E[r²|Re])

    where K = E[r²|Bu∪Be] * P[Bu∪Be] / (E[r|Bu]*P[Bu] - E[r|Be]*P[Be])

    All moments are of the MARKET RETURN at t+1 conditional on CYCLE STATE at t.
    (State at t predicts the return earned the following month.)

    Returns aCo, aRe clipped to [0, 1] as per paper.
    """
    # r_next[t] = r[t+1] : return earned given state at t
    r_next = r.shift(-1)
    rn     = r_next[train_mask].dropna()
    cyc    = cycle[train_mask].reindex(rn.index)

    def cond_stats(states):
        """Return (E[r|s], E[r²|s], P[s]) for one or more states."""
        mask = cyc.isin(states) if isinstance(states, list) else (cyc == states)
        vals = rn[mask]
        return vals.mean(), (vals**2).mean(), mask.mean()

    mu_bu, _,     p_bu  = cond_stats('Bull')
    mu_be, _,     p_be  = cond_stats('Bear')
    mu_co, sq_co, _     = cond_stats('Correction')
    mu_re, sq_re, _     = cond_stats('Rebound')
    _,     sq_bb, p_bb  = cond_stats(['Bull', 'Bear'])

    denom = mu_bu * p_bu - mu_be * p_be

    # Guard: if condition E[r|Bu]*P[Bu] > E[r|Be]*P[Be] not met, fall back to MED
    if denom <= 0 or sq_co <= 0 or sq_re <= 0:
        return 0.5, 0.5

    K   = sq_bb * p_bb / denom
    aCo = 0.5 * (1.0 - K * mu_co / sq_co)
    aRe = 0.5 * (1.0 + K * mu_re / sq_re)

    return float(np.clip(aCo, 0.0, 1.0)), float(np.clip(aRe, 0.0, 1.0))


def build_dyn_weights(cycle, aCo, aRe):
    """
    Construct DYN weight series w_DYN[t] for given speed parameters.

    Weight applied to return at t+1:
      Bull       →  +1.0          (both signals agree long)
      Bear       →  -1.0          (both signals agree short)
      Correction →  1 - 2*aCo    (aCo=0: stay long; aCo=0.5: exit; aCo=1: go short)
      Rebound    →  2*aRe - 1    (aRe=0: stay short; aRe=0.5: exit; aRe=1: go long)
    """
    w = pd.Series(np.nan, index=cycle.index, name='wDYN')
    w[cycle == 'Bull']       =  1.0
    w[cycle == 'Bear']       = -1.0
    w[cycle == 'Correction'] =  1.0 - 2.0 * aCo
    w[cycle == 'Rebound']    =  2.0 * aRe - 1.0
    return w


def find_opt_speeds(r, cycle, eval_mask, grid_size=101):
    """
    Vectorised grid search for the ex-post optimal (aCo*, aRe*) over the
    evaluation window that maximises the Sharpe ratio.

    The DYN return decomposes linearly in aCo and aRe:
      r_dyn[t] = r_bull[t] - r_bear[t]
               + (1 - 2*aCo) * r_corr[t]
               + (2*aRe - 1) * r_reb[t]

    so we pre-build the four contribution series and vectorise across the grid.
    """
    r_ev       = r[eval_mask]
    cyc_lagged = cycle.shift(1)[eval_mask]     # state that set the weight for r_ev[t]

    r_bull = (r_ev *  (cyc_lagged == 'Bull').astype(float)).values
    r_bear = (r_ev *  (cyc_lagged == 'Bear').astype(float)).values
    r_corr = (r_ev *  (cyc_lagged == 'Correction').astype(float)).values
    r_reb  = (r_ev *  (cyc_lagged == 'Rebound').astype(float)).values

    r_base = r_bull - r_bear                        # shape (n,)

    grid             = np.linspace(0, 1, grid_size)
    aCo_g, aRe_g     = np.meshgrid(grid, grid)     # each (G, G)
    aCo_flat         = aCo_g.ravel()               # (G²,)
    aRe_flat         = aRe_g.ravel()

    # Build all returns at once: (n, G²)
    r_all = (r_base[:, None]
             + (1 - 2*aCo_flat)[None, :] * r_corr[:, None]
             + (2*aRe_flat - 1)[None, :] * r_reb[:, None])

    means = r_all.mean(axis=0)
    stds  = r_all.std(axis=0, ddof=1)
    srs   = np.where(stds > 0, means / stds * np.sqrt(12), -np.inf)

    best  = np.argmax(srs)
    return float(srs[best]), float(aCo_flat[best]), float(aRe_flat[best])


# ---------------------------------------------------------------------------

def print_table6(r, cycle, train_start=None, eval_end=EVAL_END):
    """
    Replicate Table 6: DYN Strategy Performance across evaluation windows.

    For each window:
      - Estimate aCo, aRe from training data (1926-07 to month before eval start)
      - Apply those FIXED speeds over the evaluation period → DYN Sharpe
      - Grid-search ex-post optimal speeds → OPT Sharpe
      - Efficiency = DYN / OPT
    """
    if train_start is None:
        train_start = DATA_START

    eval_starts = ['1969-01', '1974-01', '1979-01', '1984-01',
                   '1989-01', '1994-01', '1999-01', '2004-01']

    paper_t6 = {
        '1969-01': (0.524, 0.00, 0.58, 0.570, 0.920),
        '1974-01': (0.547, 0.07, 0.59, 0.572, 0.956),
        '1979-01': (0.611, 0.08, 0.65, 0.626, 0.977),
        '1984-01': (0.614, 0.22, 0.67, 0.623, 0.985),
        '1989-01': (0.688, 0.26, 0.69, 0.721, 0.954),
        '1994-01': (0.675, 0.11, 0.71, 0.684, 0.988),
        '1999-01': (0.564, 0.17, 0.69, 0.579, 0.975),
        '2004-01': (0.611, 0.16, 0.69, 0.621, 0.984),
    }

    width = 85
    print("\n" + "=" * width)
    print("TABLE 6: DYN Strategy Performance  (training always starts", train_start + ")")
    print("=" * width)
    print(f"  {'Eval Start':>10} {'Yrs':>5} {'aCo':>6} {'aRe':>6} "
          f"{'DYN SR':>8} {'OPT SR':>8} {'Effic':>8}  "
          f"  (paper: DYN / OPT / Eff)")
    print("-" * width)

    results = []
    for es in eval_starts:
        train_mask = (r.index >= train_start) & (r.index < es)
        eval_mask  = (r.index >= es) & (r.index <= eval_end)
        yrs        = eval_mask.sum() / 12

        aCo, aRe = estimate_dyn_speeds(r, cycle, train_mask)

        w_dyn    = build_dyn_weights(cycle, aCo, aRe)
        r_dyn_ev = w_dyn.shift(1)[eval_mask] * r[eval_mask]
        dyn_sr   = r_dyn_ev.mean() / r_dyn_ev.std(ddof=1) * np.sqrt(12)

        opt_sr, opt_aCo, opt_aRe = find_opt_speeds(r, cycle, eval_mask)
        eff = dyn_sr / opt_sr if opt_sr > 0 else np.nan

        p = paper_t6[es]
        print(f"  {es:>10} {yrs:>5.1f} {aCo:>6.2f} {aRe:>6.2f} "
              f"{dyn_sr:>8.3f} {opt_sr:>8.3f} {eff:>8.3f}  "
              f"  ({p[0]:.3f} / {p[3]:.3f} / {p[4]:.3f})")

        results.append(dict(eval_start=es, yrs=yrs, aCo=aCo, aRe=aRe,
                            dyn_sr=dyn_sr, opt_sr=opt_sr, efficiency=eff,
                            w_dyn=w_dyn, r_dyn=r_dyn_ev))

    print("=" * width)
    return results


# ---------------------------------------------------------------------------

def plot_figure7(r, cycle, strats, aCo_main, aRe_main,
                 eval_start=EVAL_START, eval_end=EVAL_END):
    """
    Replicate Figure 7: Sharpe ratio from month t to end of sample.

    For each starting month t in 1969-01 to 2004-01, compute the annualised
    Sharpe ratio of DYN, MED, SLOW, and FAST over [t, 2018-12].

    DYN uses the fixed speeds estimated from the main training window
    (1926-07 to 1968-12) throughout — no re-estimation as t advances.
    """
    em = (r.index >= eval_start) & (r.index <= eval_end)

    # DYN with main training-window speeds
    w_dyn  = build_dyn_weights(cycle, aCo_main, aRe_main)
    r_dyn  = w_dyn.shift(1) * r

    # Collect all strategy series restricted to eval period
    series = {
        'DYN':  r_dyn[em],
        'MED':  strats[0.5][em],
        'SLOW': strats[0.0][em],
        'FAST': strats[1.0][em],
    }

    # Rolling starting months: 1969-01 to 2004-01
    start_periods = [idx for idx in r[em].index
                     if idx <= pd.Period('2004-01', 'M')]

    def sharpe_from(r_series, start):
        rs = r_series[r_series.index >= start].dropna()
        return rs.mean() / rs.std(ddof=1) * np.sqrt(12) if len(rs) >= 12 else np.nan

    dates   = [p.to_timestamp() for p in start_periods]
    colors  = {'DYN': '#e74c3c', 'MED': '#3498db',
               'SLOW': '#95a5a6', 'FAST': '#f39c12'}
    widths  = {'DYN': 1.8, 'MED': 1.4, 'SLOW': 1.0, 'FAST': 1.0}
    dashes  = {'DYN': '-',  'MED': '-', 'SLOW': '--', 'FAST': '--'}

    fig, ax = plt.subplots(figsize=(12, 5))

    for name, rs in series.items():
        srs = [sharpe_from(rs, s) for s in start_periods]
        ax.plot(dates, srs, color=colors[name], linewidth=widths[name],
                linestyle=dashes[name], label=name)

    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_ylabel('Sharpe Ratio (annualized)', fontsize=9)
    ax.set_xlabel(f'Beginning Month  (through {eval_end})', fontsize=9)
    ax.set_title(
        'Figure 7: Sharpe Ratios (Month t to End of Sample) — DYN vs. Static Strategies\n'
        f'DYN trained on {DATA_START}–1968-12:  aCo = {aCo_main:.2f}  '
        f'(paper 0.00)   aRe = {aRe_main:.2f}  (paper 0.58)',
        fontsize=10, fontweight='bold',
    )
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3, linestyle='--')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

    plt.tight_layout()
    if SAVE_FIGS:
        path = os.path.join(OUTPUT_DIR, 'figure7_rolling_sharpe_dyn.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.show()


# =============================================================================
# SECTION 10: CYCLE STATE TIME-SERIES (bonus visual)
# =============================================================================

def plot_cycle_time_series(cycle, eval_start=EVAL_START, eval_end=EVAL_END):
    """
    Plot the cycle state as a colour-coded bar chart over time.
    Useful for visual inspection of Bull/Correction/Bear/Rebound sequences.
    """
    em       = (cycle.index >= eval_start) & (cycle.index <= eval_end)
    cyc_ev   = cycle[em]

    color_map = {'Bull': '#2ecc71', 'Correction': '#3498db',
                 'Bear': '#e74c3c',  'Rebound':    '#95a5a6'}
    num_map   = {'Bull': 3, 'Correction': 2, 'Bear': 0, 'Rebound': 1}

    fig, ax = plt.subplots(figsize=(14, 2.5))
    times   = np.arange(len(cyc_ev))

    for t, (idx, s) in enumerate(cyc_ev.items()):
        ax.bar(t, 1, color=color_map.get(s, 'white'), width=1.0,
               edgecolor='none')

    # x-ticks every 5 years
    tick_pos   = [i for i, idx in enumerate(cyc_ev.index)
                  if idx.month == 1 and idx.year % 5 == 0]
    tick_labels = [str(cyc_ev.index[i].year) for i in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_yticks([])
    ax.set_title('Market Cycle States Over Time', fontsize=10, fontweight='bold')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=v, label=k) for k, v in color_map.items()]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8,
              ncol=4, framealpha=0.9)

    plt.tight_layout()
    if SAVE_FIGS:
        path = os.path.join(OUTPUT_DIR, 'cycle_time_series.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("MOMENTUM TURNING POINTS REPLICATION")
    print("Goulding, Harvey & Mazzoleni (JFE 2021)")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    ff = get_french_data()
    r  = ff['Mkt-RF']    # monthly excess return, in percent (%)

    # ------------------------------------------------------------------
    # 2. Signals (computed on ALL available data — no look-ahead risk)
    # ------------------------------------------------------------------
    print("\nBuilding signals and classifying cycles...")
    w_slow, w_fast = build_signals(r)
    cycle          = classify_cycles(w_slow, w_fast)

    # Sanity check: SLOW needs 12 months to initialise
    first_valid = w_slow.first_valid_index()
    print(f"  First valid SLOW signal: {first_valid}")
    print(f"  First valid strategy return: {pd.Period(str(first_valid), 'M') + 1}")

    # ------------------------------------------------------------------
    # 3. Strategy returns  (all speeds)
    # ------------------------------------------------------------------
    strats, weights = build_strategies(r, w_slow, w_fast)

    # Estimate DYN speeds now so every downstream section can use them
    train_mask_main = (r.index >= DATA_START) & (r.index < EVAL_START)
    aCo_main, aRe_main = estimate_dyn_speeds(r, cycle, train_mask_main)

    em = (r.index >= EVAL_START) & (r.index <= EVAL_END)
    print(f"\nEvaluation period: {EVAL_START} to {EVAL_END}  (N={em.sum()} months)")

    # Quick cycle frequency check
    cyc_ev = cycle[em]
    freq   = cyc_ev.value_counts(normalize=True).mul(100).round(1)
    print("\nCycle frequencies in eval period:")
    for s in ['Bull', 'Correction', 'Bear', 'Rebound']:
        f = freq.get(s, 0.0)
        p = PAPER_FIG1[s][3]
        print(f"  {s:<12s}: {f:>5.1f}%  (paper: {p:.1f}%)")

    # ------------------------------------------------------------------
    # 4. Figure 1
    # ------------------------------------------------------------------
    print("\n" + "─" * 65)
    print("FIGURE 1: Market Cycles Conditional Return Stats")
    print("─" * 65)
    plot_figure1(r, cycle)

    # ------------------------------------------------------------------
    # 5. Table 2
    # ------------------------------------------------------------------
    print("\n" + "─" * 65)
    print("TABLE 2: Performance Summary by Speed")
    print("─" * 65)
    all_metrics = print_table2(r, strats, weights)
    print_vs_buyandhold(all_metrics)

    # ------------------------------------------------------------------
    # 5b. Cumulative PnL: MED vs. Buy-and-Hold
    # ------------------------------------------------------------------
    print("\n" + "─" * 65)
    print("FIGURE 2: Cumulative PnL — DYN & MED vs. Buy-and-Hold")
    print("─" * 65)
    # Build DYN return series using main training-window speeds
    _w_dyn_main = build_dyn_weights(cycle, aCo_main, aRe_main)
    _r_dyn_main = _w_dyn_main.shift(1) * r
    plot_cumulative_pnl(r, strats, r_dyn=_r_dyn_main)

    # ------------------------------------------------------------------
    # 6. Table 3
    # ------------------------------------------------------------------
    print("\n" + "─" * 65)
    print("TABLE 3: Cycle-Conditional Return Decomposition")
    print("─" * 65)
    print_table3(r, strats, cycle)

    # ------------------------------------------------------------------
    # 7. Table 7
    # ------------------------------------------------------------------
    print("\n" + "─" * 65)
    print("TABLE 7: Cycle Transition Probabilities")
    print("─" * 65)
    print_table7(cycle)

    # ------------------------------------------------------------------
    # 8. DYN strategy
    # ------------------------------------------------------------------
    print("\n" + "─" * 65)
    print("DYN: DYNAMIC SPEED SELECTION  (Paper Section 5)")
    print("─" * 65)

    print(f"\n  Main training window: {DATA_START} to 1968-12")
    print(f"  Estimated speeds:  aCo = {aCo_main:.2f}  (paper: 0.00)")
    print(f"                     aRe = {aRe_main:.2f}  (paper: 0.58)")
    print(f"\n  Interpretation:")
    print(f"    aCo={aCo_main:.2f} → after Correction, weight = {1-2*aCo_main:+.2f}  "
          f"({'stay long' if aCo_main < 0.4 else 'reduce long' if aCo_main < 0.6 else 'go short'})")
    print(f"    aRe={aRe_main:.2f} → after Rebound,    weight = {2*aRe_main-1:+.2f}  "
          f"({'slightly long' if aRe_main > 0.5 else 'slightly short' if aRe_main < 0.5 else 'flat'})")

    print("\n  Running Table 6 (8 evaluation windows, grid search for OPT)...")
    table6_results = print_table6(r, cycle)

    print("\n" + "─" * 65)
    print("FIGURE 7: Rolling Sharpe — DYN vs. Static Strategies")
    print("─" * 65)
    plot_figure7(r, cycle, strats, aCo_main, aRe_main)

    # ------------------------------------------------------------------
    # 9. Bonus: cycle time-series bar chart
    # ------------------------------------------------------------------
    print("\n" + "─" * 65)
    print("BONUS: Cycle State Time-Series Plot")
    print("─" * 65)
    plot_cycle_time_series(cycle)

    print("\n" + "=" * 65)
    print("DONE.  All outputs saved to:", OUTPUT_DIR)
    print("=" * 65)


if __name__ == '__main__':
    main()
