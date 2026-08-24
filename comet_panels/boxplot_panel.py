"""
Comet Python Panel: Metric Boxplot
===================================

Group experiments by hyperparameters and compare metric distributions
using interactive boxplots arranged in a 2D subplot grid.

Features
--------
- Select any logged metric from the project
- Aggregate per-experiment time series (average / min / max / sum)
- Choose a window over the time series (full range, first N, last N, custom range)
- Map up to four parameters to visual axes:
    * X-axis  — each unique value becomes a box group
    * Color   — overlaid boxes within each x group
    * Rows    — one subplot row per unique value
    * Columns — one subplot column per unique value
- Filter experiments by any parameter not on an axis
- Show / hide individual groups for each axis parameter
- Summary statistics table

Usage
-----
Paste this entire file into a Comet Python Panel editor.
The panel runs on Comet Compute (CPython 3.9) with streamlit,
plotly, numpy, and pandas pre-installed.
"""

import streamlit as st
from comet_ml import API
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────

MAX_SUBPLOT_ROWS = 5
MAX_SUBPLOT_COLS = 5
AGG_FNS = {
    "average": np.nanmean,
    "min": np.nanmin,
    "max": np.nanmax,
    "sum": np.nansum,
}
PALETTE = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]
NONE_OPT = "\u2014 (single group) \u2014"


def _smart_sort(vals):
    """Sort values numerically when possible, otherwise alphabetically."""
    try:
        return sorted(vals, key=lambda v: float(v))
    except (ValueError, TypeError):
        return sorted(vals, key=str)


def _apply_window(values, steps, mode, size, start, end):
    """Slice a metric time series according to the selected window."""
    if mode == "Last N" and size is not None:
        return values[-size:]
    if mode == "First N" and size is not None:
        return values[:size]
    if mode == "Custom range" and start is not None and end is not None:
        sa = np.array(steps, dtype=float) if len(steps) else np.arange(len(values), dtype=float)
        mask = (sa >= start) & (sa <= end)
        return values[mask]
    return values


# ── Data Fetching (cached) ────────────────────────────────────────────────────

# NOTE  We intentionally do NOT cache the two panel-scoped API calls
#       (get_panel_metrics_names / get_panel_experiment_keys) because
#       their result depends on the current panel filter, which may
#       change between reruns.  The expensive per-experiment calls are
#       cached with a 5-minute TTL and can be force-refreshed via the
#       sidebar button.

@st.cache_data(ttl=300, show_spinner="Fetching experiment data\u2026")
def _fetch_details(exp_keys_tuple, metric_name, refresh_version):
    """Fetch parameters and metric time-series for all experiments.

    *exp_keys_tuple* must be a tuple of strings so it is hashable.
    *refresh_version* changes when the user clicks "Refresh Data",
    forcing a cache miss.
    """
    api = API()
    experiments = api.get_panel_experiments()
    params_by_key = {}
    for exp in experiments:
        if exp.key in exp_keys_tuple:
            raw = exp.get_parameters_summary()
            if raw is None:
                raw = []
            params = {}
            for p in raw:
                pname = p.get("name", p.get("parameterName", ""))
                pval = p.get("valueCurrent", p.get("value", p.get("valueDefault", "")))
                if pname:
                    params[pname] = pval
            params_by_key[exp.key] = params
    metric_data = api.get_metrics_for_chart(list(exp_keys_tuple), [metric_name])
    return params_by_key, metric_data


# ── Sidebar — Project data ────────────────────────────────────────────────────

api = API()
metric_names = sorted(api.get_panel_metrics_names())
exp_keys = api.get_panel_experiment_keys()

if not metric_names:
    st.error("No metrics found in this project.")
    st.stop()

if not exp_keys:
    st.error("No experiments in the current panel scope.")
    st.stop()

# ── Sidebar — Metric & Aggregation ────────────────────────────────────────────

st.sidebar.header("Metric & Aggregation")

sel_metric = st.sidebar.selectbox("Metric", metric_names, index=0)
sel_agg = st.sidebar.selectbox(
    "Aggregation",
    list(AGG_FNS.keys()),
    index=0,
    format_func=lambda x: x.capitalize(),
)

# ── Sidebar — Window ──────────────────────────────────────────────────────────

st.sidebar.header("Window")

w_mode = st.sidebar.radio("Mode", ("Full", "Last N", "First N", "Custom range"), index=0)

w_size = w_start = w_end = None

if w_mode == "Last N":
    w_size = st.sidebar.number_input("Window size (steps)", min_value=1, value=50, step=1)
elif w_mode == "First N":
    w_size = st.sidebar.number_input("Window size (steps)", min_value=1, value=50, step=1)
elif w_mode == "Custom range":
    w_start = st.sidebar.number_input("Start step", min_value=0, value=0, step=1)
    w_end = st.sidebar.number_input("End step (inclusive)", min_value=0, value=100, step=1)

# ── Sidebar — Grouping ────────────────────────────────────────────────────────

# We need parameters before showing grouping widgets.
# Use a placeholder version so that _fetch_details has a cache key.
if "data_version" not in st.session_state:
    st.session_state.data_version = 0

params_by_key, metric_data = _fetch_details(
    tuple(exp_keys), sel_metric, st.session_state.data_version
)

# # ── DEBUG ──────────────────────────────────────────────────────────────────────
#
# with st.expander("Debug info", expanded=False):
#     st.write("### Experiment keys from API")
#     st.write(f"Total keys: {len(exp_keys)}")
#
#     st.write("### Parameters per experiment")
#     for ek in exp_keys:
#         p = params_by_key.get(ek, {})
#         st.write(f"  `{ek[:16]}` → {p}")
#
#     st.write("### Metric data coverage")
#     keys_in_md = [k for k in exp_keys if k in metric_data]
#     md_with_values = []
#     for k in keys_in_md:
#         em = metric_data.get(k, {})
#         metrics = em.get("metrics", [])
#         names = [m.get("metricName", "?") for m in metrics]
#         md_with_values.append((k, names))
#     st.write(f"Keys in metric_data: {len(metric_data)} / {len(exp_keys)}")
#     for ek, names in md_with_values:
#         st.write(f"  `{ek[:16]}` → metrics: {names[:5]}{'…' if len(names) > 5 else ''}")
#
#     st.write("### Raw get_parameters_summary() sample")
#     api_dbg = API()
#     exps_dbg = api_dbg.get_panel_experiments()
#     if exps_dbg:
#         raw = exps_dbg[0].get_parameters_summary()
#         st.write(f"Sample (first exp, {len(raw) if raw else 0} params):")
#         if raw:
#             st.write(raw[:3])
#
# # ── END DEBUG ──────────────────────────────────────────────────────────────────

all_params = sorted(
    set().union(*(d.keys() for d in params_by_key.values())) if params_by_key else []
)

if not all_params:
    st.error("No parameters found across experiments.")
    st.stop()

param_opts = [NONE_OPT] + all_params

st.sidebar.header("Grouping")
x_sel = st.sidebar.selectbox(
    "X-axis parameter",
    param_opts,
    index=0,
    help="Each unique value becomes a box group on the x-axis.",
)
color_sel = st.sidebar.selectbox(
    "Color parameter",
    param_opts,
    index=0,
    help="Overlay differently coloured boxes within each x group.",
)
row_sel = st.sidebar.selectbox(
    "Row parameter (subplot rows)",
    param_opts,
    index=0,
    help="Create one subplot row per unique value.",
)
col_sel = st.sidebar.selectbox(
    "Column parameter (subplot cols)",
    param_opts,
    index=0,
    help="Create one subplot column per unique value.",
)

x_param = None if x_sel == NONE_OPT else x_sel
color_param = None if color_sel == NONE_OPT else color_sel
row_param = None if row_sel == NONE_OPT else row_sel
col_param = None if col_sel == NONE_OPT else col_sel

axis_params = {p for p in [x_param, color_param, row_param, col_param] if p is not None}

# ── Sidebar — Filter ───────────────────────────────────────────────────────────

st.sidebar.header("Filter")

filter_candidates = [p for p in all_params if p not in axis_params]
filter_sel = st.sidebar.multiselect(
    "Filter by parameters",
    filter_candidates,
    help="Select parameters to filter experiments. Only experiments matching all selected values are shown.",
) if filter_candidates else []

param_filters = {}
for p in filter_sel:
    p_all = _smart_sort(set(
        params_by_key[k].get(p, "N/A")
        for k in exp_keys
        if k in params_by_key
    ))
    if not p_all:
        continue
    param_filters[p] = st.sidebar.multiselect(
        f"Filter {p}",
        p_all,
        default=list(p_all),
    )

# ── Sidebar — Visibility ──────────────────────────────────────────────────────

st.sidebar.header("Visibility")
vis = {}
for p in [x_param, color_param, row_param, col_param]:
    if p is None:
        continue
    p_all = _smart_sort(set(
        params_by_key[k].get(p, "N/A")
        for k in exp_keys
        if k in params_by_key
    ))
    if not p_all:
        continue
    vis[p] = st.sidebar.multiselect(
        f"Show {p}",
        p_all,
        default=list(p_all),
    )

if not vis:
    vis = None

# ── Sidebar — Refresh ─────────────────────────────────────────────────────────

st.sidebar.header("Data")
if st.sidebar.button("Refresh data"):
    st.session_state.data_version = st.session_state.get("data_version", 0) + 1
    st.rerun()

# ── Data Processing ────────────────────────────────────────────────────────────

agg_fn = AGG_FNS[sel_agg]
rows = []
_skipped = {"no_metric_data": [], "no_values": [], "empty_window": [], "metric_not_found": []}

for ek in exp_keys:
    if ek not in metric_data:
        _skipped["no_metric_data"].append(ek)
        continue
    em = metric_data.get(ek, {})
    if "metrics" not in em or not em["metrics"]:
        _skipped["no_metric_data"].append(ek)
        continue

    found = False
    for entry in em["metrics"]:
        if entry.get("metricName") != sel_metric:
            continue

        values = entry.get("values", [])
        steps = entry.get("steps", [])
        if not values:
            _skipped["no_values"].append(ek)
            continue

        va = np.array(values, dtype=float)
        windowed = _apply_window(va, steps, w_mode, w_size, w_start, w_end)
        if len(windowed) == 0:
            _skipped["empty_window"].append(ek)
            continue

        agg_val = float(agg_fn(windowed))
        ep = params_by_key.get(ek, {})
        found = True

        row = {
            "experiment_key": ek,
            "agg_value": agg_val,
            "_group": "All",
        }
        if x_param:
            row[x_param] = ep.get(x_param, "N/A")
        if color_param:
            row[color_param] = ep.get(color_param, "N/A")
        if row_param:
            row[row_param] = ep.get(row_param, "N/A")
        if col_param:
            row[col_param] = ep.get(col_param, "N/A")
        for fp in filter_sel:
            row[fp] = ep.get(fp, "N/A")

        rows.append(row)
        break

    if not found:
        _skipped["metric_not_found"].append(ek)

if not rows:
    st.warning(f"No data for metric **{sel_metric}** with the current window settings.")
    st.stop()

df = pd.DataFrame(rows)

for p, allowed in (vis or {}).items():
    if p in df.columns:
        df = df[df[p].isin(allowed)]

for p, allowed in param_filters.items():
    if p in df.columns:
        df = df[df[p].isin(allowed)]

if df.empty:
    st.warning("No data remaining after group filter.")
    st.stop()

# # ── DEBUG: Skipping reasons ───────────────────────────────────────────────────
#
# with st.expander("Skipped experiments", expanded=False):
#     st.write(f"**{len(rows)}** experiments processed · **{len(exp_keys)}** total")
#     for reason, keys in _skipped.items():
#         st.write(f"  **{reason}**: {len(keys)} experiments")
#         for k in keys[:5]:
#             ep = params_by_key.get(k, {})
#             st.write(f"    `{k[:16]}` → params: {ep}")
#         if len(keys) > 5:
#             st.write(f"    … and {len(keys) - 5} more")
#
#     if x_param:
#         st.write(f"### `{x_param}` value distribution in processed rows")
#         if not df.empty and x_param in df.columns:
#             st.dataframe(df[x_param].value_counts().to_frame("count"))
#         st.write(f"### `{x_param}` value distribution in ALL experiment params")
#         all_vals = [params_by_key.get(k, {}).get(x_param, "MISSING") for k in exp_keys]
#         from collections import Counter
#         st.write(dict(Counter(all_vals)))

# ── Subplot dimension checks ──────────────────────────────────────────────────

n_rows = df[row_param].nunique() if row_param else 1
n_cols = df[col_param].nunique() if col_param else 1

if n_rows > MAX_SUBPLOT_ROWS:
    st.error(f"Row parameter has {n_rows} unique values (max {MAX_SUBPLOT_ROWS}). Choose a coarser parameter.")
    st.stop()
if n_cols > MAX_SUBPLOT_COLS:
    st.error(f"Column parameter has {n_cols} unique values (max {MAX_SUBPLOT_COLS}). Choose a coarser parameter.")
    st.stop()

# ── Build Figure ───────────────────────────────────────────────────────────────

title_parts = [f"{sel_metric} ({sel_agg})"]
if w_mode != "Full":
    part = w_mode
    if w_mode in ("Last N", "First N") and w_size:
        part += f" = {w_size}"
    elif w_mode == "Custom range" and w_start is not None:
        part += f" [{w_start}\u2026{w_end}]"
    title_parts.append(part)
title = " \u00b7 ".join(title_parts)

x_col = x_param if x_param else "_group"

cat_orders = {}
if x_param:
    cat_orders[x_param] = vis.get(x_param, _smart_sort(df[x_param].unique())) if vis else _smart_sort(df[x_param].unique())
else:
    cat_orders["_group"] = ["All"]

if color_param:
    cat_orders[color_param] = vis.get(color_param, _smart_sort(df[color_param].unique())) if vis else _smart_sort(df[color_param].unique())
if row_param:
    cat_orders[row_param] = vis.get(row_param, _smart_sort(df[row_param].unique())) if vis else _smart_sort(df[row_param].unique())
if col_param:
    cat_orders[col_param] = vis.get(col_param, _smart_sort(df[col_param].unique())) if vis else _smart_sort(df[col_param].unique())

color_map = None
if color_param:
    unique_colors = _smart_sort(df[color_param].unique())
    color_map = {v: PALETTE[i % len(PALETTE)] for i, v in enumerate(unique_colors)}

box_kwargs = dict(
    data_frame=df,
    x=x_col,
    y="agg_value",
    points="all",
    title=title,
    category_orders=cat_orders,
    labels={"agg_value": f"{sel_metric} ({sel_agg})"},
)
if color_param:
    box_kwargs["color"] = color_param
    box_kwargs["color_discrete_map"] = color_map
if row_param:
    box_kwargs["facet_row"] = row_param
if col_param:
    box_kwargs["facet_col"] = col_param

fig = px.box(**box_kwargs)

fig.update_traces(jitter=0.3, pointpos=-1.8, marker=dict(size=4))
if color_param:
    fig.update_layout(boxmode="group")

# Clean up facet annotation labels (remove "param=" prefix)
if row_param or col_param:
    fig.for_each_annotation(
        lambda a: a.update(text=a.text.split("=")[-1] if "=" in a.text else a.text)
    )

# Improve facet spacing
fig.update_layout(
    height=max(400, 300 * n_rows) if (n_rows > 1 or n_cols > 1) else None,
)

st.plotly_chart(fig, use_container_width=True)

# ── Summary Statistics ─────────────────────────────────────────────────────────

with st.expander("Summary statistics", expanded=False):
    group_cols = [c for c in [x_param, color_param, row_param, col_param] if c and c in df.columns]
    if group_cols:
        summary = (
            df.groupby(group_cols, observed=True)["agg_value"]
            .agg(["count", "mean", "std", "min", lambda s: s.quantile(0.5), "max"])
            .round(6)
        )
        summary.columns = ["Count", "Mean", "Std", "Min", "Median", "Max"]
    else:
        summary = pd.DataFrame(
            {
                "Count": [df["agg_value"].count()],
                "Mean": [df["agg_value"].mean()],
                "Std": [df["agg_value"].std()],
                "Min": [df["agg_value"].min()],
                "Median": [df["agg_value"].median()],
                "Max": [df["agg_value"].max()],
            },
            index=[sel_metric],
        ).round(6)
    st.dataframe(summary)