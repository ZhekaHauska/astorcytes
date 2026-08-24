"""
Comet Boxplot Dashboard (Panel + Seaborn)
=============================================

A standalone interactive dashboard for comparing metric distributions
across experiment groups, fetched from the Comet API.

Features
--------
- Connect to any Comet workspace/project
- Select any logged metric and aggregation method
- Choose a window over the time series (full, last N, first N, custom range)
- Map parameters to visual axes (x, color, row, column)
- Filter experiments by non-axis parameters
- Show/hide groups per axis parameter
- Summary statistics table

Usage
-----
    panel serve boxplot_dashboard.py --show

Requires: panel, seaborn, matplotlib, comet_ml, numpy, pandas, diskcache
"""

import os

import panel as pn
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from comet_ml import API
import diskcache
import numpy as np
import pandas as pd

pn.extension()

_CACHE_DIR = os.environ.get(
    "COMET_BOXPLOT_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "comet_boxplot"),
)
_CACHE_TTL = int(os.environ.get("COMET_BOXPLOT_CACHE_TTL", 86400))

_cache = None


def _get_cache():
    global _cache
    if _cache is None:
        _cache = diskcache.Cache(_CACHE_DIR)
    return _cache


def _project_tag(workspace, project):
    return f"{workspace}:{project}"

MAX_GRID_ROWS = 5
MAX_GRID_COLS = 5
AGG_FNS = {
    "average": np.nanmean,
    "min": np.nanmin,
    "max": np.nanmax,
    "sum": np.nansum,
}
NONE_OPT = "\u2014 (single group) \u2014"
PALETTE = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]


def _smart_sort(vals):
    try:
        return sorted(vals, key=lambda v: float(v))
    except (ValueError, TypeError):
        return sorted(vals, key=str)


def _apply_window(values, steps, mode, size, start, end):
    if mode == "Last N" and size is not None:
        return values[-size:]
    if mode == "First N" and size is not None:
        return values[:size:]
    if mode == "Custom range" and start is not None and end is not None:
        sa = np.array(steps, dtype=float) if len(steps) else np.arange(len(values), dtype=float)
        mask = (sa >= start) & (sa <= end)
        return values[mask]
    return values


def _fetch_all(workspace, project):
    cache = _get_cache()
    key = f"all:{workspace}:{project}"
    tag = _project_tag(workspace, project)
    cached = cache.get(key)
    if cached is not None:
        return cached

    api = API()
    experiments = api.get_experiments(workspace, project)
    exp_keys = [exp.key for exp in experiments]

    metric_names = set()
    params_by_key = {}
    for exp in experiments:
        m_summary = exp.get_metrics_summary()
        if m_summary:
            for m in m_summary:
                metric_names.add(m.get("name", m.get("metricName", "")))
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

    metric_names.discard("")
    result = (exp_keys, sorted(metric_names), params_by_key)
    cache.set(key, result, expire=_CACHE_TTL, tag=tag)
    return result


def _fetch_metric(workspace, project, exp_keys, metric_name):
    cache = _get_cache()
    key = f"metric:{workspace}:{project}:{metric_name}"
    tag = _project_tag(workspace, project)
    cached = cache.get(key)
    if cached is not None:
        return cached

    api = API()
    result = api.get_metrics_for_chart(exp_keys, [metric_name])
    cache.set(key, result, expire=_CACHE_TTL, tag=tag)
    return result


def _build_df(exp_keys, params_by_key, metric_data, metric, agg_key,
              w_mode, w_size, w_start, w_end,
              x_param, color_param, row_param, col_param, filter_params):
    agg_fn = AGG_FNS[agg_key]
    rows = []
    for ek in exp_keys:
        if ek not in metric_data:
            continue
        em = metric_data.get(ek, {})
        if "metrics" not in em or not em["metrics"]:
            continue
        for entry in em["metrics"]:
            if entry.get("metricName") != metric:
                continue
            values = entry.get("values", [])
            steps = entry.get("steps", [])
            if not values:
                continue
            va = np.array(values, dtype=float)
            windowed = _apply_window(va, steps, w_mode, w_size, w_start, w_end)
            if len(windowed) == 0:
                continue
            agg_val = float(agg_fn(windowed))
            ep = params_by_key.get(ek, {})
            row = {"experiment_key": ek, "agg_value": agg_val, "_group": "All"}
            if x_param:
                row[x_param] = ep.get(x_param, "N/A")
            if color_param:
                row[color_param] = ep.get(color_param, "N/A")
            if row_param:
                row[row_param] = ep.get(row_param, "N/A")
            if col_param:
                row[col_param] = ep.get(col_param, "N/A")
            for fp in filter_params:
                row[fp] = ep.get(fp, "N/A")
            rows.append(row)
            break
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _make_plot(df, x_param, color_param, row_param, col_param,
               vis, param_filters, title, agg_label):
    if df.empty:
        fig = Figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No data remaining after filters.",
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    x_col = x_param if x_param else "_group"

    has_row = row_param and row_param in df.columns
    has_col = col_param and col_param in df.columns
    has_color = color_param and color_param in df.columns

    if has_row and has_col:
        row_vals = _smart_sort(df[row_param].unique())
        col_vals = _smart_sort(df[col_param].unique())
        if len(row_vals) > MAX_GRID_ROWS or len(col_vals) > MAX_GRID_COLS:
            fig = Figure(figsize=(12, 6))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5,
                    f"Grid too large: {len(row_vals)}\u00d7{len(col_vals)} cells.\n"
                    f"Max is {MAX_GRID_ROWS}\u00d7{MAX_GRID_COLS}.",
                    ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_axis_off()
            return fig

    elif has_row:
        row_vals = _smart_sort(df[row_param].unique())
        if len(row_vals) > MAX_GRID_ROWS:
            fig = Figure(figsize=(12, 6))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Too many row values: {len(row_vals)}.",
                    ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_axis_off()
            return fig

    elif has_col:
        col_vals = _smart_sort(df[col_param].unique())
        if len(col_vals) > MAX_GRID_COLS:
            fig = Figure(figsize=(12, 6))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Too many column values: {len(col_vals)}.",
                    ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_axis_off()
            return fig

    n_x = max(len(df[x_col].unique()), 1) if x_param else 1
    n_hue = len(df[color_param].unique()) if has_color else 0
    n_rows = len(row_vals) if has_row else 1
    n_cols = len(col_vals) if has_col else 1

    facet_height = max(4.0, min(7.0, 2.5 + n_x * 0.3))
    aspect = max(1.5, min(n_x * 0.5 + 0.8 + (0.3 if n_hue > 2 else 0), 5.0))

    total_w = n_cols * facet_height * aspect
    total_h = n_rows * facet_height
    if total_w > 40:
        aspect *= 40 / total_w
    if total_h > 30:
        facet_height *= 30 / total_h

    kwargs = dict(
        data=df, x=x_col, y="agg_value",
        kind="box", palette=PALETTE,
        height=facet_height, aspect=aspect,
    )
    if has_color:
        kwargs["hue"] = color_param
    if has_row:
        kwargs["row"] = row_param
    if has_col:
        kwargs["col"] = col_param

    g = sns.catplot(**kwargs)
    g.figure.suptitle(title, y=1.02)
    g.set_ylabels(agg_label)
    if not x_param:
        g.set_xlabels("")

    g.figure.tight_layout()
    return g.figure


def _build_title(metric, agg_key, w_mode, w_size, w_start, w_end):
    parts = [f"{metric} ({agg_key})"]
    if w_mode != "Full":
        p = w_mode
        if w_mode in ("Last N", "First N") and w_size:
            p += f" = {w_size}"
        elif w_mode == "Custom range" and w_start is not None:
            p += f" [{w_start}\u2026{w_end}]"
        parts.append(p)
    return " \u00b7 ".join(parts)


class BoxplotDashboard:

    def __init__(self):
        self._state = {
            "connected": False,
            "exp_keys": [],
            "metrics": [],
            "params_by_key": {},
            "metric_data": {},
        }
        self._metric_data_cache = {}
        self._build_ui()

    def _build_ui(self):
        self.workspace_input = pn.widgets.TextInput(
            name="Workspace", value="", placeholder="e.g. my-workspace"
        )
        self.project_input = pn.widgets.TextInput(
            name="Project", value="", placeholder="e.g. my-project"
        )
        self.connect_btn = pn.widgets.Button(
            name="\u2728 Connect", button_type="primary", width=200
        )
        self.connect_btn.on_click(self._on_connect)
        self.status = pn.widgets.StaticText(
            value="Enter workspace and project, then click Connect."
        )

        self.metric_select = pn.widgets.Select(
            name="Metric", options=["(connect first)"], disabled=True
        )
        self.agg_select = pn.widgets.Select(
            name="Aggregation", options=list(AGG_FNS.keys()), value="average"
        )

        self.window_mode = pn.widgets.RadioButtonGroup(
            name="Window mode",
            options=["Full", "Last N", "First N", "Custom range"],
            value="Full",
        )
        self.window_size = pn.widgets.IntInput(
            name="Window size (steps)", value=50, start=1, visible=False
        )
        self.window_start = pn.widgets.IntInput(
            name="Start step", value=0, start=0, visible=False
        )
        self.window_end = pn.widgets.IntInput(
            name="End step (inclusive)", value=100, start=0, visible=False
        )

        self.x_select = pn.widgets.Select(
            name="X-axis parameter", options=[NONE_OPT], value=NONE_OPT, disabled=True
        )
        self.color_select = pn.widgets.Select(
            name="Color parameter", options=[NONE_OPT], value=NONE_OPT, disabled=True
        )
        self.row_select = pn.widgets.Select(
            name="Row parameter", options=[NONE_OPT], value=NONE_OPT, disabled=True
        )
        self.col_select = pn.widgets.Select(
            name="Column parameter", options=[NONE_OPT], value=NONE_OPT, disabled=True
        )

        self.filter_params = pn.widgets.MultiChoice(
            name="Filter by parameters",
            options=[],
            disabled=True,
        )
        self._filter_value_widgets = {}
        self.filter_values_column = pn.Column()

        self._vis_widgets = {}
        self.visibility_column = pn.Column()

        self.refresh_btn = pn.widgets.Button(
            name="\U0001f504 Refresh data", button_type="light", width=200
        )
        self.refresh_btn.on_click(self._on_refresh)

        self.error_panel = pn.pane.Markdown("", visible=False)
        _fig = Figure(figsize=(12, 6))
        _ax = _fig.add_subplot(111)
        _ax.text(0.5, 0.5, "Connect to a Comet project to get started.",
                 ha='center', va='center', fontsize=12, transform=_ax.transAxes)
        _ax.set_axis_off()
        self.plot_pane = pn.pane.Matplotlib(_fig, sizing_mode="stretch_width", dpi=144)
        self.stats_pane = pn.pane.DataFrame(sizing_mode="stretch_width")

        self._bind_window_visibility()
        self._bind_grouping_updates()
        self._bind_filter_updates()
        self._bind_metric()
        self._bind_plot()

    def _bind_window_visibility(self):
        def update_window(*events):
            mode = self.window_mode.value
            self.window_size.visible = mode in ("Last N", "First N")
            self.window_start.visible = mode == "Custom range"
            self.window_end.visible = mode == "Custom range"

        pn.bind(update_window, self.window_mode, watch=True)

    def _bind_grouping_updates(self):
        def update_grouping(*events):
            if not self._state["connected"]:
                return
            all_p = self._state.get("all_params", [])
            selects = [self.x_select, self.color_select, self.row_select, self.col_select]
            filter_vals = set(self.filter_params.value)

            for sel in selects:
                other_vals = {s.value for s in selects if s is not sel}
                other_vals.discard(NONE_OPT)
                excluded = other_vals | filter_vals
                available = [NONE_OPT] + [p for p in all_p if p not in excluded]
                old = sel.value
                sel.options = available
                if old in available:
                    sel.value = old
                elif NONE_OPT in available:
                    sel.value = NONE_OPT

            self._update_visibility()

        for w in [self.x_select, self.color_select, self.row_select, self.col_select, self.filter_params]:
            pn.bind(update_grouping, w, watch=True)

    def _bind_filter_updates(self):
        def update_filter(*events):
            if not self._state["connected"]:
                return
            selected = self.filter_params.value
            params_by_key = self._state["params_by_key"]
            exp_keys = self._state["exp_keys"]

            old_keys = set(self._filter_value_widgets.keys())
            new_keys = set(selected)

            for k in old_keys - new_keys:
                self._filter_value_widgets.pop(k, None)

            for k in new_keys:
                if k not in self._filter_value_widgets:
                    p_all = _smart_sort(set(
                        params_by_key[e].get(k, "N/A")
                        for e in exp_keys
                        if e in params_by_key
                    ))
                    w = pn.widgets.MultiChoice(
                        name=f"Filter {k}",
                        options=p_all,
                        value=list(p_all),
                    )
                    pn.bind(self._trigger_plot, w, watch=True)
                    self._filter_value_widgets[k] = w

            self.filter_values_column[:] = list(self._filter_value_widgets.values())
            self._trigger_plot()

        pn.bind(update_filter, self.filter_params, watch=True)

    def _update_visibility(self):
        params_by_key = self._state["params_by_key"]
        exp_keys = self._state["exp_keys"]

        axis_params = {
            self.x_select.value, self.color_select.value,
            self.row_select.value, self.col_select.value
        }
        axis_params.discard(NONE_OPT)

        old_keys = set(self._vis_widgets.keys())
        new_keys = axis_params

        for k in old_keys - new_keys:
            self._vis_widgets.pop(k, None)

        for k in new_keys:
            if k not in self._vis_widgets:
                p_all = _smart_sort(set(
                    params_by_key[e].get(k, "N/A")
                    for e in exp_keys
                    if e in params_by_key
                ))
                w = pn.widgets.MultiChoice(
                    name=f"Show {k}",
                    options=p_all,
                    value=list(p_all),
                )
                pn.bind(self._trigger_plot, w, watch=True)
                self._vis_widgets[k] = w

        self.visibility_column[:] = list(self._vis_widgets.values())

    def _bind_plot(self):
        for w in [
            self.metric_select, self.agg_select,
            self.window_mode, self.window_size, self.window_start, self.window_end,
            self.x_select, self.color_select, self.row_select, self.col_select,
        ]:
            pn.bind(self._trigger_plot, w, watch=True)

    def _bind_metric(self):
        def on_metric_change(*events):
            if self._state["connected"]:
                self._fetch_metric_data()
        pn.bind(on_metric_change, self.metric_select, watch=True)

    def _trigger_plot(self, *events):
        self._update_plot()

    def _on_connect(self, event):
        workspace = self.workspace_input.value.strip()
        project = self.project_input.value.strip()
        if not workspace or not project:
            self.status.value = "\u26a0\ufe0f Please enter both workspace and project."
            return

        self.status.value = "\u23f3 Fetching experiments\u2026"
        try:
            exp_keys, metrics, params_by_key = _fetch_all(workspace, project)
        except Exception as e:
            self.status.value = f"\u274c Error: {e}"
            return

        if not exp_keys:
            self.status.value = "\u26a0\ufe0f No experiments found."
            return

        all_p = sorted(
            set().union(*(d.keys() for d in params_by_key.values()))
            if params_by_key else []
        )

        self._state.update({
            "connected": True,
            "workspace": workspace,
            "project": project,
            "exp_keys": exp_keys,
            "metrics": metrics,
            "params_by_key": params_by_key,
            "all_params": all_p,
            "metric_data": {},
        })
        self._metric_data_cache.clear()

        param_opts = [NONE_OPT] + all_p

        self.metric_select.options = metrics
        self.metric_select.value = metrics[0] if metrics else None
        self.metric_select.disabled = False

        for sel in [self.x_select, self.color_select, self.row_select, self.col_select]:
            sel.options = param_opts
            sel.value = NONE_OPT
            sel.disabled = False

        self.filter_params.options = all_p
        self.filter_params.disabled = False
        self._filter_value_widgets.clear()
        self.filter_values_column[:] = []

        self.status.value = (
            f"\u2705 Connected: {len(exp_keys)} experiments, "
            f"{len(metrics)} metrics, {len(all_p)} parameters."
        )

        self._fetch_metric_data()
        self._update_plot()

    def _on_refresh(self, event):
        self._metric_data_cache.clear()
        if self._state["connected"]:
            workspace = self._state["workspace"]
            project = self._state["project"]
            cache = _get_cache()
            cache.evict(tag=_project_tag(workspace, project))
            self.status.value = "\u23f3 Refreshing\u2026"
            exp_keys, metrics, params_by_key = _fetch_all(workspace, project)
            all_p = sorted(
                set().union(*(d.keys() for d in params_by_key.values()))
                if params_by_key else []
            )
            self._state.update({
                "exp_keys": exp_keys,
                "metrics": metrics,
                "params_by_key": params_by_key,
                "all_params": all_p,
                "metric_data": {},
            })
            self._metric_data_cache.clear()
            self._fetch_metric_data()
            self._update_plot()
            self.status.value = "\u2705 Data refreshed."
        else:
            self.status.value = "\u26a0\ufe0f Not connected."

    def _fetch_metric_data(self):
        metric = self.metric_select.value
        if not metric or not self._state["connected"]:
            return
        cache_key = metric
        if cache_key not in self._metric_data_cache:
            self._metric_data_cache[cache_key] = _fetch_metric(
                self._state["workspace"], self._state["project"],
                self._state["exp_keys"], metric
            )
        self._state["metric_data"] = self._metric_data_cache[cache_key]

    def _update_plot(self):
        if not self._state["connected"]:
            return

        metric = self.metric_select.value
        if not metric:
            return

        metric_data = self._state.get("metric_data", {})
        if not metric_data:
            self._fetch_metric_data()
            metric_data = self._state["metric_data"]

        x_param = None if self.x_select.value == NONE_OPT else self.x_select.value
        color_param = None if self.color_select.value == NONE_OPT else self.color_select.value
        row_param = None if self.row_select.value == NONE_OPT else self.row_select.value
        col_param = None if self.col_select.value == NONE_OPT else self.col_select.value

        vis = {}
        for k, w in self._vis_widgets.items():
            vis[k] = w.value

        filter_params_list = list(self.filter_params.value)
        param_filters = {}
        for k, w in self._filter_value_widgets.items():
            param_filters[k] = w.value

        agg_key = self.agg_select.value
        w_mode = self.window_mode.value
        w_size = self.window_size.value
        w_start = self.window_start.value
        w_end = self.window_end.value

        df = _build_df(
            self._state["exp_keys"],
            self._state["params_by_key"],
            metric_data,
            metric,
            agg_key,
            w_mode,
            w_size if w_mode in ("Last N", "First N") else None,
            w_start if w_mode == "Custom range" else None,
            w_end if w_mode == "Custom range" else None,
            x_param, color_param, row_param, col_param,
            filter_params_list,
        )

        if df.empty:
            _fig = Figure(figsize=(12, 6))
            _ax = _fig.add_subplot(111)
            _ax.text(0.5, 0.5, f"No data for metric **{metric}** with current settings.",
                     ha='center', va='center', fontsize=12, transform=_ax.transAxes)
            _ax.set_axis_off()
            self.plot_pane.object = _fig
            self.stats_pane.object = None
            return

        for p, allowed in vis.items():
            if p in df.columns:
                df = df[df[p].isin(allowed)]
        for p, allowed in param_filters.items():
            if p in df.columns:
                df = df[df[p].isin(allowed)]

        if df.empty:
            _fig = Figure(figsize=(12, 6))
            _ax = _fig.add_subplot(111)
            _ax.text(0.5, 0.5, "No data remaining after filters.",
                     ha='center', va='center', fontsize=12, transform=_ax.transAxes)
            _ax.set_axis_off()
            self.plot_pane.object = _fig
            self.stats_pane.object = None
            return

        title = _build_title(metric, agg_key, w_mode,
                              w_size if w_mode in ("Last N", "First N") else None,
                              w_start if w_mode == "Custom range" else None,
                              w_end if w_mode == "Custom range" else None)
        agg_label = f"{metric} ({agg_key})"

        try:
            fig = _make_plot(
                df, x_param, color_param, row_param, col_param,
                vis, param_filters, title, agg_label
            )
        except Exception as e:
            _fig = Figure(figsize=(12, 6))
            _ax = _fig.add_subplot(111)
            _ax.text(0.5, 0.5, f"Plot error: {e}",
                     ha='center', va='center', fontsize=12, transform=_ax.transAxes)
            _ax.set_axis_off()
            self.plot_pane.object = _fig
            self.stats_pane.object = None
            return

        self.plot_pane.object = fig
        self._update_stats(df, x_param, color_param, row_param, col_param)

    def _update_stats(self, df, x_param, color_param, row_param, col_param):
        group_cols = [
            c for c in [x_param, color_param, row_param, col_param]
            if c and c in df.columns
        ]
        if group_cols:
            summary = (
                df.groupby(group_cols, observed=True)["agg_value"]
                .agg(["count", "mean", "std", "min", lambda s: s.quantile(0.5), "max"])
                .round(6)
            )
            summary.columns = ["Count", "Mean", "Std", "Min", "Median", "Max"]
        else:
            summary = pd.DataFrame({
                "Count": [df["agg_value"].count()],
                "Mean": [df["agg_value"].mean()],
                "Std": [df["agg_value"].std()],
                "Min": [df["agg_value"].min()],
                "Median": [df["agg_value"].median()],
                "Max": [df["agg_value"].max()],
            })

        self.stats_pane.object = summary

    def servable(self):
        sidebar = pn.Column(
            pn.pane.Markdown("## Connection"),
            self.workspace_input,
            self.project_input,
            pn.Row(self.connect_btn, self.refresh_btn),
            self.status,
            pn.layout.Divider(),
            pn.pane.Markdown("## Metric & Aggregation"),
            self.metric_select,
            self.agg_select,
            pn.layout.Divider(),
            pn.pane.Markdown("## Window"),
            self.window_mode,
            self.window_size,
            self.window_start,
            self.window_end,
            pn.layout.Divider(),
            pn.pane.Markdown("## Grouping"),
            self.x_select,
            self.color_select,
            self.row_select,
            self.col_select,
            pn.layout.Divider(),
            pn.pane.Markdown("## Filter"),
            self.filter_params,
            self.filter_values_column,
            pn.layout.Divider(),
            pn.pane.Markdown("## Visibility"),
            self.visibility_column,
            sizing_mode="stretch_width",
        )

        main = pn.Column(
            self.error_panel,
            self.plot_pane,
            pn.layout.Divider(),
            pn.pane.Markdown("## Summary Statistics"),
            self.stats_pane,
            sizing_mode="stretch_width",
        )

        template = pn.template.MaterialTemplate(
            title="Comet Boxplot Dashboard",
            sidebar=[sidebar],
            main=[main],
        )
        return template


dashboard = BoxplotDashboard()
template = dashboard.servable()
template.servable()