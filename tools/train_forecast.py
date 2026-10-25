"""Train and evaluate the 60-minute close-crossing forecast on 9.5 years of gage history.

Protocol: leave-one-year-out. For each year, the model is fitted on the other years and scored on that year.
Event metrics: for every crossing event (stage reaches 5.0 ft after being below 4.5), the warning lead time is
how long before the crossing the alarm first fired (within the previous 6 hours and staying on until the crossing
or for at least 30 min). False alarms: alarm episodes not followed by a crossing within 3 hours, per year.

Compared alarms:
  gage_threshold   the gage itself crosses 5.0 ft (lead 0 by definition: what we have without forecasting)
  local_trend      linear extrapolation of the last 15-min rise reaches 5.0 ft within 60 min
  upstream_rule    the upstream gage rose more than 1 ft in 2 hours
  logistic         riverwatch.forecast model (shipped), probability >= threshold
  boosting         gradient boosting on the same features (reference, not shipped)

usage: train_forecast.py      writes models/forecast.json and data/eval/forecast_summary.json
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from riverwatch import forecast  # noqa: E402

DECK, CLOSE, H = 3.99, 5.0, 4  # horizon: 4 steps of 15 min


def load():
    A = np.load(ROOT / "data/history/08101300.npz"); B = np.load(ROOT / "data/history/08101000.npz")
    t = A["t"]; d = A["v"]
    u = np.interp(t, B["t"], B["v"])
    ub = np.interp(t, B["t"], np.isfinite(B["v"]).astype(float), left=0, right=0)
    u[ub < 1] = np.nan
    return t, d, u


def build(t, d, u):
    n = len(t)
    X = np.full((n, len(forecast.FEATURES)), np.nan)
    for i in range(96, n):
        X[i] = forecast.features_at(d, u, i, DECK)
    fut = np.full(n, np.nan)
    for i in range(n - H):
        w = d[i + 1:i + 1 + H]
        fut[i] = np.nanmax(w) if np.isfinite(w).any() else np.nan
    y = (fut >= CLOSE).astype(float)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(fut) & (d < CLOSE)
    return X, y, valid


def events(t, d):
    ev, last, below = [], -1e18, True
    for i in range(len(t)):
        if not np.isfinite(d[i]):
            continue
        if d[i] < 4.5:
            below = True
        if below and d[i] >= CLOSE and t[i] - last > 86400:
            ev.append(i); last = t[i]; below = False
    return ev


def score_alarm(alarm, t, d, ev, years, yr):
    """Lead times per event and false-alarm episodes per year for a boolean alarm series."""
    leads = []
    for i in ev:
        if yr[i] not in years:
            continue
        a = max(0, i - 24)
        on = np.where(alarm[a:i])[0]
        if len(on) == 0:
            leads.append(0.0)
            continue
        # first alarm in the 6 h window that is not followed by a gap longer than 60 min before the crossing
        idx = a + on
        start = idx[0]
        for k in range(1, len(idx)):
            if idx[k] - idx[k - 1] > 4:
                start = idx[k]
        leads.append((t[i] - t[start]) / 60.0)
    # false alarm episodes: an episode (alarm on while below the close level; gaps under 1 h merged) whose
    # stage does not reach the close level within 3 hours of the episode start
    n = len(t)
    dd = np.nan_to_num(d, nan=-1.0)
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.r_[dd, np.full(12, -1.0)]
    exceed3h = sliding_window_view(pad[1:], 12)[:n].max(axis=1) >= CLOSE
    fa, last_on = 0, -10**9
    for i in range(n):
        if yr[i] not in years or not (alarm[i] and d[i] < CLOSE):
            continue
        if i - last_on > 4 and not exceed3h[i]:
            fa += 1
        last_on = i
    return leads, fa


def below_caution(alarm, t, d, span_years):
    """Alarm episodes after which the water stays below the caution level (4.5 ft) for 3 hours."""
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(t); pad = np.r_[np.nan_to_num(d, nan=-1.0), np.full(12, -1.0)]
    mx = sliding_window_view(pad[1:], 12)[:n].max(axis=1)
    c, last = 0, -99
    for i in np.where(alarm & (d < CLOSE))[0]:
        if i - last > 4 and mx[i] < 4.5:
            c += 1
        last = i
    return round(c / span_years, 1)


def main():
    t, d, u = load()
    X, y, valid = build(t, d, u)
    yr = np.array([datetime.fromtimestamp(x, timezone.utc).year for x in t])
    ev = events(t, d)
    years = sorted(set(yr))
    print("rows", int(valid.sum()), "positives", int(y[valid].sum()), "events", len(ev))

    p_log = np.full(len(t), np.nan); p_gb = np.full(len(t), np.nan)
    for yv in years:
        tr = valid & (yr != yv); te = (yr == yv) & np.isfinite(X).all(axis=1)
        if tr.sum() == 0 or te.sum() == 0:
            continue
        Z = forecast.transform(X[tr]); mu, sd = Z.mean(0), Z.std(0) + 1e-9
        lr = LogisticRegression(C=0.5, class_weight={0: 1, 1: 20}, max_iter=2000).fit((Z - mu) / sd, y[tr])
        p_log[te] = lr.predict_proba((forecast.transform(X[te]) - mu) / sd)[:, 1]
        gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                            class_weight={0: 1, 1: 20}).fit(X[tr], y[tr])
        p_gb[te] = gb.predict_proba(X[te])[:, 1]
        print("fold", yv)

    finite = np.isfinite(X).all(axis=1)
    rise15 = X[:, 1]; up120 = X[:, 5]
    alarms = {
        "gage_threshold": np.zeros(len(t), bool),
        "local_trend": finite & (d + 4 * np.maximum(rise15, 0) >= CLOSE),
        "upstream_rule": finite & (up120 >= 1.0),
    }
    for thr in (0.5, 0.7):
        alarms[f"logistic_p{int(thr * 100)}"] = np.nan_to_num(p_log) >= thr
        alarms[f"boosting_p{int(thr * 100)}"] = np.nan_to_num(p_gb) >= thr

    span_years = (t[-1] - t[0]) / (365.25 * 86400)
    res = {}
    for name, al in alarms.items():
        leads, fa = score_alarm(al, t, d, ev, set(years), yr)
        L = np.array(leads)
        res[name] = {"events": len(L), "warned_ahead": int((L > 0).sum()),
                     "warned_15min_ahead": int((L >= 15).sum()), "warned_30min_ahead": int((L >= 30).sum()),
                     "median_lead_min": float(np.median(L)), "mean_lead_min": round(float(L.mean()), 1),
                     "false_alarm_episodes_per_year": round(fa / span_years, 1)}
        print(f"{name:20s} {json.dumps(res[name])}")

    alarms["shipped_boosting_p50_or_upstream"] = (np.nan_to_num(p_gb) >= 0.5) | (np.nan_to_num(up120) >= forecast.UPSTREAM_RULE_FT)
    leads, fa = score_alarm(alarms["shipped_boosting_p50_or_upstream"], t, d, ev, set(years), yr)
    L = np.array(leads)
    res["shipped_boosting_p50_or_upstream"] = {"events": len(L), "warned_ahead": int((L > 0).sum()),
        "warned_15min_ahead": int((L >= 15).sum()), "warned_30min_ahead": int((L >= 30).sum()),
        "median_lead_min": float(np.median(L)), "mean_lead_min": round(float(L.mean()), 1),
        "false_alarm_episodes_per_year": round(fa / span_years, 1),
        "alarms_with_water_below_caution_per_year": below_caution(alarms["shipped_boosting_p50_or_upstream"], t, d, span_years),
        "per_event_lead_min": {datetime.fromtimestamp(t[i], timezone.utc).strftime("%Y-%m-%d %H:%M"): round(float(l))
                               for i, l in zip(ev, L)}}
    for name in res:
        if "below" not in str(res[name].keys()):
            res[name]["alarms_with_water_below_caution_per_year"] = below_caution(alarms[name], t, d, span_years)
    print("shipped", json.dumps({k: v for k, v in res["shipped_boosting_p50_or_upstream"].items() if k != "per_event_lead_min"}))

    # final shipped model: boosting on all years, exported as JSON trees
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                        class_weight={0: 1, 1: 20}).fit(X[valid], y[valid])
    trees = []
    for (pred,) in gb._predictors:
        nd = pred.nodes
        trees.append({"feature": nd["feature_idx"].tolist(), "threshold": nd["num_threshold"].tolist(),
                      "left": nd["left"].tolist(), "right": nd["right"].tolist(), "is_leaf": nd["is_leaf"].tolist(),
                      "value": nd["value"].tolist(), "missing_left": nd["missing_go_to_left"].tolist()})
    params = {"type": "gbm", "baseline": float(np.ravel(gb._baseline_prediction)[0]), "trees": trees, "threshold": 0.5,
              "features": forecast.FEATURES, "deck_ft": DECK, "close_ft": CLOSE, "horizon_min": H * 15,
              "upstream_rule_ft": forecast.UPSTREAM_RULE_FT,
              "trained_on": [datetime.fromtimestamp(t[0], timezone.utc).date().isoformat(),
                             datetime.fromtimestamp(t[-1], timezone.utc).date().isoformat()], "events": len(ev)}
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "models/forecast.json").write_text(json.dumps(params))
    m = forecast.Model(params)
    idx = np.where(valid)[0][::997]
    diff = max(abs(m.prob(X[i]) - gb.predict_proba(X[i:i + 1])[0, 1]) for i in idx)
    print("numpy export max |p - sklearn p| =", diff)
    assert diff < 1e-6
    out = {"protocol": "leave-one-year-out; event = stage reaches 5.0 ft after being below 4.5 ft",
           "years": [int(x) for x in years], "span_years": round(span_years, 2), "alarms": res}
    (ROOT / "data/eval/forecast_summary.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(ROOT / "data/eval/forecast_cv.npz", t=t, d=d, u=u, p_log=p_log, p_gb=p_gb)


if __name__ == "__main__":
    main()
