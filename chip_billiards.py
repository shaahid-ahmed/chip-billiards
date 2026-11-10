#!/usr/bin/env python3
"""Chip Billiards — where does a TE/RB chip send the pass rusher, and does it matter?

Treat every PFF-labeled chip block (pff_blockType == 'CH') as a collision: compare the chipped rusher's
velocity in the 0.3 s before vs after contact with the chipper, classify the deflection
(braked = closing speed toward the QB dropped / no change / ran through it), and relate it to whether that rusher
recorded a pressure (PFF hit, hurry or sack).

Run:  python get_data.py && python chip_billiards.py      (outputs land in ./outputs)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("BDB_DATA", os.path.join(HERE, "data"))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

SKILL = ["TE", "RB", "FB", "WR"]
WIN = 25          # search for contact within 2.5 s (25 frames) after the snap
CONTACT_YD = 1.5  # chipper–rusher distance that counts as contact
K = 3             # 0.3 s before/after contact for velocity
MIN_SPEED = 1.5   # rusher must be moving (yd/s) before contact, else direction is noise
BRAKE_T = 0.3     # yd/s change in closing speed toward the QB to call a chip effective / ineffective
SNAP_EV = {"ball_snap", "autoevent_ballsnap"}
END_EV = {"pass_forward", "autoevent_passforward", "qb_sack", "qb_strip_sack", "run", "fumble"}
CATS = ["Braked him", "No change", "Ran through it"]
COLORS = {"Braked him": "#2a9d8f", "No change": "#8d99ae", "Ran through it": "#e76f51"}


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# ---------------------------------------------------------------- labels
pff = pd.read_csv(os.path.join(DATA, "pffScoutingData.csv"))
players = pd.read_csv(os.path.join(DATA, "players.csv"))
plays = pd.read_csv(os.path.join(DATA, "plays.csv"))
pos = players.set_index("nflId")["officialPosition"]
name = players.set_index("nflId")["displayName"]

pff["pressure"] = (pff[["pff_hit", "pff_hurry", "pff_sack"]].fillna(0).sum(axis=1) > 0).astype(int)
press = pff[["gameId", "playId", "nflId", "pressure"]]

ch = pff[pff.pff_blockType == "CH"].copy()
ch["chip_pos"] = ch.nflId.map(pos)
ch = ch[ch.chip_pos.isin(SKILL) & ch.pff_nflIdBlockedPlayer.notna()].copy()
ch["rusher"] = ch.pff_nflIdBlockedPlayer.astype(int)
qb = (pff[pff.pff_role == "Pass"][["gameId", "playId", "nflId"]]
      .drop_duplicates(["gameId", "playId"]).rename(columns={"nflId": "qb"}))
ch = ch.merge(qb, on=["gameId", "playId"], how="inner")
ch = ch.merge(press.rename(columns={"nflId": "rusher", "pressure": "rusher_pressure"}),
              on=["gameId", "playId", "rusher"], how="left")
ch = ch.merge(plays[["gameId", "playId", "possessionTeam", "passResult", "pff_passCoverageType"]],
              on=["gameId", "playId"], how="left")
ch = ch[["gameId", "playId", "nflId", "chip_pos", "rusher", "qb", "rusher_pressure",
         "possessionTeam", "passResult", "pff_passCoverageType"]].rename(columns={"nflId": "chipper"})
print(f"CH chips by skill players: {len(ch)} on {ch[['gameId', 'playId']].drop_duplicates().shape[0]} plays")

# baselines: all rushers, and the same chipped rushers on rushes where they were NOT chipped
rush = pff[pff.pff_role == "Pass Rush"][["gameId", "playId", "nflId", "pressure"]].copy()
chipped_keys = set(zip(ch.gameId, ch.playId, ch.rusher))
rush["chipped"] = [k in chipped_keys for k in zip(rush.gameId, rush.playId, rush.nflId)]
base_all = rush.pressure.mean()
base_same_unchipped = rush[(~rush.chipped) & rush.nflId.isin(set(ch.rusher))].pressure.mean()
base_same_chipped = rush[rush.chipped].pressure.mean()

# ---------------------------------------------------------------- tracking
need = pd.concat([ch[["gameId", "playId", c]].rename(columns={c: "nflId"}) for c in ("chipper", "rusher", "qb")]).drop_duplicates()
trk_parts, play_info = [], {}
CACHE = os.path.join(OUT, "_tracking_cache.pkl")
cached = pd.read_pickle(CACHE) if os.path.exists(CACHE) else None
for g in ([] if cached is not None else sorted(ch.gameId.unique())):
    t = pd.read_csv(os.path.join(DATA, "tracking", f"tracking_{g}.csv"),
                    usecols=["gameId", "playId", "nflId", "frameId", "playDirection", "x", "y", "event"])
    t = t[t.playId.isin(need.playId[need.gameId == g].unique())]
    for p, tp in t.groupby("playId"):
        ev = tp.dropna(subset=["event"]).groupby("frameId").event.first()
        snaps = ev[ev.isin(SNAP_EV)].index
        if len(snaps) == 0:
            continue
        snap = int(snaps.min())
        ends = ev[ev.isin(END_EV) & (ev.index > snap)].index
        play_info[(g, p)] = (snap, int(ends.min()) if len(ends) else int(tp.frameId.max()))
    t = t.merge(need[need.gameId == g], on=["gameId", "playId", "nflId"])
    left = t.playDirection == "left"
    t.loc[left, "x"] = 120 - t.loc[left, "x"]
    t.loc[left, "y"] = 160 / 3 - t.loc[left, "y"]
    trk_parts.append(t[["gameId", "playId", "nflId", "frameId", "x", "y"]])
if cached is None:
    trk = pd.concat(trk_parts)
    pd.to_pickle((trk, play_info), CACHE)
else:
    trk, play_info = cached
paths = {k: v.set_index("frameId")[["x", "y"]].sort_index() for k, v in trk.groupby(["gameId", "playId", "nflId"])}


# ---------------------------------------------------------------- per-chip physics
def analyze(r):
    g, p = r.gameId, r.playId
    key = (g, p)
    if key not in play_info:
        return {"status": "no_snap"}
    c, ru, q = (paths.get((g, p, getattr(r, c_))) for c_ in ("chipper", "rusher", "qb"))
    if c is None or ru is None or q is None:
        return {"status": "missing_tracking"}
    snap, end = play_info[key]
    frames = [f for f in range(snap, min(snap + WIN, end) + 1) if f in c.index and f in ru.index]
    if not frames:
        return {"status": "no_frames"}
    d = np.hypot(c.loc[frames, "x"].values - ru.loc[frames, "x"].values,
                 c.loc[frames, "y"].values - ru.loc[frames, "y"].values)
    i = int(np.argmin(d))
    cf, dmin = frames[i], float(d[i])
    out = {"contact_frame": cf, "contact_dist": dmin, "t_contact": (cf - snap) / 10, "snap": snap, "end": end}
    if dmin > CONTACT_YD:
        return {**out, "status": "no_contact"}
    if i == len(frames) - 1 and frames[-1] < end:
        return {**out, "status": "closest_after_window"}
    if not all(f in ru.index for f in (cf - K, cf + K)) or cf not in q.index:
        return {**out, "status": "edge_frames"}
    p0, p1, p2 = (ru.loc[f, ["x", "y"]].values.astype(float) for f in (cf - K, cf, cf + K))
    qp = q.loc[cf, ["x", "y"]].values.astype(float)
    vb, va = (p1 - p0) / (K * 0.1), (p2 - p1) / (K * 0.1)
    sb, sa = np.linalg.norm(vb), np.linalg.norm(va)
    if sb < MIN_SPEED:
        return {**out, "status": "rusher_not_moving", "speed_before": sb}
    side = 1.0 if p1[1] >= qp[1] else -1.0          # which side of the QB the rusher is on
    u = (qp - p1) / max(np.linalg.norm(qp - p1), 1e-6)  # unit vector rusher -> QB
    e_out = np.array([-u[1], u[0]])
    if e_out[1] * side < 0:
        e_out = -e_out                                    # lateral unit vector pointing away from the QB
    dlat = (va - vb) @ e_out                             # + = pushed wider
    dclose = va @ u - vb @ u                             # - = slower toward the QB
    ang = lambda v: np.degrees(np.arccos(np.clip(v @ u / max(np.linalg.norm(v), 1e-6), -1, 1)))
    cat = "Braked him" if dclose <= -BRAKE_T else "Ran through it" if dclose >= BRAKE_T else "No change"
    direction = "wider" if dlat > 0.3 else "inside" if dlat < -0.3 else "same line"
    return {**out, "status": "ok", "category": cat, "speed_before": sb, "speed_after": sa, "dspeed": sa - sb,
            "dlat": dlat, "dclose": dclose, "dtheta": ang(va) - ang(vb),
            "vb_out": vb @ e_out, "vb_up": vb @ u, "va_out": va @ e_out, "va_up": va @ u,
            "rusher_to_qb_yd": float(np.linalg.norm(qp - p1)), "direction": direction}


res = pd.DataFrame([analyze(r) for r in ch.itertuples()], index=ch.index)
chips = pd.concat([ch, res], axis=1)
chips["chipper_name"] = chips.chipper.map(name)
chips["rusher_name"] = chips.rusher.map(name)
chips.to_csv(os.path.join(OUT, "chips.csv"), index=False)
ok = chips[chips.status == "ok"].copy()
print(chips.status.value_counts().to_string())

# ---------------------------------------------------------------- summary table
rows = []
for cat in CATS:
    s = ok[ok.category == cat]
    k, n = int(s.rusher_pressure.sum()), len(s)
    lo, hi = wilson(k, n)
    rows.append({"category": cat, "n": n, "share": n / len(ok), "pressures": k, "pressure_rate": k / n if n else np.nan,
                 "ci_lo": lo, "ci_hi": hi, "mean_dspeed": s.dspeed.mean(), "mean_dlat": s.dlat.mean()})
summ = pd.DataFrame(rows)
summ.to_csv(os.path.join(OUT, "pressure_by_deflection.csv"), index=False)

# ---------------------------------------------------------------- hero figure
figS, ax1 = plt.subplots(figsize=(8, 7.2))

# panel 1: shot chart in the rusher's frame (up = straight at the QB, right = wider / away from QB)
rng = np.random.default_rng(7)
samp = ok.sample(min(260, len(ok)), random_state=7)
for r in samp.itertuples():
    ax1.add_patch(FancyArrowPatch((0, 0), (r.va_out, r.va_up), arrowstyle="-|>", mutation_scale=7,
                                  color=COLORS[r.category], alpha=0.35 if not r.rusher_pressure else 0.95,
                                  lw=0.8 if not r.rusher_pressure else 1.6))
mb = ok[["vb_out", "vb_up"]].mean().values
ax1.add_patch(FancyArrowPatch((-mb[0], -mb[1]), (0, 0), arrowstyle="-|>", mutation_scale=18, color="black", lw=3))
ax1.text(-mb[0] - 0.2, -mb[1] - 0.6, "average rusher\nbefore the chip", ha="center", fontsize=9)
for cat in CATS:
    s = ok[ok.category == cat]
    if len(s):
        m = s[["va_out", "va_up"]].mean().values
        ax1.add_patch(FancyArrowPatch((0, 0), m, arrowstyle="-|>", mutation_scale=20, color=COLORS[cat], lw=4))
ax1.scatter([0], [0], s=160, color="white", edgecolor="black", zorder=5)
ax1.annotate("QB", xy=(0, 7.6), ha="center", fontsize=11, fontweight="bold")
ax1.annotate("", xy=(0, 7.4), xytext=(0, 6.4), arrowprops=dict(arrowstyle="->", lw=1.5))
ax1.text(6.8, -0.3, "wider\n(away from QB) →", ha="right", fontsize=9)
ax1.text(-6.8, -0.3, "← inside\n(toward QB)", ha="left", fontsize=9)
ax1.set_xlim(-7, 7); ax1.set_ylim(-5.5, 8); ax1.set_aspect("equal")
ax1.set_xlabel("lateral velocity after the chip (yd/s)"); ax1.set_ylabel("velocity toward the QB after the chip (yd/s)")
ax1.set_title(f"Shot chart: rusher velocity 0.3 s after contact ({len(samp)} random chips;\nbold = rusher got a pressure, thick = category average)", fontsize=10)
ax1.grid(alpha=0.2)
plt.tight_layout()
figS.savefig(os.path.join(OUT, "shot_chart.png"), dpi=150)
plt.close(figS)

# hero: (left) what each position's chips do to the rusher, (right) pressure by chip outcome
POS_ORDER = ["RB", "FB", "TE", "WR"]
pos_tab = (ok.groupby("chip_pos").category.value_counts(normalize=True).unstack().reindex(POS_ORDER)[CATS].fillna(0))
pos_n = ok.chip_pos.value_counts().reindex(POS_ORDER)
pos_tab.to_csv(os.path.join(OUT, "chip_outcome_by_position.csv"))
fig, (axp, ax2) = plt.subplots(1, 2, figsize=(16, 6.6), gridspec_kw={"width_ratios": [1, 1.1]})
ran = ok[ok.category == "Ran through it"].rusher_pressure.mean()
held = ok[ok.category != "Ran through it"].rusher_pressure.mean()
fig.suptitle(f"Chip Billiards: when a pass rusher runs through a chip, he gets pressure {ran:.1%} of the time vs {held:.1%} when the chip holds.\n"
             f"TE chips get run through {pos_tab.loc['TE', 'Ran through it']:.0%} of the time, RBs {pos_tab.loc['RB', 'Ran through it']:.0%}.",
             fontsize=14, fontweight="bold", x=0.01, ha="left")
yp = np.arange(len(POS_ORDER))[::-1]
leftp = np.zeros(len(POS_ORDER))
for cat in CATS:
    vals = pos_tab[cat].values * 100
    axp.barh(yp, vals, left=leftp, color=COLORS[cat], height=0.62, label=cat)
    for yi, l, v in zip(yp, leftp, vals):
        if v >= 7:
            axp.text(l + v / 2, yi, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    leftp += vals
axp.set_yticks(yp); axp.set_yticklabels([f"{p_}  (n={int(pos_n[p_])})" for p_ in POS_ORDER], fontsize=11)
axp.set_xlim(0, 100); axp.set_xlabel("share of chips")
axp.set_title("What the chip did to the rusher's closing speed toward the QB, by chipper position", fontsize=11)
axp.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=10)

# panel 2: pressure rate by chip outcome
y = np.arange(len(summ))[::-1]
ax2.barh(y, summ.pressure_rate * 100, color=[COLORS[c] for c in summ.category], height=0.6)
ax2.errorbar(summ.pressure_rate * 100, y, xerr=[(summ.pressure_rate - summ.ci_lo) * 100, (summ.ci_hi - summ.pressure_rate) * 100],
             fmt="none", ecolor="black", capsize=4, lw=1)
for yi, r in zip(y, summ.itertuples()):
    ax2.text(r.ci_hi * 100 + 0.6, yi, f"{r.pressure_rate:.1%}  (n={r.n}, {r.share:.0%} of chips)", va="center", fontsize=10)
ax2.axvline(base_same_unchipped * 100, ls="--", color="black", lw=1.2)
ax2.text(base_same_unchipped * 100 + 0.3, -0.55, f"same rushers when NOT chipped: {base_same_unchipped:.1%}", fontsize=9, va="center")
ax2.set_yticks(y); ax2.set_yticklabels(summ.category, fontsize=11)
ax2.set_xlabel("chipped rusher's pressure rate on the play (PFF hit / hurry / sack), 95% CI")
ax2.set_xlim(0, max(35, (summ.ci_hi.max() * 100) + 14))
ax2.set_title(f"Rusher pressure rate by what the chip did to his closing speed toward the QB\n({len(ok)} chips with measurable contact)", fontsize=11)
ax2.grid(axis="x", alpha=0.2)
fig.text(0.01, 0.035, "Data: NFL Big Data Bowl 2023 (2021 wks 1-8) tracking + PFF scouting, 1,560 PFF-labeled chips (pff_blockType = CH). Contact = closest chipper-rusher approach <1.5 yd within 2.5 s of snap.", fontsize=8.5, color="#555")
fig.text(0.01, 0.01, "Braked = rusher closing speed toward QB fell >=0.3 yd/s (0.3 s after vs before contact); Ran through = rose >=0.3. Associations, not causal effects.", fontsize=8, color="#555")
ax2.set_ylim(-0.85, len(summ) - 0.4)
plt.tight_layout(rect=(0, 0.06, 1, 0.9))
fig.savefig(os.path.join(OUT, "chip_billiards_hero.png"), dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- chipper leaderboard
MIN_CHIPS = 12
team = ok.groupby("chipper").possessionTeam.agg(lambda s: s.mode().iat[0])
lb = (ok.groupby("chipper").agg(chips=("category", "size"),
                                 braked=("category", lambda s: (s == "Braked him").mean()),
                                 nochange=("category", lambda s: (s == "No change").mean()),
                                 ranthrough=("category", lambda s: (s == "Ran through it").mean()),
                                 dclose=("dclose", "mean"), pressure=("rusher_pressure", "mean"))
      .query("chips >= @MIN_CHIPS").sort_values(["braked", "dclose"], ascending=[False, True]))
lb.insert(0, "player", lb.index.map(name)); lb.insert(1, "pos", lb.index.map(pos)); lb.insert(2, "team", lb.index.map(team))
lb.to_csv(os.path.join(OUT, "chipper_leaderboard.csv"))

fig, ax = plt.subplots(figsize=(12, 0.4 * len(lb) + 1.8))
yy = np.arange(len(lb))[::-1]
left = np.zeros(len(lb))
for col_name, cat in [("braked", "Braked him"), ("nochange", "No change"), ("ranthrough", "Ran through it")]:
    ax.barh(yy, lb[col_name] * 100, left=left, color=COLORS[cat], height=0.65, label=cat)
    left += lb[col_name].values * 100
for yi, r in zip(yy, lb.itertuples()):
    ax.text(102, yi, f"n={r.chips:>2}  avg Δclose {r.dclose:+.2f} yd/s  rusher pressure {r.pressure:.0%}", va="center", fontsize=9, family="monospace")
ax.set_yticks(yy); ax.set_yticklabels([f"{r.player} ({r.pos}, {r.team})" for r in lb.itertuples()], fontsize=9)
ax.set_xlim(0, 175); ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("share of the player's chips")
ax.set_title(f"Chipper report card: whose chips actually brake the rusher? (TE/RB/FB/WR with ≥{MIN_CHIPS} measurable chips)",
             fontsize=12, fontweight="bold", loc="left")
ax.legend(loc="lower right", fontsize=9, frameon=False)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "chipper_leaderboard.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- validation: 6 example chips
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
ex = pd.concat([ok[ok.category == c].sample(min(2, (ok.category == c).sum()), random_state=3) for c in CATS[:3]])
for ax, r in zip(axes.flat, ex.itertuples()):
    for who, col, lab in [(r.chipper, "#1d3557", "chipper"), (r.rusher, "#e63946", "rusher"), (r.qb, "black", "QB")]:
        pth = paths[(r.gameId, r.playId, who)]
        pth = pth.loc[r.snap:min(r.end, r.snap + 30)]
        ax.plot(pth.x, pth.y, color=col, lw=2, label=lab)
        ax.scatter(pth.x.iloc[0], pth.y.iloc[0], color=col, s=25)
        if r.contact_frame in pth.index:
            ax.scatter(pth.loc[r.contact_frame, "x"], pth.loc[r.contact_frame, "y"], marker="*", s=180, color="gold", edgecolor="k", zorder=5)
    ax.set_aspect("equal"); ax.grid(alpha=0.2)
    ax.set_title(f"{r.chipper_name} chips {r.rusher_name}\n{r.category}, Δspeed {r.dspeed:+.1f}, pressure={int(r.rusher_pressure)}  "
                 f"(game {r.gameId} play {r.playId})", fontsize=9)
axes.flat[0].legend(fontsize=8)
fig.suptitle("Validation: snap → 3 s paths, ★ = detected contact frame", fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "validation_examples.png"), dpi=120)
plt.close(fig)

# ---------------------------------------------------------------- text summary
lines = [f"skill-player chips (CH): {len(chips)}; measurable contact: {len(ok)} ({len(ok) / len(chips):.0%})",
         f"pressure rate - all rushers: {base_all:.1%} | chipped rushers when chipped: {base_same_chipped:.1%} | same rushers when NOT chipped: {base_same_unchipped:.1%}",
         summ.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
         "", "top chippers:", lb.head(10).to_string(float_format=lambda v: f"{v:.2f}")]
open(os.path.join(OUT, "summary.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
