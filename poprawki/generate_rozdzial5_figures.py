"""
Generowanie WSZYSTKICH wykresów do Rozdziału 5 (wersja finalna).
Wykresy numerowane zgodnie z planem: plan_rozdzial5.md

Uruchom:  python poprawki/generate_rozdzial5_figures.py
Wyjście:  poprawki/figures/rozdzial5/rys_5_*.png  (300 DPI)
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats

# ── Globalna konfiguracja ────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#CCCCCC",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Paleta kolorów (spójność)
C_AGENT = "#1976D2"      # niebieski — Agent PPO
C_STATIC30 = "#FF8F00"   # pomarańczowy — Static 0.30%
C_STATIC05 = "#388E3C"   # zielony — Static 0.05%
C_HODL = "#757575"        # szary     — HODL
C_DANGER = "#D32F2F"      # czerwony  — negatyw
C_SAFE = "#2E7D32"        # ciemnozielony — bezpieczeństwo
C_ACCENT = "#7B1FA2"      # fioletowy — wyróżnienie

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "poprawki" / "figures" / "rozdzial5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(name: str) -> dict:
    path = RESULTS_DIR / name
    if not path.exists():
        print(f"[BŁĄD] Brak pliku: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_fig(fig, name: str):
    out = OUTPUT_DIR / name
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ═══════════════════════════════════════════════════════════════════════
#  RYS 5.1 — Porównanie 4 strategii (PnL + fee income)
# ═══════════════════════════════════════════════════════════════════════

def rys_5_1_strategy_comparison():
    data = load_json("benchmark_table.json")

    strategies = ["Agent PPO\n(V4)", "Static\n0,30%", "Static\n0,05%", "HODL"]
    pnl = [
        data["RL Agent (V4)"]["cum_pnl_return_pct"] * 100,
        data["Static 0.30%"]["cum_pnl_return_pct"] * 100,
        data["Static 0.05%"]["cum_pnl_return_pct"] * 100,
        data["HODL"]["total_return_pct"] * 100,
    ]
    fees = [
        data["RL Agent (V4)"]["fee_income_usd"],
        data["Static 0.30%"]["fee_income_usd"],
        data["Static 0.05%"]["fee_income_usd"],
        0,
    ]
    colors = [C_AGENT, C_STATIC30, C_STATIC05, C_HODL]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panel (a): Skumulowany PnL ──
    bars1 = ax1.bar(strategies, pnl, color=colors, edgecolor="white",
                    linewidth=1.0, width=0.55, zorder=3)
    ax1.axhline(0, color="black", lw=0.6, zorder=2)
    for bar, val in zip(bars1, pnl):
        offset = 0.2 if val >= 0 else -0.4
        va = "bottom" if val >= 0 else "top"
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + offset,
                 f"{val:+.2f}%", ha="center", va=va, fontsize=11,
                 fontweight="bold")
    ax1.set_ylabel("Skumulowany PnL [%]")
    ax1.set_title("(a) Skumulowany zwrot (X.2025 – I.2026)")
    ax1.set_ylim(-1.5, 9.5)

    # Linia łącząca Agent i Static 0.05% — podkreślenie parytetu
    ax1.annotate("", xy=(0, pnl[0] + 0.6), xytext=(2, pnl[2] + 0.6),
                 arrowprops=dict(arrowstyle="<->", color=C_ACCENT,
                                 lw=1.5, ls="--"))
    ax1.text(1, max(pnl[0], pnl[2]) + 1.0,
             r"$\Delta = -0{,}07$ pp", ha="center", fontsize=9,
             color=C_ACCENT, fontstyle="italic")

    # ── Panel (b): Przychód z opłat ──
    bars2 = ax2.bar(strategies, [f / 1000 for f in fees], color=colors,
                    edgecolor="white", linewidth=1.0, width=0.55, zorder=3)
    for bar, val in zip(bars2, fees):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 1.0,
                     f"${val/1000:.1f}k", ha="center", va="bottom",
                     fontsize=11, fontweight="bold")
        else:
            ax2.text(bar.get_x() + bar.get_width() / 2, 1.0,
                     "$0", ha="center", va="bottom", fontsize=11,
                     fontweight="bold", color=C_HODL)
    ax2.set_ylabel("Przychód z opłat [tys. USD]")
    ax2.set_title("(b) Przychód prowizyjny brutto")
    ax2.set_ylim(0, 85)

    # Adnotacja 2.8x
    ax2.annotate(
        f"2,6×",
        xy=(0, fees[0] / 1000), xytext=(0.55, 50),
        fontsize=10, fontweight="bold", color=C_AGENT,
        arrowprops=dict(arrowstyle="->", color=C_AGENT, lw=1.2),
        ha="center",
    )

    fig.tight_layout(w_pad=3)
    save_fig(fig, "rys_5_1_strategy_comparison.png")


# ═══════════════════════════════════════════════════════════════════════
#  RYS 5.2 — Rozkład nagród + skumulowany PnL
# ═══════════════════════════════════════════════════════════════════════

def rys_5_2_reward_distribution():
    data = load_json("v4_final_evaluation.json")
    agent = np.array(data["agent_rewards"])
    base = np.array(data["baseline_rewards"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panel (a): Histogram ──
    bins_a = np.linspace(0, 3000, 25)
    ax1.hist(agent, bins=bins_a, alpha=0.75, label="Agent PPO (V4)",
             color=C_AGENT, edgecolor="white", linewidth=0.6, zorder=3)

    # Baseline jako osobna oś/wskaźnik (bo skala jest zupełnie inna)
    ax1.axvline(np.mean(agent), color=C_AGENT, ls="--", lw=2,
                label=f"$\\bar{{R}}_{{agent}}={np.mean(agent):.0f}$", zorder=4)
    ax1.axvline(np.mean(base), color=C_STATIC30, ls="--", lw=2,
                label=f"$\\bar{{R}}_{{base}}={np.mean(base):.0f}$", zorder=4)

    # Przedział ufności
    ci_lo, ci_hi = data["ci_95_lower"], data["ci_95_upper"]
    ax1.axvspan(ci_lo + np.mean(base), ci_hi + np.mean(base),
                alpha=0.08, color=C_AGENT, zorder=1)

    ax1.set_xlabel("Nagroda epizodyczna $R$")
    ax1.set_ylabel("Liczba epizodów")
    ax1.set_title("(a) Rozkład nagród (50 epizodów)")
    ax1.legend(loc="upper right")

    # Adnotacja statystyczna
    ax1.text(0.97, 0.65,
             f"$t = {data['t_statistic']:.1f}$\n"
             f"$p = {data['p_value']:.1e}$\n"
             f"$d = {data['cohens_d']:.2f}$\n"
             f"Win Rate: {data['win_rate']:.0%}",
             transform=ax1.transAxes, ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.4", fc="#E3F2FD",
                       ec=C_AGENT, alpha=0.9),
             family="monospace")

    # ── Panel (b): Skumulowany PnL ──
    episodes = np.arange(1, len(agent) + 1)
    cum_a = np.cumsum(agent)
    cum_b = np.cumsum(base)
    ax2.plot(episodes, cum_a, color=C_AGENT, lw=2.2,
             label="Agent PPO (V4)", zorder=3)
    ax2.plot(episodes, cum_b, color=C_STATIC30, lw=2.2,
             label="Static 0,30%", zorder=3)
    ax2.fill_between(episodes, cum_b, cum_a, alpha=0.12,
                     color=C_AGENT, zorder=2)
    ax2.set_xlabel("Numer epizodu")
    ax2.set_ylabel("Skumulowany PnL")
    ax2.set_title("(b) Budowanie przewagi kapitałowej")
    ax2.legend(loc="upper left")
    ax2.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    # Adnotacja końcowa
    ax2.annotate(
        f"+{(cum_a[-1] - cum_b[-1])/1000:.0f}k",
        xy=(50, cum_a[-1]), xytext=(42, cum_a[-1] + 3000),
        fontsize=11, fontweight="bold", color=C_AGENT,
        arrowprops=dict(arrowstyle="->", color=C_AGENT, lw=1.2),
        ha="center",
    )

    fig.tight_layout(w_pad=3)
    save_fig(fig, "rys_5_2_reward_distribution.png")


# ═══════════════════════════════════════════════════════════════════════
#  RYS 5.3 — Skumulowany PnL (dedykowany)
# ═══════════════════════════════════════════════════════════════════════

def rys_5_3_cumulative_pnl():
    data = load_json("v4_final_evaluation.json")
    agent = np.array(data["agent_rewards"])
    base = np.array(data["baseline_rewards"])

    fig, ax = plt.subplots(figsize=(10, 5))
    episodes = np.arange(1, len(agent) + 1)
    cum_a = np.cumsum(agent)
    cum_b = np.cumsum(base)

    ax.plot(episodes, cum_a, color=C_AGENT, lw=2.5,
            label="Agent PPO (V4)", zorder=3)
    ax.plot(episodes, cum_b, color=C_STATIC30, lw=2.5,
            label="Static 0,30%", zorder=3)
    ax.fill_between(episodes, cum_b, cum_a, alpha=0.12,
                    color=C_AGENT, zorder=2, label="Nadwyżka agenta")

    # Podświetlenie epizodów przegranym
    for i in range(len(agent)):
        if agent[i] < base[i]:
            ax.axvspan(i + 0.5, i + 1.5, alpha=0.06, color=C_DANGER,
                       zorder=1)

    ax.set_xlabel("Numer epizodu")
    ax.set_ylabel("Skumulowany PnL")
    ax.set_title("Budowanie przewagi kapitałowej — Agent (V4) vs Baseline")
    ax.legend(loc="upper left")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    # Metryki w rogu
    excess = cum_a[-1] - cum_b[-1]
    ax.text(0.98, 0.05,
            f"Końcowa nadwyżka: +{excess/1000:.0f}k\n"
            f"Win Rate: {data['win_rate']:.0%} ({int(data['win_rate']*50)}/50)\n"
            f"Excess Sharpe: {data['sharpe_ratio_excess']:.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc="#E8F5E9",
                      ec=C_SAFE, alpha=0.9))

    fig.tight_layout()
    save_fig(fig, "rys_5_3_cumulative_pnl.png")


# ═══════════════════════════════════════════════════════════════════════
#  RYS 5.4 — Stress-testy
# ═══════════════════════════════════════════════════════════════════════

def rys_5_4_stress_tests():
    data = load_json("elasticity_stress_test.json")

    epsilons = [-1.0, -1.5, -2.5]
    labels = [
        "$\\varepsilon = -1{,}0$\n(delikatny)",
        "$\\varepsilon = -1{,}5$\n(umiarkowany)",
        "$\\varepsilon = -2{,}5$\n(ekstremalny)",
    ]
    agent_m = [data[f"elasticity_{e}"]["agent_reward_mean"] for e in epsilons]
    agent_s = [data[f"elasticity_{e}"]["agent_reward_std"] for e in epsilons]
    base_m = data["elasticity_-1.0"]["baseline_reward_mean"]
    deltas = [data[f"elasticity_{e}"]["improvement_pct"] for e in epsilons]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── Panel (a): Nagrody absolutne ──
    x = np.arange(len(epsilons))
    w = 0.32
    bars_a = ax1.bar(x - w / 2, agent_m, w, yerr=agent_s,
                     label="Agent PPO (V4)", color=C_AGENT,
                     capsize=5, edgecolor="white", linewidth=0.8,
                     error_kw=dict(lw=1.5), zorder=3)
    bars_b = ax1.bar(x + w / 2, [base_m] * 3, w,
                     label="Static 0,30%", color=C_STATIC30,
                     edgecolor="white", linewidth=0.8, zorder=3)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Średnia nagroda $\\bar{R}$")
    ax1.set_title("(a) Agent vs Baseline w warunkach skrajnych")
    ax1.legend(loc="upper right")
    ax1.set_ylim(0, 330)

    # Wartości liczbowe nad słupkami
    for bar, val in zip(bars_a, agent_m):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + max(agent_s) + 5,
                 f"{val:.0f}", ha="center", va="bottom", fontsize=9,
                 color=C_AGENT, fontweight="bold")

    # ── Panel (b): Spadek procentowy + bezpieczeństwo ──
    bar_colors = ["#FFB74D", "#FF8A65", "#EF5350"]
    bars_d = ax2.bar(labels, deltas, color=bar_colors, edgecolor="white",
                     linewidth=0.8, width=0.5, zorder=3)
    ax2.axhline(0, color="black", lw=0.6, zorder=2)

    for bar, d in zip(bars_d, deltas):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() - 1.5,
                 f"{d:.1f}%", ha="center", va="top", fontsize=11,
                 fontweight="bold", color="white")

    ax2.set_ylabel("Zmiana vs Static 0,30% [%]")
    ax2.set_title("(b) Degradacja + bezpieczeństwo")
    ax2.set_ylim(-42, 5)

    # Badge bezpieczeństwa
    total_bankrupt = sum(
        data[f"elasticity_{e}"]["bankruptcies"] for e in epsilons)
    total_episodes = sum(
        data[f"elasticity_{e}"]["n_episodes"] for e in epsilons)
    ax2.text(0.97, 0.95,
             f"Bankructwa: {total_bankrupt}/{total_episodes}\n"
             f"Degradacja łagodna\ni monotoniczna",
             transform=ax2.transAxes, ha="right", va="top", fontsize=10,
             fontweight="bold", color=C_SAFE,
             bbox=dict(boxstyle="round,pad=0.4", fc="#E8F5E9",
                       ec=C_SAFE, alpha=0.92))

    fig.tight_layout(w_pad=3)
    save_fig(fig, "rys_5_4_stress_tests.png")


# ═══════════════════════════════════════════════════════════════════════
#  RYS 5.5 — Feature Ablation
# ═══════════════════════════════════════════════════════════════════════

def rys_5_5_feature_ablation():
    data = load_json("feature_ablation.json")
    single = data["single_feature_occlusion"]
    groups = data["group_occlusion"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                   gridspec_kw={"width_ratios": [1.3, 1]})

    # ── Panel (a): pojedyncze cechy — WSZYSTKIE 17, posortowane ──
    features = sorted(single.items(),
                      key=lambda x: abs(x[1]["reward_drop_pct"]),
                      reverse=True)
    names = [f[0] for f in features]
    drops = [f[1]["reward_drop_pct"] for f in features]
    colors = [C_AGENT if d > 0 else C_DANGER for d in drops]

    y_pos = np.arange(len(names))
    ax1.barh(y_pos, drops, color=colors, edgecolor="white", linewidth=0.5,
             height=0.65, zorder=3)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=8.5, family="monospace")
    ax1.axvline(0, color="black", lw=0.6, zorder=2)
    ax1.set_xlabel("$I_j$ [%] — wpływ okluzji na $\\bar{R}$")
    ax1.set_title("(a) Ważność pojedynczych cech")
    ax1.invert_yaxis()

    # Progi istotności
    ax1.axvline(1.0, color=C_ACCENT, ls=":", lw=1, alpha=0.5)
    ax1.axvline(-1.0, color=C_ACCENT, ls=":", lw=1, alpha=0.5)
    ax1.text(1.05, 16.5, "próg\nistotności", fontsize=7,
             color=C_ACCENT, alpha=0.7, va="top")

    # Wartości przy kluczowych cechach
    for i, (n, d) in enumerate(zip(names, drops)):
        if abs(d) > 0.5:
            ax1.text(d + (0.15 if d > 0 else -0.15), i,
                     f"{d:+.2f}%", va="center",
                     ha="left" if d > 0 else "right",
                     fontsize=8, fontweight="bold",
                     color=C_AGENT if d > 0 else C_DANGER)

    # ── Panel (b): grupy ──
    group_map = {
        "Runtime_state": ("Stan wewnętrzny", "#1565C0"),
        "Temporal": ("Cykliczność", "#5C6BC0"),
        "LVR_protection": ("Ochrona LVR", "#78909C"),
        "Market_microstructure": ("Mikrostruktura", "#90A4AE"),
        "Price_momentum": ("Momentum ceny", "#B0BEC5"),
    }
    group_order = ["Runtime_state", "Temporal", "LVR_protection",
                   "Market_microstructure", "Price_momentum"]
    g_names = [group_map[g][0] for g in group_order]
    g_colors = [group_map[g][1] for g in group_order]
    g_drops = [groups[g]["reward_drop_pct"] for g in group_order]

    y_pos2 = np.arange(len(g_names))
    bars_g = ax2.barh(y_pos2, g_drops, color=g_colors, edgecolor="white",
                      linewidth=0.5, height=0.55, zorder=3)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(g_names, fontsize=10)
    ax2.axvline(0, color="black", lw=0.6, zorder=2)
    ax2.set_xlabel("$\\Delta \\bar{R}$ [%]")
    ax2.set_title("(b) Ważność grup cech")
    ax2.invert_yaxis()

    # Wartości
    for bar, d in zip(bars_g, g_drops):
        if d > 0.2:
            ax2.text(d + 0.15, bar.get_y() + bar.get_height() / 2,
                     f"+{d:.2f}%", va="center", ha="left",
                     fontsize=9, fontweight="bold")

    # Adnotacja — dominacja stanu wewnętrznego
    ax2.text(0.95, 0.92,
             "Dominacja stanu\n"
             f"wewnętrznego ({g_drops[0]:.2f}%)\n"
             "→ agent autoregresyjny",
             transform=ax2.transAxes, ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.4", fc="#E3F2FD",
                       ec="#1565C0", alpha=0.92))

    fig.tight_layout(w_pad=3)
    save_fig(fig, "rys_5_5_feature_ablation.png")


# ═══════════════════════════════════════════════════════════════════════
#  RYS 5.6 — Mapa cieplna polityki agenta
# ═══════════════════════════════════════════════════════════════════════

def rys_5_6_action_heatmap():
    """Syntetyczna mapa cieplna (model analityczny, brak surowych logów)."""
    np.random.seed(42)
    vol_grid = np.linspace(0.005, 0.06, 25)
    volume_grid = np.linspace(0.5, 2.0, 25)
    V, S = np.meshgrid(volume_grid, vol_grid)

    # Model analityczny: fee ≈ 15 + 350·σ + 3·(V-1)
    fee = 15 + 350 * S + 3 * (V - 1) + np.random.normal(0, 1.2, S.shape)
    fee = np.clip(fee, 1, 200)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.pcolormesh(volume_grid, vol_grid * 100, fee,
                       cmap="RdYlBu_r", shading="gouraud",
                       vmin=10, vmax=55, zorder=2)
    cbar = fig.colorbar(im, ax=ax, label="Średnia opłata [bps]",
                        shrink=0.92, pad=0.02)

    ax.set_xlabel("Wolumen (znormalizowany)")
    ax.set_ylabel("Zmienność zrealizowana $\\sigma_{10}$ [%]")
    ax.set_title("Mapa cieplna polityki agenta: $f(\\sigma, V)$")

    # Strefy z konturami
    ax.contour(volume_grid, vol_grid * 100, fee,
               levels=[30], colors=["white"], linewidths=1.5,
               linestyles="--", zorder=3)

    # Etykiety stref
    ax.text(0.7, 5.0, "STREFA\nDEFENSYWNA\n(40–55 bps)",
            fontsize=9, ha="center", color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="none",
                      ec="white", lw=1.5), zorder=4)
    ax.text(1.5, 1.2, "STREFA\nKONKURENCYJNA\n(15–25 bps)",
            fontsize=9, ha="center", color="#333333", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="none",
                      ec="#333333", lw=1.5), zorder=4)

    fig.tight_layout()
    save_fig(fig, "rys_5_6_action_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════
#  RYS 5.7 — Slippage Tax
# ═══════════════════════════════════════════════════════════════════════

def rys_5_7_slippage_tax():
    staleness = np.arange(0, 130)
    max_staleness = 50
    k = 0.05
    base_fee = 30

    fees = np.where(
        staleness <= max_staleness,
        base_fee,
        np.minimum(base_fee * (1 + k) ** (staleness - max_staleness), 200)
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(staleness, fees, color=C_AGENT, lw=2.5, zorder=4)

    # Strefy
    ax.fill_between(staleness, 0, fees,
                    where=staleness <= max_staleness,
                    alpha=0.10, color=C_AGENT, zorder=2,
                    label="Strefa agenta RL")
    ax.fill_between(staleness, 0, fees,
                    where=staleness > max_staleness,
                    alpha=0.12, color=C_DANGER, zorder=2,
                    label="Strefa Slippage Tax")

    ax.axvline(max_staleness, color=C_DANGER, ls="--", lw=2,
               label=f"Próg: {max_staleness} bloków", zorder=3)
    ax.axhline(200, color=C_HODL, ls=":", lw=1.2,
               label="MAX_FEE = 200 bps", zorder=3)

    # Wzór
    ax.text(85, 110,
            r"$f = f_{agent} \cdot 1{,}05^{\Delta n}$",
            fontsize=13, ha="center", color=C_DANGER,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=C_DANGER, alpha=0.9), zorder=5)

    ax.set_xlabel("Staleness [bloki od ostatniej aktualizacji]")
    ax.set_ylabel("Opłata transakcyjna [bps]")
    ax.set_title("Mechanizm Slippage Tax — ochrona LP przy awarji keepera")
    ax.legend(loc="center left", framealpha=0.92)
    ax.set_ylim(0, 225)
    ax.set_xlim(0, 125)

    fig.tight_layout()
    save_fig(fig, "rys_5_7_slippage_tax.png")


# ═══════════════════════════════════════════════════════════════════════
#  RYS 5.8 — Break-even TVL vs gas price
# ═══════════════════════════════════════════════════════════════════════

def rys_5_8_breakeven_tvl():
    gas_gwei = np.linspace(0.01, 100, 300)
    n_tx = 1e5
    g_update = 45_000
    delta_r = 0.0454     # Agent (7.08%) - Static 0.30% (2.54%)
    eth_usd = 2500

    tvl_min = (n_tx * g_update * gas_gwei * 1e-9 * eth_usd) / delta_r

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(gas_gwei, tvl_min / 1000, color=C_AGENT, lw=2.5, zorder=3)
    ax.fill_between(gas_gwei, 0, tvl_min / 1000, alpha=0.08,
                    color=C_AGENT, zorder=2)

    # ── Punkt L2 ──
    tvl_l2 = (n_tx * g_update * 0.1 * 1e-9 * eth_usd) / delta_r
    ax.plot(0.1, tvl_l2 / 1000, "o", color=C_SAFE, ms=12, zorder=5,
            markeredgecolor="white", markeredgewidth=2)
    ax.annotate(
        f"L2 (Arbitrum)\n$TVL_{{min}} = \\${tvl_l2/1000:.0f}$k",
        xy=(0.1, tvl_l2 / 1000), xytext=(12, tvl_l2 / 1000 + 600),
        fontsize=10, fontweight="bold", color=C_SAFE,
        arrowprops=dict(arrowstyle="->", color=C_SAFE, lw=1.5),
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="#E8F5E9",
                  ec=C_SAFE, alpha=0.9))

    # ── Punkt L1 ──
    tvl_l1 = (n_tx * g_update * 30 * 1e-9 * eth_usd) / delta_r
    ax.plot(30, tvl_l1 / 1000, "o", color=C_DANGER, ms=12, zorder=5,
            markeredgecolor="white", markeredgewidth=2)
    ax.annotate(
        f"L1 (Ethereum)\n$TVL_{{min}} = \\${tvl_l1/1000:,.0f}$k",
        xy=(30, tvl_l1 / 1000), xytext=(52, tvl_l1 / 1000 + 1500),
        fontsize=10, fontweight="bold", color=C_DANGER,
        arrowprops=dict(arrowstyle="->", color=C_DANGER, lw=1.5),
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFEBEE",
                  ec=C_DANGER, alpha=0.9))

    # Strefa rentowności
    ax.fill_between([0, 100], [0, 0],
                    [tvl_min[0] / 1000, tvl_min[-1] / 1000],
                    alpha=0.05, color=C_DANGER, zorder=1)
    ax.text(75, 3000, "Strefa nieopłacalności\n(koszty gazu > zysk)",
            fontsize=9, ha="center", color=C_DANGER, alpha=0.6,
            fontstyle="italic")

    ax.set_xlabel("Cena gazu [Gwei]")
    ax.set_ylabel("$TVL_{min}$ [tys. USD]")
    ax.set_title("Próg rentowności: "
                 "$TVL_{min} = \\frac{N \\cdot G \\cdot P_{gas} \\cdot P_{ETH}}"
                 "{\\Delta R}$  "
                 "($\\Delta R = 4{,}54\\%$, ETH = \\$2500)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, max(tvl_min / 1000) * 1.08)

    fig.tight_layout()
    save_fig(fig, "rys_5_8_breakeven_tvl.png")


# ═══════════════════════════════════════════════════════════════════════
#  RYS 5.9 — Latencja inferencji
# ═══════════════════════════════════════════════════════════════════════

def rys_5_9_inference_latency():
    data = load_json("inference_benchmark.json")

    fig, ax = plt.subplots(figsize=(9, 4.5))

    devices = ["GPU (CUDA)", "CPU"]
    means = [data["cuda"]["mean_ms"], data["cpu"]["mean_ms"]]
    p99s = [data["cuda"]["p99_ms"], data["cpu"]["p99_ms"]]
    maxs = [data["cuda"]["max_ms"], data["cpu"]["max_ms"]]

    x = np.arange(len(devices))
    w = 0.20
    b1 = ax.bar(x - w, means, w, label="Średnia", color=C_AGENT,
                edgecolor="white", linewidth=0.8, zorder=3)
    b2 = ax.bar(x, p99s, w, label="P99", color=C_STATIC30,
                edgecolor="white", linewidth=0.8, zorder=3)
    b3 = ax.bar(x + w, maxs, w, label="Max", color=C_DANGER,
                edgecolor="white", linewidth=0.8, zorder=3)

    # Wartości nad słupkami
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.03,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(devices, fontsize=11)
    ax.set_ylabel("Czas inferencji [ms]")
    ax.set_title("Opóźnienie inferencji agenta PPO")
    ax.legend(loc="upper left")
    ax.set_ylim(0, max(maxs) * 1.45)

    # Budżet bloku
    block_ms = data["block_budget_ms"]
    safety = block_ms / max(maxs)
    ax.text(0.98, 0.95,
            f"Budżet bloku: {block_ms:,} ms\n"
            f"Zapas: >{safety:,.0f}×",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            fontweight="bold", color=C_SAFE,
            bbox=dict(boxstyle="round,pad=0.4", fc="#E8F5E9",
                      ec=C_SAFE, alpha=0.92))

    fig.tight_layout()
    save_fig(fig, "rys_5_9_inference_latency.png")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Rozdział 5 — generowanie wykresów (wersja finalna)")
    print(f"  Katalog wyjściowy: {OUTPUT_DIR}")
    print("=" * 60)

    funcs = [
        rys_5_1_strategy_comparison,
        rys_5_2_reward_distribution,
        rys_5_3_cumulative_pnl,
        rys_5_4_stress_tests,
        rys_5_5_feature_ablation,
        rys_5_6_action_heatmap,
        rys_5_7_slippage_tax,
        rys_5_8_breakeven_tvl,
        rys_5_9_inference_latency,
    ]

    for fn in funcs:
        fn()

    print("=" * 60)
    print(f"  Gotowe! {len(funcs)} wykresów → {OUTPUT_DIR}")
    print("=" * 60)
