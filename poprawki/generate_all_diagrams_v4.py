"""
Generator finalnych diagramow i tabel statycznych.

Uruchom: python poprawki/generate_all_diagrams_v4.py
Wyniki:  poprawki/figures/
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Kolory ──
C_BLUE   = "#2E86AB"
C_PURPLE = "#A23B72"
C_ORANGE = "#F18F01"
C_RED    = "#C73E1D"
C_DARK   = "#2C3E50"
C_GREEN  = "#27AE60"
C_YELLOW = "#F39C12"
C_VIOLET = "#8E44AD"
C_HDR    = "#34495E"
C_ALT    = "#F7F9FC"

# A4 textwidth ≈ 160mm ≈ 6.3". Uzywamy 7.5" bo bbox='tight' tnie marginesy.
W = 7.5  # szerokosc docelowa

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Segoe UI'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'axes.titleweight': 'bold',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def save(fig, name):
    fig.savefig(os.path.join(OUTPUT_DIR, name), facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [OK] {name}")


def src(fig, text="Opracowanie wlasne"):
    fig.text(0.98, 0.005, f"Zrodlo: {text}",
             ha='right', fontsize=5.5, style='italic', color='#AAAAAA')


def rbox(ax, cx, cy, w, h, text, bg, ec, fs=8, fw='bold', lh=1.25):
    """Zaokraglony prostokat ze srodkiem (cx, cy)."""
    r = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle="round,pad=0.08", facecolor=bg,
                       edgecolor=ec, linewidth=1.2, zorder=2)
    ax.add_patch(r)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            fontweight=fw, linespacing=lh, color=C_DARK, zorder=3)


def arr(ax, x1, y1, x2, y2, color=None, lw=1.2, style='->', ls='-'):
    color = color or C_DARK
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                linestyle=ls), zorder=4)


def style_tbl(table, nc, nr, fs=8):
    table.auto_set_font_size(False)
    table.set_fontsize(fs)
    for j in range(nc):
        table[0, j].set_facecolor(C_HDR)
        table[0, j].set_text_props(color='white', fontweight='bold', fontsize=fs)
    for i in range(nr):
        bg = C_ALT if i % 2 == 0 else '#FFFFFF'
        for j in range(nc):
            table[i+1, j].set_facecolor(bg)
            table[i+1, j].set_edgecolor('#E0E0E0')
            if j == 0:
                table[i+1, j].set_text_props(fontweight='bold')


# ══════════════════════════════════════════════════════════════
# 1. TABELA POROWNAWCZA V2 / V3 / V4
# ══════════════════════════════════════════════════════════════
def fig01():
    fig, ax = plt.subplots(figsize=(W, 4.8))
    ax.axis('off')

    cols = ["Cecha", "Uniswap V2", "Uniswap V3", "Uniswap V4"]
    rows = [
        ["Architektura",             "1 kontrakt\nna pare", "1 kontrakt\nna pare",         "Singleton\n(1 kontrakt)"],
        ["Plynnosc",                 "Pelny zakres",       "Skoncentrowana\n(tick range)", "Skoncentrowana\n+ dyn. oplaty"],
        ["Oplaty",                   "Stale 0,30%",        "4 progi\n(0,01/0,05/\n0,30/1,00%)", "Dowolne via Hooks\n(0,01 -- 2,00%)"],
        ["Rozszerzalnosc",           "Brak",               "Ograniczona",                  "Hooks (before/\nafterSwap, ...)"],
        ["Koszt wdrozenia\npuli",    "ok. 5M gas",         "ok. 5M gas",                   "ok. 30K gas\n(redukcja 99%)"],
        ["Flash Accounting",         "Nie",                "Nie",                          "Tak (EIP-1153)"],
    ]

    table = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    style_tbl(table, len(cols), len(rows), fs=7.5)
    table.scale(1.0, 2.0)

    for i in range(len(rows)):
        table[i+1, 3].set_facecolor('#EBF5FB' if i % 2 == 0 else '#D6EAF8')

    ax.set_title("Tabela 1.1. Ewolucja architektoniczna protokolu Uniswap",
                 fontsize=10, pad=10, loc='left', fontweight='bold')
    src(fig, "opracowanie wlasne na podst. Adams et al. (2021, 2023, 2024)")
    save(fig, "tab_1_1_uniswap_versions.png")


# ══════════════════════════════════════════════════════════════
# 2. CYKL TRANSAKCJI Z HOOKS
# ══════════════════════════════════════════════════════════════
def fig02():
    fig, ax = plt.subplots(figsize=(W, 4.0))
    ax.axis('off')
    ax.set_xlim(-0.2, 7.7)
    ax.set_ylim(-1.6, 2.5)

    labels = [
        "Uzytkownik\nwysyla tx",
        "PoolManager\nswap()",
        "beforeSwap\n(Hook)",
        "Logika AMM\nswap()",
        "afterSwap\n(Hook)",
        "Rozliczenie\nflash acct.",
    ]
    bgs = ['#EAECF0', '#D6EAF8', '#FADBD8', '#D5F5E3', '#FFF3CD', '#E8DAEF']
    ecs = [C_DARK, C_BLUE, C_RED, C_GREEN, C_YELLOW, C_VIOLET]

    bw, bh = 1.05, 0.7
    gap = 1.33  # odleglosc miedzy srodkami
    xs = [0.5 + i * gap for i in range(6)]

    for i, x in enumerate(xs):
        rbox(ax, x, 1.0, bw, bh, labels[i], bgs[i], ecs[i], fs=6.5)

    for i in range(5):
        arr(ax, xs[i] + bw/2 + 0.02, 1.0, xs[i+1] - bw/2 - 0.02, 1.0, lw=1.5)

    # Numery
    for i, x in enumerate(xs):
        ax.text(x, 1.0 + bh/2 + 0.15, str(i+1), ha='center', va='center',
                fontsize=8, fontweight='bold', color='white',
                bbox=dict(boxstyle='circle,pad=0.1', facecolor=C_DARK, edgecolor='none'))

    # Adnotacja beforeSwap
    ax.text(xs[2], -0.3, "Modyfikacja oplaty\nWalidacja staleness\nSlippage tax",
            ha='center', va='top', fontsize=6, color=C_RED, style='italic',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#FEF9E7',
                      edgecolor=C_RED, alpha=0.9, linewidth=0.8))
    arr(ax, xs[2], 1.0 - bh/2, xs[2], -0.25, color=C_RED, lw=0.8)

    # Adnotacja afterSwap
    ax.text(xs[4], -0.3, "Aktualizacja EMA\nZapis stanu oplat\nEmisja zdarzenia",
            ha='center', va='top', fontsize=6, color='#B7950B', style='italic',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#FEF9E7',
                      edgecolor=C_YELLOW, alpha=0.9, linewidth=0.8))
    arr(ax, xs[4], 1.0 - bh/2, xs[4], -0.25, color=C_YELLOW, lw=0.8)

    ax.set_title("Rysunek 1.1. Cykl zycia transakcji swap z mechanizmem Hooks",
                 fontsize=10, pad=10, loc='left', fontweight='bold')
    src(fig)
    save(fig, "rys_1_1_hook_lifecycle.png")


# ══════════════════════════════════════════════════════════════
# 3. ARCHITEKTURA ACTOR-CRITIC PPO
# ══════════════════════════════════════════════════════════════
def fig03():
    fig, ax = plt.subplots(figsize=(W, 5.5))
    ax.axis('off')
    ax.set_xlim(-0.2, 7.7)
    ax.set_ylim(-0.3, 5.5)

    # Cechy wejsciowe (lewa strona)
    features = [
        "Zmiennosc zrealiz. (10 bl.)",
        "Wsp. zmiennosci (10/100)",
        "Log-stopa zwrotu",
        "Zmiana ticka",
        "Cykl godzinowy (sin/cos)",
        "Cykl tygodniowy (sin/cos)",
        "Znorm. biezaca oplata",
    ]

    ax.text(0.85, 5.3, "Wektor obserwacji s(t) — 7 wymiarow",
            fontsize=8, fontweight='bold', color=C_BLUE, ha='center')

    fw, fh = 1.7, 0.42
    feat_cx = 0.85
    feat_ys = [4.8 - i * 0.6 for i in range(7)]

    for i, (label, fy) in enumerate(zip(features, feat_ys)):
        rbox(ax, feat_cx, fy, fw, fh, label, '#EBF5FB', C_BLUE, fs=5.5, fw='normal')

    # Warstwa wspoldzielona
    sh_cx, sh_cy = 2.8, 2.7
    sh_w, sh_h = 1.4, 1.3
    rbox(ax, sh_cx, sh_cy, sh_w, sh_h,
         "Wspoldzielona\nwarstwa MLP\n128 neuronow\nReLU", '#FEF9E7', C_YELLOW, fs=6.5)

    for fy in feat_ys:
        arr(ax, feat_cx + fw/2, fy, sh_cx - sh_w/2, sh_cy, color='#CCCCCC', lw=0.4)

    # Galaz aktora (gorna)
    ax.text(4.0, 5.1, "AKTOR", fontsize=7.5, fontweight='bold', color=C_GREEN)
    a1_cx, a1_cy = 4.5, 4.4
    a2_cx, a2_cy = 4.5, 3.6
    bw_b, bh_b = 1.5, 0.55

    rbox(ax, a1_cx, a1_cy, bw_b, bh_b, "Warstwa ukryta\n128, ReLU", '#D5F5E3', C_GREEN, fs=6)
    rbox(ax, a2_cx, a2_cy, bw_b, bh_b, "Glowica aktora\nmu, log(sigma)", '#D5F5E3', C_GREEN, fs=6)
    arr(ax, a1_cx, a1_cy - bh_b/2, a2_cx, a2_cy + bh_b/2, color=C_GREEN, lw=1)

    # Galaz krytyka (dolna)
    ax.text(4.0, 2.0, "KRYTYK", fontsize=7.5, fontweight='bold', color=C_RED)
    c1_cx, c1_cy = 4.5, 1.4
    c2_cx, c2_cy = 4.5, 0.6
    rbox(ax, c1_cx, c1_cy, bw_b, bh_b, "Warstwa ukryta\n128, ReLU", '#FADBD8', C_RED, fs=6)
    rbox(ax, c2_cx, c2_cy, bw_b, bh_b, "Glowica krytyka\nV(s_t)", '#FADBD8', C_RED, fs=6)
    arr(ax, c1_cx, c1_cy - bh_b/2, c2_cx, c2_cy + bh_b/2, color=C_RED, lw=1)

    # Strzalki z warstwy wspoldzielonej
    arr(ax, sh_cx + sh_w/2, sh_cy + 0.3, a1_cx - bw_b/2, a1_cy, color=C_DARK, lw=1.5)
    arr(ax, sh_cx + sh_w/2, sh_cy - 0.3, c1_cx - bw_b/2, c1_cy, color=C_DARK, lw=1.5)

    # Wyjscia
    out_cx = 6.5
    out_w, out_h = 1.6, 0.6
    rbox(ax, out_cx, a2_cy, out_w, out_h,
         "Akcja a ~ N(mu, sigma)\na w [-1, 1]", '#E8DAEF', C_VIOLET, fs=6)
    arr(ax, a2_cx + bw_b/2, a2_cy, out_cx - out_w/2, a2_cy, color=C_GREEN, lw=1.5)

    rbox(ax, out_cx, c2_cy, out_w, out_h,
         "Estymata wartosci\nV(s_t)", '#E8DAEF', C_VIOLET, fs=6)
    arr(ax, c2_cx + bw_b/2, c2_cy, out_cx - out_w/2, c2_cy, color=C_RED, lw=1.5)

    # Mapowanie
    map_cy = 2.3
    rbox(ax, out_cx, map_cy, out_w, 0.7,
         "Mapowanie:\nfee = 100 +\na * 19 900\n[100, 20 000]", '#FCF3CF', '#B7950B', fs=5.5)
    arr(ax, out_cx, a2_cy - out_h/2, out_cx, map_cy + 0.35, color=C_DARK, lw=1)

    ax.set_title("Rysunek 2.1. Architektura sieci Actor-Critic PPO",
                 fontsize=10, pad=10, loc='left', fontweight='bold')
    src(fig)
    save(fig, "rys_2_1_ppo_architecture.png")


# ══════════════════════════════════════════════════════════════
# 4. TABELA FAZ CURRICULUM LEARNING
# ══════════════════════════════════════════════════════════════
def fig04():
    fig, ax = plt.subplots(figsize=(W, 3.8))
    ax.axis('off')

    cols = ["Parametr", "Faza 1\n(bazowa)", "Faza 2\n(posrednia)", "Faza 3\n(pelna)"]
    rows = [
        ["Krok treningowy",             "0 — 2M",           "2M — 5M",           "5M — 10M"],
        ["Szum akcji (sigma)",          "0,5",               "0,3",                "0,1"],
        ["Opoznienie akcji",            "0 blokow",          "1 blok",             "1 blok"],
        ["Randomizacja\nelastycznosci", "eps ~ U[-2; -1]",   "eps ~ U[-2,5; -0,7]","eps ~ U[-3; -0,5]"],
        ["Cel dydaktyczny",             "Stabilnosc",        "Generalizacja",      "Robustnosc"],
    ]

    table = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    style_tbl(table, len(cols), len(rows), fs=8)
    table.scale(1.0, 1.9)

    phase_bg = ['#FFFFFF', '#DBEAFE', '#FDE8E8', '#D1FAE5']
    for i in range(len(rows)):
        for j in range(1, len(cols)):
            table[i+1, j].set_facecolor(phase_bg[j])

    ax.set_title("Tabela 2.1. Konfiguracja fazowa uczenia programowego (Curriculum Learning)",
                 fontsize=10, pad=10, loc='left', fontweight='bold')
    src(fig)
    save(fig, "tab_2_1_curriculum_phases.png")


# ══════════════════════════════════════════════════════════════
# 5. DRZEWO DECYZYJNE _beforeSwap
# ══════════════════════════════════════════════════════════════
def fig05():
    fig, ax = plt.subplots(figsize=(W, 8.5))
    ax.axis('off')
    ax.set_xlim(-0.2, 8.2)
    ax.set_ylim(-0.8, 8.5)

    def diamond(cx, cy, hw, hh, text, bg='#FFF3CD'):
        verts = [(cx, cy + hh), (cx + hw, cy), (cx, cy - hh), (cx - hw, cy)]
        poly = plt.Polygon(verts, facecolor=bg, edgecolor=C_DARK, linewidth=1.2, zorder=2)
        ax.add_patch(poly)
        ax.text(cx, cy, text, ha='center', va='center', fontsize=6.5,
                fontweight='bold', color=C_DARK, zorder=3, linespacing=1.15)
        return {'cx': cx, 'cy': cy, 'top': cy+hh, 'bot': cy-hh,
                'left': cx-hw, 'right': cx+hw}

    def box(cx, cy, w, h, text, bg='#D5F5E3'):
        rbox(ax, cx, cy, w, h, text, bg, C_DARK, fs=6.5)
        return {'cx': cx, 'cy': cy, 'top': cy+h/2, 'bot': cy-h/2,
                'left': cx-w/2, 'right': cx+w/2}

    def lbl(x, y, text, color):
        ax.text(x, y, text, ha='center', va='center', fontsize=6.5,
                fontweight='bold', color=color, zorder=5)

    main_x = 2.8
    side_x = 6.5
    gap = 1.5

    # Start
    y0 = 8.0
    s = box(main_x, y0, 2.0, 0.5, "Wywolanie swap()\nprzez PoolManager", '#EAECF0')

    # Wezel 1: emergencyMode?
    y1 = y0 - gap
    arr(ax, main_x, s['bot'], main_x, y1 + 0.45, lw=1.2)
    d1 = diamond(main_x, y1, 1.2, 0.4, "Tryb awaryjny\n(emergencyMode)?")
    arr(ax, d1['right'], y1, side_x - 1.0, y1, lw=1.2)
    lbl(d1['right'] + 0.4, y1 + 0.2, "TAK", C_RED)
    r1 = box(side_x, y1, 2.0, 0.5, "EMERGENCY_FEE\n= 5 000 (0,50%)", '#FADBD8')

    # Wezel 2: niezainicjalizowana?
    y2 = y1 - gap
    arr(ax, main_x, d1['bot'], main_x, y2 + 0.45, lw=1.2)
    lbl(main_x - 0.5, d1['bot'] - 0.15, "NIE", C_GREEN)
    d2 = diamond(main_x, y2, 1.2, 0.4, "Pula\nniezainicjalizowana?")
    arr(ax, d2['right'], y2, side_x - 1.0, y2, lw=1.2)
    lbl(d2['right'] + 0.4, y2 + 0.2, "TAK", C_RED)
    r2 = box(side_x, y2, 2.0, 0.5, "DEFAULT_FEE\n= 3 000 (0,30%)", '#FADBD8')

    # Wezel 3: staleness?
    y3 = y2 - gap
    arr(ax, main_x, d2['bot'], main_x, y3 + 0.5, lw=1.2)
    lbl(main_x - 0.5, d2['bot'] - 0.15, "NIE", C_GREEN)
    d3 = diamond(main_x, y3, 1.3, 0.45, "Bloki od aktualizacji\n> MAX_STALENESS?", '#FFE0B2')
    arr(ax, d3['right'], y3, side_x - 1.1, y3, lw=1.2)
    lbl(d3['right'] + 0.4, y3 + 0.2, "TAK", C_RED)
    r3 = box(side_x, y3, 2.2, 0.55, "Podatek staleness:\nfee * 1,05^delta", '#FFE0B2')

    # Wezel 4: agentFee
    y4 = y3 - gap
    arr(ax, main_x, d3['bot'], main_x, y4 + 0.3, lw=1.2)
    lbl(main_x - 0.5, d3['bot'] - 0.15, "NIE", C_GREEN)
    r4 = box(main_x, y4, 2.2, 0.55, "Zastosuj oplate agenta\n(z ostatniego updateFee())", '#D6EAF8')

    # Clamp
    y5 = y4 - 1.2
    arr(ax, main_x, r4['bot'], main_x, y5 + 0.3, lw=1.2)
    r5 = box(main_x, y5, 2.4, 0.55, "Clamp: max(MIN_FEE,\nmin(fee, MAX_FEE))  [100 — 20 000]", '#E8DAEF')

    # Wynik
    y6 = y5 - 1.0
    arr(ax, main_x, r5['bot'], main_x, y6 + 0.2, lw=1.2)
    box(main_x, y6, 2.2, 0.4, "Zwroc: OVERRIDE_FEE_FLAG | fee", '#D5F5E3')

    # Linia zbiorcza → clamp
    col_x = 7.7
    ax.plot([col_x, col_x], [y1, y5], color='#BBBBBB', lw=0.8, ls='--', zorder=1)
    for ry in [y1, y2, y3]:
        ax.plot([r1['right'] if ry == y1 else (r2['right'] if ry == y2 else r3['right']),
                 col_x], [ry, ry], color='#BBBBBB', lw=0.8, ls='--', zorder=1)
    ax.plot([col_x, r5['right'] + 0.15], [y5, y5], color='#BBBBBB', lw=0.8, ls='--', zorder=1)
    ax.annotate('', xy=(r5['right'] + 0.05, y5), xytext=(r5['right'] + 0.5, y5),
                arrowprops=dict(arrowstyle='->', color='#BBBBBB', lw=0.8, linestyle='--'))
    ax.text(col_x + 0.1, (y1+y3)/2, "Wszystkie\nsciezki ->\nclamp",
            fontsize=5.5, color='#AAAAAA', style='italic', ha='left', va='center')

    ax.set_title("Rysunek 3.1. Drzewo decyzyjne funkcji _beforeSwap\n"
                 "w kontrakcie VolatilityFeeHook",
                 fontsize=10, pad=10, loc='left', fontweight='bold')
    src(fig, "opracowanie wlasne na podst. kodu kontraktu")
    save(fig, "rys_3_1_beforeswap_flowchart.png")


# ══════════════════════════════════════════════════════════════
# 6. ARCHITEKTURA HYBRYDOWA
# ══════════════════════════════════════════════════════════════
def fig06():
    fig, ax = plt.subplots(figsize=(W, 5.5))
    ax.axis('off')
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 5.8)

    # Strefy
    zone_off = FancyBboxPatch((-0.3, -0.2), 4.1, 5.5,
                              boxstyle="round,pad=0.15", facecolor='#F0F8FF',
                              edgecolor=C_BLUE, linewidth=2, linestyle='--', zorder=0)
    ax.add_patch(zone_off)
    ax.text(1.75, 5.5, "OFF-CHAIN (Python)",
            ha='center', fontsize=9, fontweight='bold', color=C_BLUE)

    zone_on = FancyBboxPatch((4.7, -0.2), 3.6, 5.5,
                             boxstyle="round,pad=0.15", facecolor='#FFF5F0',
                             edgecolor=C_RED, linewidth=2, linestyle='--', zorder=0)
    ax.add_patch(zone_on)
    ax.text(6.5, 5.5, "ON-CHAIN (Solidity / EVM)",
            ha='center', fontsize=9, fontweight='bold', color=C_RED)

    bw, bh = 1.6, 0.8
    # Off-chain
    rbox(ax, 0.6, 4.0, bw, bh, "Dane CEX\n(Binance OHLCV)", '#D6EAF8', C_BLUE, fs=6.5)
    rbox(ax, 2.7, 4.0, bw, bh, "Inzynieria cech\n(sigma, OFI, log-r)", '#D6EAF8', C_BLUE, fs=6.5)
    rbox(ax, 0.6, 2.3, bw, bh, "Agent PPO\n(Actor-Critic\n128 x 128)", '#D5F5E3', C_GREEN, fs=6.5)
    rbox(ax, 2.7, 2.3, bw, bh, "Most Web3\n(web3.py)", '#FEF9E7', C_YELLOW, fs=6.5)
    rbox(ax, 1.65, 0.7, 3.0, 0.6, "Pula cieni (Shadow Pool) — symulacja LVR", '#E8DAEF', C_VIOLET, fs=6)

    # On-chain
    rbox(ax, 6.5, 4.0, 2.8, bh, "VolatilityFeeHook.sol\n(beforeSwap / afterSwap)", '#FADBD8', C_RED, fs=6.5)
    rbox(ax, 6.5, 2.3, 2.8, bh, "PoolManager (Singleton)\nswap() -> Hook -> settle()", '#FADBD8', C_RED, fs=6.5)
    rbox(ax, 6.5, 0.7, 2.8, 0.6, "Wyrocznia EMA (afterSwap)", '#FFE0B2', '#E67E22', fs=6.5)

    # Strzalki off-chain
    arr(ax, 0.6 + bw/2, 4.0, 2.7 - bw/2, 4.0, color=C_BLUE, lw=1)
    arr(ax, 2.7, 4.0 - bh/2, 0.6, 2.3 + bh/2, color=C_BLUE, lw=1)
    arr(ax, 0.6 + bw/2, 2.3, 2.7 - bw/2, 2.3, color=C_GREEN, lw=1.5)
    ax.text(1.65, 2.55, "a(t)", fontsize=7, fontweight='bold', color=C_GREEN, ha='center')

    # Strzalka most → hook
    arr(ax, 2.7 + bw/2, 2.5, 6.5 - 2.8/2, 3.8, color=C_YELLOW, lw=2.5)
    ax.text(4.3, 3.5, "updateFee(fee)",
            fontsize=7, fontweight='bold', color='#B7950B', rotation=20,
            bbox=dict(facecolor='white', edgecolor=C_YELLOW,
                      boxstyle='round,pad=0.15', alpha=0.95))

    # Strzalki on-chain
    arr(ax, 6.5, 4.0 - bh/2, 6.5, 2.3 + bh/2, color=C_RED, lw=1, style='<->')
    arr(ax, 6.5, 2.3 - bh/2, 6.5, 0.7 + 0.3, color='#E67E22', lw=1)

    ax.set_title("Rysunek 3.2. Architektura hybrydowa systemu adaptacyjnych oplat",
                 fontsize=10, pad=10, loc='left', fontweight='bold')
    src(fig)
    save(fig, "rys_3_2_hybrid_architecture.png")


# ══════════════════════════════════════════════════════════════
# 7. OS CZASU PODZIALU DANYCH
# ══════════════════════════════════════════════════════════════
def fig07():
    fig, ax = plt.subplots(figsize=(W, 2.8))

    bar_y = 1.0
    bh = 0.5

    ax.barh(bar_y, 21, left=0,  height=bh, color='#D6EAF8', edgecolor=C_BLUE,  linewidth=1.5)
    ax.barh(bar_y, 6,  left=21, height=bh, color='#D5F5E3', edgecolor=C_GREEN, linewidth=1.5)
    ax.barh(bar_y, 3,  left=27, height=bh, color='#FADBD8', edgecolor=C_RED,   linewidth=1.5)

    ax.text(10.5, bar_y, "ZBIOR TRENINGOWY\n(lip. 2023 — mar. 2025)\n38,1 mln rek.  |  ~21 mies.",
            ha='center', va='center', fontsize=7, fontweight='bold', color=C_DARK, linespacing=1.2)

    ax.text(24, bar_y + bh/2 + 0.35, "WALIDACYJNY\n(kw.—wrz. 2025)\n6 mies.",
            ha='center', va='center', fontsize=6.5, fontweight='bold', color=C_GREEN, linespacing=1.15)
    arr(ax, 24, bar_y + bh/2 + 0.1, 24, bar_y + bh/2 + 0.01, color=C_GREEN, lw=0.8)

    ax.text(28.5, bar_y + bh/2 + 0.35, "TESTOWY\n(paz. 2025—\nsty. 2026)",
            ha='center', va='center', fontsize=6.5, fontweight='bold', color=C_RED, linespacing=1.15)
    arr(ax, 28.5, bar_y + bh/2 + 0.1, 28.5, bar_y + bh/2 + 0.01, color=C_RED, lw=0.8)

    dates = [(0,'07.2023'), (6,'01.2024'), (12,'07.2024'), (18,'01.2025'),
             (21,'04.2025'), (24,'07.2025'), (27,'10.2025'), (30,'01.2026')]
    for pos, lbl in dates:
        ax.text(pos, bar_y - bh/2 - 0.12, lbl, ha='center', va='top', fontsize=6, color=C_DARK)
        ax.plot([pos, pos], [bar_y - bh/2 + 0.03, bar_y + bh/2 - 0.03],
                color='#CCCCCC', lw=0.4, ls=':', zorder=0)

    ax.set_xlim(-1, 31.5)
    ax.set_ylim(-0.2, 2.0)
    ax.axis('off')
    ax.set_title("Rysunek 4.1. Temporalny podzial zbioru danych (walk-forward validation)",
                 fontsize=10, pad=10, loc='left', fontweight='bold')
    src(fig)
    save(fig, "rys_4_1_data_timeline.png")


# ══════════════════════════════════════════════════════════════
# 8. TABELA WYNIKOW BENCHMARKOWYCH
# Dane: benchmark_table.json (zweryfikowane)
# Agent PnL=7.08%, fee=$71 106, LVR=0.0042, DD=-0.005%, Sharpe=5.95
# Static 0.30%: PnL=2.54%, fee=$27 658, Sharpe=218.56
# Static 0.05%: PnL=7.15%, fee=$71 502, DD=-0.005%, Sharpe=365.64
# HODL: PnL=-0.16%, Sharpe=-0.02
# ══════════════════════════════════════════════════════════════
def fig08():
    fig, ax = plt.subplots(figsize=(W, 3.8))
    ax.axis('off')

    cols = ["Strategia", "Skumul.\nPnL (%)", "Dochod z oplat\n(USD)",
            "LVR\n(pkt baz.)", "Maks.\nobsun.", "Wsp.\nSharpe'a"]
    rows = [
        ["Agent RL (V4)",        "7,08",  "71 106", "0,0042", "0,005%", "5,95"],
        ["Statyczna 0,30%",      "2,54",  "27 658", "0,0042", "0,000%",  "—"],
        ["Statyczna 0,05%",      "7,15",  "71 502", "0,0042", "0,005%", "—"],
        ["HODL (kup i trzymaj)", "-0,16", "—",      "—",      "—",       "-0,02"],
    ]

    table = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    style_tbl(table, len(cols), len(rows), fs=8)
    table.scale(1.0, 1.9)

    for j in range(len(cols)):
        table[1, j].set_facecolor('#DBEAFE')
        table[1, j].set_text_props(fontweight='bold')
    for j in range(len(cols)):
        table[3, j].set_facecolor('#FEF9E7')

    ax.set_title("Tabela 5.1. Porownanie strategii na zbiorze testowym\n"
                 "(pazdziernik 2025 — styczen 2026)",
                 fontsize=10, pad=10, loc='left', fontweight='bold')
    fig.text(0.02, 0.02,
             "Uwaga: Statyczna 0,05% marginalnie lepsza (7,15% vs 7,08%) — artefakt niskiej zmiennosci.\n"
             "Sharpe dla strategii statycznych pominieto z powodu quasi-zerowej wariancji.",
             ha='left', fontsize=5.5, style='italic', color='#999999')
    src(fig)
    save(fig, "tab_5_1_benchmark_results.png")


# ══════════════════════════════════════════════════════════════
# 9. ROZKLAD NAGROD
# Dane: v4_final_evaluation.json (50 epizodow)
# agent_mean=1304.35, agent_std=808.67
# baseline_mean=253.84, baseline_std=1.56
# t=9.09, p=4.28e-12, d=1.84, win_rate=0.86
# ══════════════════════════════════════════════════════════════
def fig09():
    agent_rewards = [
        615.52, 139.37, 298.01, 2402.39, 1089.96, 421.05, 229.22, 2308.87,
        1478.16, 1525.50, 819.52, 1240.77, 1502.63, 202.68, 1225.11, 2144.17,
        1058.22, 1135.11, 1736.57, 132.14, 427.88, 2430.32, 2624.68, 2019.84,
        787.72, 340.79, 1141.02, 2416.80, 1559.27, 2361.61, 136.94, 2478.65,
        1587.48, 917.90, 255.78, 2603.87, 195.75, 682.70, 1702.32, 1415.76,
        2122.06, 1634.04, 187.93, 2428.51, 1628.75, 2204.90, 1040.44, 2004.76,
        1919.24, 254.70,
    ]
    baseline_mean = 253.84

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W, 3.8),
                                   gridspec_kw={'width_ratios': [2.2, 1]})
    fig.subplots_adjust(top=0.85, wspace=0.35)

    ax1.hist(agent_rewards, bins=14, color=C_BLUE, alpha=0.75,
             edgecolor='white', linewidth=0.6, label='Agent RL', zorder=2)
    ax1.axvline(np.mean(agent_rewards), color=C_BLUE, ls='--', lw=1.5,
                label=f'Srednia agenta = {np.mean(agent_rewards):.0f}', zorder=3)
    ax1.axvline(baseline_mean, color=C_RED, ls='-', lw=1.5,
                label=f'Srednia baseline = {baseline_mean:.0f}', zorder=3)
    ax1.set_xlabel("Nagroda epizodyczna", fontsize=8)
    ax1.set_ylabel("Liczba epizodow", fontsize=8)
    ax1.legend(fontsize=6.5, loc='upper right')
    ax1.set_title("A) Rozklad nagrod (50 epiz.)", fontweight='bold', fontsize=9)
    ax1.grid(axis='y', alpha=0.2)
    ax1.tick_params(labelsize=7)

    bp = ax2.boxplot(
        [agent_rewards, [baseline_mean]*50],
        tick_labels=['Agent RL', 'Baseline\n(0,30%)'],
        patch_artist=True,
        medianprops=dict(color='black', linewidth=1.5),
        whiskerprops=dict(color=C_DARK),
        capprops=dict(color=C_DARK),
    )
    bp['boxes'][0].set_facecolor(C_BLUE); bp['boxes'][0].set_alpha(0.5)
    bp['boxes'][1].set_facecolor(C_RED);  bp['boxes'][1].set_alpha(0.5)
    ax2.set_ylabel("Nagroda", fontsize=8)
    ax2.set_title("B) Porownanie", fontweight='bold', fontsize=9)
    ax2.grid(axis='y', alpha=0.2)
    ax2.tick_params(labelsize=7)

    stats = "t-Welch = 9,09\np = 4,28e-12\nd Cohena = 1,84\nWin rate = 86%"
    ax2.text(0.95, 0.95, stats, transform=ax2.transAxes, fontsize=6.5,
             va='top', ha='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEF9E7',
                       edgecolor=C_YELLOW, alpha=0.9))

    fig.suptitle("Rysunek 5.1. Rozklad nagrod epizodycznych: Agent RL vs Baseline",
                 fontsize=10, fontweight='bold')
    src(fig)
    save(fig, "rys_5_1_reward_distribution.png")


# ══════════════════════════════════════════════════════════════
# 10. TABELA STRESS TESTOW
# Dane: elasticity_stress_test.json + v4_final_evaluation.json
# ══════════════════════════════════════════════════════════════
def fig10():
    fig, ax = plt.subplots(figsize=(W, 3.8))
    ax.axis('off')

    cols = ["Scenariusz", "eps", "Nagroda agenta\n(mi +/- sigma)",
            "Nagroda baseline\n(mi +/- sigma)", "Roznica\n(%)",
            "Bankr.", "Sr. oplata\n(pkt baz.)"]
    rows = [
        ["Niska elast.",    "-1,0",        "268,9 +/- 2,2",  "276,7 +/- 1,2", "-2,8%",   "0/100", "41,2"],
        ["Srednia elast.",  "-1,5",        "233,7 +/- 12,1", "276,7 +/- 1,2", "-15,5%",  "0/100", "41,2"],
        ["Wysoka elast.",   "-2,5",        "181,3 +/- 25,1", "276,7 +/- 1,2", "-34,5%",  "0/100", "41,2"],
        ["Randomizacja\n(tren.)", "U[-3; -0,5]", "1 304 +/- 809", "254 +/- 1,6", "+413,8%","0/50", "—"],
    ]

    table = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    style_tbl(table, len(cols), len(rows), fs=7)
    table.scale(1.0, 1.9)

    for i in range(3):
        table[i+1, 4].set_facecolor('#FADBD8')
    table[4, 4].set_facecolor('#D1FAE5')
    table[4, 0].set_text_props(fontweight='bold')

    ax.set_title("Tabela 5.2. Test warunkow skrajnych: ustalone wartosci elastycznosci",
                 fontsize=10, pad=10, loc='left', fontweight='bold')
    fig.text(0.02, 0.01,
             "Agent przewyzsza baseline wylacznie przy zrandomizowanej eps. Przy sztywnej eps — suboptymalny, lecz 0 bankructw.",
             ha='left', fontsize=5.5, style='italic', color='#999999')
    src(fig)
    save(fig, "tab_5_2_stress_tests.png")


# ══════════════════════════════════════════════════════════════
# 11. WYKRES ABLACJI CECH
# Dane: feature_ablation.json (reward_drop_pct)
# ══════════════════════════════════════════════════════════════
def fig11():
    data = [
        ('fee_income_rate',    6.74,  "Tempo dochodu z oplat"),
        ('current_fee_norm',  -5.12,  "Znorm. biezaca oplata"),
        ('hour_sin',           0.20,  "Cykl godzinowy (sin)"),
        ('dow_cos',            0.12,  "Cykl tygodniowy (cos)"),
        ('realized_vol_10',    0.083, "Zmiennosc zrealiz. (10 bl.)"),
        ('tick_delta',         0.049, "Zmiana ticka"),
        ('dow_sin',           -0.038, "Cykl tygodniowy (sin)"),
        ('lvr_rate_bps',       0.022, "Stopa LVR (pkt baz.)"),
        ('vol_ratio_10_100',   0.013, "Wsp. zmiennosci 10/100"),
        ('log_return_5',       0.006, "Log-zwrot (5 bl.)"),
        ('hour_cos',           0.006, "Cykl godzinowy (cos)"),
        ('log_return_1',       0.002, "Log-zwrot (1 bl.)"),
        ('log_return_100',     0.002, "Log-zwrot (100 bl.)"),
        ('toxic_flow_ratio',   0.001, "Toksyczn. przeplywu"),
        ('cumulative_pnl',    -0.014, "Skumulowany PnL"),
        ('ofi_10',            -0.016, "OFI (10 bl.)"),
        ('log_return_20',     -0.004, "Log-zwrot (20 bl.)"),
    ]

    data_s = sorted(data, key=lambda x: abs(x[1]), reverse=True)
    names  = [d[0] for d in data_s]
    drops  = [d[1] for d in data_s]
    labels = [d[2] for d in data_s]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W, 5.5),
                                   gridspec_kw={'width_ratios': [3, 2]})
    fig.subplots_adjust(top=0.92, wspace=0.55, left=0.22, right=0.97)

    colors = [C_RED if d > 0 else C_GREEN for d in drops]
    ax1.barh(range(len(names)), drops, color=colors, alpha=0.8,
             edgecolor='white', linewidth=0.6)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(labels, fontsize=5.5)
    ax1.set_xlabel("Zmiana nagrody po okluzji (%)", fontsize=7)
    ax1.axvline(0, color='black', lw=0.6)
    ax1.set_title("A) Okluzja pojedynczych cech", fontweight='bold', fontsize=9)
    ax1.grid(axis='x', alpha=0.2)
    ax1.invert_yaxis()
    ax1.tick_params(labelsize=6)

    legend_el = [
        Patch(facecolor=C_RED, alpha=0.8, label='Usuniecie poprawia nagrode\n(cecha szumowa)'),
        Patch(facecolor=C_GREEN, alpha=0.8, label='Usuniecie pogarsza nagrode\n(cecha istotna)'),
    ]
    ax1.legend(handles=legend_el, fontsize=5.5, loc='lower right', framealpha=0.9)

    # Grupy
    groups = [
        ("Stan biezacy\n(Runtime state)", 12.85),
        ("Cyklicznosc\n(Temporal)",        0.25),
        ("Ochrona LVR",                   0.12),
        ("Mikrostruktura\nrynku",          0.04),
        ("Momentum\ncenowy",             -0.001),
    ]
    g_n = [g[0] for g in groups]
    g_d = [g[1] for g in groups]
    gc  = [C_RED if d > 0 else C_GREEN for d in g_d]

    ax2.barh(range(len(g_n)), g_d, color=gc, alpha=0.8, edgecolor='white', height=0.5)
    ax2.set_yticks(range(len(g_n)))
    ax2.set_yticklabels(g_n, fontsize=7)
    ax2.set_xlabel("Zmiana nagrody po okluzji grupy (%)", fontsize=7)
    ax2.axvline(0, color='black', lw=0.6)
    ax2.set_title("B) Okluzja grup cech", fontweight='bold', fontsize=9)
    ax2.grid(axis='x', alpha=0.2)
    ax2.invert_yaxis()
    ax2.tick_params(labelsize=6)

    ax2.text(0.95, 0.90,
             "Stan biezacy:\n12,85% wplywu\n— dominujaca\ngrupa cech",
             transform=ax2.transAxes, fontsize=6.5, va='top', ha='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEF9E7',
                       edgecolor=C_YELLOW, alpha=0.9))

    fig.suptitle("Rysunek 5.2. Analiza ablacyjna cech (okluzja cech)",
                 fontsize=10, fontweight='bold')
    src(fig)
    save(fig, "rys_5_2_feature_ablation.png")


# ══════════════════════════════════════════════════════════════
# 12. TABELA BREAK-EVEN L1 vs L2
# ══════════════════════════════════════════════════════════════
def fig12():
    fig, ax = plt.subplots(figsize=(W, 4.5))
    ax.axis('off')

    cols = ["Parametr", "Ethereum L1", "Arbitrum / Base (L2)"]
    rows = [
        ["Zuzycie gazu updateFee()",            "ok. 50 000 gas",    "ok. 50 000 gas"],
        ["Cena gasu",                           "20 — 80 gwei",      "0,01 — 0,1 gwei"],
        ["Koszt transakcji\n(ETH = 2 500 USD)", "2,50 — 10,00 USD", "0,001 — 0,01 USD"],
        ["Czas bloku",                          "12 s",              "ok. 0,25 s"],
        ["Aktualizacje / godz.",                "300",               "14 400"],
        ["Koszt / godz.",                       "750 — 3 000 USD",   "0,01 — 0,14 USD"],
        ["Prog rentownosci\n(dochod / blok)",   "2,50 — 10 USD",    "0,001 USD"],
        ["WNIOSEK",                             "Nieoplacalne\nTVL < 50 mln USD",
                                                                     "Oplacalne\nTVL > 10 tys. USD"],
    ]

    table = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    style_tbl(table, len(cols), len(rows), fs=7.5)
    table.scale(1.0, 1.7)

    n = len(rows)
    table[n, 0].set_text_props(fontweight='bold', color=C_DARK)
    table[n, 1].set_facecolor('#FADBD8'); table[n, 1].set_text_props(fontweight='bold')
    table[n, 2].set_facecolor('#D1FAE5'); table[n, 2].set_text_props(fontweight='bold')

    ax.set_title("Tabela 5.3. Analiza progu rentownosci: koszty operacyjne L1 vs L2",
                 fontsize=10, pad=10, loc='left', fontweight='bold')
    src(fig, "opracowanie wlasne na podst. danych etherscan.io, arbiscan.io")
    save(fig, "tab_5_3_breakeven_l1_l2.png")


# ══════════════════════════════════════════════════════════════
# 13. LATENCJA INFERENCJI
# Dane: inference_benchmark.json
# CPU:  mean=0.401ms, p99=0.725ms, max=0.922ms
# CUDA: mean=0.576ms, p99=1.178ms, max=1.587ms
# ══════════════════════════════════════════════════════════════
def fig13():
    fig, ax = plt.subplots(figsize=(W, 4.0))

    devices = ['CPU', 'CUDA (GPU)']
    means = [0.401, 0.576]
    p99s  = [0.725, 1.178]
    maxs  = [0.922, 1.587]

    x = np.arange(len(devices))
    w = 0.22

    b1 = ax.bar(x - w, means, w, label='Srednia',       color=C_BLUE,   alpha=0.9, zorder=2)
    b2 = ax.bar(x,     p99s,  w, label='Percentyl 99.', color=C_ORANGE, alpha=0.9, zorder=2)
    b3 = ax.bar(x + w, maxs,  w, label='Maksimum',      color=C_RED,    alpha=0.9, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(devices, fontsize=9)
    ax.set_ylabel("Czas inferencji (ms)", fontsize=8)
    ax.set_title("Rysunek 5.3. Latencja inferencji modelu PPO",
                 fontsize=10, loc='left', fontweight='bold')
    ax.legend(fontsize=7.5)
    ax.grid(axis='y', alpha=0.2)
    ax.tick_params(labelsize=7)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold')

    ax.text(0.97, 0.93,
            "Budzet bloku ETH = 12 000 ms\nInferencja < 0,002% budzetu",
            transform=ax.transAxes, fontsize=7, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#D1FAE5',
                      edgecolor=C_GREEN, alpha=0.9))
    src(fig)
    fig.tight_layout()
    save(fig, "rys_5_3_inference_latency.png")


# ══════════════════════════════════════════════════════════════
# 14. BAYESOWSKI ESTYMATOR ELASTYCZNOSCI
# ══════════════════════════════════════════════════════════════
def fig14():
    fig, ax = plt.subplots(figsize=(W, 4.5))
    ax.axis('off')
    ax.set_xlim(-0.2, 8.2)
    ax.set_ylim(-0.2, 4.5)

    bw, bh = 1.7, 1.0

    # Gorny wiersz
    rbox(ax, 0.9, 3.2, bw, bh,
         "Prior bayesowski\neps_0 = -1,5\nwaga w_0 = 1,0", '#D6EAF8', C_BLUE, fs=6.5)
    rbox(ax, 3.2, 3.2, 1.9, bh,
         "Zanik wykladniczy\nw(t)=w_0*exp(-lambda*t)\nlambda=0,15\nt_1/2 ~ 4,6 bl.", '#FEF9E7', C_YELLOW, fs=6)
    rbox(ax, 5.6, 3.2, bw, bh,
         "Estymata eps(t)\n-> funkcja nagrody\n-> kara LVR~|eps|", '#FADBD8', C_RED, fs=6.5)

    # Dolny wiersz
    rbox(ax, 0.9, 1.2, bw, bh,
         "Obserwacja rynk.\ndelta_V / delta_fee\n(reakcja wolumenu)", '#D5F5E3', C_GREEN, fs=6.5)
    rbox(ax, 3.2, 1.2, 1.9, bh,
         "Aktualizacja\na posteriori\neps=w*eps_0\n+(1-w)*eps_obs", '#E8DAEF', C_VIOLET, fs=6)
    rbox(ax, 5.6, 1.2, bw, bh,
         "Min. prog wagi\nmin(w_t) = 0,01\nPrior nie zanika\ncalkowicie", '#FFE0B2', '#E67E22', fs=6.5)

    # Wyjscie
    rbox(ax, 7.5, 2.2, 1.0, 1.0,
         "Nagroda\nr_t = fee\n- |eps|*LVR", '#FCF3CF', '#B7950B', fs=6.5)

    # Strzalki
    arr(ax, 0.9 + bw/2, 3.2, 3.2 - 1.9/2, 3.2, lw=1.5)
    arr(ax, 3.2 + 1.9/2, 3.2, 5.6 - bw/2, 3.2, lw=1.5)
    arr(ax, 0.9 + bw/2, 1.2, 3.2 - 1.9/2, 1.2, lw=1.5)
    arr(ax, 3.2 + 1.9/2, 1.2, 5.6 - bw/2, 1.2, lw=1.5)
    arr(ax, 3.2, 3.2 - bh/2, 3.2, 1.2 + bh/2, lw=1)
    arr(ax, 5.6, 1.2 + bh/2, 5.6, 3.2 - bh/2, color='#E67E22', lw=1, ls='--')
    arr(ax, 5.6 + bw/2, 3.0, 7.5 - 0.5, 2.5, lw=1.5)

    ax.text(2.85, 2.2, "wagi\nprioru", fontsize=6, color=C_DARK, style='italic',
            ha='center', va='center')

    ax.set_title("Rysunek 3.3. Bayesowski estymator elastycznosci cenowej\n"
                 "z zanikajacym priorem (exponential decay)",
                 fontsize=10, pad=10, loc='left', fontweight='bold')
    src(fig)
    save(fig, "rys_3_3_bayesian_estimator.png")


# ══════════════════════════════════════════════════════════════
# 15. POROWNANIE PnL
# Dane: benchmark_table.json
# ══════════════════════════════════════════════════════════════
def fig15():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W, 3.8))
    fig.subplots_adjust(top=0.85, wspace=0.4)

    strats = ['Agent RL\n(V4)', 'Statyczna\n0,30%', 'Statyczna\n0,05%', 'HODL']
    pnl  = [7.08, 2.54, 7.15, -0.16]
    fees = [71106, 27658, 71502]
    colors = [C_BLUE, C_PURPLE, C_ORANGE, C_RED]

    bars1 = ax1.bar(strats, pnl, color=colors, alpha=0.85,
                    edgecolor='white', linewidth=1, zorder=2)
    ax1.set_ylabel("Skumulowany PnL (%)", fontsize=8)
    ax1.set_title("A) Skumulowany zwrot", fontweight='bold', fontsize=9)
    ax1.grid(axis='y', alpha=0.2)
    ax1.axhline(0, color='black', lw=0.6)
    ax1.tick_params(labelsize=7)
    for bar, val in zip(bars1, pnl):
        yo = 0.12 if val >= 0 else -0.3
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + yo,
                 f'{val:.2f}%', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    bars2 = ax2.bar(strats[:3], fees, color=colors[:3], alpha=0.85,
                    edgecolor='white', linewidth=1, zorder=2)
    ax2.set_ylabel("Dochod z oplat (USD)", fontsize=8)
    ax2.set_title("B) Dochod prowizyjny", fontweight='bold', fontsize=9)
    ax2.grid(axis='y', alpha=0.2)
    ax2.tick_params(labelsize=7)
    for bar, val in zip(bars2, fees):
        fmt = f'{val:,}'.replace(',', ' ')
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 700,
                 f'{fmt} USD', ha='center', va='bottom', fontsize=7, fontweight='bold')

    fig.suptitle("Rysunek 5.4. Porownanie strategii: PnL i dochod z oplat",
                 fontsize=10, fontweight='bold')
    src(fig)
    save(fig, "rys_5_4_pnl_comparison.png")


# ══════════════════════════════════════════════════════════════
FINAL_ASSETS = [
    ("Tabela 1.1", fig01),
    ("Rysunek 1.1", fig02),
    ("Rysunek 2.1", fig03),
    ("Rysunek 3.1", fig05),
    ("Rysunek 3.2", fig06),
    ("Rysunek 3.3", fig14),
    ("Rysunek 4.1", fig07),
]

STALE_OUTPUTS = [
    "tab_2_1_curriculum_phases.png",
    "tab_5_1_benchmark_results.png",
    "rys_5_1_reward_distribution.png",
    "tab_5_2_stress_tests.png",
    "rys_5_2_feature_ablation.png",
    "tab_5_3_breakeven_l1_l2.png",
    "rys_5_3_inference_latency.png",
    "rys_5_4_pnl_comparison.png",
]


def cleanup_stale_outputs():
    for name in STALE_OUTPUTS:
        path = os.path.join(OUTPUT_DIR, name)
        if os.path.exists(path):
            os.remove(path)
            print(f"  [DEL] {name}")


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  Finalne diagramy statyczne")
    print("=" * 60)
    print(f"  Katalog: {OUTPUT_DIR}\n")

    cleanup_stale_outputs()

    for _, func in FINAL_ASSETS:
        func()

    print(f"\n  Wygenerowano {len(FINAL_ASSETS)} plikow finalnych w: {OUTPUT_DIR}")
    print("=" * 60)
