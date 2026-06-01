"""Embedded DSO catalog — Messier + brightest NGC/IC objects.

Each entry: (name, ra_deg, dec_deg, size_arcmin, type_code)
type_code: G=Galaxy, N=Nebula, OC=Open Cluster, GC=Globular Cluster,
           PN=Planetary Nebula, SNR=Supernova Remnant, EN=Emission Nebula
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DSOEntry:
    name: str
    ra_deg: float
    dec_deg: float
    size_arcmin: float
    type_code: str


# ── Catalog ────────────────────────────────────────────────────────────────────
# fmt: off
_RAW = [
    # Messier objects
    ("M1",    83.633,  22.015,  7.0,  "SNR"),   # Crab Nebula
    ("M2",   323.363,  -0.823, 12.9,  "GC"),
    ("M3",   205.548,  28.377, 18.6,  "GC"),
    ("M4",   245.897, -26.526, 36.0,  "GC"),
    ("M5",   229.638,   2.082, 23.4,  "GC"),
    ("M6",   265.083, -32.217, 33.0,  "OC"),
    ("M7",   268.462, -34.793, 80.0,  "OC"),
    ("M8",   270.920, -24.383, 90.0,  "EN"),    # Lagoon Nebula
    ("M11",  282.767,  -6.267, 14.0,  "OC"),
    ("M13",  250.423,  36.461, 36.0,  "GC"),    # Hercules Cluster
    ("M15",  322.493,  12.167, 18.0,  "GC"),
    ("M16",  274.700, -13.800, 35.0,  "EN"),    # Eagle Nebula
    ("M17",  275.197, -16.172, 46.0,  "EN"),    # Omega Nebula
    ("M20",  270.633, -23.033, 29.0,  "EN"),    # Trifid Nebula
    ("M22",  279.100, -23.905, 32.0,  "GC"),
    ("M27",  299.901,  22.721,  8.0,  "PN"),    # Dumbbell Nebula
    ("M31",  10.685,  41.269, 190.0,  "G"),     # Andromeda Galaxy
    ("M32",  10.674,  40.866,  8.0,   "G"),
    ("M33",  23.462,  30.660,  73.0,  "G"),     # Triangulum Galaxy
    ("M42",  83.822,  -5.391, 85.0,   "EN"),    # Orion Nebula
    ("M43",  83.885,  -5.267,  20.0,  "EN"),
    ("M44",  130.100,  19.667, 95.0,  "OC"),    # Beehive Cluster
    ("M45",  56.750,  24.117, 110.0,  "OC"),    # Pleiades
    ("M51",  202.470,  47.195, 11.0,  "G"),     # Whirlpool Galaxy
    ("M57",  283.396,  33.029,  1.4,  "PN"),    # Ring Nebula
    ("M63",  198.955,  42.032, 12.6,  "G"),     # Sunflower Galaxy
    ("M64",  194.182,  21.682,  9.3,  "G"),     # Black Eye Galaxy
    ("M74",  24.174,  15.783, 10.5,   "G"),
    ("M78",  86.685,   0.079,  8.0,   "EN"),
    ("M81",  148.888,  69.065, 26.9,  "G"),     # Bode's Galaxy
    ("M82",  148.969,  69.680, 11.2,  "G"),     # Cigar Galaxy
    ("M83",  204.254, -29.866, 13.0,  "G"),     # Southern Pinwheel
    ("M92",  259.281,  43.134, 14.0,  "GC"),
    ("M97",  168.699,  55.019,  3.4,  "PN"),    # Owl Nebula
    ("M101", 210.802,  54.349, 28.8,  "G"),     # Pinwheel Galaxy
    ("M104", 189.998,  -11.623, 9.0,  "G"),     # Sombrero Galaxy
    ("M106", 184.740,  47.304, 18.6,  "G"),
    ("M108", 167.881,  55.674, 8.7,   "G"),
    ("M109", 179.401,  53.374, 7.6,   "G"),
    ("M110", 10.092,  41.685, 21.9,   "G"),
    # Bright NGC objects
    ("NGC 224",   10.685,  41.269, 190.0, "G"),   # = M31
    ("NGC 869",   34.750,  57.133,  30.0, "OC"),  # h Persei
    ("NGC 884",   35.602,  57.133,  30.0, "OC"),  # χ Persei
    ("NGC 1499",  60.492,  36.374, 160.0, "EN"),  # California Nebula
    ("NGC 1502",  61.081,  62.333,  20.0, "OC"),
    ("NGC 1976",  83.822,  -5.391,  85.0, "EN"),  # = M42
    ("NGC 2024",  85.424,  -1.912,  30.0, "EN"),  # Flame Nebula
    ("NGC 2070", 84.676,  -69.101, 40.0,  "EN"),  # Tarantula Nebula
    ("NGC 2237", 97.960,   4.967,  80.0,  "EN"),  # Rosette Nebula
    ("NGC 2244", 97.979,   4.833,  29.0,  "OC"),
    ("NGC 2264", 100.238,  9.895,  20.0,  "OC"),  # Christmas Tree / Cone
    ("NGC 2392", 112.295,  20.911,  0.7,  "PN"),  # Eskimo Nebula
    ("NGC 2403", 114.215,  65.603,  21.9, "G"),
    ("NGC 2682", 132.825,  11.814,  30.0, "OC"),  # M67
    ("NGC 3031", 148.888,  69.065,  26.9, "G"),   # = M81
    ("NGC 3034", 148.969,  69.680,  11.2, "G"),   # = M82
    ("NGC 3372",  160.990, -59.868, 120.0,"EN"),  # Eta Carinae Nebula
    ("NGC 3628",  170.070,  13.589,  14.8, "G"),  # Hamburger Galaxy
    ("NGC 4038",  180.471, -18.868,   5.2, "G"),  # Antennae
    ("NGC 4565",  189.087,  25.988,  15.9, "G"),  # Needle Galaxy
    ("NGC 4631",  190.533,  32.542,  15.5, "G"),  # Whale Galaxy
    ("NGC 5128",  201.365, -43.019,  25.7, "G"),  # Centaurus A
    ("NGC 5194",  202.470,  47.195,  11.0, "G"),  # = M51
    ("NGC 5457",  210.802,  54.349,  28.8, "G"),  # = M101
    ("NGC 6188",  248.936, -53.722,  20.0, "EN"), # Rim Nebula
    ("NGC 6334",  260.833, -35.967,  35.0, "EN"), # Cat's Paw Nebula
    ("NGC 6357",  262.695, -34.200,  50.0, "EN"), # War & Peace Nebula
    ("NGC 6514",  270.633, -23.033,  29.0, "EN"), # = M20 Trifid
    ("NGC 6523",  270.920, -24.383,  90.0, "EN"), # = M8 Lagoon
    ("NGC 6611",  274.700, -13.800,  35.0, "EN"), # = M16 Eagle
    ("NGC 6618",  275.197, -16.172,  46.0, "EN"), # = M17 Omega
    ("NGC 6720",  283.396,  33.029,   1.4, "PN"), # = M57 Ring
    ("NGC 6888",  303.152,  38.350,  20.0, "EN"), # Crescent Nebula
    ("NGC 6992",  313.875,  31.728,  60.0, "SNR"),# Eastern Veil
    ("NGC 6960",  312.562,  30.720,  70.0, "SNR"),# Western Veil
    ("NGC 6979",  313.213,  32.350,  40.0, "SNR"),# Veil Nebula centre
    ("NGC 7000",  314.750,  44.417, 120.0, "EN"), # North America Nebula
    ("NGC 7009",  323.360,  -11.362,  1.7, "PN"), # Saturn Nebula
    ("NGC 7293",  337.411,  -20.837,  16.0,"PN"), # Helix Nebula
    ("NGC 7331",  339.267,  34.416,  10.5, "G"),
    ("NGC 7380",  341.700,  58.117,  12.0, "OC"),
    ("NGC 7789",  359.333,  56.717,  16.0, "OC"), # Caroline's Rose
    ("IC 405",    79.720,  34.267,  30.0,  "EN"), # Flaming Star Nebula
    ("IC 410",    82.200,  33.417,  40.0,  "EN"), # Tadpoles Nebula
    ("IC 434",    83.750,  -2.600,  60.0,  "EN"), # Horsehead region
    ("IC 443",    94.211,  22.583,  45.0,  "SNR"),# Jellyfish Nebula
    ("IC 1318",  305.000,  40.000,  80.0,  "EN"), # Gamma Cygni Nebula
    ("IC 1805",   38.175,  61.467,  60.0,  "EN"), # Heart Nebula
    ("IC 1848",   43.267,  60.467,  60.0,  "EN"), # Soul Nebula
    ("IC 2118",   76.700,  -7.217,  180.0, "EN"), # Witch Head Nebula
    ("IC 2177",  108.965,  -10.400,  80.0, "EN"), # Seagull Nebula
    ("IC 5070",  312.950,  44.367,  60.0,  "EN"), # Pelican Nebula
    ("IC 5146",  328.344,  47.261,  12.0,  "EN"), # Cocoon Nebula
    ("Sh2-132",  330.000,  56.000,  50.0,  "EN"), # Lion Nebula
    ("Sh2-155",  342.900,  62.633,  50.0,  "EN"), # Cave Nebula
    ("Sh2-308",  107.000,  -26.333,  40.0, "PN"), # Dolphin Nebula
]
# fmt: on

_CATALOG: list[DSOEntry] = [
    DSOEntry(name=r[0], ra_deg=r[1], dec_deg=r[2], size_arcmin=r[3], type_code=r[4])
    for r in _RAW
]


def query_dso_in_field(
    ra_center: float,
    dec_center: float,
    fov_deg: float,
) -> list[DSOEntry]:
    """Return DSOs within fov_deg/2 radius of the field centre."""
    results = []
    cos_dec = np.cos(np.radians(dec_center))
    half = fov_deg / 2.0
    for obj in _CATALOG:
        dra  = abs(obj.ra_deg  - ra_center) * cos_dec
        ddec = abs(obj.dec_deg - dec_center)
        if dra < half and ddec < half:
            results.append(obj)
    return results
