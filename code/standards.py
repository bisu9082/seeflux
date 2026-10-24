# Standard / prior simplified models of the >10 MeV atmospheric neutron flux (n/cm2/h).
# Every constant below is taken from the cited primary source (see comments); nothing is fitted here.
import numpy as np, pandas as pd, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
FT_PER_M = 1 / 0.3048

# ---- S1: IEC 62396-1 / Boeing simplified model -------------------------------------------
# IEC 62396-1 (as described by Cecchetto et al. 2019, Aerospace 6:119, Sec. 2): the Boeing-model differential
# flux at about 12 km (40,000 ft) and 45 deg geographic latitude integrated above 10 MeV gives 5600 n/cm2/h of
# neutrons, rounded to 6000 to include a 6.7% proton share; the standard calls it conservative by about a
# factor of 2 relative to in-flight measurements, and applies the 1-10 MeV Boeing altitude variation to all
# neutrons above 1 MeV. PARMA here is neutrons only, so the neutron value 5600 is the primary reference (S1);
# 6000 is kept as a variant (S1_6000). Shape: Boeing altitude curve (Normand 1996 Fig. 2) and latitude curve
# (Fig. 3), separable, digitized by code/digitize_normand.py. Consistency check: Normand 1996 Eq.(2)
# integrated 10-1000 MeV gives 5813 n/cm2/h at the same reference point.
IEC_REF_NEUTRON, IEC_REF_TOTAL = 5600.0, 6000.0
_fig2 = pd.read_csv(ROOT / "data/step4/normand1996_fig2_digitized.csv")
_fig3 = pd.read_csv(ROOT / "data/step4/normand1996_fig3_digitized.csv")


def _boeing_alt(alt_kft):
    return np.interp(alt_kft, _fig2.alt_kft, _fig2.flux_1to10_ncm2s)


def _boeing_lat(lat_deg):
    return np.interp(np.abs(lat_deg), _fig3.lat_deg, _fig3.flux_1to10_ncm2s)


def s1_iec_boeing(alt_km, lat_deg, ref=IEC_REF_NEUTRON):
    a = np.asarray(alt_km) * 1000 * FT_PER_M / 1000
    return ref * _boeing_alt(a) / _boeing_alt(40.0) * _boeing_lat(lat_deg) / _boeing_lat(45.0)


def s1_iec_boeing_6000(alt_km, lat_deg):
    """Variant with the rounded neutron-plus-proton design value 6000 n/cm2/h."""
    return s1_iec_boeing(alt_km, lat_deg, IEC_REF_TOTAL)


def s1b_iec_boeing_eq3(alt_km, lat_deg):
    """Variant using Normand's own latitude fit, Eq.(3) of Normand 1996, instead of the curve."""
    eq3 = lambda L: 0.6252 * np.exp(-0.461 * np.cos(np.radians(2 * L)) ** 2 - 0.94 * np.cos(np.radians(2 * L)) + 0.252)
    a = np.asarray(alt_km) * 1000 * FT_PER_M / 1000
    return IEC_REF_NEUTRON * _boeing_alt(a) / _boeing_alt(40.0) * eq3(np.abs(lat_deg)) / eq3(45.0)


# ---- S2: JESD89A Annex A / Gordon et al. 2004 ---------------------------------------------
# F = F_NYC * F_alt(d) * F_BSYD(Rc, h, I); F_NYC = 13 n/cm2/h (>10 MeV, NYC sea level, mid solar);
# JESD89B (2021) Annex A: F_NYC = 12.946 n/cm2/h above 10 MeV (A.2); F_alt = exp((1033.2 - d)/131.3) (A.3);
# (Gordon et al. 2004 Eq. 4 printed 1033.7 and the rounded reference 13); F_BSYD per Gordon Eqs. (A-1)-(A-6), h = 9.8025e-4 d [bar];
# solar: Smin = (A-1), Smax = (A-2), Smid = mean of (A-1) and (A-2) as defined in the Gordon Appendix (Table A-I).
# The conversion h [bar] = 9.8025e-4 d [g/cm2] is Gordon's printed constant. As printed, these equations give
# values 1.2-2.4% below Gordon's own Table A-I at sea level (check_bsyd_tableA1.py); the equations are used unchanged.
def _bsyd(rc, d, which):
    h = 9.8025e-4 * np.asarray(d, float)
    rc = np.maximum(np.asarray(rc, float), 1e-6)
    a1 = np.exp(1.84 + 0.094 * h - 0.09 * np.exp(-1.1 * h)); k1 = 1.4 - 0.56 * h + 0.24 * np.exp(-8.8 * h)
    a2 = np.exp(1.93 + 0.15 * h - 0.18 * np.exp(-10 * h));   k2 = 1.32 - 0.49 * h + 0.18 * np.exp(-9.5 * h)
    smin = 1.098 * (1 - np.exp(-a1 / rc ** k1))
    smax = 1.098 * (1 - np.exp(-a2 / rc ** k2)) * (1 - np.exp(-a1 / 50 ** k1)) / (1 - np.exp(-a2 / 50 ** k2))
    return {"min": smin, "max": smax, "mid": 0.5 * (smin + smax)}[which]


JESD_REF, JESD_D0 = 12.946, 1033.2  # JESD89B Annex A.2 and A.3


def s2_jesd89(depth, rc, solar="mid"):
    return JESD_REF * np.exp((JESD_D0 - np.asarray(depth)) / 131.3) * _bsyd(rc, depth, solar)


# ---- S4: Microchip AN4589 rev D hand formulas ---------------------------------------------
# Eq.3-1 latitude: 0.857 + 0.559 tanh((lat-42)/11.5); Eq.3-3 ABB altitude: 13.3 exp[(1-(1-h/44300)^5.26)/0.143];
# Eq.3-4 CoolCAD: 3600 * 10^(0.2093 - 2.41511 L + 0.50386 L^2), L = log10(h*3.28); h in m, valid to 15 km.
def _an_lat(lat_deg):
    return 0.857 + 0.559 * np.tanh((np.abs(lat_deg) - 42) / 11.5)


def s4_an4589_abb(alt_km, lat_deg):
    h = np.asarray(alt_km) * 1000
    return 13.3 * np.exp((1 - (1 - h / 44300) ** 5.26) / 0.143) * _an_lat(lat_deg)


def s4b_an4589_coolcad(alt_km, lat_deg):
    h = np.maximum(np.asarray(alt_km) * 1000, 300.0)
    L = np.log10(h * 3.28)
    return 3600 * 10 ** (0.2093 - 2.41511 * L + 0.50386 * L * L) * _an_lat(lat_deg)


def boeing_spectrum_integral(e1, e2):
    """Integral of Normand 1996 Eq.(2) (40 kft, 45 deg) between e1 and e2 MeV, in n/cm2/h."""
    from scipy.integrate import quad
    f = lambda l: 0.3459 * np.exp(l) ** -0.9219 * np.exp(-0.01522 * l * l) * np.exp(l)
    return quad(f, np.log(e1), np.log(e2))[0] * 3600


if __name__ == "__main__":
    print("Normand 1996 Eq.(2) integral 10-1000 MeV: %.0f n/cm2/h (IEC reference 6000); 1-10 MeV: %.3f n/cm2/s"
          % (boeing_spectrum_integral(10, 1000), boeing_spectrum_integral(1, 10) / 3600))
    # sanity checks at the anchor points of each model
    print("S1  40kft,45deg  :", float(s1_iec_boeing(12.192, 45)))
    print("S2  NYC sea level, Rc=2.08 GV (JESD89 NYC cutoff):", float(s2_jesd89(1033.2, 2.08)))
    print("S4  sea level 45 deg:", float(s4_an4589_abb(0, 45)), " 40kft 45deg:", float(s4_an4589_abb(12.192, 45)))
    print("S4b 40kft 45deg:", float(s4b_an4589_coolcad(12.192, 45)))
    print("AN lat factor at 45:", float(_an_lat(45)))
