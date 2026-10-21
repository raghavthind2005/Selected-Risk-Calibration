"""One seed of the matched-cost comparison, end to end.

    python3.12 run_seed.py <seed>

Per seed: build the context (fit, calibrate, sanity gate), build the full
two-stage decision-aware method and the cmicl_refit control, then sweep every
arm over its frozen knob grid and record the whole (cost, risk) frontier on
D_perf. Nothing is cost-matched by search; frontiers are compared afterwards.
"""
from __future__ import annotations

import json, math, pathlib, sys, time, warnings
import multiprocessing as mp
import numpy as np
from scipy.optimize import minimize
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).parent
BASE = pathlib.Path(__import__("os").environ.get(
    "LLM_OCL_BASE", str(pathlib.Path(__file__).resolve().parents[2] / "benchmarks")))
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
# Benchmark is selected by the BENCHMARK environment variable; "methane" is the default
# path is unchanged from the run that produced seed 9.
BENCH = __import__("os").environ.get("BENCHMARK", "methane")
sys.path.insert(0, str(BASE / {"methane": "methane_reactor",
                               "williams": "williams_otto",
                               "hanoi":    "hanoi_network"}[BENCH]))
import benchmark as BM
from ocl.ocl_solve import solve_ocl_multistart
from ocl.decision_envelope import (
    DeployedCorrection, fit_smooth_max_affine_envelope, fit_constant_upper_level)

ALPHA, FIT_TAU = 0.10, 0.90
N_FIT, N_CAL, N_ENV, N_SANE = 1000, 300, 400, 800
N_PROP_PL, N_PERF_PL = 20, 40
LAWS = ["random_linear", "corner_pressure", "boundary_pressure",
        "residual_hotspot_pressure", "boundary_residual_pressure"]
CAP_FRACTIONS = ([0.25, 0.40, 0.55, 0.70, 1.00] if BENCH == "methane"
                 else [0.25, 0.55, 1.00])
# Primary budget 60. 20 starts measured NON-monotone in strength
# (7-8/40 nesting violations) but 60 and 120 both monotone and agreeing within
# 3% on every rung -- so 60 is the smallest budget the evidence supports, and
# CHECK_STARTS=120 now verifies it UPWARD on the decision rungs, which is a
# stronger check than the downward 60-vs-120 one it replaces.
PRIMARY_STARTS, CHECK_STARTS = 60, 120
REFINE_TOL, REFINE_ITERS = 0.02, 4
N_SPAN_RUNGS = 3      # spanning rungs per arm; readout+pinned rungs are added on top

ARMS = ["scalar", "trust_region_box", "alpha_tightening",
        "cmicl", "cmicl_refit", "law_blind_envelope",
        "wmicl_p5", "cqr_scaled", "trust_region_hull"]
if BENCH == "hanoi":
    # STRUCTURAL EXCLUSION, written before any hanoi Stage-2 row exists.
    # trust_region_hull is not dropped because it is unfavourable -- it cannot be
    # BUILT at DIM=8. ConvexHull on N_FIT=1000 points in 8-D SIGKILLed the kernel
    # in [1b]; 400 points already give 474,818 facets and 322MB (5-D/1000 gives
    # 6,050 facets in 0.03s), and every constraint evaluation takes a max over
    # all facets. This is a limit of OptiCL's geometry, reported as such in
    # OPTICL_COMPARISON.md, NOT a baseline weakened to help us.
    # It stays reachable by name below, so asking for it yields the infeasibility
    # rather than a misleading "unknown arm".
    ARMS = [a for a in ARMS if a != "trust_region_hull"]
# Arms the frozen decision rule actually tests: they get readout rungs and the
# exact cost-matching pass. The rest are D1 peer-set CONTEXT -- reported on the
# frontier (the other comparison D3 permits) but never refined, because 8 cores
# at n=200 x 120 starts cannot afford exact matching for all eight.
# alpha_tightening promoted from context to decision arm: "why not just tighten
# alpha?" is the first reviewer question and it was REFUSED at every cap on both
# methane seeds for want of readout rungs, so we have no cost-matched answer to
# the scalar FAMILY -- only to pure_scalar.
# trust_region_box is a baseline too: OptiCL's framework embeds a convex-hull
# trust region, so it comes from the literature rather than being a control.
# All four literature baselines are decision arms: each
# gets readout rungs and the exact cost-matching pass, because a baseline that is
# only ever on the frontier can never enter a paired test.
DECISION_ARMS = ("law_blind_envelope", "cmicl", "cmicl_refit",
                 "scalar", "alpha_tightening", "trust_region_box",
                 "wmicl_p5", "cqr_scaled", "trust_region_hull")
N_SPAN_CONTEXT = 3

# ---------------------------------------------------------------- RESUME ----
# RESUME=1 continues a run from its own .partial.json instead of starting over.
# The checkpoint is gated on a fingerprint of this file, so a changed runner
# refuses to resume rather than mixing two code versions in one run.
RESUME = bool(__import__("os").environ.get("RESUME"))
_CODE_MD5 = __import__("hashlib").md5(
    pathlib.Path(__file__).read_bytes()).hexdigest()


def _code_md5():
    """Hashed ONCE at import, never re-read.

    Re-reading at checkpoint time was a real defect: editing run_seed.py while a
    run is in flight would stamp later checkpoints with the NEW hash while the
    process still executes the OLD code, and a resume would then "verify"
    against code that never produced those rows."""
    return _CODE_MD5


def _constants(arms):
    """Every constant that can change a produced number. Compared exactly."""
    return dict(bench=BENCH, dim=DIM, alpha=ALPHA, fit_tau=FIT_TAU,
                n_fit=N_FIT, n_cal=N_CAL, n_env=N_ENV, n_sane=N_SANE,
                n_prop_pl=N_PROP_PL, n_perf_pl=N_PERF_PL,
                cap_fractions=list(CAP_FRACTIONS),
                primary_starts=PRIMARY_STARTS, check_starts=CHECK_STARTS,
                refine_tol=REFINE_TOL, refine_iters=REFINE_ITERS,
                n_span_rungs=N_SPAN_RUNGS, n_span_context=N_SPAN_CONTEXT,
                cal_obj=CAL_OBJ, cal_starts=CAL_STARTS,
                grid_doublings=GRID_DOUBLINGS,
                dry=bool(DRY),
                laws=list(LAWS), arms=list(arms),
                decision_arms=list(DECISION_ARMS),
                cmicl=CMICL, wmicl=WMICL, cqr=CQR, hullc=HULLC, envc=ENVC,
                bounds=[[float(a), float(b)] for a, b in BOUNDS])


def _exact(a, b, path=""):
    """Bit-exact structural equality, with NaN == NaN (NaN is a legitimate value
    here -- an unreachable cost is nan -- so the IEEE rule would spuriously abort
    a valid resume). Returns the first differing path, or None."""
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return None
        return None if a == b else f"{path}: {a!r} != {b!r}"
    if isinstance(a, bool) != isinstance(b, bool):
        return f"{path}: {a!r} != {b!r}"
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return _exact(float(a), float(b), path) if not (
            isinstance(a, bool) or isinstance(b, bool)) else (
            None if a == b else f"{path}: {a!r} != {b!r}")
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return (f"{path}: key sets differ "
                    f"(only-checkpoint={sorted(set(b)-set(a))}, "
                    f"only-now={sorted(set(a)-set(b))})")
        for k in a:
            r = _exact(a[k], b[k], f"{path}.{k}" if path else str(k))
            if r: return r
        return None
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            r = _exact(x, y, f"{path}[{i}]")
            if r: return r
        return None
    return None if a == b else f"{path}: {a!r} != {b!r}"


def _ck_all(d, *names):
    """Every copy of a value the checkpoint carries, as (key, value) pairs.

    The checkpoint stores the context TWICE -- `ctxinfo` from the early
    checkpoint("context") and `context` from the later ck.update(...) -- and the
    method likewise as `minfo` and `method`. Returning only the first match was
    a real hole: a resume tampered with (or corrupted in) the *other* copy
    passed verification silently. Caught by the tamper suite, which perturbed
    `context.r2` by 1e-15 and was NOT aborted. Every copy is now checked, and
    the copies must also agree with each other."""
    return [(n, d[n]) for n in names if n in d]


CMICL = dict(hidden=(32, 32), wd=0.01, max_iter=2000, tol=1e-6, eps=0.01)
# ---- The literature baselines. Every constant here is fixed in advance. ----
# W-MICL: C-MICL's own ensemble-scenario baseline (regression.py train_ensembles
# + update_non_conformal_model). Members are bootstrap resamples of HALF the
# training set -- their bootstrap_size = int(0.5*len(X_train)) -- refitted with
# the identical loss and restart protocol. Requiring >= ceil((1-alpha)P) members
# feasible is exactly the k-th smallest member prediction, no binaries needed.
WMICL = dict(sizes={"wmicl_p5": 5}, frac=0.5, restarts=12)
# NOT CQR -- read this before citing it.
#
# Published CQR (Romano, Patterson & Candes, NeurIPS 2019) adds the conformal
# correction ADDITIVELY to the fitted quantile function:
#     C(x) = [q_lo(x) - Q_{1-a}(E),  q_hi(x) + Q_{1-a}(E)],  E_i = Y_i - q_a(X_i)
# The adaptivity lives in the quantile regressor, not in a scaled correction.
# With FIT_TAU = 0.90 and ALPHA = 0.10, our `static` arm -- h_hat + q_hat <= 0,
# h_hat a tau=0.90 pinball fit and q_hat the conformal quantile of h - h_hat --
# IS one-sided split CQR exactly. We have been running CQR all along under
# another name; see analyze.py, which now says so.
#
# THIS arm is the locally-weighted / normalised-nonconformity family
# (Papadopoulos et al.; Lei et al. 2018), which is what C-MICL also belongs to.
# It differs from C-MICL only in the choice of scale function: quantile spread
# h_hat_{0.90} - h_hat_{0.50} instead of an MLP fit to |h - h_hat|. Published
# family, our choice of scale -- label it that way, never as "CQR".
CQR = dict(tau_lo=0.50, eps_frac=0.01)
# OptiCL (Maragno et al. 2021) convex-hull trust region.
#
# Their SS2.3 defines the trust region as CH(X) = {x | sum_i lambda_i x_i = x,
# sum lambda_i = 1, lambda >= 0} -- eq (2). Facet form is equivalent for a single
# CH, and scipy returns UNIT-NORMALISED facet normals, so max_j(A_j x + b_j) is
# exactly the signed Euclidean distance to the hull.
#
# Their SS3.2 then gives the trust region its own published dial: the eps-CH,
# eq (5), which ENLARGES the region so points within distance eps of the hull are
# admitted. eps = 0 is plain CH; eps > 0 relaxes. Our dial is that same signed
# distance with the sign flipped, so t = 0 IS OptiCL's published CH and t > 0 is
# the inward offset -- the natural continuation of their own geometry in the
# direction that buys safety. An earlier version shrank the hull toward its
# centroid, which is NOT a parameter OptiCL defines; it has been removed.
#
# Distances are taken in BOX-NORMALISED coordinates (x - LO)/(HI - LO). OptiCL
# uses ||s||_p in raw feature space, but our coordinates differ by three orders
# of magnitude (v0 ~ 450-1500 against dt ~ 0.5-2), so a raw-space ball is
# dominated by one variable. The convex hull is affine-equivariant, so the SET at
# t = 0 is identical either way; only the offset metric changes. Declared choice.
HULLC = dict(hi0=0.02)
ENVC = dict(n_pieces_grid=[2, 4, 8], l2=1e-3, floor_weight=0.5, coverage=0.90,
            smooth_frac=0.05, restarts=4, max_iter=500,
            gen_strength=1.0, fp_share=0.25, pool=40000, keep=2000)
BOUNDS = BM.bounds
LO = np.array([b[0] for b in BOUNDS]); HI = np.array([b[1] for b in BOUNDS])
# The harness hardcoded dimension 5 in six places, which silently restricted it
# to methane and williams. DIM generalises it; for every dim-5 benchmark the
# generated numbers are byte-identical, since len(BOUNDS) is 5 there.
DIM = len(BOUNDS)
G: dict = {}
DRY = False        # set True by _dryrun.py; redirects output to runs_dry/


# ------------------------------------------------------------------ helpers
def q_official(s, a):
    s = np.asarray(s, float); n = s.size
    lvl = math.ceil((n + 1) * (1 - a)) / n
    return float("inf") if lvl > 1 else float(np.quantile(s, lvl, method="higher"))


def _hb_methane(X, P):
    a = (X[:,0]-450.)/1050.; b = (X[:,1]-450.)/1050.; c = (X[:,2]-997.18)/350.94
    d = (X[:,3]-.5)/1.5; e = np.maximum((X[:,4]-10.)/90., 0.)
    return (P[0]*a+P[1]*b+P[2]*c+P[3]*d+P[4]*np.sqrt(e+1e-8)+P[5]*e
            + P[6]*a*b + P[7]*c*(a+b) + P[8]*d*e + P[9]-2.5)


def _hb_williams(X, P):
    """The symbolic surrogate the search returned (score -0.03305, R2 0.7591,
    pinball 0.0253).

    Two guards, both the same class of defect as methane's sqrt: SLSQP's line
    search steps marginally outside the box, and there
      (a) 1 + P[8]*x5_n can cross zero -> inf/NaN from the division, and
      (b) exp(-P[10]*x5_n) can overflow,
    either of which silently corrupts every solve. The denominator guard is
    sign-preserving so it introduces no discontinuity; inside the box both are
    inert.
    """
    x1_n = (X[:,0]-0.03)/0.07; x2_n = (X[:,1]-5.8)/1.0; x3_n = (X[:,2]-1.0)/4.0
    x4_n = (X[:,3]-1.0)/4.0;   x5_n = (X[:,4]-0.1)/0.8
    den = 1.0 + P[8]*x5_n
    den = np.where(den >= 0, 1.0, -1.0) * np.maximum(np.abs(den), 1e-6)
    ex  = np.exp(-np.clip(P[10]*x5_n, -50.0, 50.0))
    return (P[0]*x1_n + P[1]*x2_n + P[2]*x2_n**2 + P[3]*x3_n + P[4]*x2_n*x3_n
            + P[5]*x4_n + P[6]*x4_n**2 + P[7]*x4_n/den
            + P[9]*(1.0 - ex) + P[11])


def _hb_hanoi(X, P):
    """Physics-motivated STAND-IN for the Hanoi water network -- NOT LLM-SR.

    Hazen-Williams head loss along a pipe goes as Q^1.852 / D^4.87. In the
    harness coordinate x = D^1.5 that is x^-3.247, so the basis is the eight
    group reciprocal-powers plus aggregate terms. Symbolic regression finds
    power laws readily, so this is the honest stand-in for a discovered form
    rather than a deliberately weak one.

    OLS: R2 0.9988 fitted / 0.9985 held-out -- but that number is dominated by
    the tail (h ranges to +7700 m when pipes are far too small). What the method
    actually operates on is the boundary, and there the SAME fit has residual
    sd 14.5 m and p90 |resid| 27.1 m against a 30 m head requirement. Global R2
    and boundary error point in opposite directions here; the boundary number is
    the one that matters.
    """
    u = (np.asarray(X, float) / 100.0) ** (-3.247)
    return (u @ np.asarray(P[:8], float)
            + P[8] * np.sum(u**2, axis=1)
            + P[9] * np.sum(u, axis=1)**2
            + P[10] * np.max(u, axis=1) + P[11])


_HB = {"methane": _hb_methane, "williams": _hb_williams,
       "hanoi": _hb_hanoi}[BENCH]
P0_FILE = {"methane": "p0.json", "williams": "p0_williams.json",
           "hanoi": "p0_hanoi.json"}[BENCH]


def hb(X, P=None):
    P = G["P"] if P is None else P
    return _HB(np.atleast_2d(np.asarray(X, float)), P)


def _hs_methane(x, P):
    x1, x2, x3, x4, x5 = x
    a=(x1-450.)/1050.; b=(x2-450.)/1050.; c=(x3-997.18)/350.94
    d=(x4-.5)/1.5; e=(x5-10.)/90.
    if e < 0.: e = 0.
    return (P[0]*a+P[1]*b+P[2]*c+P[3]*d+P[4]*math.sqrt(e+1e-8)+P[5]*e
            + P[6]*a*b + P[7]*c*(a+b) + P[8]*d*e + P[9]-2.5)


def _hs_williams(x, P):
    """Scalar twin of _hb_williams. Must stay numerically identical to it --
    the batch form fits the surrogate, the scalar form is what the solver sees,
    and a divergence between them would be invisible and corrupting."""
    x1, x2, x3, x4, x5 = x
    x1_n=(x1-0.03)/0.07; x2_n=(x2-5.8)/1.0; x3_n=(x3-1.0)/4.0
    x4_n=(x4-1.0)/4.0;   x5_n=(x5-0.1)/0.8
    den = 1.0 + P[8]*x5_n
    den = (1.0 if den >= 0 else -1.0) * max(abs(den), 1e-6)
    ex  = math.exp(-max(-50.0, min(50.0, P[10]*x5_n)))
    return (P[0]*x1_n + P[1]*x2_n + P[2]*x2_n**2 + P[3]*x3_n + P[4]*x2_n*x3_n
            + P[5]*x4_n + P[6]*x4_n**2 + P[7]*x4_n/den
            + P[9]*(1.0 - ex) + P[11])


def _hs_hanoi(x, P):
    """Scalar twin of _hb_hanoi. Must stay numerically identical to it."""
    tot = P[11]; sq = 0.0; su = 0.0; mx = -1e300
    for i in range(8):
        u = (x[i] / 100.0) ** (-3.247)
        tot += P[i] * u
        sq += u * u; su += u
        if u > mx: mx = u
    return tot + P[8] * sq + P[9] * su * su + P[10] * mx


_HS = {"methane": _hs_methane, "williams": _hs_williams,
       "hanoi": _hs_hanoi}[BENCH]


def hs(x): return _HS(x, G["P"])


def mlp(X, k):
    m = G[k]
    z = (np.atleast_2d(np.asarray(X, float)) - m["mean"]) / m["scale"]
    z = np.maximum(z @ m["W"][0] + m["B"][0], 0.)
    z = np.maximum(z @ m["W"][1] + m["B"][1], 0.)
    return np.maximum((z @ m["W"][2] + m["B"][2]).reshape(-1), CMICL["eps"])


if hasattr(BM, "operability_constraints"):
    def oper(x): return max(BM.operability_constraints(np.asarray(x, float)))
else:
    # Williams-Otto declares no operability constraints; the box bounds are the
    # whole feasible domain and the solver already enforces them. Return a
    # value that can never bind in max(h_hat + c, oper).
    def oper(x): return -1e9
def orange(w): return float(np.sum(np.abs(w) * (HI - LO)))
def rank01(v):
    v = np.asarray(v, float).reshape(-1)
    return np.argsort(np.argsort(v)) / max(len(v) - 1, 1)
def unit(W): return W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)


def laws_weights(law, n, seed):
    rng = np.random.default_rng(seed)
    if law == "random_linear": W = rng.standard_normal((n, DIM))
    elif law == "corner_pressure":
        W = rng.choice([-1., 1.], size=(n, DIM)) + rng.standard_normal((n, DIM)) * .1
    elif law == "boundary_pressure":
        r = rng.standard_normal((n, DIM)); W = np.sign(r) * np.abs(r) ** 3
    else:
        Xc, res = G["X_cal"], G["resid_cal"]
        rr, br = rank01(res), rank01(-np.abs(hb(Xc) + G["q_hat"]))
        if law == "residual_hotspot_pressure":
            npool = max(1, min(len(res), max(10, int(math.ceil(.10*len(res))))))
            cand = np.argsort(res)[-npool:]
            w = np.maximum(res[cand]-res[cand].min(), 0.)+1e-6
        else:
            cand = np.flatnonzero((rr >= .6) & (br >= .6))
            if cand.size < max(3, min(10, len(res))):
                npool = max(1, min(len(res), max(10, int(math.ceil(.15*len(res))))))
                cand = np.argsort(rr+br)[-npool:]
            sc = rr[cand]+br[cand]; w = np.maximum(sc-sc.min(), 0.)+1e-6
        T = Xc[rng.choice(cand, size=n, replace=True, p=w/w.sum())]
        span = np.maximum(HI-LO, 1e-12); dirs = (T-LO)/span - .5
        bad = np.linalg.norm(dirs, axis=1) <= 1e-12
        if bad.any(): dirs[bad] = rng.standard_normal((int(bad.sum()), DIM))
        ns = .20 if law == "residual_hotspot_pressure" else .12
        W = unit(unit(-dirs + ns*rng.standard_normal(dirs.shape)) / span[None, :])
    return unit(W)


# ----------------------------------------------------- baseline arm assets
def _fit_pinball(X, h, tau, p0, rng, restarts):
    """Pinball fit with the SAME protocol as the primary surrogate in
    build_context: p0 plus (restarts-1) perturbed starts, L-BFGS-B, keep the
    lowest finite loss.

    This deliberately DUPLICATES build_context's fit loop rather than factoring
    it out. Refactoring the primary path risks perturbing a recorded result, and
    every existing seed must stay byte-reproducible.
    """
    def loss(pp):
        try: pr = hb(X, pp)
        except Exception: return 1e12
        if not np.all(np.isfinite(pr)): return 1e12
        r = h - pr
        return float(np.mean(np.maximum(tau*r, (tau-1)*r)))
    best, nconv = None, 0
    for s in [p0] + [p0*(1+.25*rng.standard_normal(p0.size)) for _ in range(restarts-1)]:
        rr = minimize(loss, s, method="L-BFGS-B",
                      options={"maxiter": 2000, "maxfun": 20000, "ftol": 1e-12})
        if np.isfinite(rr.fun):
            nconv += int(rr.success)
            if best is None or rr.fun < best[0]: best = (float(rr.fun), rr.x)
    if best is None:
        raise RuntimeError("pinball fit produced no finite loss on any restart")
    return np.asarray(best[1], float), best[0], nconv


def prepare_arm_assets(seed, arms, log):
    """Fit whatever the baseline arms need, and ONLY what the selected arms
    need, so a run without them costs exactly what it costs today.

    Every random draw here comes from its own generator seeded off `seed` with a
    distinct offset. Nothing touches the generators used by build_context, so
    adding these arms cannot move any existing number.
    """
    p0 = np.array(json.load(open(HERE / P0_FILE)), float)

    for arm, P in WMICL["sizes"].items():
        if arm not in arms: continue
        Xf, hf = G["X_fit"], G["h_fit"]
        m = int(WMICL["frac"] * len(Xf))          # their bootstrap_size
        rb = np.random.default_rng(seed + 9001)   # independent stream
        members, losses, convs = [], [], []
        for _ in range(P):
            idx = rb.integers(0, len(Xf), size=m)
            pm, lm, nc = _fit_pinball(Xf[idx], hf[idx], FIT_TAU, p0, rb,
                                      WMICL["restarts"])
            members.append([float(v) for v in pm]); losses.append(lm); convs.append(nc)
        k = int(math.ceil((1.0 - ALPHA) * P))
        G.setdefault("wmicl", {})[arm] = members
        G.setdefault("wmicl_k", {})[arm] = k
        # Published operating point diagnostic: risk of the ensemble constraint
        # with no pad, on the sanity split.
        log(f"  {arm}: {P} bootstrap members (n={m} of {len(Xf)}, with replacement), "
            f"k={k}=ceil((1-alpha)P) | member pinball {min(losses):.4f}..{max(losses):.4f} "
            f"(primary {G['pinball_primary']:.4f}) | restarts converged "
            f"{min(convs)}-{max(convs)} of {WMICL['restarts']} per member "
            f"(an unconverged member would silently weaken this baseline)")

    if "cqr_scaled" in arms:
        rq = np.random.default_rng(seed + 9101)   # independent stream
        P_lo, l_lo, nc_lo = _fit_pinball(G["X_fit"], G["h_fit"], CQR["tau_lo"],
                                         p0, rq, 12)
        G["P_cqr_lo"] = [float(v) for v in P_lo]
        hi_c = hb(G["X_cal"]); lo_c = hb(G["X_cal"], P_lo)
        gap = hi_c - lo_c
        # Floor the scale at a fixed fraction of its own median. Without it the
        # quantile fits can cross, the scale collapses to ~0, and the constraint
        # degenerates to h_hat <= 0 exactly there -- a zero-margin pocket the
        # optimizer will drive straight into, a failure mode measured earlier.
        # C-MICL
        # carries the same protection as its eps = 0.01 on u_hat, whose measured
        # ratio to median(u_hat) is ~0.006; 0.01 here is the same order and makes
        # this competitor STRONGER, never weaker.
        floor = CQR["eps_frac"] * float(np.median(np.maximum(gap, 0.0)))
        G["cqr_floor"] = floor
        s_cal = np.maximum(gap, floor)
        G["q_cqr"] = q_official(G["resid_cal"] / s_cal, ALPHA)
        # CQR degenerates toward a constant if the two quantile fits cross.
        # Measure it and report whatever it is.
        cross = float(np.mean(hb(G["X_env"]) - hb(G["X_env"], P_lo) <= floor))
        log(f"  cqr_scaled: tau={CQR['tau_lo']} fit pinball={l_lo:.4f} | q_cqr={G['q_cqr']:.4f} "
            f"| median scale={float(np.median(s_cal)):.4f} floor={floor:.5f} "
            f"| quantile-crossing fraction={cross:.4f} "
            f"(high means the local scale is pinned at eps and CQR collapses "
            f"toward a constant pad)")

    if "trust_region_hull" in arms:
        from scipy.spatial import ConvexHull
        Hu = ConvexHull((G["X_fit"] - LO) / (HI - LO))      # box-normalised
        G["hull_A"] = np.ascontiguousarray(Hu.equations[:, :-1])
        G["hull_b"] = np.ascontiguousarray(Hu.equations[:, -1])
        rh = np.random.default_rng(seed + 9201)   # independent stream
        Yn = rh.uniform(0.0, 1.0, size=(20000, DIM))
        d = (Yn @ G["hull_A"].T + G["hull_b"]).max(1)       # signed distance
        log(f"  trust_region_hull: {len(G['hull_b'])} facets on {len(G['X_fit'])} "
            f"fit points (box-normalised) | hull covers {float(np.mean(d <= 0)):.4f} "
            f"of the box, median inward depth {-float(np.median(d[d <= 0])):.4f} "
            f"| t=0 is OptiCL's published CH; t>0 is the inward offset of their "
            f"eps-CH geometry")


# --------------------------------------------------------------------- arms
def constraint(arm, knob):
    q = G["q_hat"]
    if arm == "static":  return lambda x: max(hs(x) + q, oper(x))
    if arm == "scalar":  return lambda x: max(hs(x) + knob, oper(x))
    if arm == "alpha_tightening": return lambda x: max(hs(x) + knob, oper(x))
    if arm == "cmicl":
        qn = G["q_norm"]
        return lambda x: max(hs(x) + knob*qn*float(mlp(np.atleast_2d(x), "u")[0]), oper(x))
    if arm == "cmicl_refit":
        qn = G["q_norm_refit"]
        return lambda x: max(hs(x) + knob*qn*float(mlp(np.atleast_2d(x), "u2")[0]), oper(x))
    if arm == "trust_region_box":
        lo = np.quantile(G["X_fit"], knob, axis=0); hi = np.quantile(G["X_fit"], 1-knob, axis=0)
        def f(x):
            p = np.asarray(x, float)
            return max(hs(p)+q, oper(p), float(max((lo-p).max(), (p-hi).max())))
        return f
    if arm in WMICL["sizes"]:
        # W-MICL: >= k of the P ensemble members must be feasible, i.e. the k-th
        # SMALLEST member prediction is <= 0. `knob` is the swept additive pad;
        # knob = 0 is the published operating point.
        Pm = G["wmicl"][arm]; k = G["wmicl_k"][arm]
        def f(x):
            return max(sorted(_HS(x, pp) for pp in Pm)[k-1] + knob, oper(x))
        return f
    if arm == "cqr_scaled":
        # Locally-weighted conformal with the scale taken from quantile spread.
        # NOT published CQR -- see the CQR dict above.
        Plo = G["P_cqr_lo"]; qc = G["q_cqr"]; ep = G["cqr_floor"]
        def f(x):
            hi = hs(x)
            return max(hi + knob*qc*max(hi - _HS(x, Plo), ep), oper(x))
        return f
    if arm == "trust_region_hull":
        # OptiCL's convex-hull trust region, dialled by the signed distance to
        # the hull in box-normalised coordinates. knob = 0 is exactly CH(X_fit),
        # the published form; knob > 0 requires the decision to sit at least
        # `knob` inside it. This is their eps-CH parameter with the sign flipped.
        A, b = G["hull_A"], G["hull_b"]
        def f(x):
            pt = np.asarray(x, float)
            xn = (pt - LO) / (HI - LO)
            return max(hs(pt) + q, oper(pt), float(np.max(A @ xn + b)) + knob)
        return f
    if arm == "law_blind_envelope":
        c = DeployedCorrection(G["env"], G["floor"], ENVC["floor_weight"], knob)
        return lambda x: max(hs(x)+q+float(c.batch(np.atleast_2d(x))[0]), oper(x))
    raise ValueError(arm)


_J: dict = {}
def _init(j): _J.update(j)
def _one(i):
    w = _J["W"][i]
    r = solve_ocl_multistart(lambda x: float(np.dot(np.asarray(x).ravel(), w)),
                             _J["con"], BOUNDS, n_multistart=_J["st"],
                             seed=int(_J["sd"][i]), max_iter=200)
    x = np.asarray(r["x_star"], float)
    return dict(i=i, dep=bool(r["feasible"]), val=float(np.dot(x, w)), x=x.tolist())


# Match the pool to the machine. Hardcoding 8 oversubscribed a 4 vCPU machine
# 2:1; it also under-used anything larger.
N_PROC = int(__import__('os').environ.get('N_PROC',
                                          __import__('os').cpu_count() or 8))


def solve_all(arm, knob, W, sd, st, procs=None):
    procs = N_PROC if procs is None else procs
    j = dict(W=W, sd=sd, st=st, con=constraint(arm, knob))
    with mp.get_context("fork").Pool(procs, initializer=_init, initargs=(j,)) as p:
        out = p.map(_one, range(len(W)), chunksize=2)
    return sorted(out, key=lambda r: r["i"])


def cost_of(rows, base, W):
    c = np.full(len(W), np.nan)
    for k, (r, b, w) in enumerate(zip(rows, base, W)):
        if r["dep"] and b["dep"]: c[k] = (r["val"] - b["val"]) / orange(w)
    return c


# ------------------------------------------------------------------ context
def build_context(seed, log):
    rng = np.random.default_rng(seed)
    n = N_FIT + N_CAL + N_ENV + N_SANE
    # The oracle is not total: Williams-Otto's simulation fails to converge on a
    # small corner (~0.5% of the box, low x4 / high x2) and returns NaN. A NaN
    # in h makes the pinball loss NaN and every restart fail -- which is exactly
    # the "success: false on all 12 fits" signature. Draw until we
    # have n points where the oracle is actually defined, and record the rate.
    Xs, hs_, tried, bad = [], [], 0, 0
    while sum(len(a) for a in Xs) < n and tried < 40:
        m = int((n - sum(len(a) for a in Xs)) * 1.25) + 32
        Xc = rng.uniform(LO, HI, size=(m, DIM))
        hc = np.array([BM.true_h(x) for x in Xc])
        ok = np.isfinite(hc); bad += int((~ok).sum()); tried += 1
        Xs.append(Xc[ok]); hs_.append(hc[ok])
    X = np.vstack(Xs)[:n]; h = np.concatenate(hs_)[:n]
    if len(X) < n:
        raise SystemExit(f"oracle undefined too often: only {len(X)}/{n} usable")
    log(f"  oracle: {bad} undefined draw(s) rejected "
        f"({bad/max(bad+n,1):.2%} of draws); {n} usable points")
    i = 0
    def take(k):
        nonlocal i
        s = slice(i, i+k); i += k; return X[s], h[s]
    Xf, hf = take(N_FIT); Xc, hc = take(N_CAL); Xe, he = take(N_ENV); Xs, hsn = take(N_SANE)
    p0 = np.array(json.load(open(HERE / P0_FILE)), float)
    def loss(p):
        try: pr = hb(Xf, p)
        except Exception: return 1e12
        if not np.all(np.isfinite(pr)): return 1e12
        r = hf - pr
        return float(np.mean(np.maximum(FIT_TAU*r, (FIT_TAU-1)*r)))
    best, nconv = None, 0
    r0 = np.random.default_rng(seed)
    for s in [p0] + [p0*(1+.25*r0.standard_normal(p0.size)) for _ in range(11)]:
        rr = minimize(loss, s, method="L-BFGS-B",
                      options={"maxiter": 2000, "maxfun": 20000, "ftol": 1e-12})
        if np.isfinite(rr.fun):
            nconv += int(rr.success)
            if best is None or rr.fun < best[0]: best = (float(rr.fun), rr.x)
    if best is None:
        raise RuntimeError("surrogate fit produced no finite loss on any restart")
    G["P"] = np.asarray(best[1], float)
    G["pinball_primary"] = float(best[0])   # for the W-MICL member comparison
    pf = hb(Xf); r2 = 1 - float(np.sum((hf-pf)**2))/float(np.sum((hf-hf.mean())**2))
    res = hc - hb(Xc)
    G.update(X_fit=Xf, h_fit=hf, X_cal=Xc, h_cal=hc, X_env=Xe, h_env=he,
             resid_cal=res, q_hat=q_official(res, ALPHA))
    sc = StandardScaler().fit(Xf)
    m = MLPRegressor(hidden_layer_sizes=CMICL["hidden"], activation="relu", solver="adam",
                     alpha=CMICL["wd"], max_iter=CMICL["max_iter"], tol=CMICL["tol"],
                     random_state=seed).fit(sc.transform(Xf), np.abs(hf-pf))
    G["u"] = dict(mean=sc.mean_, scale=sc.scale_, W=m.coefs_, B=m.intercepts_)
    ucal = mlp(Xc, "u")
    G["q_norm"] = q_official(np.abs(res)/ucal, ALPHA)
    cov_s = float(np.mean((hsn - hb(Xs)) <= G["q_hat"]))
    cov_c = float(np.mean((hsn - hb(Xs)) <= G["q_norm"]*mlp(Xs, "u")))
    log(f"  fit: pinball={best[0]:.4f} R2={r2:.4f} converged={nconv}/12 | "
        f"q_hat={G['q_hat']:.4f} q_norm={G['q_norm']:.4f} med_u={float(np.median(ucal)):.4f}")
    log(f"  SANITY (n={N_SANE}): scalar={cov_s:.4f} cmicl={cov_c:.4f} target=0.90")
    # A1 localization: how concentrated is the DANGEROUS error (false-safe:
    # truly infeasible but predicted safe under the deployed static constraint)?
    # The guidelines' advantage scales with 1/mass_hot; if it is diffuse, a
    # shaped correction cannot beat a constant and NULL is the true answer.
    pred_safe = (hb(Xs) + G["q_hat"]) <= 0
    false_safe = pred_safe & (hsn > 0)
    mass_hot = float(false_safe.mean())
    frac_of_safe = float(false_safe.sum() / max(pred_safe.sum(), 1))
    exc = (hsn - hb(Xs) - G["q_hat"])[false_safe]
    log(f"  A1 uniform-measure: mass_hot={mass_hot:.4f} "
        f"share_of_predicted_safe={frac_of_safe:.4f} "
        f"(near-zero is EXPECTED: pinball tau=0.90 biases h_hat conservative, so the "
        f"predicted-safe set sits inside the feasible set almost everywhere. The "
        f"optimizer does not sample uniformly -- it drives to the deployed boundary, "
        f"a Lebesgue-null set. See the boundary diagnostic in [2].)")
    return dict(r2=r2, pinball=best[0], n_converged=nconv, q_hat=G["q_hat"],
                q_norm=G["q_norm"], median_u=float(np.median(ucal)),
                cov_scalar=cov_s, cov_cmicl=cov_c,
                mass_hot=mass_hot, false_safe_share_of_predicted_safe=frac_of_safe,
                feasible_fraction=float(np.mean(h <= 0)))


def build_method(seed, Wpr, spr, log):
    e0 = G["h_env"] - hb(G["X_env"]) - G["q_hat"]
    sm = max(ENVC["smooth_frac"]*float(np.std(e0)), 1e-9)
    G["floor"] = max(float(fit_constant_upper_level(e0, coverage_level=ENVC["coverage"])[0]), 0.)
    # n_pieces selected on D_prop
    picks = {}
    for npc in ENVC["n_pieces_grid"]:
        env, _ = fit_smooth_max_affine_envelope(
            G["X_env"], e0, bounds=BOUNDS, n_pieces=npc, l2=ENVC["l2"], smoothing=sm,
            coverage_level=ENVC["coverage"], seed=seed, n_restarts=ENVC["restarts"],
            max_iter=ENVC["max_iter"])
        G["env"] = env
        rows = solve_all("law_blind_envelope", ENVC["gen_strength"], Wpr, spr, PRIMARY_STARTS)
        Xs = np.array([r["x"] for r in rows]); dep = np.array([r["dep"] for r in rows])
        ht = np.array([BM.true_h(x) for x in Xs])
        dep = dep & np.isfinite(ht)      # unscoreable, not safe
        bad = (ht > 0) & dep
        risk = float(bad.sum() / max(dep.sum(), 1))
        sev = float(ht[bad].mean()) if bad.any() else 0.0
        picks[npc] = (risk, env, Xs, ht, dep, sev)
        log(f"  n_pieces={npc}: D_prop selected risk={risk:.4f} mean_severity={sev:.4f}")
    # Binary risk ties often on 100 objectives; mean positive severity is a finer
    # signal on the SAME legitimate split. Parsimony breaks a full tie.
    npc = min(picks, key=lambda k: (picks[k][0], picks[k][5], k))
    risk0, env0, Xs_all, ht_all, dep, sev0 = picks[npc]
    # Only DEPLOYED decisions are real selected points; a failed solve returns an
    # arbitrary incumbent whose residual excess would poison the refit.
    Xs, ht = Xs_all[dep], ht_all[dep]
    G["env"] = env0
    log(f"  selected n_pieces={npc} (D_prop risk {risk0:.4f}); floor={G['floor']:.4f}; "
        f"deployed {int(dep.sum())}/{len(dep)} of D_prop")

    rng = np.random.default_rng(seed + 77)
    pool = rng.uniform(LO, HI, size=(ENVC["pool"], DIM))
    c0 = DeployedCorrection(env0, G["floor"], ENVC["floor_weight"], ENVC["gen_strength"])
    g = hb(pool) + G["q_hat"] + c0.batch(pool)
    Xb = pool[np.argsort(np.abs(g))[:ENVC["keep"]]]
    hbd = np.array([BM.true_h(x) for x in Xb])
    okb = np.isfinite(hbd)
    if not okb.all():
        log(f"  boundary pool: dropped {int((~okb).sum())}/{len(Xb)} points where "
            f"the oracle is undefined")
        Xb, hbd = Xb[okb], hbd[okb]

    # --- BOUNDARY localization (the quantity that decides shape-vs-constant).
    # Along the deployed boundary, how much does the danger VARY? A shaped
    # correction can only beat a constant if the required margin is
    # heterogeneous here. Uniform-measure mass_hot is blind to this (T2: the
    # selected law concentrates on a null set).
    exb = hbd - hb(Xb) - G["q_hat"]          # residual excess on the boundary
    pos = exb[exb > 0]
    b_unsafe = float(np.mean(hbd > 0))
    cv = float(pos.std() / pos.mean()) if pos.size > 1 and pos.mean() > 0 else 0.0
    p50 = float(np.percentile(pos, 50)) if pos.size else 0.0
    p90 = float(np.percentile(pos, 90)) if pos.size else 0.0
    # --- C-MICL SHAPE QUALITY (decides shape-vs-C-MICL, which mass_hot does not).
    # The separation condition is about shape vs SCALAR only. Beating C-MICL is
    # a different question: its u_hat is fit to |h - h_hat| by MSE on the TRAINING
    # distribution, so it wins only if u_hat actually TRACKS the required margin
    # at the points the optimizer selects. Measured directly on the boundary at
    # ZERO extra oracle cost -- Xb and its true h are already paid for.
    #   rho_cmicl  ~0 or negative => u_hat misallocates => C-MICL is beatable
    #   waste      share of u_hat's budget landing where there is no excess
    ub = mlp(Xb, "u")
    rho_c = (float(np.corrcoef(rank01(ub), rank01(exb))[0, 1])
             if len(Xb) > 3 else float("nan"))
    waste = float(ub[exb <= 0].sum() / max(float(ub.sum()), 1e-12))
    log(f"  C-MICL shape quality on the boundary: rho(u_hat, excess)={rho_c:+.3f} "
        f"| budget landing where excess<=0: {waste:.3f} "
        f"-> rho near 0 or negative means C-MICL's shape is misallocated (beatable); "
        f"rho high means its u_hat already tracks the danger")
    log(f"  BOUNDARY localization (n={len(Xb)} pts on h_hat+q_hat=0): "
        f"truly-infeasible={b_unsafe:.4f} | positive-excess CV={cv:.3f} "
        f"p50={p50:.4f} p90={p90:.4f} ratio_p90/p50={(p90/p50 if p50>0 else float('nan')):.2f}")
    log(f"    -> CV>~0.5 or p90/p50>~2 means the required margin is heterogeneous "
        f"along the boundary, i.e. shape CAN beat a constant; near-flat means it cannot.")

    Xall = np.vstack([G["X_env"], Xs, Xb])
    hall = np.concatenate([G["h_env"], ht, hbd])
    yall = hall - hb(Xall) - G["q_hat"]
    isfp = np.zeros(len(Xall), bool)
    isfp[len(G["X_env"]):len(G["X_env"])+len(Xs)] = ht > 0
    w = np.ones(len(Xall))
    if isfp.any():
        w[isfp] = ENVC["fp_share"]*int((~isfp).sum())/(int(isfp.sum())*(1-ENVC["fp_share"]))
    env1, _ = fit_smooth_max_affine_envelope(
        Xall, yall, bounds=BOUNDS, n_pieces=npc, l2=ENVC["l2"],
        smoothing=max(ENVC["smooth_frac"]*float(np.std(yall)), 1e-9),
        coverage_level=ENVC["coverage"], weights=w, seed=seed,
        n_restarts=ENVC["restarts"], max_iter=ENVC["max_iter"])
    G["env"] = env1
    log(f"  round1: {len(Xall)} pts (env {len(G['X_env'])} + sel {len(Xs)} + bnd {len(Xb)}), "
        f"{int(isfp.sum())} false positives up-weighted x{w[isfp][0] if isfp.any() else 1:.1f}")

    Xex = np.vstack([Xs, Xb]); hex_ = np.concatenate([ht, hbd])
    Xr = np.vstack([G["X_fit"], Xex]); hr = np.concatenate([G["h_fit"], hex_])
    sc = StandardScaler().fit(Xr)
    m = MLPRegressor(hidden_layer_sizes=CMICL["hidden"], activation="relu", solver="adam",
                     alpha=CMICL["wd"], max_iter=CMICL["max_iter"], tol=CMICL["tol"],
                     random_state=seed).fit(sc.transform(Xr), np.abs(hr - hb(Xr)))
    G["u2"] = dict(mean=sc.mean_, scale=sc.scale_, W=m.coefs_, B=m.intercepts_)
    G["q_norm_refit"] = q_official(np.abs(G["resid_cal"])/mlp(G["X_cal"], "u2"), ALPHA)

    log(f"  cmicl_refit: +{len(Xex)} same oracle pts | q_norm'={G['q_norm_refit']:.4f} "
        f"(was {G['q_norm']:.4f})")
    return dict(boundary_unsafe_frac=b_unsafe, rho_cmicl=rho_c, cmicl_waste=waste,
                boundary_excess_cv=cv,
                boundary_excess_p50=p50, boundary_excess_p90=p90,
                n_pieces=npc, dprop_risk_by_pieces={str(k): [picks[k][0], picks[k][5]] for k in picks},
                floor=G["floor"], n_selected=int(len(Xs)),
                n_dprop_not_deployed=int((~dep).sum()), n_fp=int((ht > 0).sum()),
                n_boundary=len(Xb), q_norm_refit=G["q_norm_refit"],
                frac_resid_pos=float(np.mean(G["resid_cal"] > 0)),
                oracle_queries=dict(dprop=len(Xs), boundary=len(Xb)))


# ------------------------------------------------------------------- sweep
KNOB_HI0 = dict(scalar=None, cmicl=1.0, cmicl_refit=1.0,
                law_blind_envelope=1.0, trust_region_box=0.45, alpha_tightening=1.0,
                wmicl_p5=None,                   # additive pad, like scalar
                cqr_scaled=1.0, trust_region_hull=HULLC["hi0"])


def knob_of(arm, t):
    """Map a scalar dial t>=0 to the arm's native knob (monotone increasing)."""
    # pure_scalar is c(x)=s swept from 0, NOT q_hat+s.
    # Mapping it to q_hat+t truncated
    # its frontier at the low end, exactly where the comparison is decided.
    if arm == "scalar":            return t
    if arm in ("cmicl", "cmicl_refit", "law_blind_envelope"): return t
    if arm == "trust_region_box":  return min(t, 0.49)
    if arm in WMICL["sizes"]:      return t           # additive pad, like scalar
    if arm == "cqr_scaled":        return t           # multiplier on q_cqr
    if arm == "trust_region_hull": return t   # inward offset, unbounded above
    if arm == "alpha_tightening":
        s = np.sort(G["resid_cal"]); n = s.size
        a = ALPHA * (1 - 0.995 * min(t, 1.0))
        return float(np.quantile(s, min(math.ceil((n+1)*(1-a))/n, 1.), method="higher"))
    raise ValueError(arm)


CAL_OBJ, CAL_STARTS = 40, 20      # grid placement only; oracle-free, not a claim
# Dial doublings allowed while growing an arm's grid toward 1.5*C*.
# A normalized arm can saturate when its scale sits at the floor over most of the box, so
# scaling the dial mostly buys margin only where u3 is already large): it hit
# the old limit of 8 at cost 0.057 vs a 0.064 target. A larger limit can only
# extend an arm's reachable cost range, i.e. make competitors stronger.
GRID_DOUBLINGS = 12
def calibrate_grid(arm, target, Wm, sm, base_m, log, readout_costs=()):
    """Bracket the dial range on D_match. ZERO oracle.

    A dial is USABLE only if the deployed constraint still admits a solution on
    most objectives. A non-finite mean cost means the arm went INFEASIBLE, i.e.
    the dial is too HIGH -- the previous version treated that as 'cost still
    below target' and doubled into 0..512, silently destroying the scalar and
    frontiers. We now expand only through usable dials and bisect
    back to the feasibility ceiling.
    """
    # Stratified across laws: D_match is ordered 20-per-law, so a head slice
    # would calibrate on random_linear + corner_pressure only.
    idx = np.unique(np.linspace(0, len(Wm) - 1, CAL_OBJ).astype(int))
    W, sd, base = Wm[idx], sm[idx], [base_m[i] for i in idx]

    def probe(t):
        rows = solve_all(arm, knob_of(arm, t), W, sd, CAL_STARTS)
        c = cost_of(rows, base, W)
        rate = float(np.mean([r["dep"] for r in rows]))
        return (float(np.nanmean(c)) if np.isfinite(c).any() else float("nan"), rate)

    usable = lambda cost, rate: np.isfinite(cost) and rate >= 0.5
    DIAL_CAP = dict(trust_region_box=0.49, alpha_tightening=1.0)   # knob saturates
    cap = DIAL_CAP.get(arm, float("inf"))
    hi = min(KNOB_HI0[arm] if KNOB_HI0[arm] else 1.0, cap)
    c, r = probe(hi)
    if not usable(c, r):                       # too high already -> bisect down
        lo_t, hi_t = 0.0, hi
        for _ in range(8):
            mid = .5 * (lo_t + hi_t); cm, rm = probe(mid)
            if usable(cm, rm): lo_t, c, r = mid, cm, rm
            else: hi_t = mid
        hi = lo_t
        status = "ceiling-infeasible"
    else:                                      # usable -> grow toward target
        tries = 0
        while c < target and tries < GRID_DOUBLINGS and hi < cap:
            nh = min(hi * 2.0, cap); cn, rn = probe(nh)
            if not usable(cn, rn):
                lo_t, hi_t = hi, nh           # feasibility wall between hi and nh
                for _ in range(6):
                    mid = .5 * (lo_t + hi_t); cm, rm = probe(mid)
                    if usable(cm, rm): lo_t, c, r = mid, cm, rm
                    else: hi_t = mid
                hi = lo_t
                break
            hi, c, r = nh, cn, rn
            tries += 1
        status = "spans" if c >= target else "ceiling-below"
    # The comparison happens at fixed COSTS, but a dial-uniform grid bunches
    # rungs wherever dial->cost is steep, so a readout cost can fall between
    # rungs and be silently skipped -- a NULL caused by grid placement.
    # Measure the dial->cost curve here (oracle-free) so main() can place rungs
    # ON the readout costs.
    curve = [(0.0, 0.0)]
    grid = list(np.linspace(0.0, hi, 5)[1:])
    if hi > 0:                      # geometric leg resolves the steep low end
        grid += [hi * (0.5 ** k) for k in (1, 2, 3, 4, 5)]
    for t in sorted(set(float(x) for x in grid if x > 0)):
        ct, rt = probe(t)
        if np.isfinite(ct):
            curve.append((float(t), float(ct)))
    curve.sort(key=lambda z: z[0])
    # bisect onto each readout cost so [6] starts from a bracket, not a void
    for tc in readout_costs:
        cs = [c for _, c in curve]
        if not cs or not (min(cs) <= tc <= max(cs)):
            continue
        lo_b = max([d for d, c in curve if c <= tc], default=0.0)
        hi_b = min([d for d, c in curve if c >= tc], default=hi)
        for _ in range(4):
            mid = 0.5 * (lo_b + hi_b)
            cm, rm = probe(mid)
            if not np.isfinite(cm): break
            curve.append((float(mid), float(cm)))
            if cm < tc: lo_b = mid
            else:       hi_b = mid
        curve.sort(key=lambda z: z[0])
    log(f"    grid {arm:20} dial 0..{hi:.5g}  {status:18} "
        f"cost@max={c:.5f} solve_rate={r:.2f} target={target:.5f}")
    return curve, status, c, hi


def main():
    seed = int(sys.argv[1]); t0 = time.time()
    # Dry runs must NEVER write to the evaluation path. A throwaway artifact
    # sitting at runs/seed9.json is how a dry run gets mistaken for a result.
    sub = ("runs_dry" if DRY else "runs") \
          + ("" if BENCH == "methane" else f"_{BENCH}")
    out_dir = HERE / sub; out_dir.mkdir(exist_ok=True)
    _sfx = ""
    logf = open(out_dir / f"seed{seed}{_sfx}.log", "w")
    def log(m):
        m = f"[{time.time()-t0:7.0f}s] {m}"
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    ck = {}
    def checkpoint(stage):
        """Durably dump everything known so far. The log carries risks but NOT
        the per-objective masks the paired tests need, so a crash before [done]
        used to lose 6h of oracle calls irrecoverably."""
        try:
            tmp = out_dir / f"seed{seed}{_sfx}.partial.json.tmp"
            json.dump(dict(stage=stage, seed=seed, elapsed_s=time.time()-t0,
                           code_md5=_code_md5(), **ck),
                      open(tmp, "w"))
            tmp.replace(out_dir / f"seed{seed}{_sfx}.partial.json")
        except Exception as e:
            log(f"  [checkpoint {stage} failed: {e}]")

    # ---- RESUME: load the checkpoint and gate on the code fingerprint ------
    # Everything else is verified as it is re-derived below. Nothing from the
    # checkpoint is trusted until the corresponding stage reproduces it exactly.
    prev, done_keys = None, set()
    def _abort(why):
        log(f"  [RESUME ABORT] {why}")
        log("  Refusing to continue: a resumed run that does not reproduce its "
            "own checkpoint would silently mix two different estimands.")
        raise SystemExit(2)
    if RESUME:
        pp = out_dir / f"seed{seed}{_sfx}.partial.json"
        if not pp.exists():
            raise SystemExit(f"resume requested but no checkpoint at {pp}")
        prev = json.load(open(pp))
        log(f"=== RESUME from {pp.name} (stage '{prev.get('stage')}', "
            f"{prev.get('elapsed_s', 0)/3600:.2f}h already spent) ===")
        if int(prev.get("seed", -1)) != seed:
            _abort(f"checkpoint is for seed {prev.get('seed')}, not {seed}")
        if prev.get("code_md5") != _code_md5():
            _abort(f"run_seed.py changed since the checkpoint "
                   f"(md5 {prev.get('code_md5')} -> {_code_md5()})")
        done_keys = set((prev.get("frontier") or {}).keys())
        log(f"  code fingerprint matches; {len(done_keys)} rows already computed")

    log(f"=== seed {seed} ===")
    log("[1] context + calibration + sanity gate")
    ctxinfo = build_context(seed, log)
    if prev is not None:
        copies = _ck_all(prev, "ctxinfo", "context")
        if not copies:
            _abort("checkpoint carries no context to verify against")
        for _k, _v in copies:
            d = _exact(ctxinfo, _v, f"ctxinfo(vs checkpoint '{_k}')")
            if d: _abort(f"[1] context does not reproduce -> {d}")
        log(f"  [RESUME] [1] context reproduces bit-identically "
            f"({len(copies)} stored cop{'y' if len(copies)==1 else 'ies'} checked)")
    ck["ctxinfo"] = ctxinfo; checkpoint("context")

    Wpr = np.vstack([laws_weights(l, N_PROP_PL, 3000+seed*17+i) for i, l in enumerate(LAWS)])
    Wm  = np.vstack([laws_weights(l, 20,        1000+seed*17+i) for i, l in enumerate(LAWS)])
    Wp  = np.vstack([laws_weights(l, N_PERF_PL, 2000+seed*17+i) for i, l in enumerate(LAWS)])
    lawp = np.array(sum([[l]*N_PERF_PL for l in LAWS], []))
    rg = np.random.default_rng(seed)
    spr = rg.integers(0, 2**31, size=len(Wpr))
    sm  = rg.integers(0, 2**31, size=len(Wm))
    sp  = rg.integers(0, 2**31, size=len(Wp))
    log(f"  D_prop {len(Wpr)} | D_match {len(Wm)} (cost only) | D_perf {len(Wp)}")

    # Arm selection moved ahead of [2] so [1b] fits ONLY what the selected arms
    # need. ALL_ARMS is the default roster plus arms that are deliberately not in

    ck["constants"] = _constants(arms)
    if prev is not None:
        if "constants" not in prev:
            _abort("checkpoint predates the constants fingerprint; it cannot be "
                   "verified and must not be resumed")
        d = _exact(ck["constants"], prev["constants"], "constants")
        if d: _abort(f"a constant that shapes results changed -> {d}")
        log("  [RESUME] constants match")

    log("[1b] baseline arm assets (fitted only for the selected arms)")
    prepare_arm_assets(seed, arms, log)
    ck["ctxinfo"] = ctxinfo; checkpoint("assets")

    log("[2] full two-stage decision-aware method + cmicl_refit control")
    minfo = build_method(seed, Wpr, spr, log)
    if prev is not None:
        copies = _ck_all(prev, "minfo", "method")
        if not copies:
            _abort("checkpoint carries no method info to verify against")
        for _k, _v in copies:
            d = _exact(minfo, _v, f"minfo(vs checkpoint '{_k}')")
            if d: _abort(f"[2] method does not reproduce -> {d}")
        log(f"  [RESUME] [2] method reproduces bit-identically "
            f"({len(copies)} stored cop{'y' if len(copies)==1 else 'ies'} checked)")
    ck["minfo"] = minfo; checkpoint("method")

    log("[3] anchor C* on D_match (ZERO oracle) ...")
    base_m = solve_all("static", 0., Wm, sm, PRIMARY_STARTS)
    C_star = float(np.nanmean(cost_of(solve_all("cmicl", 1.0, Wm, sm, PRIMARY_STARTS),
                                      base_m, Wm)))
    caps = {f"cap{int(f*100):03d}": f*C_star for f in CAP_FRACTIONS}
    if prev is not None and "C_star" in prev:
        d = _exact(C_star, float(prev["C_star"]), "C_star")
        if d: _abort(f"[3] C* does not reproduce -> {d}  "
                     f"(every cap is f*C*, so this would recalibrate every arm)")
        d = _exact(caps, prev.get("caps"), "caps")
        if d: _abort(f"[3] caps do not reproduce -> {d}")
        log("  [RESUME] [3] C* and caps reproduce bit-identically")
    log(f"  C* (exact C-MICL, knob=1) = {C_star:.6f}   readout costs "
        + str({k: round(v, 5) for k, v in caps.items()}))

    log("[4] knob-grid calibration on D_match (ZERO oracle)")
    # TOP-UP MODE: recompute only a SUBSET of arms and merge into an existing
    # seed result. Legitimate because every arm is independent and deterministic
    # given (seed, arm, knob): all RNGs are freshly seeded from `seed` (none
    # advance across arms) and ocl_solve.deterministic_starts seeds the multistart
    # per objective, so arms are compared under common random numbers. Running a
    # subset therefore yields byte-identical rows to running the full roster.
    # merge_arms.py re-verifies C*, caps and the objective family before merging.
    grids, gstat, curves = {}, {}, {}
    for arm in arms:
        curve, gstat[arm], _, hi = calibrate_grid(
            arm, 1.5*C_star, Wm, sm, base_m, log,
            readout_costs=list(caps.values()) if arm in DECISION_ARMS else ())
        curves[arm] = curve
        cd = [d for d, _ in curve]; cc = [c for _, c in curve]
        nspan = N_SPAN_RUNGS if arm in DECISION_ARMS else N_SPAN_CONTEXT
        dials = list(np.linspace(0.0, hi, nspan))              # spanning rungs
        if arm in DECISION_ARMS:
            for t_cost in caps.values():                       # readout rungs
                if cc and min(cc) <= t_cost <= max(cc):
                    dials.append(float(np.interp(t_cost, cc, cd)))
        if arm in ("cmicl", "cmicl_refit") and hi >= 1.0:
            dials.append(1.0)                                  # published operating point
        grids[arm] = sorted({round(d, 8) for d in dials if d >= 0})
        log(f"    -> {arm:20} {len(grids[arm])} rungs; readout costs "
            f"{'covered' if cc and max(cc) >= max(caps.values()) else 'NOT fully covered'}"
            f" (cost range {min(cc):.5f}..{max(cc):.5f})")

    # The dial grid fixes WHICH rungs exist, so a resumed run that placed
    # them differently would append rows from a different experiment. Recorded
    # and checked, not assumed.
    ck["grids"] = {a: list(map(float, g)) for a, g in grids.items()}
    ck["grid_status"] = gstat
    if prev is not None and "grids" in prev:
        d = _exact(ck["grids"], prev["grids"], "grids")
        if d: _abort(f"[4] knob grid does not reproduce -> {d}")
        d = _exact(gstat, prev.get("grid_status"), "grid_status")
        if d: _abort(f"[4] grid status does not reproduce -> {d}")
        log("  [RESUME] [4] knob grids reproduce bit-identically")
    elif prev is not None and prev.get("stage") not in ("context", "assets", "method"):
        _abort("checkpoint reached [4] or beyond but carries no `grids`; it "
               "predates the resume guard and cannot be verified")
    checkpoint("grids")

    log("[5] frontier sweep on D_perf (oracle read once per rung)")
    base = {st: solve_all("static", 0., Wp, sp, st) for st in (PRIMARY_STARTS, CHECK_STARTS)}
    frontier = {}
    for arm in arms:
        for st in (PRIMARY_STARTS,):
            pts = []
            for t in grids[arm]:
                _key = f"{arm}|{st}|{t:.6f}"
                if _key in done_keys:
                    # Already computed earlier in this run and verified above
                    # to have come from identical code, constants and models.
                    frontier[_key] = prev["frontier"][_key]
                    pts.append(frontier[_key])
                    continue
                knob = knob_of(arm, t)
                rows = solve_all(arm, knob, Wp, sp, st)
                X = np.array([r["x"] for r in rows])
                dep = np.array([r["dep"] for r in rows])
                ht = np.array([BM.true_h(x) for x in X])
                dep = dep & np.isfinite(ht)      # NaN h is unscoreable, not safe
                viol = (ht > 0) & dep
                c = cost_of(rows, base[st], Wp); ok = np.isfinite(c)
                nd = int(dep.sum())
                rec = dict(arm=arm, starts=st, dial=float(t), knob=float(knob),
                           cost=float(np.nanmean(c)) if ok.any() else float("nan"),
                           n_cost_valid=int(ok.sum()),
                           n_nesting_violations=int(np.sum(c[ok] < -1e-9)),
                           risk=(int(viol.sum())/nd) if nd else float("nan"),
                           n_viol=int(viol.sum()), n_deployed=nd, n_obj=int(len(dep)),
                           solve_rate=float(dep.mean()),
                           per_law={l: float(viol[(lawp == l) & dep].sum()
                                             / max(int(((lawp == l) & dep).sum()), 1))
                                    for l in LAWS},
                           viol_mask=viol.tolist(), dep_mask=dep.tolist())
                frontier[f"{arm}|{st}|{t:.6f}"] = rec
                pts.append(rec)
            log(f"  {arm:20} @{st:>3}: " +
                " ".join(f"({v['cost']:.4f},{v['risk']:.3f})" for v in pts))
            logf.flush()
            # [5] is the longest phase (methane: ~4h of a ~20h run) and used to
            # checkpoint only on COMPLETION. A scheduler cancelled methane seeds
            # 3 and 5 at ~10h with the checkpoint still reading stage 'method',
            # so ~20 core-hours were unsalvageable. Checkpoint per arm instead.
            ck.update(context=ctxinfo, method=minfo, C_star=C_star, caps=caps,
                      grid_status=gstat, laws=LAWS,
                      law_of_objective=lawp.tolist(),
                      primary_starts=PRIMARY_STARTS, check_starts=CHECK_STARTS,
                      frontier=frontier)
            checkpoint(f"frontier:{arm}")

    # ---- REFINEMENT PASS -------------------------------------------------
    # Rungs were placed from the D_match cost curve, but risk is measured on
    # D_perf and costs differ between splits. Re-solve each
    # arm at the dial that hits each readout cost ON D_PERF, so the paired
    # comparison is exactly cost-matched on the split where risk is read.
    # Cost is oracle-free, so using the D_perf cost curve to place these rungs
    # leaks no outcome; the oracle is still read once, at the final rung.
    ck.update(context=ctxinfo, method=minfo, C_star=C_star, caps=caps,
              grid_status=gstat, laws=LAWS, law_of_objective=lawp.tolist(),
              primary_starts=PRIMARY_STARTS, check_starts=CHECK_STARTS,
              frontier=frontier)
    checkpoint("frontier")

    log("[6] refinement: exact cost-matched rungs ON D_perf")
    for arm in [a for a in arms if a in DECISION_ARMS]:
        rs = sorted([v for v in frontier.values()
                     if v["arm"] == arm and v["starts"] == PRIMARY_STARTS
                     and np.isfinite(v["cost"])], key=lambda v: v["cost"])
        if len(rs) < 2:
            log(f"  {arm:20} too few finite rungs to refine"); continue
        cc = [v["cost"] for v in rs]; dd = [v["dial"] for v in rs]
        for cap_id, target in caps.items():
            _rkey = f"READOUT|{arm}|{cap_id}"
            if _rkey in done_keys:
                frontier[_rkey] = prev["frontier"][_rkey]
                r = frontier[_rkey]
                log(f"  {arm:20} {cap_id}: [resumed] realised={r['cost']:.5f} "
                    f"[{r.get('match_status','?')}] risk={r['n_viol']}/{r['n_deployed']}")
                continue
            if not (min(cc) <= target <= max(cc)):
                log(f"  {arm:20} {cap_id}: outside swept range "
                    f"[{min(cc):.5f},{max(cc):.5f}] -> ceiling/floor, no extrapolation")
                continue
            # One interpolation lands ~14% off (measured): the dial->cost curve
            # is noisy and not locally linear. Iterate against the REALISED cost
            # until within tolerance, keeping the best attempt.
            #
            # Every probe runs at PRIMARY_STARTS, the same budget as the seed
            # table `cc` (taken from the [5] frontier) and as the recorded
            # readout. An earlier version probed at SWEEP_STARTS and appended
            # those costs into a PRIMARY-start table: two budgets do not price a
            # knob identically, so the root-find converged on a dial that hit the
            # target at 60 starts and missed it at 120 -- every rung would come
            # back "step-gap", analyze.py would refuse all of them, and the run
            # would report NULL for want of a matched comparison. That is the
            # cost-match failure this pass exists to prevent.
            hc, hd = list(cc), list(dd)
            best = None
            lo_t = hi_t = None          # dial bracket around the target cost
            for _ in range(REFINE_ITERS):
                order = np.argsort(hc)
                t = float(np.interp(target, np.array(hc)[order], np.array(hd)[order]))
                # Pure secant chases noise: at low cost the dial->cost curve is
                # steep and the mean over D_perf is a noisy estimate, so the
                # interpolant can jump outside the interval already known to
                # bracket the target. Once a bracket exists, keep the step
                # inside it and fall back to bisection when it escapes -- that
                # converges on the dial regardless of how noisy cost is.
                if lo_t is not None and hi_t is not None:
                    if not (min(lo_t, hi_t) < t < max(lo_t, hi_t)):
                        t = 0.5*(lo_t + hi_t)
                if any(abs(t - u) < 1e-12 for u in hd[len(cc):]):
                    break                      # root-find has stopped moving
                rows = solve_all(arm, knob_of(arm, t), Wp, sp, PRIMARY_STARTS)
                c = cost_of(rows, base[PRIMARY_STARTS], Wp); ok = np.isfinite(c)
                if not ok.any():
                    break
                realised = float(np.nanmean(c))
                e = abs(realised - target) / max(target, 1e-12)
                if best is None or e < best[0]:
                    best = (e, t, rows, c, ok, realised)
                if e <= REFINE_TOL:
                    break
                if realised < target: lo_t = t
                else:                 hi_t = t
                if len(hd) > len(cc) and abs(realised - hc[-1]) < 1e-9:
                    break                      # knob saturated: target unreachable
                hc.append(realised); hd.append(t)
            if best is None:
                log(f"  {arm:20} {cap_id}: no finite cost -> unresolved"); continue
            err, t, rows, c, ok, realised = best
            knob = knob_of(arm, t)
            status = "matched" if err <= REFINE_TOL else "step-gap"
            X = np.array([r["x"] for r in rows])
            dep = np.array([r["dep"] for r in rows])
            ht = np.array([BM.true_h(x) for x in X])
            dep = dep & np.isfinite(ht)          # NaN h is unscoreable, not safe
            viol = (ht > 0) & dep
            nd = int(dep.sum())
            frontier[f"READOUT|{arm}|{cap_id}"] = dict(
                arm=arm, starts=PRIMARY_STARTS, dial=t, knob=float(knob),
                readout=cap_id, target_cost=float(target), cost=realised,
                cost_error=err, match_status=status,
                n_cost_valid=int(ok.sum()),
                n_nesting_violations=int(np.sum(c[ok] < -1e-9)),
                risk=(int(viol.sum())/nd) if nd else float("nan"),
                n_viol=int(viol.sum()), n_deployed=nd, n_obj=int(len(dep)),
                solve_rate=float(dep.mean()),
                per_law={l: float(viol[(lawp == l) & dep].sum()
                                  / max(int(((lawp == l) & dep).sum()), 1)) for l in LAWS},
                viol_mask=viol.tolist(), dep_mask=dep.tolist())
            ck["frontier"] = frontier; checkpoint(f"refine:{arm}:{cap_id}")
            log(f"  {arm:20} {cap_id}: target={target:.5f} realised={realised:.5f} "
                f"err={100*err:.1f}% [{status}] risk={int(viol.sum())}/{nd} "
                f"deployed={nd}/{len(dep)}")
        logf.flush()

    # ---- [7] UPWARD SOLVER CONVERGENCE CHECK -----------------------------
    # The primary budget is 60 starts. 60 and 120 starts agree within 3% on every
    # rung; this re-verifies that on the actual readout rungs of this run, for the
    # method and its anchor, at the ladder's two extremes.
    log(f"[7] upward convergence check at {CHECK_STARTS} starts")
    for arm in ("law_blind_envelope", "cmicl"):
        for cap_id in ("cap025", "cap100"):
            r = frontier.get(f"READOUT|{arm}|{cap_id}")
            if r is None: continue
            _ckey = f"CHECK|{arm}|{cap_id}"
            if _ckey in done_keys:
                frontier[_ckey] = prev["frontier"][_ckey]
                log(f"  {arm:20} {cap_id}: [resumed] "
                    f"risk {r['risk']:.3f} -> {frontier[_ckey]['risk']:.3f}")
                continue
            rows = solve_all(arm, r["knob"], Wp, sp, CHECK_STARTS)
            c = cost_of(rows, base[CHECK_STARTS], Wp); ok = np.isfinite(c)
            dep = np.array([r2["dep"] for r2 in rows])
            ht = np.array([BM.true_h(r2["x"]) for r2 in rows])
            dep = dep & np.isfinite(ht)          # NaN h is unscoreable, not safe
            viol = (ht > 0) & dep; nd = int(dep.sum())
            rk = (int(viol.sum())/nd) if nd else float("nan")
            frontier[f"CHECK|{arm}|{cap_id}"] = dict(
                arm=arm, starts=CHECK_STARTS, knob=r["knob"], readout=cap_id,
                cost=float(np.nanmean(c)) if ok.any() else float("nan"), risk=rk,
                n_viol=int(viol.sum()), n_deployed=nd,
                viol_mask=viol.tolist(), dep_mask=dep.tolist())
            log(f"  {arm:20} {cap_id}: risk {r['risk']:.3f} @{PRIMARY_STARTS} -> "
                f"{rk:.3f} @{CHECK_STARTS} (d={rk-r['risk']:+.3f}) | cost "
                f"{r['cost']:.5f} -> {float(np.nanmean(c)):.5f}")
        logf.flush()

    # PLATFORM FINGERPRINT. A top-up merged into a base produced on a different
    # machine is INVALID: the pinball surrogate fit runs through a different BLAS
    # (Accelerate vs OpenBLAS), which changes the local optimum. Merging a
    # top-up from another machine into a methane seed 6 base gave C* differing by
    # 5.1% and caps 5% apart -- arms calibrated to different cost targets, the
    # exact failure the protocol exists to prevent. A merge now refuses
    # unless this matches, so the mismatch is caught before any compute is spent
    # rather than after.
    import platform as _plat
    _fp = dict(python=_plat.python_version(), machine=_plat.machine(),
               system=_plat.system(), numpy=np.__version__,
               blas=str(getattr(np.__config__, "CONFIG", {})
                        .get("Build Dependencies", {})
                        .get("blas", {}).get("name", "unknown")))
    json.dump(dict(seed=seed, platform=_fp,
                   context=ctxinfo, method=minfo, C_star=C_star, caps=caps,
                   grid_status=gstat, laws=LAWS, law_of_objective=lawp.tolist(),
                   primary_starts=PRIMARY_STARTS, sweep_starts=PRIMARY_STARTS,
                   check_starts=CHECK_STARTS,
                   frontier=frontier, elapsed_s=time.time()-t0),
              open(out_dir / (f"seed{seed}.arms.json" if _pick
                              else f"seed{seed}.json"), "w"))
    log(f"[done] {time.time()-t0:.0f}s -> runs/seed{seed}.json")


if __name__ == "__main__":
    main()
