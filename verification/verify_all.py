"""Top-level reproducible certificate for the Krauth-Mezard verification.

Checks, in order:
  1. Block 1  -- the Ding-Sun parameter rectangle (Prop 1.3), 13 checks
                 (runs block1_gardner.py).
  2. Huang Condition 1.3, Region I -- the star region around the degenerate
     maximizer, from results/huang_region1.json (run huang_region1.py).
  3. Huang Condition 1.3, Region II -- the bulk sweeps, from
     results/huang_sweep.log (huang_sweep.py, K minus EXCL_OLD) and
     results/huang_sweep2.log (huang_sweep2.py, EXCL_OLD minus the star).
  4. Consistency of the degenerate point: the one-sided upper bound
     S_*(1,0) <= 1e-4 (the analytic value is 0 by Huang's identities).
  5. Ding-Sun Condition 1.2: block2 (near one), block3a grid logs, and
     block3bc (middle interval / degenerate zero) results.

Each long computation writes its own log/JSON under results/; this driver
re-checks their outcomes and re-runs the cheap ones inline.

Usage:  python verify_all.py
"""

import warnings
warnings.filterwarnings('ignore')
import json
import subprocess
import sys
import os
from flint import arb
from core import set_prec, dec, endpoints, PSI, Q


def banner(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70, flush=True)


def check_log(path, needle_fail="FAIL"):
    """(exists, completed_line, nfails) for a sweep-style log."""
    if not os.path.exists(path):
        return False, None, None
    fails = 0
    last = ''
    with open(path) as f:
        for line in f:
            if line.startswith(needle_fail):
                fails += 1
            if line.strip():
                last = line.strip()
    return True, last, fails


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(here, 'results')
    outcomes = {}

    banner("Block 1: Ding-Sun parameter rectangle (Proposition 1.3)")
    r = subprocess.run([sys.executable, os.path.join(here, "block1_gardner.py")])
    outcomes['DS Block 1 (rectangle)'] = (r.returncode == 0)

    banner("Huang Condition 1.3 / Region I: star region certificates")
    p = os.path.join(res, 'huang_region1.json')
    ok = False
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
        nfail = d.get('fails', None)
        n = len(d.get('results', []))
        ok = (nfail == 0) and n > 0
        print(f"huang_region1.json: {n} band jobs, fails={nfail}")
        if ok:
            st = d['star']
            print(f"  star: T_LONG={st['T_LONG']} T_MID={st['T_MID']} "
                  f"T_CORE={st['T_CORE']} (W_ANG={st['W_ANG']})")
    else:
        print("MISSING results/huang_region1.json -- run huang_region1.py")
    outcomes['Huang Region I (star)'] = ok

    banner("Huang Condition 1.3 / Region II: bulk sweeps")
    ex, last, nf = check_log(os.path.join(res, 'huang_sweep.log'))
    ok1 = ex and nf == 0 and last and last.startswith('total_leaves')
    print(f"huang_sweep.log: exists={ex} fails={nf} last='{last}'")
    ex, last, nf = check_log(os.path.join(res, 'huang_sweep2.log'))
    ok2 = ex and nf == 0 and last and last.startswith('total_leaves')
    print(f"huang_sweep2.log: exists={ex} fails={nf} last='{last}'")
    outcomes['Huang Region II (sweeps)'] = ok1 and ok2

    banner("Sweep rectangle contains the moment body: K subset R")
    set_prec(60)
    import huanggrid as hg
    # K subset {|a1| <= h(1,0) = E|X|, |a2| <= h(0,1) = E|M|} (support box);
    # the sweeps cover R = [-1.31, 1.31] x [-0.70, 0.70].
    h10 = hg.h_support_upper(1, 0)
    h01 = hg.h_support_upper(0, 1)
    okK = (h10 < dec('1.31')) and (h01 < dec('0.70'))
    print(f"E|X| <= {h10} < 1.31 and E|M| <= {h01} < 0.70: {okK}")
    outcomes['K inside sweep rectangle'] = bool(okK)

    banner("Degenerate point: S_*(1,0) upper bound (analytic value 0)")
    gx = hg.get_x_grid()
    gz = hg.get_zt_grid()
    s0 = (1 - Q).sqrt()
    a1s = PSI * (1 - Q)
    a2s = Q
    val = hg.S_moment(a1s, a2s, s0,
                      duals=[(arb(1), arb(0), hg.Phi_of(arb(1), arb(0)))],
                      gz=gz, gx=gx)[0]
    _, vhi = endpoints(val)
    near0 = vhi < dec('0.0001')
    print(f"S_*(1,0) upper enclosure: {val}  (<= 1e-4: {near0})")
    outcomes['S_*(1,0) <= 1e-4'] = bool(near0)

    banner("Ding-Sun Condition 1.2: near-one block (block2)")
    r = subprocess.run([sys.executable, os.path.join(here, "block2_near_one.py")])
    outcomes['DS Block 2 (near one)'] = (r.returncode == 0)

    banner("Ding-Sun Condition 1.2: grid logs (block3a) and middle (block3bc)")
    ex, last, nf = check_log(os.path.join(res, 'block3a.log'))
    npass = 0
    if ex:
        with open(os.path.join(res, 'block3a.log')) as f:
            npass = sum(1 for ln in f if ln.startswith('PASS'))
    print(f"block3a.log: exists={ex} PASS={npass} FAIL={nf}")
    outcomes['DS Block 3a (grids)'] = ex and nf == 0 and npass >= 115
    ex3, last3, nf3 = check_log(os.path.join(res, 'block3bc.log'))
    print(f"block3bc.log: exists={ex3} fails={nf3} last='{last3}'")
    outcomes['DS Block 3bc (middle)'] = bool(ex3 and nf3 == 0 and last3
                                             and 'ALL PASS' in last3)

    banner("Summary")
    allok = True
    for k, v in outcomes.items():
        print(f"{'PASS' if v else 'FAIL':4}  {k}")
        allok = allok and v
    print(f"\n{'ALL CERTIFICATES PASS' if allok else 'INCOMPLETE / FAILURES'}")
    if not allok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
