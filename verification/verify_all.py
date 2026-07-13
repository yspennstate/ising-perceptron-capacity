"""Top-level reproducible certificate for the Krauth-Mezard verification.

Checks, in order:
  1. Block 1  -- the Ding-Sun parameter rectangle (Prop 1.3), 13 checks
                 (runs block1_gardner.py).
  2. Huang Condition 1.3, Region I -- the constructive moment-body interior
     certificate and the star around the degenerate maximizer, from
     results/huang_star_interior.json and results/huang_region1.json.
  3. Huang Condition 1.3, Region II -- canonical exact rectangle-proof
     trees from results/huang_sweep.json and results/huang_sweep2.json,
     including the full Sweep1 -> Sweep2 -> Region-I delegation chain.
  4. The source-bound support-box guards and Huang's analytic identity
     S_*(1,0) = 0 carried by the canonical Sweep1 certificate.
  5. Ding-Sun Condition 1.2: block2 (near one), block3a grid logs, and
     block3bc (middle interval / degenerate zero) results.

Each long computation writes canonical JSON under results/.  This driver
accepts the proof artifacts by path and never promotes legacy summary logs.

Usage:  python verify_all.py
"""

import warnings
warnings.filterwarnings('ignore')
import subprocess
import sys
import os
os.environ.setdefault('HUANG_GRID_N', '2700')   # the resolution of record
CREATE_FLAGS = (getattr(subprocess, 'CREATE_NO_WINDOW', 0)
                | getattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS', 0))


def banner(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70, flush=True)


def validate_region1_manifest(path_or_data, r1=None):
    """Accept only the canonical schema-3 exact leaf-proof certificate."""
    del r1  # Retained only for compatibility with older callers.
    if not isinstance(path_or_data, (str, bytes, os.PathLike)):
        return False, 'path_required_for_canonical_certificate'
    try:
        import huang_region1_verify as verifier
        verifier.verify_certificate(path_or_data)
        return True, 'ok'
    except Exception as exc:
        return False, f'exception:{type(exc).__name__}:{exc}'


def validate_star_interior_certificate(path):
    """Replay the constructive inball certificate exactly."""
    try:
        import huang_star_interior as interior
        payload = interior.verify_certificate(path)
        return True, 'ok', payload
    except Exception as exc:
        return False, f'exception:{type(exc).__name__}:{exc}', None


def validate_sweep_manifest(path, module, stage):
    """Accept only a canonical schema-3 exact rectangle-proof certificate."""
    del module  # The exact verifier selects and source-binds the stage module.
    if not isinstance(path, (str, bytes, os.PathLike)):
        return False, 'path_required_for_canonical_certificate', None
    try:
        import huang_sweep_verify as verifier
        data, _ = verifier.verify_certificate(path, stage)
        return True, 'ok', data
    except Exception as exc:
        return False, f'exception:{type(exc).__name__}:{exc}', None


def verify_block3a_certificate(path, expected_model=None):
    """Fail-closed dispatch between the two frozen Block-3a evidence modes."""
    import block3a_assemble as legacy
    import block3a_singlerun as single_run

    data = legacy.load_certificate(path)
    evidence_model = data.get('evidence_model')
    legacy_model = data.get('policy', {}).get('evidence_model')
    selected = evidence_model or legacy_model
    if expected_model is not None and selected != expected_model:
        raise ValueError('Block3a evidence model does not match final policy')
    if evidence_model == single_run.EVIDENCE_MODEL:
        return single_run.verify_certificate(path)
    if legacy_model == 'source-bound-trusted-execution':
        return legacy.verify_certificate(path)
    raise ValueError('unknown Block3a evidence model')


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(here, 'results')
    outcomes = {}

    banner("Block 1: Ding-Sun parameter rectangle (Proposition 1.3)")
    r = subprocess.run([sys.executable, '-B',
                        os.path.join(here, "block1_gardner.py")],
                       creationflags=CREATE_FLAGS)
    outcomes['DS Block 1 (rectangle)'] = (r.returncode == 0)

    banner("Huang Condition 1.3 / Region I: moment-body interior")
    interior_path = os.path.join(res, 'huang_star_interior.json')
    interior_ok, interior_reason, interior_data = \
        validate_star_interior_certificate(interior_path)
    print(f"huang_star_interior.json: "
          f"{'PASS' if interior_ok else 'FAIL'} ({interior_reason})"
          + (f" {interior_data['certificate_sha256']}"
             if interior_ok else ''))
    outcomes['Huang Region I (star interior)'] = interior_ok

    banner("Huang Condition 1.3 / Region I: star region certificates")
    p = os.path.join(res, 'huang_region1.json')
    ok = False
    d = None
    if os.path.exists(p):
        try:
            import huang_region1_verify as _region1_verify
            replay_workers = int(os.environ.get(
                'HUANG_REGION1_REPLAY_WORKERS', '1'))
            d = _region1_verify.verify_certificate_full(
                p, workers=replay_workers,
                progress=lambda done, total, index: print(
                    f'  Region-I numerical replay {done}/{total} '
                    f'(last job {index})', flush=True)
                if done == 1 or done % 25 == 0 or done == total else None)
            ok = True
            manifest_reason = 'ok'
            summary = d['derived_summary']
            print(f"huang_region1.json: {summary['jobs']} band jobs, "
                  f"fails={summary['failures']}, manifest=ok")
        except Exception as exc:
            manifest_reason = f'{type(exc).__name__}:{exc}'
            print(f"huang_region1.json: FAIL manifest={manifest_reason}")
        if ok:
            st = d['star']
            print("  exact star policy and all curvature leaves verified")
    else:
        print("MISSING results/huang_region1.json -- run huang_region1.py")
    outcomes['Huang Region I (star)'] = ok

    banner("Huang Condition 1.3 / Region II: bulk sweeps")
    import huang_sweep_verify as _sweep_verify
    sweep1_path = os.path.join(res, 'huang_sweep.json')
    sweep2_path = os.path.join(res, 'huang_sweep2.json')
    try:
        data1, delegated1 = _sweep_verify.verify_certificate(sweep1_path, 1)
        ok1, why1 = True, 'ok'
    except Exception as exc:
        data1, delegated1 = None, None
        ok1, why1 = False, f'{type(exc).__name__}:{exc}'
    try:
        data2, delegated2 = _sweep_verify.verify_certificate(sweep2_path, 2)
        ok2, why2 = True, 'ok'
    except Exception as exc:
        data2, delegated2 = None, None
        ok2, why2 = False, f'{type(exc).__name__}:{exc}'
    summary1 = data1['derived_summary'] if ok1 else None
    summary2 = data2['derived_summary'] if ok2 else None
    leaves1 = (sum(summary1[key] for key in
                   ('negative_leaves', 'outside_K_leaves',
                    'delegated_leaves')) if ok1 else None)
    leaves2 = (sum(summary2[key] for key in
                   ('negative_leaves', 'outside_K_leaves',
                    'region1_delegated_leaves')) if ok2 else None)
    print(f"huang_sweep.json: {'PASS' if ok1 else 'FAIL'} ({why1})"
          + (f" leaves={leaves1}" if ok1 else ''))
    print(f"huang_sweep2.json: {'PASS' if ok2 else 'FAIL'} ({why2})"
          + (f" leaves={leaves2}" if ok2 else ''))
    pair_ok = False
    pair_reason = 'component_certificate_failed'
    if ok1 and ok2 and ok and interior_ok:
        try:
            components = _sweep_verify.verify_pair_components(
                data1, delegated1, data2, delegated2, d)
            _sweep_verify.verify_bundle(
                os.path.join(res, 'huang_bundle.json'),
                (interior_data, components))
            pair_ok = True
            pair_reason = 'ok; canonical bundle bound'
        except Exception as exc:
            pair_reason = f'{type(exc).__name__}:{exc}'
    elif ok1 and ok2:
        pair_reason = 'complete Region-I exact evidence unavailable'
    print(f"Region-II delegation chain: "
          f"{'PASS' if pair_ok else 'FAIL'} ({pair_reason})")
    outcomes['Huang Region II (sweeps)'] = pair_ok

    banner("Sweep rectangle contains the moment body: K subset R")
    guards = data1['domain_guards'] if ok1 else None
    if ok1:
        print("PASS (canonical Sweep1 h(1,0), h(0,1) upper packets are "
              "strictly below the exact rectangle limits)")
        okK = True
    else:
        print("FAIL (no valid canonical Sweep1 domain guards)")
        okK = False
    outcomes['K inside sweep rectangle'] = okK

    banner("Degenerate point: S_*(1,0) analytic identity")
    degenerate_ok = bool(
        ok1 and guards['degenerate_identity']
        == 'huang-analytic-Sstar(1,0)=0')
    print(f"{'PASS' if degenerate_ok else 'FAIL'} "
          "(certificate binds the analytic identity S_*(1,0)=0)")
    outcomes['S_*(1,0) = 0'] = degenerate_ok

    banner("Ding-Sun Condition 1.2: near-one block (block2)")
    r = subprocess.run([sys.executable, '-B',
                        os.path.join(here, "block2_near_one.py")],
                       creationflags=CREATE_FLAGS)
    outcomes['DS Block 2 (near one)'] = (r.returncode == 0)

    banner("Ding-Sun Condition 1.2: grid logs (block3a) and middle (block3bc)")
    try:
        import block3a_singlerun as _b3a_single
        cert3a = verify_block3a_certificate(
            os.path.join(res, 'block3a_certificate.json'),
            expected_model=_b3a_single.EVIDENCE_MODEL)
        ok3a = True
        print("block3a_certificate.json: PASS "
              f"{cert3a['certificate_sha256']}")
    except Exception as exc:
        ok3a = False
        print("block3a_certificate.json: INCOMPLETE/FAIL "
              f"({type(exc).__name__}: {exc})")
    outcomes['DS Block 3a (grids)'] = ok3a
    try:
        import block3bc_assemble as _b3bc_assemble
        cert3 = _b3bc_assemble.verify_certificate(
            os.path.join(res, 'block3bc_certificate.json'))
        ok3 = True
        print("block3bc_certificate.json: PASS "
              f"{cert3['certificate_sha256']}")
    except Exception as exc:
        ok3 = False
        print("block3bc_certificate.json: INCOMPLETE/FAIL "
              f"({type(exc).__name__}: {exc})")
    outcomes['DS Block 3bc (middle)'] = ok3

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
