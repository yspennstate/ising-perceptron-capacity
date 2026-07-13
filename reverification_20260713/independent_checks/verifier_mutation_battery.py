"""Falsifiability battery for the published certificate verifiers.

Runs each canonical verifier against (a) the pristine certificate and (b)
deliberately corrupted copies, and requires: pristine accepted, every
corruption rejected.  A verifier that accepts a corrupted certificate is a
silent alarm; this battery is the direct empirical test that the green
verify_all rows are load-bearing.

Corruptions are chosen to hit distinct layers of each verifier:
envelope hash, packet endpoint, cover structure, sign claim.

Run from the verification/ directory of a checkout whose verify_all is
green (the attested runtime), so a rejection is attributable to the
corruption rather than to environment drift:

    python verifier_mutation_battery.py
"""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import traceback

sys.path.insert(0, os.getcwd())

RESULTS = os.path.join(os.getcwd(), "results")


def write_temp(data):
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8", dir=os.getcwd())
    json.dump(data, handle)
    handle.close()
    return handle.name


def expect_reject(label, fn):
    try:
        fn()
    except Exception as exc:
        print(f"PASS {label}: rejected ({type(exc).__name__}: {str(exc)[:80]})")
        return True
    print(f"FAIL {label}: corruption was ACCEPTED")
    return False


def expect_accept(label, fn):
    try:
        fn()
        print(f"PASS {label}: pristine certificate accepted")
        return True
    except Exception as exc:
        print(f"FAIL {label}: pristine certificate rejected "
              f"({type(exc).__name__}: {str(exc)[:120]})")
        traceback.print_exc(limit=2)
        return False


def main():
    import huang_star_interior
    import huang_region1_verify
    import huang_sweep_verify

    ok = True
    tempfiles = []

    # --- star interior ----------------------------------------------------
    star_path = os.path.join(RESULTS, "huang_star_interior.json")
    star = json.load(open(star_path))
    ok &= expect_accept(
        "star_interior pristine",
        lambda: huang_star_interior.verify_certificate(star_path))

    corrupted = copy.deepcopy(star)
    corrupted["matrix"][0][0]["mid10"] = str(
        int(corrupted["matrix"][0][0]["mid10"]) - 10 ** 63)
    path = write_temp(corrupted)
    tempfiles.append(path)
    ok &= expect_reject(
        "star_interior matrix entry shifted 1e-2",
        lambda: huang_star_interior.verify_certificate(path))

    # --- Region I (structural verifier over the full tree) ----------------
    r1_path = os.path.join(RESULTS, "huang_region1.json")
    r1 = json.load(open(r1_path))
    ok &= expect_accept(
        "region1 pristine (structural)",
        lambda: huang_region1_verify.verify_certificate(r1_path))

    corrupted = copy.deepcopy(r1)
    corrupted["records"][0]["B_certificate"]["b11_packet"]["rad10"] = str(
        int(corrupted["records"][0]["B_certificate"]["b11_packet"]["mid10"]) + 1)
    path = write_temp(corrupted)
    tempfiles.append(path)
    ok &= expect_reject(
        "region1 b11 radius swallows midpoint",
        lambda: huang_region1_verify.verify_certificate(path))

    corrupted = copy.deepcopy(r1)
    leaves = corrupted["records"][0]["localization_certificate"]["leaves"]
    removed = leaves.pop(0)
    path = write_temp(corrupted)
    tempfiles.append(path)
    ok &= expect_reject(
        f"region1 localization leaf dropped (edge {removed['edge']})",
        lambda: huang_region1_verify.verify_certificate(path))

    # --- Region II sweep 1 -------------------------------------------------
    s1_path = os.path.join(RESULTS, "huang_sweep.json")
    s1 = json.load(open(s1_path))
    ok &= expect_accept(
        "sweep1 pristine",
        lambda: huang_sweep_verify.verify_certificate(s1_path, 1))

    def first_upper_packet(node):
        if isinstance(node, dict):
            if "upper_packet" in node:
                return node["upper_packet"]
            for value in node.values():
                found = first_upper_packet(value)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for value in node:
                found = first_upper_packet(value)
                if found is not None:
                    return found
        return None

    corrupted = copy.deepcopy(s1)
    packet = first_upper_packet(corrupted)
    if packet is None:
        print("FAIL sweep1: no upper_packet found to corrupt")
        ok = False
    else:
        packet["mid10"] = str(abs(int(packet["mid10"])))
        path = write_temp(corrupted)
        tempfiles.append(path)
        ok &= expect_reject(
            "sweep1 negativity packet sign flipped",
            lambda: huang_sweep_verify.verify_certificate(path, 1))

    for path in tempfiles:
        try:
            os.unlink(path)
        except OSError:
            pass

    print("BATTERY " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
