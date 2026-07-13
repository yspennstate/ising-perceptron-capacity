"""Update the manuscript inventory from exact, canonical proof certificates."""

import argparse
import json
import os
import pathlib
from fractions import Fraction

import block3a_singlerun
import block3bc_assemble
import block3bc_aux_verify
import block3bc_exact
import huang_sweep_verify
import verify_all


HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
TEX = HERE.parent / "paper" / "main.tex"
START = "% BEGIN VERIFIED CERTIFICATE INVENTORY"
END = "% END VERIFIED CERTIFICATE INVENTORY"


def verified_counts():
    _, (_, components) = huang_sweep_verify.verify_bundle(
        RESULTS / "huang_bundle.json")
    sweep1, sweep2, region1, _, _ = components
    block3a = verify_all.verify_block3a_certificate(
        RESULTS / "block3a_certificate.json",
        expected_model=block3a_singlerun.EVIDENCE_MODEL)
    block3bc = block3bc_assemble.verify_certificate(
        RESULTS / "block3bc_certificate.json")
    aux = block3bc_aux_verify.verify_manifest(
        RESULTS / block3bc["aux_manifest"], require_complete=True)

    k_run = block3bc_exact.fraction_from_record(block3bc["k_run"])
    if k_run != Fraction(21, 2):
        raise ValueError(f"release requires K_run=21/2, got {k_run}")
    k_nodes = len(aux["manifest"]["k_records"])
    if k_nodes != 59:
        raise ValueError(
            f"release requires the canonical 59-point I'' grid, got {k_nodes}")

    s1 = sweep1["derived_summary"]
    s2 = sweep2["derived_summary"]
    r1 = region1["derived_summary"]
    counts = {
        "sweep1_jobs": s1["jobs"],
        "sweep1_leaves": sum(s1[key] for key in (
            "negative_leaves", "outside_K_leaves", "delegated_leaves")),
        "sweep2_jobs": s2["jobs"],
        "sweep2_leaves": sum(s2[key] for key in (
            "negative_leaves", "outside_K_leaves",
            "region1_delegated_leaves")),
        "region1_jobs": r1["jobs"],
        "region1_angular_leaves": r1["certified_angular_leaves"],
        "region1_radial_pieces": r1["radial_pieces"],
        "block3a_cells": block3a["schedule"]["cells"],
        "block3a_leaves": block3a["coverage"]["total_recursive_leaves"],
        "block3bc": block3bc["summary"],
        "k_nodes": k_nodes,
        "k_run": block3bc_exact.fraction_text(k_run),
    }
    validate_counts(counts)
    return counts


def validate_counts(counts):
    if counts["region1_jobs"] != 1404:
        raise ValueError("release requires all 1,404 Region-I jobs")
    if counts["block3a_cells"] != 247:
        raise ValueError("release requires all 247 Block3a cells")
    expected = {"b_pos": 24, "b_neg": 331, "c": 16}
    actual = {
        part: counts["block3bc"][part]["top_cells"] for part in expected}
    if actual != expected:
        raise ValueError(
            f"release requires Block3bc top-cell counts {expected}, got {actual}")
    if counts["k_nodes"] != 59 or counts["k_run"] != "21/2":
        raise ValueError("release requires 59 K nodes and K_run=21/2")


def render_inventory(counts):
    validate_counts(counts)
    b3 = counts["block3bc"]
    return (
        r"\emph{Inventory.} The final source-bound runs comprise: "
        f"the parameter rectangle, 13 checks; Huang Region~II stage~1, "
        f"${counts['sweep1_jobs']}$ top cells and "
        f"${counts['sweep1_leaves']}$ verified leaves; stage~2, "
        f"${counts['sweep2_jobs']}$ top cells and "
        f"${counts['sweep2_leaves']}$ verified leaves; Huang Region~I, "
        f"${counts['region1_jobs']}$ band jobs, "
        f"${counts['region1_angular_leaves']}$ certified angular leaves, and "
        f"${counts['region1_radial_pieces']}$ radial pieces; Ding--Sun "
        f"Block~3a, ${counts['block3a_cells']}$ top cells and "
        f"${counts['block3a_leaves']}$ recursive leaves; the corrected "
        r"near-one block; and Block~3b/c at "
        f"$K_{{\\mathrm{{run}}}}={counts['k_run']}$, with "
        f"${b3['b_pos']['top_cells']}$ positive-branch top cells "
        f"(${b3['b_pos']['leaves']}$ leaves), "
        f"${b3['b_neg']['top_cells']}$ negative-branch top cells "
        f"(${b3['b_neg']['leaves']}$ leaves), and "
        f"${b3['c']['top_cells']}$ central top cells "
        f"(${b3['c']['leaves']}$ leaves), supported by a certified "
        f"${counts['k_nodes']}$-point $I''$ grid. The canonical JSON "
        "certificates and their complete raw record trees are included with "
        "the source."
    )


def replace_inventory(tex, inventory):
    if tex.count(START) != 1 or tex.count(END) != 1:
        raise ValueError("manuscript must contain exactly one inventory marker pair")
    start = tex.index(START) + len(START)
    end = tex.index(END)
    if start >= end:
        raise ValueError("manuscript inventory markers are inverted")
    return tex[:start] + "\n" + inventory + "\n" + tex[end:]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    counts = verified_counts()
    current = TEX.read_text(encoding="utf-8")
    expected = replace_inventory(current, render_inventory(counts))
    print(json.dumps(counts, sort_keys=True, indent=2))
    if args.check:
        if current != expected:
            raise SystemExit("paper inventory differs from verified certificates")
        print("PASS paper inventory matches verified certificates")
        return 0

    if current == expected:
        print("PASS paper inventory already current")
        return 0
    temporary = TEX.with_name(TEX.name + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(expected)
    os.replace(temporary, TEX)
    print(f"UPDATED {TEX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
