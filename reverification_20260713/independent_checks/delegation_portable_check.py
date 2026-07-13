"""Portable check of the Region-II delegation-chain geometry.

On a non-attested runtime, verify_all's delegation-chain row fails only
because its bundle check first replays the star-interior certificate, which
is runtime-bound by design.  This checker re-establishes the delegation
geometry portably, from the stored certificates alone, in exact rational
arithmetic (fractions; no python-flint, no floats):

  1. Sweep-1 root cells exactly tile the stage-1 domain rectangle EXCL
     (coordinate-compressed exact cover: every elementary cell covered
     exactly once, no overlaps, no gaps).
  2. Every stage-1 `delegate_sweep2` leaf rectangle lies inside the stage-2
     domain EXCL_OLD, mirroring verify_pair_components.
  3. Sweep-2 root cells exactly tile EXCL_OLD.
  4. Policy linkage: sweep2.policy.region1_star equals the Region-I star
     record field-for-field, and the sdot model and coefficients match,
     mirroring the Sweep2/Region-I policy comparison.

The one bundle condition NOT re-checked here is the star-interior norm link
(max Region-I leaf radius-box norm against the required-radius enclosure):
its margin is a few parts in 1e5 and its verification needs certified
trigonometric enclosures, which is exactly what Arb provides on the attested
runtime.  It passes there; a portable rational-trigonometry version is left
as future work.

Exit 0 with PASS lines on success; any failure raises.
"""

from __future__ import annotations

import json
import os
import sys
from fractions import Fraction


def frac(record) -> Fraction:
    if not isinstance(record, dict) or set(record) != {'num', 'den'}:
        raise ValueError('invalid rational record schema')
    num, den = int(record['num']), int(record['den'])
    if den <= 0 or (str(num), str(den)) != (record['num'], record['den']):
        raise ValueError('rational record is not canonical')
    out = Fraction(num, den)
    if (out.numerator, out.denominator) != (num, den):
        raise ValueError('rational record is not reduced')
    return out


def rect_of(cell):
    r = tuple(frac(v) for v in cell)
    if not (r[0] < r[1] and r[2] < r[3]):
        raise ValueError('empty or inverted rectangle')
    return r


def collect_delegates(node, out):
    kind = node['kind']
    if kind == 'split':
        for child in node['children']:
            collect_delegates(child, out)
    elif kind in ('delegate_sweep2', 'delegate_region1'):
        out.append(rect_of(node['cell']))


def exact_cover(rects, domain, label, hole=None):
    """Rectangles are disjoint, inside `domain`, and cover it exactly --
    except that uncovered elementary cells, if any, must lie inside `hole`.
    Returns the exact uncovered area."""
    for r in rects:
        if not (domain[0] <= r[0] and r[1] <= domain[1]
                and domain[2] <= r[2] and r[3] <= domain[3]):
            raise ValueError(f'{label}: rectangle escapes the domain')
    xs = sorted({v for r in rects for v in (r[0], r[1])}
                | {domain[0], domain[1]})
    ys = sorted({v for r in rects for v in (r[2], r[3])}
                | {domain[2], domain[3]})
    x_index = {v: i for i, v in enumerate(xs)}
    y_index = {v: i for i, v in enumerate(ys)}
    counts = [[0] * (len(ys) - 1) for _ in range(len(xs) - 1)]
    for r in rects:
        for i in range(x_index[r[0]], x_index[r[1]]):
            row = counts[i]
            for j in range(y_index[r[2]], y_index[r[3]]):
                row[j] += 1
    uncovered_area = Fraction(0)
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            c = counts[i][j]
            if c > 1:
                raise ValueError(
                    f'{label}: elementary cell '
                    f'[{xs[i]},{xs[i+1]}]x[{ys[j]},{ys[j+1]}] covered '
                    f'{c} times')
            if c == 0:
                if hole is None or not (
                        hole[0] <= xs[i] and xs[i + 1] <= hole[1]
                        and hole[2] <= ys[j] and ys[j + 1] <= hole[3]):
                    raise ValueError(
                        f'{label}: uncovered cell '
                        f'[{xs[i]},{xs[i+1]}]x[{ys[j]},{ys[j+1]}] outside '
                        f'the permitted hole')
                uncovered_area += (xs[i + 1] - xs[i]) * (ys[j + 1] - ys[j])
    area = sum((r[1] - r[0]) * (r[3] - r[2]) for r in rects)
    if area + uncovered_area != ((domain[1] - domain[0])
                                 * (domain[3] - domain[2])):
        raise ValueError(f'{label}: area accounting mismatch')
    print(f'PASS {label}: {len(rects)} disjoint cells; uncovered area '
          f'{uncovered_area} confined to the permitted hole '
          f'({len(xs)-1}x{len(ys)-1} elementary grid)')
    return uncovered_area


def main():
    results = sys.argv[1] if len(sys.argv) > 1 else 'results'
    s1 = json.load(open(os.path.join(results, 'huang_sweep.json')))
    s2 = json.load(open(os.path.join(results, 'huang_sweep2.json')))
    r1 = json.load(open(os.path.join(results, 'huang_region1.json')))

    # 1. stage-1 root cells: disjoint, inside [-A1MAX,A1MAX]x[-A2MAX,A2MAX],
    #    uncovered region confined to the EXCL hole around the maximizer
    a1max = frac(s1['policy']['A1MAX'])
    a2max = frac(s1['policy']['A2MAX'])
    dom1 = (-a1max, a1max, -a2max, a2max)
    excl = tuple(frac(v) for v in s1['policy']['EXCL'])
    roots1 = [rect_of(rec['cell']) for rec in s1['records']]
    exact_cover(roots1, dom1, 'sweep1 root cover of R minus EXCL', hole=excl)

    # 2. delegated stage-1 rectangles lie inside EXCL_OLD
    dom_old = tuple(frac(v) for v in s2['policy']['EXCL_OLD'])
    delegates1 = []
    for rec in s1['records']:
        collect_delegates(rec['tree'], delegates1)
    for r in delegates1:
        if not (dom_old[0] <= r[0] and r[1] <= dom_old[1]
                and dom_old[2] <= r[2] and r[3] <= dom_old[3]):
            raise ValueError('sweep1 delegation escapes sweep2 domain')
    print(f'PASS sweep1 delegation containment: {len(delegates1)} delegated '
          f'rectangles inside EXCL_OLD')

    # 3. stage-2 root cells tile EXCL_OLD
    roots2 = [rect_of(rec['cell']) for rec in s2['records']]
    exact_cover(roots2, dom_old, 'sweep2 root tiling of EXCL_OLD')

    # 4. policy linkage with Region-I
    star_fields = sorted(s2['policy']['region1_star'])
    linked = {name: r1['star'][name] for name in star_fields}
    if s2['policy']['region1_star'] != linked:
        raise ValueError('sweep2/region1 star policy mismatch')
    if (s2['policy']['region1_sdot_model']
            != r1['certificate_policy']['sdot_model']
            or s2['policy']['region1_sdot_coefficients']
            != r1['certificate_policy']['sdot_coefficients']):
        raise ValueError('sweep2/region1 sdot policy mismatch')
    print(f'PASS sweep2/region1 policy linkage: star fields {star_fields} '
          f'and sdot model/coefficients agree')

    delegates2 = []
    for rec in s2['records']:
        collect_delegates(rec['tree'], delegates2)
    print(f'INFO sweep2 delegates {len(delegates2)} rectangles to Region I; '
          f'their per-leaf star witnesses are checked by the stage-2 '
          f'verifier, and the star-interior norm link is attested-runtime '
          f'scope (see module docstring)')
    print('DELEGATION GEOMETRY PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
