"""Exact tree verifier for Huang Region-II sweep certificates."""

from __future__ import annotations

import argparse
import math
import pathlib
import re
from fractions import Fraction

import block3bc_exact as exact
import huang_sweep as sweep1
import huang_sweep2 as sweep2
import huanggrid as hg
from core import dec, set_prec


_RUN_ID_RE = re.compile(r'[0-9a-f]{32}\Z')
_OUTSIDE_FAN_CACHE = {}
BUNDLE_SCHEMA_VERSION = 2
BUNDLE_KIND = 'huang_condition_certificate_bundle'
BUNDLE_EVIDENCE_MODEL = 'four-artifact-exact-delegation-v2'
_BUNDLE_FILES = {
    'star_interior': 'huang_star_interior.json',
    'region1': 'huang_region1.json',
    'sweep1': 'huang_sweep.json',
    'sweep2': 'huang_sweep2.json',
}


def _fraction(record):
    return exact.fraction_from_record(record)


def _packet(record):
    return exact.packet_fraction_endpoints(record)


def _strict_equal(left, right):
    return exact.canonical_json_bytes(left) == exact.canonical_json_bytes(right)


def _cell(records):
    values = tuple(_fraction(value) for value in records)
    if len(values) != 4 or not values[0] < values[1] or not values[2] < values[3]:
        raise ValueError('invalid rectangle geometry')
    return values


def _negative(packet):
    if _packet(packet)[1] >= 0:
        raise ValueError('rectangle value packet is not strictly negative')


def _bounded_decimal_choice(record, digits):
    value = _fraction(record)
    if (value * 10 ** digits).denominator != 1:
        raise ValueError('majorant choice exceeds its decimal precision')
    return value


def _replay_negative_leaf(tree, rectangle):
    """Independently rebuild the fixed-choice Arb majorant for one leaf."""
    from flint import arb
    from core import ALPHA, PSI, endpoints
    import huang_hessian as hh

    witness = tree['majorant_witness']
    method = tree['method']
    if not isinstance(witness, dict):
        raise ValueError('negative majorant witness is not an object')
    decimals = (sweep1.DIRECT_CHOICE_DIGITS if method == 'direct'
                else sweep1.MEAN_VALUE_CHOICE_DIGITS)
    s_value = _bounded_decimal_choice(witness.get('tilt_s'), decimals)
    if not sweep1.TILT_MIN <= s_value <= sweep1.TILT_MAX:
        raise ValueError('negative majorant tilt is outside its domain')
    s = exact.fraction_arb(s_value)
    a1lo, a1hi, a2lo, a2hi = (
        exact.fraction_arb(value) for value in rectangle)
    cc1, rr1 = (a1lo + a1hi) / 2, (a1hi - a1lo) / 2
    cc2, rr2 = (a2lo + a2hi) / 2, (a2hi - a2lo) / 2
    a1b = a1lo.union(a1hi)
    a2b = a2lo.union(a2hi)

    if method == 'direct':
        dual_mode = witness.get('dual_mode')
        if dual_mode == 'fixed_tangent':
            if set(witness) != {
                    'dual_mode', 'lambda1', 'lambda2', 'tilt_s'}:
                raise ValueError('fixed-tangent witness schema mismatch')
            b1_value = _bounded_decimal_choice(
                witness['lambda1'], sweep1.DIRECT_CHOICE_DIGITS)
            b2_value = _bounded_decimal_choice(
                witness['lambda2'], sweep1.DIRECT_CHOICE_DIGITS)
            if (abs(b1_value) > sweep1.DIRECT_LAMBDA_ABS_MAX
                    or abs(b2_value) > sweep1.DIRECT_LAMBDA_ABS_MAX):
                raise ValueError('fixed-tangent choice exceeds proof policy')
            b1 = exact.fraction_arb(b1_value)
            b2 = exact.fraction_arb(b2_value)
            hub = hg.Phi_of(b1, b2) - b1 * a1b - b2 * a2b
        elif dual_mode == 'entropy_cap':
            if set(witness) != {'dual_mode', 'tilt_s'}:
                raise ValueError('entropy-cap witness schema mismatch')
            hub = arb(0).union(hg.LOG2)
        else:
            raise ValueError('unknown fixed majorant dual mode')
        t_bound = hg.T_meanvalue(cc1, cc2, rr1, rr2, s)
        if t_bound is None:
            raise ValueError('direct majorant replay left the T domain')
        value = hub + s * s * PSI / 2 + ALPHA * t_bound
    elif method == 'mean_value':
        if set(witness) != {'lambda1', 'lambda2', 'tilt_s'}:
            raise ValueError('mean-value witness schema mismatch')
        b1_value = _bounded_decimal_choice(
            witness['lambda1'], sweep1.MEAN_VALUE_CHOICE_DIGITS)
        b2_value = _bounded_decimal_choice(
            witness['lambda2'], sweep1.MEAN_VALUE_CHOICE_DIGITS)
        if (abs(b1_value) >= sweep1.MEAN_VALUE_LAMBDA_ABS_STRICT_MAX
                or abs(b2_value)
                >= sweep1.MEAN_VALUE_LAMBDA_ABS_STRICT_MAX):
            raise ValueError('mean-value tangent choice exceeds proof policy')
        b1 = exact.fraction_arb(b1_value)
        b2 = exact.fraction_arb(b2_value)
        center_t = hg.T_of(cc1, cc2, s)
        if center_t is None:
            raise ValueError('mean-value replay left the T domain')
        center = (hg.Phi_of(b1, b2) - b1 * cc1 - b2 * cc2
                  + s * s * PSI / 2 + ALPHA * center_t)
        derivatives = hh._T_derivs(a1b, a2b, hg.get_zt_grid(), s)
        if derivatives is None:
            raise ValueError('mean-value replay derivative failed')
        g1 = -b1 + ALPHA * derivatives['dT1']
        g2 = -b2 + ALPHA * derivatives['dT2']
        _, g1_abs_upper = endpoints(abs(g1))
        _, g2_abs_upper = endpoints(abs(g2))
        slack = g1_abs_upper * rr1 + g2_abs_upper * rr2
        value = center + slack
    else:
        raise ValueError('unknown negative majorant method')
    expected = exact.arb_packet(value)
    if not _strict_equal(tree['value_packet'], expected):
        raise ValueError('negative majorant packet does not replay')
    if not value < 0:
        raise ValueError('replayed majorant is not strictly negative')


def _outside_witness(witness, rectangle):
    from flint import ctx

    required = {
        'fan_index', 'sign', 'u_packet', 'v_packet',
        'support_upper_packet', 'box_min_packet'}
    if (not isinstance(witness, dict) or set(witness) != required
            or not isinstance(witness['fan_index'], int)
            or isinstance(witness['fan_index'], bool)
            or not 0 <= witness['fan_index'] < 360
            or witness['sign'] not in (-1, 1)
            or isinstance(witness['sign'], bool)):
        raise ValueError('outside-K witness schema mismatch')
    fan_index = witness['fan_index']
    sign = witness['sign']
    cache_key = (fan_index, ctx.prec, hg.GRID_N)
    if cache_key not in _OUTSIDE_FAN_CACHE:
        angle = math.pi * fan_index / 360
        u = dec(str(round(math.cos(angle), 8)))
        v = dec(str(round(math.sin(angle), 8)))
        support = hg.h_support_upper(u, v)
        _OUTSIDE_FAN_CACHE[cache_key] = (u, v, support)
    u, v, support = _OUTSIDE_FAN_CACHE[cache_key]
    su, sv = (u, v) if sign == 1 else (-u, -v)
    a1lo, a1hi, a2lo, a2hi = (
        dec(str(float(value))) for value in rectangle)
    box_min = hg._corner_min(su, sv, a1lo, a1hi, a2lo, a2hi)
    expected = {
        'u_packet': exact.arb_packet(su),
        'v_packet': exact.arb_packet(sv),
        'support_upper_packet': exact.arb_packet(support),
        'box_min_packet': exact.arb_packet(box_min),
    }
    if any(not _strict_equal(witness[name], value)
           for name, value in expected.items()):
        raise ValueError('outside-K witness is not canonical for its rectangle')
    support_lo, support_hi = _packet(witness['support_upper_packet'])
    min_lo, _ = _packet(witness['box_min_packet'])
    if support_lo < 0 or min_lo <= support_hi:
        raise ValueError('outside-K support separation is not strict')


def _star_witness(witness, rectangle):
    if not isinstance(witness, dict):
        raise ValueError('star witness is not an object')
    expected = sweep2.star_witness(*(float(value) for value in rectangle))
    if expected is None or not _strict_equal(witness, expected):
        raise ValueError('star witness is not canonical for its rectangle')
    mode = witness.get('mode')
    if mode == 'core_radius':
        if set(witness) != {'mode', 'radius_max_packet', 'radius_limit'}:
            raise ValueError('core star witness schema mismatch')
        expected_limit = sweep1._float_fraction(sweep2.T_CORE - sweep2.SAFE)
    elif mode in ('long_cone', 'mid_cone'):
        required = {
            'mode', 'axis_sign', 'half_angle', 'radius_max_packet',
            'radius_limit', 'projection_min_packet', 'left_cone_packet',
            'right_cone_packet'}
        if (set(witness) != required or witness['axis_sign'] not in (-1, 1)
                or isinstance(witness['axis_sign'], bool)):
            raise ValueError('cone star witness schema mismatch')
        if mode == 'long_cone':
            expected_half = sweep1._float_fraction(
                sweep2.WEDGE_HALF - 0.012 - sweep2.SAFE)
            expected_limit = sweep1._float_fraction(
                sweep2.T_LONG - sweep2.SAFE)
        else:
            expected_half = sweep1._float_fraction(
                sweep2.CONE_MID - 0.030 - sweep2.SAFE)
            expected_limit = sweep1._float_fraction(
                sweep2.T_MID - sweep2.SAFE)
        if _fraction(witness['half_angle']) != expected_half:
            raise ValueError('star cone half-angle policy mismatch')
        for name in ('projection_min_packet', 'left_cone_packet',
                     'right_cone_packet'):
            if _packet(witness[name])[0] <= 0:
                raise ValueError('star cone guard is not strictly positive')
    else:
        raise ValueError('unknown star witness mode')
    if _fraction(witness['radius_limit']) != expected_limit:
        raise ValueError('star radius policy mismatch')
    radius_lo, radius_hi = _packet(witness['radius_max_packet'])
    if radius_lo < 0 or radius_hi >= expected_limit:
        raise ValueError('star radius guard is not strict')


def _validate_tree(tree, rectangle, stage, delegates):
    if not isinstance(tree, dict) or _cell(tree.get('cell')) != rectangle:
        raise ValueError('rectangle proof node geometry mismatch')
    kind = tree.get('kind')
    if kind == 'split':
        if set(tree) != {'kind', 'cell', 'axis', 'split_at', 'children'}:
            raise ValueError('rectangle split schema mismatch')
        axis = tree['axis']
        if axis not in (1, 2) or isinstance(axis, bool):
            raise ValueError('rectangle split axis mismatch')
        split = _fraction(tree['split_at'])
        lower, upper = ((rectangle[0], rectangle[1]) if axis == 1
                        else (rectangle[2], rectangle[3]))
        if (not lower < split < upper or not isinstance(tree['children'], list)
                or len(tree['children']) != 2):
            raise ValueError('rectangle split point mismatch')
        if axis == 1:
            children = ((rectangle[0], split, rectangle[2], rectangle[3]),
                        (split, rectangle[1], rectangle[2], rectangle[3]))
        else:
            children = ((rectangle[0], rectangle[1], rectangle[2], split),
                        (rectangle[0], rectangle[1], split, rectangle[3]))
        left = _validate_tree(tree['children'][0], children[0], stage, delegates)
        right = _validate_tree(tree['children'][1], children[1], stage, delegates)
        return tuple(a + b for a, b in zip(left, right))
    if kind == 'negative':
        if (set(tree) != {
                'kind', 'cell', 'method', 'majorant_witness', 'value_packet'}
                or tree['method'] not in ('direct', 'mean_value')):
            raise ValueError('negative rectangle leaf schema mismatch')
        _negative(tree['value_packet'])
        _replay_negative_leaf(tree, rectangle)
        return 1, 0, 0
    if kind == 'outside_K':
        if set(tree) != {'kind', 'cell', 'witness'}:
            raise ValueError('outside-K rectangle leaf schema mismatch')
        _outside_witness(tree['witness'], rectangle)
        return 0, 1, 0
    if stage == 1 and kind == 'delegate_sweep2':
        if set(tree) != {'kind', 'cell'}:
            raise ValueError('Sweep1 delegation leaf schema mismatch')
        delegates.append(rectangle)
        return 0, 0, 1
    if stage == 2 and kind == 'delegate_region1':
        if set(tree) != {'kind', 'cell', 'witness'}:
            raise ValueError('Sweep2 delegation leaf schema mismatch')
        _star_witness(tree['witness'], rectangle)
        delegates.append(rectangle)
        return 0, 0, 1
    raise ValueError('failed/unknown rectangle leaf')


def _expected_policy(stage):
    if stage == 1:
        return {
            'coarse': 48,
            'A1MAX': sweep1._float_record(sweep1.A1MAX),
            'A2MAX': sweep1._float_record(sweep1.A2MAX),
            'EXCL': [sweep1._float_record(x) for x in sweep1.EXCL],
            'MIN_SIDE': sweep1._float_record(sweep1.MIN_SIDE),
            'huang_grid_n': sweep1.GRID_N,
            'majorant_witness': sweep1.majorant_witness_policy(),
        }
    _, n1, n2 = sweep2.build_jobs()
    return {
        'n1': n1,
        'n2': n2,
        'EXCL_OLD': [sweep1._float_record(x) for x in sweep2.EXCL_OLD],
        'MIN_SIDE': sweep1._float_record(sweep2.MIN_SIDE),
        'SAFE': sweep1._float_record(sweep2.SAFE),
        'huang_grid_n': sweep1.GRID_N,
        'majorant_witness': sweep1.majorant_witness_policy(),
        'region1_star': sweep2.region1_star_policy(),
        'region1_sdot_model': sweep2.r1.SDOT_MODEL,
        'region1_sdot_coefficients': [
            sweep2.r1._decimal_record(text)
            for text in sweep2.r1.SDOT_C_TEXT],
    }


def _validate_top_schedule(schedule, stage):
    if stage == 1:
        n1 = 48
        if len(schedule) % n1:
            raise ValueError('Sweep1 top schedule is not rectangular')
        n2 = len(schedule) // n1
        bounds = tuple(sweep1._float_fraction(value) for value in
                       (-sweep1.A1MAX, sweep1.A1MAX,
                        -sweep1.A2MAX, sweep1.A2MAX))
    else:
        _, n1, n2 = sweep2.build_jobs()
        bounds = tuple(sweep1._float_fraction(value)
                       for value in sweep2.EXCL_OLD)
    if len(schedule) != n1 * n2:
        raise ValueError('sweep top schedule has the wrong dimensions')
    cells = []
    for index, record in enumerate(schedule):
        if (not isinstance(record, dict)
                or set(record) != {'index', 'cell'}
                or not isinstance(record['index'], int)
                or isinstance(record['index'], bool)
                or record['index'] != index):
            raise ValueError('sweep top schedule record schema mismatch')
        cells.append(_cell(record['cell']))
    if ((cells[0][0], cells[-1][1], cells[0][2], cells[n2 - 1][3])
            != bounds):
        raise ValueError('sweep top schedule outer boundary mismatch')
    for i in range(n1):
        for j in range(n2):
            cell = cells[i * n2 + j]
            if (cell[0:2] != cells[i * n2][0:2]
                    or cell[2:4] != cells[j][2:4]):
                raise ValueError('sweep top schedule is not a tensor grid')
            if j + 1 < n2 and cell[3] != cells[i * n2 + j + 1][2]:
                raise ValueError('sweep top schedule has a vertical gap')
            if i + 1 < n1 and cell[1] != cells[(i + 1) * n2 + j][0]:
                raise ValueError('sweep top schedule has a horizontal gap')


def verify_certificate(path, stage):
    if stage not in (1, 2):
        raise ValueError('unknown sweep stage')
    set_prec(50)
    module = sweep1 if stage == 1 else sweep2
    data = exact.load_json(exact.require_plain_regular_file(path))
    required = {
        'schema_version', 'kind', 'evidence_model', 'run_id',
        'source_sha256', 'source_sha256_after', 'source_set_sha256',
        'runtime', 'policy', 'schedule', 'schedule_sha256', 'records',
        'derived_summary', 'certificate_sha256'}
    if stage == 1:
        required.add('domain_guards')
    expected_kind = f'huang_sweep{stage}_certificate'
    if (not isinstance(data, dict) or set(data) != required
            or data['schema_version'] != module.SCHEMA_VERSION
            or data['kind'] != expected_kind
            or data['evidence_model'] != module.EVIDENCE_MODEL
            or not isinstance(data['run_id'], str)
            or _RUN_ID_RE.fullmatch(data['run_id']) is None
            or data['certificate_sha256'] != exact.payload_sha256(
                data, omit=('certificate_sha256',))):
        raise ValueError('invalid sweep certificate schema/identity')
    if hg.GRID_N != sweep1.GRID_N:
        raise ValueError('Huang grid resolution does not match proof policy')
    sources = exact.source_hashes(module.proof_source_paths())
    if (data['source_sha256'] != sources
            or data['source_sha256_after'] != sources
            or data['source_set_sha256']
            != exact.payload_sha256(sources, omit=())):
        raise ValueError('stale sweep proof source')
    exact.validate_runtime_record(data['runtime'], 50, allow_any_workers=True)
    if not _strict_equal(data['policy'], _expected_policy(stage)):
        raise ValueError('sweep policy mismatch')
    schedule = (sweep1.schedule_records(48) if stage == 1
                else sweep2.schedule_records())
    if (not _strict_equal(data['schedule'], schedule)
            or data['schedule_sha256'] != exact.payload_sha256(
                schedule, omit=())):
        raise ValueError('sweep schedule mismatch')
    _validate_top_schedule(schedule, stage)
    records = data['records']
    if (not isinstance(records, list) or len(records) != len(schedule)
            or any(not isinstance(row.get('index'), int)
                   or isinstance(row.get('index'), bool)
                   for row in records)
            or [row.get('index') for row in records]
            != list(range(len(schedule)))):
        raise ValueError('sweep record index mismatch')
    totals = (0, 0, 0)
    delegates = []
    for target, row in zip(schedule, records):
        required_row = {
            'index', 'cell', 'verdict', 'tree', 'runtime_milliseconds',
            'worker_source_sha256_before', 'worker_source_sha256_after',
            'worker_runtime', 'run_id'}
        if (not isinstance(row, dict) or set(row) != required_row
                or not _strict_equal(row['cell'], target['cell'])
                or row['verdict'] != 'PASS'
                or row['run_id'] != data['run_id']
                or row['worker_source_sha256_before'] != sources
                or row['worker_source_sha256_after'] != sources
                or not isinstance(row['runtime_milliseconds'], int)
                or isinstance(row['runtime_milliseconds'], bool)
                or row['runtime_milliseconds'] < 0):
            raise ValueError('invalid sweep job record')
        exact.validate_runtime_record(row['worker_runtime'], 50, workers=1)
        if (exact.runtime_identity(row['worker_runtime'])
                != exact.runtime_identity(data['runtime'])):
            raise ValueError('sweep worker/producer runtime mismatch')
        count = _validate_tree(row['tree'], _cell(row['cell']), stage, delegates)
        totals = tuple(a + b for a, b in zip(totals, count))
    summary = {
        'jobs': len(records),
        'failures': 0,
        'negative_leaves': totals[0],
        'outside_K_leaves': totals[1],
        ('delegated_leaves' if stage == 1 else
         'region1_delegated_leaves'): totals[2],
    }
    if not _strict_equal(data['derived_summary'], summary):
        raise ValueError('sweep derived summary mismatch')
    if stage == 1:
        guards = data['domain_guards']
        if not isinstance(guards, dict) or set(guards) != {
                'h10_packet', 'h10_limit', 'h01_packet', 'h01_limit',
                'degenerate_identity'}:
            raise ValueError('sweep domain guard schema mismatch')
        h10_limit = _fraction(guards['h10_limit'])
        h01_limit = _fraction(guards['h01_limit'])
        if (not _strict_equal(guards['h10_limit'], data['policy']['A1MAX'])
                or not _strict_equal(
                    guards['h01_limit'], data['policy']['A2MAX'])
                or _packet(guards['h10_packet'])[1] >= h10_limit
                or _packet(guards['h01_packet'])[1] >= h01_limit
                or guards['degenerate_identity']
                != 'huang-analytic-Sstar(1,0)=0'):
            raise ValueError('sweep domain guard failed')
        expected_h10 = exact.arb_packet(hg.h_support_upper(1, 0))
        expected_h01 = exact.arb_packet(hg.h_support_upper(0, 1))
        if (guards['h10_packet'] != expected_h10
                or guards['h01_packet'] != expected_h01):
            raise ValueError('sweep domain guards are not canonical')
    return data, delegates


def verify_pair_components(
        first, delegated, second, region1_delegated, region1):
    domain = tuple(_fraction(value) for value in second['policy']['EXCL_OLD'])
    for rectangle in delegated:
        if not (domain[0] <= rectangle[0] < rectangle[1] <= domain[1]
                and domain[2] <= rectangle[2] < rectangle[3] <= domain[3]):
            raise ValueError('Sweep1 delegation escapes Sweep2 domain')
    region1_star = {
        name: region1['star'][name]
        for name in sweep2._REGION1_STAR_FIELDS}
    if (not _strict_equal(
            second['policy']['region1_star'], region1_star)
            or second['policy']['region1_sdot_model']
            != region1['certificate_policy']['sdot_model']
            or not _strict_equal(
                second['policy']['region1_sdot_coefficients'],
                region1['certificate_policy']['sdot_coefficients'])):
        raise ValueError('Sweep2/Region-I policy mismatch')
    return first, second, region1, delegated, region1_delegated


def verify_pair(stage1_path, stage2_path, region1_path):
    first, delegated = verify_certificate(stage1_path, 1)
    second, region1_delegated = verify_certificate(stage2_path, 2)
    import huang_region1_verify
    region1 = huang_region1_verify.verify_certificate(region1_path)
    return verify_pair_components(
        first, delegated, second, region1_delegated, region1)


def _certified_region1_leaves(node):
    kind = node.get('kind')
    if kind == 'split':
        for child in node['children']:
            yield from _certified_region1_leaves(child)
    elif kind == 'certified':
        yield node


def _verify_star_interior_link(star_interior, region1):
    """Prove that the replayed Region-I interval boxes fit in the inball.

    Region-I stores independent sine/cosine and radius interval boxes, so the
    geometric ray radius alone is not a sufficient bound.  Rebuild a norm
    bound from every certified leaf packet and compare its maximum with the
    independently replayed star-interior required-radius enclosure.
    """
    from flint import arb

    set_prec(100)
    try:
        policy = star_interior['policy']
        bounds = star_interior['bounds']
        origin = _fraction(region1['star']['origin_radius'])
        if (Fraction(policy['star_radius'])
                != _fraction(region1['star']['T_LONG'])
                or Fraction(policy['origin_coordinate_radius']) != origin
                or policy['center_interval_diameter_factor'] != 2):
            raise ValueError('star-interior/Region-I geometry policy mismatch')
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith('star-interior/'):
            raise
        raise ValueError('invalid star-interior/Region-I geometry policy') from exc

    center_bound = 2 * arb(2).sqrt() * exact.fraction_arb(origin)
    maximum_packet = None
    maximum_hi = None
    leaf_count = 0
    piece_count = 0
    for row in region1.get('records', []):
        for root in row.get('angular_roots', []):
            for leaf in _certified_region1_leaves(root):
                leaf_count += 1
                theta = exact.packet_ball(leaf['theta_box'])
                cos_upper = exact.packet_abs_upper(
                    exact.arb_packet(theta.cos()))
                sin_upper = exact.packet_abs_upper(
                    exact.arb_packet(theta.sin()))
                direction_norm = (
                    cos_upper * cos_upper + sin_upper * sin_upper).sqrt()
                for piece in leaf['radial_pieces']:
                    piece_count += 1
                    radius_upper = exact.packet_abs_upper(piece['radius_box'])
                    candidate = radius_upper * direction_norm + center_bound
                    packet = exact.arb_packet(candidate)
                    _, candidate_hi = exact.packet_fraction_endpoints(packet)
                    if maximum_hi is None or candidate_hi > maximum_hi:
                        maximum_packet = packet
                        maximum_hi = candidate_hi
    if maximum_packet is None or leaf_count == 0 or piece_count == 0:
        raise ValueError('Region-I has no certified interval boxes')

    _, required_hi = exact.packet_fraction_endpoints(
        bounds['required_radius'])
    inradius_lo, _ = exact.packet_fraction_endpoints(bounds['inradius'])
    if maximum_hi > required_hi or required_hi >= inradius_lo:
        raise ValueError('Region-I interval boxes escape the certified inball')
    return {
        'certified_angular_leaves': leaf_count,
        'radial_pieces': piece_count,
        'region1_interval_box_radius_bound': maximum_packet,
        'star_interior_required_radius': bounds['required_radius'],
        'certified_clearance_lower': exact.fraction_record(
            inradius_lo - maximum_hi),
    }


def _artifact_identity(filename, data):
    identity = {
        'file': filename,
        'schema_version': data['schema_version'],
        'kind': data['kind'],
        'certificate_sha256': data['certificate_sha256'],
    }
    for name in ('evidence_model', 'run_id', 'source_set_sha256'):
        if name in data:
            identity[name] = data[name]
    if 'source_sha256' in data:
        identity['source_identity_sha256'] = exact.payload_sha256(
            data['source_sha256'], omit=())
    return identity


def bundle_payload(
        star_interior, first, second, region1,
        delegated, region1_delegated):
    import huang_region1_verify

    payload = {
        'schema_version': BUNDLE_SCHEMA_VERSION,
        'kind': BUNDLE_KIND,
        'evidence_model': BUNDLE_EVIDENCE_MODEL,
        'verifier_source_sha256': exact.source_hashes({
            'huang_region1_verify.py': huang_region1_verify.__file__,
            'huang_sweep_verify.py': __file__,
        }),
        'artifacts': {
            'star_interior': _artifact_identity(
                _BUNDLE_FILES['star_interior'], star_interior),
            'region1': _artifact_identity(_BUNDLE_FILES['region1'], region1),
            'sweep1': _artifact_identity(_BUNDLE_FILES['sweep1'], first),
            'sweep2': _artifact_identity(_BUNDLE_FILES['sweep2'], second),
        },
        'delegation_summary': {
            'sweep1_to_sweep2_rectangles': len(delegated),
            'sweep2_to_region1_rectangles': len(region1_delegated),
        },
        'interior_geometry_link': _verify_star_interior_link(
            star_interior, region1),
    }
    payload['bundle_sha256'] = exact.payload_sha256(
        payload, omit=('bundle_sha256',))
    return payload


def write_bundle(
        output, star_interior_path, stage1_path, stage2_path, region1_path):
    output = pathlib.Path(output).absolute()
    if output.name != 'huang_bundle.json':
        raise ValueError('canonical Huang bundle filename is required')
    root = exact.require_plain_directory(output.parent)
    if output.exists():
        exact.require_plain_regular_file(output)
    supplied = {
        'star_interior': star_interior_path,
        'region1': region1_path,
        'sweep1': stage1_path,
        'sweep2': stage2_path,
    }
    resolved = {
        name: exact.resolve_regular_file_under(root, filename)
        for name, filename in _BUNDLE_FILES.items()
    }
    for name, supplied_path in supplied.items():
        if exact.require_plain_regular_file(supplied_path) != resolved[name]:
            raise ValueError(
                'Huang bundle inputs must be canonical sibling artifacts')
    import huang_star_interior

    star_interior = huang_star_interior.verify_certificate(
        resolved['star_interior'])
    components = verify_pair(
        resolved['sweep1'], resolved['sweep2'], resolved['region1'])
    payload = bundle_payload(
        star_interior, components[0], components[1], components[2],
        components[3], components[4])
    exact.write_json_atomic(output, payload)
    return payload


def verify_bundle(path, prevalidated=None):
    path = exact.require_plain_regular_file(path)
    if path.name != 'huang_bundle.json':
        raise ValueError('canonical Huang bundle filename is required')
    data = exact.load_json(path)
    required = {
        'schema_version', 'kind', 'evidence_model', 'artifacts',
        'verifier_source_sha256', 'delegation_summary',
        'interior_geometry_link', 'bundle_sha256'}
    if (not isinstance(data, dict) or set(data) != required
            or data['schema_version'] != BUNDLE_SCHEMA_VERSION
            or data['kind'] != BUNDLE_KIND
            or data['evidence_model'] != BUNDLE_EVIDENCE_MODEL
            or data['bundle_sha256'] != exact.payload_sha256(
                data, omit=('bundle_sha256',))):
        raise ValueError('invalid Huang bundle schema/identity')
    artifacts = data['artifacts']
    if (not isinstance(artifacts, dict)
            or set(artifacts) != set(_BUNDLE_FILES)):
        raise ValueError('invalid Huang bundle artifact map')
    root = exact.require_plain_directory(path.parent)
    resolved = {}
    for name, filename in _BUNDLE_FILES.items():
        identity = artifacts[name]
        if not isinstance(identity, dict) or identity.get('file') != filename:
            raise ValueError('Huang bundle artifact filename mismatch')
        resolved[name] = exact.resolve_regular_file_under(root, filename)
    if prevalidated is None:
        import huang_star_interior

        star_interior = huang_star_interior.verify_certificate(
            resolved['star_interior'])
        components = verify_pair(
            resolved['sweep1'], resolved['sweep2'], resolved['region1'])
    elif (not isinstance(prevalidated, tuple) or len(prevalidated) != 2
            or not isinstance(prevalidated[1], tuple)
            or len(prevalidated[1]) != 5):
        raise ValueError('prevalidated Huang bundle components are invalid')
    else:
        star_interior, components = prevalidated
        prevalidated_artifacts = {
            'star_interior': star_interior,
            'region1': components[2],
            'sweep1': components[0],
            'sweep2': components[1],
        }
        for name, expected_artifact in prevalidated_artifacts.items():
            if not _strict_equal(
                    exact.load_json(resolved[name]), expected_artifact):
                raise ValueError(
                    'prevalidated Huang artifact differs from disk')
    expected = bundle_payload(
        star_interior, components[0], components[1], components[2],
        components[3], components[4])
    if not _strict_equal(data, expected):
        raise ValueError('Huang bundle does not bind its verified artifacts')
    return data, (star_interior, components)


def _main(argv=None):
    parser = argparse.ArgumentParser(
        description='Build or verify the canonical Huang certificate bundle')
    subparsers = parser.add_subparsers(dest='command', required=True)
    build = subparsers.add_parser('bundle')
    build.add_argument('--output', required=True)
    build.add_argument('--star-interior', required=True)
    build.add_argument('--region1', required=True)
    build.add_argument('--sweep1', required=True)
    build.add_argument('--sweep2', required=True)
    check = subparsers.add_parser('verify')
    check.add_argument('path')
    args = parser.parse_args(argv)
    if args.command == 'bundle':
        write_bundle(
            args.output, args.star_interior,
            args.sweep1, args.sweep2, args.region1)
    else:
        verify_bundle(args.path)


if __name__ == '__main__':
    _main()
