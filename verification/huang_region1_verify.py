"""Structural and independent full-replay verifier for Huang Region I."""

from __future__ import annotations

import argparse
import math
import multiprocessing
import re
from fractions import Fraction

import block3bc_exact as exact
import huang_region1 as r1


_RUN_ID_RE = re.compile(r'[0-9a-f]{32}\Z')


def _fraction(record):
    return exact.fraction_from_record(record)


def _packet(record):
    return exact.packet_fraction_endpoints(record)


def _strict_equal(left, right):
    return exact.canonical_json_bytes(left) == exact.canonical_json_bytes(right)


def _positive(record, *, strict=True):
    lo, _ = _packet(record)
    if lo < 0 or (strict and lo == 0):
        raise ValueError('packet does not prove positivity')


def _negative(record):
    _, hi = _packet(record)
    if hi >= 0:
        raise ValueError('packet does not prove strict negativity')


def _contains(record, value):
    lo, hi = _packet(record)
    if not lo <= value <= hi:
        raise ValueError('packet misses its exact endpoint')


def _float(value):
    return float(value.numerator / value.denominator)


def _canonical_tmax(lo, hi):
    return r1._float_fraction(r1.T_max_over(_float(lo), _float(hi)))


def _complete_prefix_tree(paths, max_depth):
    paths = set(paths)
    if not paths:
        raise ValueError('empty binary leaf cover')
    if any(not isinstance(path, str) or set(path) - {'0', '1'}
           or len(path) > max_depth for path in paths):
        raise ValueError('invalid binary leaf path')

    def visit(prefix):
        if prefix in paths:
            if any(other.startswith(prefix) and other != prefix
                   for other in paths):
                raise ValueError('overlapping binary leaves')
            return
        if len(prefix) >= max_depth:
            raise ValueError('binary leaf cover has a gap')
        visit(prefix + '0')
        visit(prefix + '1')

    visit('')


def _validate_hull(job, hull, polygons):
    r1.set_prec(50)
    t0, t1, th0, th1 = job
    expected_polys, expected = r1.xhull_of_band(
        _float(t0), _float(t1), _float(th0), _float(th1))
    # xhull_of_band returns one flat polygon for the full-circle square,
    # while narrow sectors return a list of polygons.  edge_check canonicalizes
    # the same producer output before persisting it; mirror that normalization
    # here so the verifier compares like with like.
    if expected.get('mode') == 'full_circle_square':
        expected_polys = [expected_polys]
    expected_polygon_records = [
        [[r1._decimal_record(f'{float(x):.12f}'),
          r1._decimal_record(f'{float(y):.12f}')]
         for x, y in polygon]
        for polygon in expected_polys]
    if polygons != expected_polygon_records:
        raise ValueError('persisted localization polygon is not canonical')
    if not isinstance(hull, dict) or hull.get('polygons') != polygons:
        raise ValueError('hull/localization polygon mismatch')
    without_polygons = {key: value for key, value in hull.items()
                        if key != 'polygons'}
    if without_polygons != expected:
        raise ValueError('sector hull guard is not canonical')
    if hull.get('mode') == 'full_circle_square':
        _positive(hull['origin_padding_packet'])
        if len(polygons) != 1:
            raise ValueError('full-circle hull must have one polygon')
        return
    if hull.get('mode') != 'tangent_sector_cover':
        raise ValueError('unknown sector hull mode')
    subsectors = hull.get('subsectors')
    if not isinstance(subsectors, list) or len(subsectors) != len(polygons):
        raise ValueError('sector hull subsector mismatch')
    reach = th0
    for subsector in subsectors:
        lo = _fraction(subsector['theta_lo'])
        hi = _fraction(subsector['theta_hi'])
        if lo != reach or not lo < hi:
            raise ValueError('sector hull angular gap/overlap')
        if (_fraction(subsector['radius_lo']) != t0
                or _fraction(subsector['radius_hi'])
                != min(t1, _canonical_tmax(lo, hi))):
            raise ValueError('sector hull radial mismatch')
        _positive(subsector['cos_half_packet'])
        guards = subsector.get('tangent_slack_packets')
        if not isinstance(guards, list) or len(guards) != 2:
            raise ValueError('sector hull tangent guard mismatch')
        for guard in guards:
            _positive(guard, strict=False)
        reach = hi
    if reach != th1:
        raise ValueError('sector hull does not reach job endpoint')


def _validate_localization(localization, lbox=None):
    required = {
        'fan', 'polygons', 'initial_segments_per_edge', 'max_depth', 'leaves'}
    if not isinstance(localization, dict) or set(localization) != required:
        raise ValueError('localization certificate schema mismatch')
    fan = localization['fan']
    polygons = localization['polygons']
    leaves = localization['leaves']
    if (not isinstance(fan, list) or not fan
            or not isinstance(polygons, list) or not polygons
            or not isinstance(leaves, list) or not leaves
            or localization['initial_segments_per_edge'] != 8
            or localization['max_depth'] != 14):
        raise ValueError('localization certificate is incomplete')
    for pair in fan:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError('invalid localization fan')
        first, second = _fraction(pair[0]), _fraction(pair[1])
        if (lbox is not None and not
                (lbox[0] < first < lbox[1]
                 and lbox[2] < second < lbox[3])):
            raise ValueError('localization fan anchor escapes its box')
    for polygon in polygons:
        if not isinstance(polygon, list) or not polygon:
            raise ValueError('empty localization polygon')
        for point in polygon:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError('invalid localization polygon vertex')
            _fraction(point[0]); _fraction(point[1])
    grouped = {(edge, base): [] for edge in range(4) for base in range(8)}
    seen = set()
    for leaf in leaves:
        required_leaf = {
            'edge', 'base_segment', 'path', 'polygon_certificates'}
        if not isinstance(leaf, dict) or set(leaf) != required_leaf:
            raise ValueError('localization leaf schema mismatch')
        edge, base, path = leaf['edge'], leaf['base_segment'], leaf['path']
        if (not isinstance(edge, int) or isinstance(edge, bool)
                or not isinstance(base, int) or isinstance(base, bool)
                or (edge, base) not in grouped
                or (edge, base, path) in seen):
            raise ValueError('duplicate/unknown localization leaf')
        seen.add((edge, base, path))
        grouped[(edge, base)].append(path)
        certs = leaf['polygon_certificates']
        if not isinstance(certs, list) or len(certs) != len(polygons):
            raise ValueError('localization polygon proof mismatch')
        for cert in certs:
            if not isinstance(cert, dict) or set(cert) != {
                    'weights', 'lower_packet'}:
                raise ValueError('localization polygon proof schema mismatch')
            weights = [_fraction(value) for value in cert['weights']]
            if (len(weights) != len(fan) or any(value < 0 for value in weights)
                    or sum(weights, Fraction(0)) != 1):
                raise ValueError('invalid localization convex weights')
            _positive(cert['lower_packet'])
    for paths in grouped.values():
        _complete_prefix_tree(paths, 14)
    return polygons


def _validate_B(certificate, lbox):
    required = {
        'b11_packet', 'b12_packet', 'b22_packet', 'det_packet',
        'root_certificate'}
    if not isinstance(certificate, dict) or set(certificate) != required:
        raise ValueError('B certificate schema mismatch')
    b11lo, _ = _packet(certificate['b11_packet'])
    b12lo, b12hi = _packet(certificate['b12_packet'])
    b22lo, _ = _packet(certificate['b22_packet'])
    det_lower = (b11lo * b22lo
                 - max(abs(b12lo), abs(b12hi)) ** 2)
    if b11lo <= 0 or b22lo <= 0 or det_lower <= 0:
        raise ValueError('serialized B packets do not prove definiteness')
    _positive(certificate['det_packet'])
    root = certificate['root_certificate']
    mode = root.get('mode') if isinstance(root, dict) else None
    l1lo = exact.fraction_arb(lbox[0])
    l2lo = exact.fraction_arb(lbox[2])
    far_z = exact.fraction_arb(Fraction(5))

    def g_at(z):
        x = r1.SQ_PSI_B * z
        return l1lo * x + l2lo * x.tanh()

    def gprime_at(z):
        x = r1.SQ_PSI_B * z
        sech2 = 1 / (x.cosh() * x.cosh())
        return r1.SQ_PSI_B * (l1lo + l2lo * sech2)

    expected_mode = ('l2_nonnegative' if lbox[2] >= 0 else
                     'derivative_nonnegative'
                     if lbox[0] + lbox[2] >= 0 else
                     'positive_root_outer')
    if mode != expected_mode:
        raise ValueError('B root mode does not match its localization box')
    if not _strict_equal(
            root.get('g_far_packet'), exact.arb_packet(g_at(far_z))):
        raise ValueError('B far-root guard does not replay')
    if mode in ('l2_nonnegative', 'derivative_nonnegative'):
        if set(root) != {
                'mode', 'zq', 'g_zq_packet', 'condition_packet',
                'g_far_packet'}:
            raise ValueError('zero-root guard schema mismatch')
        if _fraction(root['zq']) != 0 or _packet(root['g_zq_packet']) != (0, 0):
            raise ValueError('zero-root identity mismatch')
        condition = l2lo if mode == 'l2_nonnegative' else l1lo + l2lo
        if not _strict_equal(
                root['condition_packet'], exact.arb_packet(condition)):
            raise ValueError('zero-root condition guard does not replay')
        _positive(root['condition_packet'], strict=False)
        _positive(root['g_far_packet'])
    elif mode == 'positive_root_outer':
        if set(root) != {
                'mode', 'zq', 'g_zq_packet', 'gprime_zq_packet',
                'g_far_packet'}:
            raise ValueError('positive-root guard schema mismatch')
        zq_value = _fraction(root['zq'])
        if (not 0 < zq_value <= 5
                or (zq_value * 10 ** 10).denominator != 1):
            raise ValueError('positive-root split is not positive')
        zq = exact.fraction_arb(zq_value)
        if (not _strict_equal(
                root['g_zq_packet'], exact.arb_packet(g_at(zq)))
                or not _strict_equal(
                    root['gprime_zq_packet'],
                    exact.arb_packet(gprime_at(zq)))):
            raise ValueError('positive-root guard does not replay')
        _positive(root['g_zq_packet'])
        _positive(root['gprime_zq_packet'])
        _positive(root['g_far_packet'])
    else:
        raise ValueError('unknown B root mode')


def _validate_angle_node(node, lo, hi, t0, t1):
    if (not isinstance(node, dict)
            or _fraction(node.get('theta_lo')) != lo
            or _fraction(node.get('theta_hi')) != hi
            or not lo < hi):
        raise ValueError('angular proof geometry mismatch')
    kind = node.get('kind')
    if kind == 'split':
        if set(node) != {
                'theta_lo', 'theta_hi', 'kind', 'split_at', 'children'}:
            raise ValueError('angular split schema mismatch')
        if _float(hi) - _float(lo) <= 2e-3:
            raise ValueError('angular split violates the producer width rule')
        midpoint = r1._float_fraction(
            0.5 * (_float(lo) + _float(hi)))
        if (_fraction(node['split_at']) != midpoint
                or not isinstance(node['children'], list)
                or len(node['children']) != 2):
            raise ValueError('angular split point mismatch')
        left = _validate_angle_node(node['children'][0], lo, midpoint, t0, t1)
        right = _validate_angle_node(node['children'][1], midpoint, hi, t0, t1)
        return tuple(a + b for a, b in zip(left, right))
    tmax = _canonical_tmax(lo, hi)
    if kind == 'outside_star':
        if (set(node) != {'theta_lo', 'theta_hi', 'kind', 'tmax'}
                or _fraction(node['tmax']) != tmax or t0 < tmax):
            raise ValueError('invalid outside-star leaf')
        return 0, 0, 1
    if kind != 'certified':
        raise ValueError('failed/unknown angular leaf')
    required = {
        'theta_lo', 'theta_hi', 'kind', 'tmax', 'radial_end',
        'theta_box', 'qB_packet', 'sdot_packet', 'radial_pieces'}
    if set(node) != required or _fraction(node['tmax']) != tmax:
        raise ValueError('certified angular leaf schema mismatch')
    theta_ball = (r1._dec(_float(lo), 8).union(r1._dec(_float(hi), 8))
                  + r1._ANG_PAD.union(-r1._ANG_PAD))
    if not _strict_equal(node['theta_box'], exact.arb_packet(theta_ball)):
        raise ValueError('angular theta packet does not replay')
    v1, v2 = theta_ball.cos(), theta_ball.sin()
    sdot = r1._sdot_box(v1, v2)
    if not _strict_equal(node['sdot_packet'], exact.arb_packet(sdot)):
        raise ValueError('angular slope packet does not replay')
    _contains(node['theta_box'], lo); _contains(node['theta_box'], hi)
    _positive(node['qB_packet'])
    _packet(node['sdot_packet'])
    end = min(t1, tmax)
    if _fraction(node['radial_end']) != end or not t0 < end:
        raise ValueError('certified angular radial endpoint mismatch')
    pieces = node['radial_pieces']
    if not isinstance(pieces, list) or not pieces:
        raise ValueError('certified angular leaf has no radial proof')
    reach = t0
    for piece in pieces:
        if not isinstance(piece, dict) or set(piece) != {
                'radius_lo', 'radius_hi', 'radius_box', 's_box',
                'curvature_packet'}:
            raise ValueError('radial leaf schema mismatch')
        left = _fraction(piece['radius_lo'])
        right = _fraction(piece['radius_hi'])
        if left != reach or not left < right:
            raise ValueError('radial proof gap/overlap')
        radius_ball = (
            r1._dec(_float(left), 12).union(r1._dec(_float(right), 12))
            + r1._RAD_PAD.union(-r1._RAD_PAD))
        if not _strict_equal(
                piece['radius_box'], exact.arb_packet(radius_ball)):
            raise ValueError('radial packet does not replay')
        s_ball = (r1._dec(r1.S0F, 10) + r1._ORIG
                  + radius_ball * sdot)
        if not _strict_equal(piece['s_box'], exact.arb_packet(s_ball)):
            raise ValueError('radial tilt packet does not replay')
        _contains(piece['radius_box'], left)
        _contains(piece['radius_box'], right)
        _positive(piece['s_box'])
        _negative(piece['curvature_packet'])
        reach = right
    if reach != end:
        raise ValueError('radial proof does not reach its endpoint')
    return 1, len(pieces), 0


def _expected_policy():
    return {
        'precision_bits': 50,
        'lbox_quant_digits': 12,
        'zq_quant_digits': 10,
        'angle_pad': r1._decimal_record('0.00000001'),
        'radius_pad': r1._decimal_record('0.000000000001'),
        'sdot_model': r1.SDOT_MODEL,
        'sdot_coefficients': [r1._decimal_record(x)
                              for x in r1.SDOT_C_TEXT],
    }


def _expected_star():
    return {
        'W_ANG': r1._float_record(r1.W_ANG),
        'WEDGE_HALF': r1._float_record(r1.WEDGE_HALF),
        'CONE_MID': r1._float_record(r1.CONE_MID),
        'T_LONG': r1._float_record(r1.T_LONG),
        'T_MID': r1._float_record(r1.T_MID),
        'T_CORE': r1._float_record(r1.T_CORE),
        'T_IN': r1._float_record(r1.T_IN),
        'N_ANG': r1.N_ANG,
        'A1S': r1._float_record(r1.A1S),
        'A2S': r1._float_record(r1.A2S),
        'S0F': r1._float_record(r1.S0F),
        'origin_radius': r1._decimal_record('0.0000001'),
    }


def verify_certificate(path):
    """Validate the canonical tree, coverage, packets, signs, and sources."""
    r1.set_prec(50)
    data = exact.load_json(exact.require_plain_regular_file(path))
    required = {
        'schema_version', 'kind', 'evidence_model', 'run_id',
        'source_sha256', 'source_sha256_after', 'source_set_sha256',
        'runtime', 'certificate_policy', 'star', 'schedule',
        'schedule_sha256', 'records', 'derived_summary',
        'certificate_sha256'}
    if (not isinstance(data, dict) or set(data) != required
            or data['schema_version'] != r1.SCHEMA_VERSION
            or data['kind'] != 'huang_region1_certificate'
            or data['evidence_model'] != r1.EVIDENCE_MODEL
            or not isinstance(data['run_id'], str)
            or _RUN_ID_RE.fullmatch(data['run_id']) is None
            or data['certificate_sha256'] != exact.payload_sha256(
                data, omit=('certificate_sha256',))):
        raise ValueError('invalid Region-I certificate schema/identity')
    sources = r1.proof_source_hashes()
    if (data['source_sha256'] != sources
            or data['source_sha256_after'] != sources
            or data['source_set_sha256']
            != exact.payload_sha256(sources, omit=())):
        raise ValueError('stale Region-I proof source')
    exact.validate_runtime_record(
        data['runtime'], 50, allow_any_workers=True)
    if not _strict_equal(data['certificate_policy'], _expected_policy()):
        raise ValueError('Region-I certificate policy mismatch')
    if not _strict_equal(data['star'], _expected_star()):
        raise ValueError('Region-I star policy mismatch')
    schedule = r1.schedule_records()
    if (not _strict_equal(data['schedule'], schedule)
            or data['schedule_sha256'] != exact.payload_sha256(
                schedule, omit=())):
        raise ValueError('Region-I top schedule mismatch')
    records = data['records']
    if (not isinstance(records, list) or len(records) != len(schedule)
            or any(not isinstance(row.get('index'), int)
                   or isinstance(row.get('index'), bool)
                   for row in records)
            or [row.get('index') for row in records]
            != list(range(len(schedule)))):
        raise ValueError('Region-I record index mismatch')
    totals = (0, 0, 0)
    for target, row in zip(schedule, records):
        required_row = {
            'index', 'geometry', 'verdict', 'failure', 'lbox',
            'pad_multiplier', 'hull_certificate',
            'localization_certificate', 'B_certificate', 'angular_roots',
            'worker_source_sha256_before',
            'worker_source_sha256_after', 'worker_runtime', 'run_id'}
        if (not isinstance(row, dict) or set(row) != required_row
                or not _strict_equal(row['geometry'], target['geometry'])
                or row['verdict'] != 'PASS' or row['failure'] is not None
                or row['run_id'] != data['run_id']
                or row['worker_source_sha256_before'] != sources
                or row['worker_source_sha256_after'] != sources):
            raise ValueError('invalid Region-I job record')
        exact.validate_runtime_record(row['worker_runtime'], 50, workers=1)
        if (exact.runtime_identity(row['worker_runtime'])
                != exact.runtime_identity(data['runtime'])):
            raise ValueError('Region-I worker/producer runtime mismatch')
        lbox = [_fraction(value) for value in row['lbox']]
        if (len(lbox) != 4 or not lbox[0] < lbox[1]
                or not lbox[2] < lbox[3] or lbox[0] <= 0
                or any((value * 10 ** 12).denominator != 1
                       for value in lbox)):
            raise ValueError('invalid Region-I localization box')
        if _fraction(row['pad_multiplier']) not in {
                Fraction(1), Fraction(8, 5), Fraction(12, 5)}:
            raise ValueError('invalid Region-I localization padding')
        polygons = _validate_localization(
            row['localization_certificate'], lbox)
        job = tuple(_fraction(value) for value in row['geometry'])
        _validate_hull(job, row['hull_certificate'], polygons)
        _validate_B(row['B_certificate'], lbox)
        roots = row['angular_roots']
        if not isinstance(roots, list) or not roots:
            raise ValueError('Region-I job has no angular roots')
        nroots = max(1, int(math.ceil(
            (_float(job[3]) - _float(job[2]))
            / (2 * math.pi / r1.N_ANG))))
        expected_roots = [
            (r1._float_fraction(
                _float(job[2])
                + (_float(job[3]) - _float(job[2])) * k / nroots),
             r1._float_fraction(
                _float(job[2])
                + (_float(job[3]) - _float(job[2])) * (k + 1) / nroots))
            for k in range(nroots)]
        actual_roots = [
            (_fraction(root.get('theta_lo')),
             _fraction(root.get('theta_hi')))
            for root in roots]
        if actual_roots != expected_roots:
            raise ValueError('Region-I angular root partition is not canonical')
        reach = job[2]
        for root in roots:
            lo = _fraction(root['theta_lo'])
            hi = _fraction(root['theta_hi'])
            if lo != reach:
                raise ValueError('Region-I angular root gap/overlap')
            count = _validate_angle_node(root, lo, hi, job[0], job[1])
            totals = tuple(a + b for a, b in zip(totals, count))
            reach = hi
        if reach != job[3]:
            raise ValueError('Region-I angular roots miss job endpoint')
    summary = {
        'jobs': len(records),
        'failures': 0,
        'certified_angular_leaves': totals[0],
        'radial_pieces': totals[1],
        'outside_star_leaves': totals[2],
    }
    if not _strict_equal(data['derived_summary'], summary):
        raise ValueError('Region-I derived summary mismatch')
    return data


def _require_packet(value, stored, label):
    expected = exact.arb_packet(value)
    if not _strict_equal(stored, expected):
        raise ValueError(f'Region-I numerical replay mismatch: {label}')
    return value


def _replay_localization(localization, lbox):
    """Recompute every accepted fixed-weight boundary witness."""
    from flint import arb

    l1lo, l1hi, l2lo, l2hi = [exact.fraction_arb(value) for value in lbox]
    fan = [tuple(exact.fraction_arb(_fraction(value)) for value in pair)
           for pair in localization['fan']]
    polygons = [[
        tuple(exact.fraction_arb(_fraction(value)) for value in point)
        for point in polygon]
        for polygon in localization['polygons']]
    phik = [r1.Phi_acb(first, second) for first, second in fan]
    tol = exact.fraction_arb(Fraction(1, 10 ** 9))
    edges = [
        (2, l2lo, l1lo, l1hi), (2, l2hi, l1lo, l1hi),
        (1, l1lo, l2lo, l2hi), (1, l1hi, l2lo, l2hi),
    ]
    for leaf in localization['leaves']:
        edge = leaf['edge']
        base = leaf['base_segment']
        fix, fixed, edge_lo, edge_hi = edges[edge]
        u0 = edge_lo + (edge_hi - edge_lo) * base / 8
        u1 = edge_lo + (edge_hi - edge_lo) * (base + 1) / 8
        for bit in leaf['path']:
            midpoint = (u0 + u1) / 2
            if bit == '0':
                u1 = midpoint
            elif bit == '1':
                u0 = midpoint
            else:
                raise ValueError('invalid localization replay path')
        midpoint = (u0 + u1) / 2
        segment = u0.union(u1)
        if fix == 2:
            e1, e2 = midpoint, fixed
            deriv, _ = r1.gradPhi_acb(segment, fixed, tol=tol)
        else:
            e1, e2 = fixed, midpoint
            _, deriv = r1.gradPhi_acb(fixed, segment, tol=tol)
        phie = r1.Phi_acb(e1, e2)
        half_length = (u1 - u0) / 2
        for polygon_index, (polygon, certificate) in enumerate(zip(
                polygons, leaf['polygon_certificates'])):
            values = []
            for x1, x2 in polygon:
                coordinate = x1 if fix == 2 else x2
                difference = deriv - coordinate
                dlo, dhi = r1.endpoints(difference)
                derivative_abs = abs(dlo).union(abs(dhi))
                slack = half_length * derivative_abs
                values.append([
                    phie - pk - (e1 - f1) * x1 - (e2 - f2) * x2
                    - slack
                    for (f1, f2), pk in zip(fan, phik)])
            weights = [_fraction(value) for value in certificate['weights']]
            one_hot = [index for index, weight in enumerate(weights)
                       if weight == 1]
            if (len(one_hot) == 1
                    and all(weight in (0, 1) for weight in weights)):
                combined = [row[one_hot[0]] for row in values]
            else:
                arb_weights = [exact.fraction_arb(value) for value in weights]
                combined = [sum((arb_weights[index] * row[index]
                                 for index in range(len(weights))), arb(0))
                            for row in values]
            lower_packet = r1._packet_union(combined)
            if not _strict_equal(lower_packet, certificate['lower_packet']):
                raise ValueError(
                    'Region-I numerical replay mismatch: localization '
                    f"edge={edge} base={base} path={leaf['path']} "
                    f"polygon={polygon_index}")
            if _packet(lower_packet)[0] <= 0:
                raise ValueError('replayed localization witness is not positive')


def _replay_B(certificate, lbox):
    exact_box = [exact.fraction_arb(value) for value in lbox]
    b11, b12, b22, root = r1.B_Lambda(*exact_box)
    _require_packet(b11, certificate['b11_packet'], 'B11')
    _require_packet(b12, certificate['b12_packet'], 'B12')
    _require_packet(b22, certificate['b22_packet'], 'B22')
    if not _strict_equal(root, certificate['root_certificate']):
        raise ValueError('Region-I numerical replay mismatch: B root guard')
    determinant = b11 * b22 - b12 * b12
    _require_packet(determinant, certificate['det_packet'], 'det(B)')
    determinant_lo, _ = r1.endpoints(determinant)
    if not (determinant_lo > 0):
        raise ValueError('replayed B matrix is not positive definite')
    return b11, b12, b22


def _replay_angle_numerics(node, B, gz):
    kind = node['kind']
    if kind == 'split':
        for child in node['children']:
            _replay_angle_numerics(child, B, gz)
        return
    if kind == 'outside_star':
        return
    lo = _fraction(node['theta_lo'])
    hi = _fraction(node['theta_hi'])
    theta = (r1._dec(_float(lo), 8).union(r1._dec(_float(hi), 8))
             + r1._ANG_PAD.union(-r1._ANG_PAD))
    v1, v2 = theta.cos(), theta.sin()
    qB = r1.Binv_form(*B, v1, v2)
    if qB is None:
        raise ValueError('replayed inverse quadratic form is undefined')
    _require_packet(qB, node['qB_packet'], 'inverse quadratic form')
    qB_lo, _ = r1.endpoints(qB)
    if not (qB_lo > 0):
        raise ValueError('replayed inverse quadratic form is not positive')
    sdot = r1._sdot_box(v1, v2)
    for piece in node['radial_pieces']:
        left = _fraction(piece['radius_lo'])
        right = _fraction(piece['radius_hi'])
        radius = (r1._dec(_float(left), 12).union(
                  r1._dec(_float(right), 12))
                  + r1._RAD_PAD.union(-r1._RAD_PAD))
        x1 = r1._dec(r1.A1S, 10) + r1._ORIG + radius * v1
        x2 = r1._dec(r1.A2S, 10) + r1._ORIG + radius * v2
        s_box = r1._dec(r1.S0F, 10) + r1._ORIG + radius * sdot
        qT = r1.quadT_box(x1, x2, s_box, v1, v2, sdot, gz)
        if qT is None:
            raise ValueError('replayed nonentropy curvature is undefined')
        curvature = qT - qB
        _require_packet(
            curvature, piece['curvature_packet'],
            f"curvature [{left},{right}]")
        _, curvature_hi = r1.endpoints(curvature)
        if not (curvature_hi < 0):
            raise ValueError('replayed curvature is not strictly negative')


def _replay_record_numerics(row, gz):
    lbox = [_fraction(value) for value in row['lbox']]
    _replay_localization(row['localization_certificate'], lbox)
    B = _replay_B(row['B_certificate'], lbox)
    for root in row['angular_roots']:
        _replay_angle_numerics(root, B, gz)


_FULL_REPLAY_DATA = None
_FULL_REPLAY_GZ = None


def _init_full_replay_worker(
        path, file_sha256, certificate_sha256, replay_sources):
    global _FULL_REPLAY_DATA, _FULL_REPLAY_GZ
    r1._init()
    if r1.proof_source_hashes() != replay_sources:
        raise ValueError('Region-I worker replay source mismatch')
    if exact.file_sha256(path) != file_sha256:
        raise ValueError('Region-I artifact differs from parent replay input')
    _FULL_REPLAY_DATA = exact.load_json(path)
    if (_FULL_REPLAY_DATA.get('certificate_sha256') != certificate_sha256
            or certificate_sha256 != exact.payload_sha256(
                _FULL_REPLAY_DATA, omit=('certificate_sha256',))):
        raise ValueError('Region-I worker certificate identity mismatch')
    _FULL_REPLAY_GZ = r1.hg.get_zt_grid(9, r1.GRID_N_RAY)


def _replay_record_index(index):
    _replay_record_numerics(
        _FULL_REPLAY_DATA['records'][index], _FULL_REPLAY_GZ)
    return index


def verify_certificate_full(path, workers=1, progress=None):
    """Recompute every numerical proof packet and compare byte-for-byte.

    This directly replays each stored fixed-weight localization witness, each
    B_Lambda integral, inverse quadratic form, and curvature quadrature from
    frozen source instead of trusting stored packet signs.  It is the
    Region-I gate used by the theorem-level verifier.
    """
    if (not isinstance(workers, int) or isinstance(workers, bool)
            or not 1 <= workers <= 4):
        raise ValueError('Region-I replay workers must be in [1,4]')
    path = exact.require_plain_regular_file(path)
    artifact_sha256 = exact.file_sha256(path)
    data = verify_certificate(path)
    replay_sources = r1.proof_source_hashes()
    records = data['records']
    completed = 0
    seen = set()
    with multiprocessing.Pool(
            workers, initializer=_init_full_replay_worker,
            initargs=(str(path), artifact_sha256,
                      data['certificate_sha256'], replay_sources)) as pool:
        for index in pool.imap_unordered(
                _replay_record_index, range(len(records))):
            if index in seen:
                raise ValueError('Region-I full replay returned a duplicate job')
            seen.add(index)
            completed += 1
            if progress is not None:
                progress(completed, len(records), index)
    if (completed != len(records)
            or seen != set(range(len(records)))):
        raise ValueError('Region-I full replay did not cover every job')
    if exact.file_sha256(path) != artifact_sha256:
        raise ValueError('Region-I artifact changed during full replay')
    if r1.proof_source_hashes() != replay_sources:
        raise ValueError('Region-I proof source changed during full replay')
    return data


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('certificate')
    parser.add_argument('--full-replay', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args(argv)

    def report(done, total, index):
        if done == 1 or done % 25 == 0 or done == total:
            print(f'REPLAY {done}/{total} last_job={index}', flush=True)

    if args.full_replay:
        data = verify_certificate_full(
            args.certificate, workers=args.workers, progress=report)
        mode = 'full numerical replay'
    else:
        data = verify_certificate(args.certificate)
        mode = 'structural replay'
    print(f"PASS Region I {mode}: {data['certificate_sha256']}")


if __name__ == '__main__':
    _main()
