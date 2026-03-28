"""
Hexagonal grid utilities using axial coordinates (q, r).
Third axis s = -q - r is implicit.

Axial directions on a hex grid (pointy-top orientation):
  Axis 0 (q-axis):  (1, 0) and (-1, 0)   — "east/west"
  Axis 1 (r-axis):  (0, 1) and (0, -1)    — "southeast/northwest"
  Axis 2 (s-axis):  (1, -1) and (-1, 1)   — "northeast/southwest"
"""

# The 6 neighbor directions in axial coordinates
DIRECTIONS = [
    (1, 0), (-1, 0),    # q-axis
    (0, 1), (0, -1),    # r-axis
    (1, -1), (-1, 1),   # s-axis
]

# The 3 line axes — each is a (dq, dr) direction; the opposite is (-dq, -dr)
LINE_AXES = [
    (1, 0),   # q-axis (east-west)
    (0, 1),   # r-axis (southeast-northwest)
    (1, -1),  # s-axis (northeast-southwest)
]


def hex_distance(q1, r1, q2, r2):
    """Manhattan distance on a hex grid (axial coordinates)."""
    dq = q1 - q2
    dr = r1 - r2
    ds = (-q1 - r1) - (-q2 - r2)  # = -(dq + dr)
    return max(abs(dq), abs(dr), abs(ds))


def hex_neighbors(q, r):
    """Return all 6 neighbors of hex (q, r)."""
    return [(q + dq, r + dr) for dq, dr in DIRECTIONS]


def count_in_direction(stones, q, r, dq, dr):
    """Count consecutive stones of the same set starting from (q,r) in direction (dq,dr),
    NOT including (q,r) itself."""
    count = 0
    cq, cr = q + dq, r + dr
    while (cq, cr) in stones:
        count += 1
        cq += dq
        cr += dr
    return count


def check_line_through(stones, q, r):
    """Check if placing a stone at (q,r) (already in `stones`) completes a line of 6+
    along any of the 3 axes. Returns True if a winning line is found."""
    for dq, dr in LINE_AXES:
        total = 1 + count_in_direction(stones, q, r, dq, dr) \
                  + count_in_direction(stones, q, r, -dq, -dr)
        if total >= 6:
            return True
    return False


def cells_within_distance(q, r, dist):
    """Yield all hex cells within `dist` steps of (q, r), excluding (q, r) itself."""
    for dq in range(-dist, dist + 1):
        for dr in range(max(-dist, -dq - dist), min(dist, -dq + dist) + 1):
            if dq == 0 and dr == 0:
                continue
            yield (q + dq, r + dr)


def bounding_box(coords):
    """Return (min_q, max_q, min_r, max_r) for a set of (q, r) coordinates."""
    if not coords:
        return (0, 0, 0, 0)
    qs = [c[0] for c in coords]
    rs = [c[1] for c in coords]
    return (min(qs), max(qs), min(rs), max(rs))
