"""Shared Go helpers: SGF parsing, rank labels, board-plane features. Pure numpy; no external data."""
import re
import numpy as np

RANKS = ['D', 'C', 'B', 'A', '1D', '2D', '3D', '4D', '5D', '6D']
RANK2I = {r: i for i, r in enumerate(RANKS)}
_MV = re.compile(r';([BW])\[([a-s]{0,2})\]')
_HD = re.compile(r'HD\[(\d+)\]')
_KM = re.compile(r'KM\[([-\d.]+)\]')


def parse(sgf):
    """-> (handicap, moves) ; moves = list of (colour 0=B/1=W, x, y) with x=y=-1 for pass."""
    m = _HD.search(sgf)
    hd = int(m.group(1)) if m else 0
    mv = []
    for c, p in _MV.findall(sgf):
        if len(p) == 2 and p[0] < 't' and p[1] < 't':
            mv.append((0 if c == 'B' else 1, ord(p[0]) - 97, ord(p[1]) - 97))
        else:
            mv.append((0 if c == 'B' else 1, -1, -1))
    return hd, mv


def own_moves(mv, colour):
    return [(x, y) for c, x, y in mv if c == colour]
