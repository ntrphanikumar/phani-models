from pgn_loader import load_games

SAMPLE_PGN = """[Event "T1"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1/2-1/2

[Event "T2"]

1. d4 Nf6 2. c4 e6 *
"""


def test_loads_games_as_san_moves(tmp_path):
    (tmp_path / "g.pgn").write_text(SAMPLE_PGN)
    games = load_games(str(tmp_path))
    assert len(games) == 2
    # first game parsed to SAN moves in order
    assert games[0] == ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]
    # no move numbers or result tokens leak in
    for g in games:
        for mv in g:
            assert isinstance(mv, str) and mv
            assert mv not in ("1/2-1/2", "1-0", "0-1", "*")
