from tokenizer import MoveTokenizer


def test_roundtrip_and_save_load(tmp_path):
    games = [["e4", "e5", "Nf3"], ["d4", "Nf6"]]
    tok = MoveTokenizer.from_games(games)

    # eos reserved at index 0
    assert tok.vocab[0] == MoveTokenizer.EOS
    assert tok.eos_id == 0
    # vocab = eos + 5 distinct moves (e4,e5,Nf3,d4,Nf6)
    assert tok.vocab_size == 6

    # round-trip ignores eos in decode
    ids = tok.encode(["e4", "e5", "Nf3"])
    assert tok.decode(ids) == ["e4", "e5", "Nf3"]

    # save + load reproduces the mapping
    p = tmp_path / "tok.json"
    tok.save(str(p))
    tok2 = MoveTokenizer.load(str(p))
    assert tok2.vocab == tok.vocab
    assert tok2.encode(["d4", "Nf6"]) == tok.encode(["d4", "Nf6"])
