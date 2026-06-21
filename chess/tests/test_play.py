import chess
import torch
from config import Config
from tokenizer import MoveTokenizer
from model import GPTLanguageModel
from play import legal_move_token_probs, play_game


def _cfg():
    c = Config()
    c.n_embd = 32; c.n_head = 4; c.n_layer = 2
    c.block_size = 32; c.dropout = 0.0; c.device = "cpu"
    return c


def test_legal_move_token_probs_filters_to_legal():
    # vocab containing some legal opening moves
    games = [["e4", "e5", "Nf3", "d4", "c4", "Nc6"]]
    tok = MoveTokenizer.from_games(games)
    board = chess.Board()
    probs = torch.ones(tok.vocab_size) / tok.vocab_size
    mapping = legal_move_token_probs(probs, board, tok)
    # every key is a legal move in the start position
    legal = set(board.legal_moves)
    for mv in mapping:
        assert mv in legal
    # e4 (a legal move present in vocab) should appear with positive prob
    e4 = board.parse_san("e4")
    assert mapping.get(e4, 0.0) > 0.0


def test_play_game_produces_only_legal_moves():
    cfg = _cfg()
    games = [["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6"]]
    tok = MoveTokenizer.from_games(games)
    model = GPTLanguageModel(tok.vocab_size, cfg)
    moves = play_game(model, tok, cfg, max_moves=10, legal_mask=True, seed=0)
    # replay through a real board: must all be legal
    board = chess.Board()
    for san in moves:
        mv = board.parse_san(san)  # raises if illegal
        board.push(mv)
    assert len(moves) >= 1
