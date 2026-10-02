from bridge.mcts.tictactoe import apply_move, is_terminal, legal_moves, winner


def test_winner_detects_row():
    board = "XXX......"
    assert winner(board) == "X"
    assert is_terminal(board)


def test_legal_moves_and_apply_move():
    board = "X........"
    assert legal_moves(board) == [1, 2, 3, 4, 5, 6, 7, 8]
    new_board = apply_move(board, 4, "O")
    assert new_board == "X...O...."
