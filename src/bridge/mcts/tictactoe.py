"""Tic-tac-toe game logic: the environment MCTS will search over in
Weeks 3-4. Board is a 9-character string: 'X', 'O', or '.' per cell,
read left-to-right, top-to-bottom."""

EMPTY_BOARD = "." * 9

WIN_LINES = [
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),  # rows
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),  # columns
    (0, 4, 8),
    (2, 4, 6),  # diagonals
]


def legal_moves(board: str) -> list[int]:
    """Indices (0-8) of empty cells."""
    return [i for i, cell in enumerate(board) if cell == "."]


def apply_move(board: str, move: int, player: str) -> str:
    """Return a new board with `player`'s mark placed at `move`."""
    return board[:move] + player + board[move + 1 :]


def winner(board: str) -> str | None:
    """Return 'X' or 'O' if that player has three in a row, else None."""
    for a, b, c in WIN_LINES:
        if board[a] != "." and board[a] == board[b] == board[c]:
            return board[a]
    return None


def is_terminal(board: str) -> bool:
    """Game over if someone won, or the board is full (a draw)."""
    return winner(board) is not None or "." not in board


def other_player(player: str) -> str:
    return "O" if player == "X" else "X"
