"""Week 4 checkpoint: play MCTS against random and greedy baselines over
many games, alternating who goes first, and report win rates."""

import random
from collections.abc import Callable

from bridge.mcts.agent import mcts_search
from bridge.mcts.baselines import greedy_move, random_move
from bridge.mcts.tictactoe import EMPTY_BOARD, apply_move, is_terminal, winner

Player = Callable[[str, str, random.Random], int]


def play_one_game(player_x: Player, player_o: Player, rng: random.Random) -> str | None:
    """Play one game, player_x moving first. Returns the winner or None."""
    board = EMPTY_BOARD
    current_player = "X"

    while not is_terminal(board):
        mover = player_x if current_player == "X" else player_o
        move = mover(board, current_player, rng)
        board = apply_move(board, move, current_player)
        current_player = "O" if current_player == "X" else "X"

    return winner(board)


def mcts_player(n_iterations: int) -> Player:
    """Wrap mcts_search so it matches the Player signature (board,
    player_to_move, rng) -> move, with n_iterations baked in."""

    def play(board: str, player_to_move: str, rng: random.Random) -> int:
        return mcts_search(board, player_to_move, n_iterations=n_iterations, rng=rng)

    return play


def run_tournament(
    opponent: Player, opponent_name: str, n_games: int = 100, n_iterations: int = 200
) -> dict[str, int]:
    """Play n_games, with MCTS and the opponent each going first half the
    time (removes first-move advantage as a confound)."""
    mcts = mcts_player(n_iterations)
    results = {"mcts_wins": 0, "opponent_wins": 0, "draws": 0}

    for i in range(n_games):
        rng = random.Random(i)
        if i % 2 == 0:
            outcome = play_one_game(mcts, opponent, rng)
            mcts_mark = "X"
        else:
            outcome = play_one_game(opponent, mcts, rng)
            mcts_mark = "O"

        if outcome is None:
            results["draws"] += 1
        elif outcome == mcts_mark:
            results["mcts_wins"] += 1
        else:
            results["opponent_wins"] += 1

    print(f"MCTS vs {opponent_name} ({n_games} games): {results}")
    return results


def main() -> None:
    run_tournament(random_move, "random", n_games=100, n_iterations=200)
    run_tournament(greedy_move, "greedy", n_games=100, n_iterations=200)


if __name__ == "__main__":
    main()
