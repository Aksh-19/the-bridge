from bridge.mcts.node import Node
from bridge.mcts.select import select_and_expand
from bridge.mcts.tictactoe import EMPTY_BOARD


def test_first_call_expands_a_child_of_root():
    root = Node(EMPTY_BOARD, player_to_move="X")
    result = select_and_expand(root)
    assert result.parent is root
    assert len(root.children) == 1


def test_repeated_calls_eventually_expand_every_root_child():
    root = Node(EMPTY_BOARD, player_to_move="X")
    for _ in range(9):  # 9 legal first moves
        select_and_expand(root)
    assert root.is_fully_expanded()
    assert len(root.children) == 9
