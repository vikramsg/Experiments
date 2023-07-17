from typing import List, Tuple

import numpy as np
import pytest
from click import testing as click_testing

from src.tic_tac_toe import _is_game_over, _make_computer_move, game_loop


@pytest.mark.parametrize(
    "moves, expected_msg",
    [
        (
            ["o", "1,1", "0,0", "1,0", "2,1", "2,2", "0,2", "0,1", "1,2", "2,0"],
            "Game finished in a draw",
        ),
        (
            ["x", "0,0", "1,1", "1,2", "1,0", "0,1", "0,2", "2,1", "2,0"],
            "Game over. Player o won.",
        ),
        (
            ["o", "2,1", "1,0", "2,2", "0,1", "2,0"],
            "Game over. Player o won.",
        ),
        (
            ["x", "1,1", "1,2", "0,0", "2,2", "0,2", "2,0", "0,1"],
            "Game over. Player x won.",
        ),
        (
            ["o", "0,0", "2,1", "0,1", "0,2", "1,1", "2,2", "1,0", "2,0"],
            "Game over. Player x won.",
        ),
    ],
)
def test_game_play(moves: List[str], expected_msg: str) -> None:
    # Given
    test_runner = click_testing.CliRunner()

    # When
    result = test_runner.invoke(game_loop, input="\n".join(moves))

    # Then
    assert expected_msg in result.output


@pytest.mark.parametrize(
    "moves, expected_msg",
    [
        (
            ["o", "0,0", "2,1", "0,1", "0,1", "0,2", "1,1", "2,2", "1,0", "2,0"],
            "Error: Invalid input. Position already occupied. Chose different position.",
        ),
        (
            ["o", "0,0", "2,1", "0,1", "1.0", "0,2", "1,1", "2,2", "1,0", "2,0"],
            "Error: Invalid input. Please enter exactly two comma-separated integers between 0 and 2.",
        ),
        (
            ["o", "0,0", "2,1", "0,1", "0,2", "0,2,1", "1,1", "2,2", "1,0", "2,0"],
            "Error: Invalid input. Please enter exactly two comma-separated integers between 0 and 2.",
        ),
        (
            ["o", "0,0", "2,1", "0,1", "0,2", "3,3", "1,1", "2,2", "1,0", "2,0"],
            "Error: Invalid input. Please enter two comma-separated integers with each integer between 0 and 2.",
        ),
    ],
)
def test_click_invalid_input_handling(moves: List[str], expected_msg: str) -> None:
    # Given
    test_runner = click_testing.CliRunner()

    # When
    result = test_runner.invoke(game_loop, input="\n".join(moves))

    # Then
    assert expected_msg in result.output


@pytest.mark.parametrize(
    "indices, position, expected_result",
    [
        # Rows
        ((0, slice(3)), (0, 2), 1),
        ((1, slice(3)), (1, 2), 1),
        ((2, slice(3)), (2, 2), 1),
        ((2, slice(2)), (2, 1), 0),
        # Columns
        ((slice(3), 0), (2, 0), 1),
        ((slice(3), 1), (2, 1), 1),
        ((slice(3), 2), (2, 2), 1),
        ((slice(2), 2), (1, 2), 0),
        # Diagonal
        (((0, 1, 2), (0, 1, 2)), (1, 1), 1),
        (((0, 1), (0, 1)), (1, 1), 0),
        # Off diagonal
        (((0, 1, 2), (2, 1, 0)), (1, 1), 1),
        (((0, 1), (2, 1)), (1, 1), 0),
    ],
)
def test_game_over(
    indices: Tuple, position: Tuple[int, int], expected_result: int
) -> None:
    # Given
    board = np.zeros((3, 3))
    board[indices] = 1

    # When
    result = _is_game_over(board, position, 1)

    # Then
    assert result == expected_result


@pytest.mark.parametrize(
    "indices, player, expected_result",
    [
        # Rows
        ((0, slice(2)), 1, (0, 2)),
        ((slice(2), 0), 1, (2, 0)),
        (((0, 1), (0, 1)), 1, (2, 2)),
        (((0, 1), (2, 1)), 1, (2, 0)),
    ],
)
def test_make_computer_move(indices, player, expected_result):
    # Given
    game_board = np.zeros((3, 3), dtype=np.short)

    # We want the computer to select 0, 2
    game_board[indices] = player

    # When
    (row, col) = _make_computer_move(game_board, player)

    # Then
    assert (row, col) == expected_result


@pytest.mark.parametrize(
    "indices, player, expected_result",
    [
        # Rows
        ((0, slice(2)), 1, (0, 2)),
    ],
)
def test_make_computer_move(indices, player, expected_result):
    # Given
    game_board = np.zeros((3, 3), dtype=np.short)

    not_player = 1 if player == 2 else 2

    # We want the computer to select 0, 2
    game_board[indices] = not_player

    # When
    (row, col) = _make_computer_move(game_board, player)

    # Then
    assert (row, col) == expected_result
