from functools import partial
from typing import Dict, Tuple

import click
import numpy as np

from random import randint
import time


def _convert_to_tic_tac_toe_input(
    value: str, game_board: np.ndarray
) -> Tuple[int, int]:
    try:
        values = value.split(",")
        assert len(values) == 2
        # ToDo: Did not catch this error, where int conversion is not possible
        tuple_values = (int(values[0]), int(values[1]))

        assert tuple_values[0] >= 0 and tuple_values[0] < 3
        assert tuple_values[1] >= 0 and tuple_values[1] < 3

        assert game_board[tuple_values] == 0

    except ValueError as e:
        if "Position already occupied" in e.args[0]:
            raise click.BadParameter(
                "Invalid input. Position already occupied. Chose different position."
            )
        elif "Please enter exactly two comma-separated integers." in e.args[0]:
            raise click.BadParameter(
                "Invalid input. Please enter exactly two comma-separated integers between 0 and 2."
            )
        elif "invalid literal for int" in e.args[0]:
            raise click.BadParameter(
                "Invalid input. Please only use integers. Input two comma-separated integers between 0 and 2."
            )

    except AssertionError:
        raise click.BadParameter(
            "Invalid input. "
            "Please enter two comma-separated integers with each integer between 0 and 2."
        )

    return tuple_values


def _is_game_over(
    game_board: np.ndarray, position: Tuple[int, int], player_id: int
) -> int:
    if np.all(game_board[position[0], :] == player_id):
        return 1
    if np.all(game_board[:, position[1]] == player_id):
        return 1
    # First condition is anti-diagonal and second for diagonal
    if np.sum(position) == 2 or position[0] == position[1]:
        if np.all(np.diag(game_board) == player_id):
            return 1
        if np.all(np.diag(np.fliplr(game_board)) == player_id):
            return 1
    return 0


def _show_game_board(
    game_board: np.ndarray, show_game_board_dict: Dict[int, str]
) -> None:
    print("Game Board status")
    line_splitter = "-------------"
    print(line_splitter)
    for row in range(3):
        row_str = "|"
        for col in range(3):
            row_str += f" {show_game_board_dict[game_board[row, col]]} |"
        print(f"{row_str}\n{line_splitter}")


def _get_winning_move(game_board, player, not_player):
    # Defensive move
    # If any of the winning moves are available for not_player
    for row in range(3):
        flag = 0
        counter = 0
        for col in range(3):
            if game_board[row, col] == not_player:
                flag = 1
                break
            elif game_board[row, col] == player:
                counter += 1
            elif game_board[row, col] == 0:
                empty_column = col
        if counter == 2 and flag == 0:
            return row, empty_column
    for col in range(3):
        flag = 0
        counter = 0
        for row in range(3):
            if game_board[row, col] == not_player:
                flag = 1
                break
            elif game_board[row, col] == player:
                counter += 1
            elif game_board[row, col] == 0:
                empty_row = row
        if counter == 2 and flag == 0:
            return empty_row, col

        # Check diagonal
    flag = 0
    counter = 0
    for i in range(3):
        if game_board[i, i] == not_player:
            flag = 1
            break
        elif game_board[i, i] == player:
            counter += 1
        elif game_board[i, i] == 0:
            empty_diag = i
    if counter == 2 and flag == 0:
        return empty_diag, empty_diag

    # Check anti-diagonal
    flag = 0
    counter = 0
    for i in range(3):
        if game_board[i, 2 - i] == not_player:
            flag = 1
            break
        elif game_board[i, 2 - i] == player:
            counter += 1
        elif game_board[i, 2 - i] == 0:
            empty_anti_diag_row = i
            empty_anti_diag_col = 2 - i
    if counter == 2 and flag == 0:
        return empty_anti_diag_row, empty_anti_diag_col

    return None


def _make_computer_move(game_board, player):
    # Check all rows, all columns, anti-diagonal and diagonal
    # Winning move
    # If any of them have 2 with player and 1 empty, select that
    # else, if they have 1 with player and rest empty, select that
    # else if all are empty, select that

    not_player = 1 if player == 2 else 2

    winning_coords = _get_winning_move(game_board, player, not_player)
    losing_coords = _get_winning_move(game_board, not_player, player)

    if winning_coords:
        return winning_coords
    elif losing_coords:
        return losing_coords
    else:
        while True:
            row = randint(0, 2)
            col = randint(0, 2)
            if game_board[row, col] == 1:
                continue
            else:
                break

    return (row, col)


@click.command()
def game_loop() -> None:
    """
    Game loop that runs each move and outputs the game board.
    It stops if the board is full or if a player has won.
    If a player wins, it reports which player has won.
    """
    player_choice = click.prompt(
        "Do you want computer to be first player?",
        type=click.Choice(["y", "n"]),
    )
    if player_choice:
        computer_player = 0
    else:
        computer_player = 1

    starting_player = click.prompt(
        "Enter starting input choice",
        type=click.Choice(["o", "x"]),
    )

    next_player = starting_player
    game_board = np.zeros((3, 3), dtype=np.short)
    player_to_number_dict = {"o": 1, "x": 2}
    show_game_board_dict = {0: " ", 1: "o", 2: "x"}
    next_player_dict = {"o": "x", "x": "o"}

    # Assumption: first move is always player
    moves_counter = 0
    while True:
        if moves_counter % 2 == computer_player:
            coordinates = _make_computer_move(
                game_board, player_to_number_dict[next_player]
            )
            # FIXME: redo
            print("Computer is thinking now")
            time.sleep(1)
            print("Computer making its move")
        else:
            injected_tic_tac_toe_input_checker_func = partial(
                _convert_to_tic_tac_toe_input, game_board=game_board
            )
            coordinates = click.prompt(
                "Enter the row and column on the Tic Tac Toe board (x,y)",
                type=click.STRING,
                value_proc=injected_tic_tac_toe_input_checker_func,
            )

        game_board[coordinates] = player_to_number_dict[next_player]

        _show_game_board(game_board, show_game_board_dict)
        if _is_game_over(game_board, coordinates, player_to_number_dict[next_player]):
            print(f"Game over. Player {next_player} won.")
            return

        next_player = next_player_dict[next_player]

        moves_counter += 1
        if moves_counter == 9:
            print("Game finished in a draw. Try again.")
            return


if __name__ == "__main__":
    game_loop()
