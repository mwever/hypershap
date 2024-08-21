"""This module contains function to quantify the level of interactions in the games."""

import copy
from typing import Optional

import numpy as np
import scipy

import shapiq
from utils import setup_game


def approximate_game_with_interactions(
    game: shapiq.Game,
    indices: Optional[list[str]] = None,
) -> dict[str, dict[int, shapiq.InteractionValues]]:
    """Approximates the game with different interaction indices and orders.

    Args:
        game: The game to approximate.
        indices: The indices to approximate the game with. Default is `None`.

    Returns:
        A dictionary containing the approximated interaction values with the indices and orders as
        keys.
    """
    if indices is None:
        indices = ["k-SII", "FSII", "FBII", "STII"]
    computer = shapiq.ExactComputer(n_players=game.n_players, game_fun=game)
    approximations = {}
    for index in indices:
        approximations[index] = {}
        for order in range(1, game.n_players + 1):
            approximations[index][order] = computer(index=index, order=order)
    return approximations


def _convert_game_to_interaction(exact_computer: shapiq.ExactComputer) -> shapiq.InteractionValues:
    """Converts the exact game values to interaction values.

    Args:
        exact_computer: The exact computer to convert the game values from.

    Returns:
        The interaction values.
    """
    baseline_value = exact_computer.game_values[exact_computer.coalition_lookup[tuple()]]
    game_values = shapiq.InteractionValues(
        values=exact_computer.game_values,
        index="Moebius",
        interaction_lookup=exact_computer.coalition_lookup,
        n_players=exact_computer.n,
        min_order=0,
        max_order=exact_computer.n,
        baseline_value=baseline_value,
    )
    return game_values


def _get_weight(n: int, coalition_size: int, uniform_weights: bool = False) -> float:
    """Computes the weight for the given coalition size.

    Args:
        n: The number of players.
        coalition_size: The size of the coalition.
        uniform_weights: Whether to use uniform weights or a Shapley kernel. Default is `False`
            (Shapley kernel).

    Returns:
        The weight for the given coalition size.
    """
    if uniform_weights:
        return (1 / 2) ** n
    if coalition_size == n or coalition_size == 0:
        sv_weight = 1
    else:
        sv_weight = scipy.special.binom(n, coalition_size) * coalition_size * (n - coalition_size)
        sv_weight = (n - 1) / sv_weight
    return sv_weight


def _get_approximation_weights(n: int, uniform_weights: bool = False) -> np.ndarray:
    """Computes the weights for the approximations.

    Args:
        n: The number of players.
        uniform_weights: Whether to use uniform weights or a Shapley kernel. Default is `False`
            (Shapley kernel).

    Returns:
        The weights for the approximations as a numpy array.
    """
    approximation_weights = np.zeros(2**n, dtype=float)
    grand_coalition_set = set(range(n))
    for coalition_pos, coalition in enumerate(shapiq.powerset(grand_coalition_set)):
        approximation_weights[coalition_pos] = _get_weight(n, len(coalition), uniform_weights)
    return approximation_weights


def get_approximation_error(
    approximations: dict[str, dict[int, shapiq.InteractionValues]],
    game_values: shapiq.InteractionValues,
    uniform_weights: bool = False,
) -> dict[str, dict[int, float]]:
    """Computes the approximation error for the approximated interaction values.

    Args:
        approximations: The approximated interaction values.
        game_values: The exact game values.
        uniform_weights: Whether to use uniform weights or a Shapley kernel. Default is `False`
            (Shapley kernel).

    Returns:
        A dictionary containing the approximation errors for the approximated interaction values.
    """
    weights = _get_approximation_weights(game_values.n_players, uniform_weights=uniform_weights)
    game_values_copy = copy.deepcopy(game_values)
    approximation_errors = {}
    for index, approximations_order in approximations.items():
        approximation_errors[index] = {}
        for order, approximation in approximations_order.items():
            # get the approximation values
            approx = copy.deepcopy(approximations[index][order])
            approx_arr = np.zeros(len(game_values.values))
            approx_arr[: len(approx.values)] = approx.values  # padded with zeros

            # get the difference between the game and si values
            game_values_copy.index = approx.index  # workaround to compare
            difference = approx - game_values_copy  # take the difference between game and si

            # compute the weighted r2 score
            error = np.sum(weights * difference.values**2)
            weighted_average = np.sum(weights * approx_arr) / np.sum(weights)
            total_sum_of_squares = np.sum(weights * (approx_arr - weighted_average) ** 2)
            r2_score = 1 - error / total_sum_of_squares

            approximation_errors[index][order] = r2_score
    return approximation_errors


def evaluate_game(
    game: shapiq.Game, indices: Optional[list[str]] = None, uniform_weights: bool = False
) -> None:
    """Evaluates the game.

    Args:
        game: The game to evaluate.
        indices: The indices to approximate the game with. Default is `None`.
        uniform_weights: Whether to use uniform weights or a Shapley kernel. Default is `False`
            (Shapley kernel).
    """
    computer = shapiq.ExactComputer(n_players=game.n_players, game_fun=game)
    game_values = _convert_game_to_interaction(computer)
    approximations = approximate_game_with_interactions(game, indices=indices)
    approximation_errors = get_approximation_error(approximations, game_values, uniform_weights)
    print("R2 Scores per Order:", approximation_errors)


if __name__ == "__main__":

    hpo_game, _, names = setup_game(
        game_type="local", benchmark_name="rbv2_ranger", normalize_loaded=True
    )
    assert len(names) == hpo_game.n_players

    evaluate_game(hpo_game, indices=["FSII"], uniform_weights=False)
