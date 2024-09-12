"""This module contains function to quantify the level of interactions in the games."""

import copy
from typing import Optional

import numpy as np
import scipy

import shapiq
from utils import setup_game

import matplotlib.pyplot as plt

from shapiq.interaction_values import InteractionValues


from shapiq.utils import powerset


def approximated_game(interaction_index):
    n_players = interaction_index.n_players
    grand_coalition_set = set(range(n_players))
    approximation_lookup = {}
    approximation_values = np.zeros(2**n_players)
    for coalition_pos, coalition in enumerate(powerset(grand_coalition_set)):
        approximation_lookup[coalition] = coalition_pos
        for interaction in powerset(coalition):
            approximation_values[coalition_pos] += interaction_index[interaction]

    baseline_value = approximation_values[approximation_lookup[tuple()]]
    approximation = InteractionValues(
        index=interaction_index.index,
        max_order=n_players,
        n_players=n_players,
        min_order=0,
        baseline_value=baseline_value,
        interaction_lookup=approximation_lookup,
        values=approximation_values,
    )
    return approximation


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
        sv_weight = 0
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
    game: shapiq.Game, indices: Optional[list[str]] = None
) -> None:
    """Evaluates the game.

    Args:
        game: The game to evaluate.
        indices: The indices to approximate the game with. Default is `None`.
    """
    computer = shapiq.ExactComputer(n_players=game.n_players, game_fun=game)
    #computer.baseline_value = float(game.normalization_value)
    game_values = _convert_game_to_interaction(computer)

    approximations = {}
    for index in indices:
        approximations[index] = {}
        for order in range(1, game.n_players + 1):
            interactions = computer.shapley_interaction(index=index, order=order)
            approximations[index][order] = approximated_game(interactions)
    approximation_errors = get_approximation_error(approximations, game_values)
    print("R2 Scores per Order:", approximation_errors)
    return approximation_errors


def plot_r2(results,game_id):
    x_vals = list(results.keys())
    y_vals = list(results.values())
    plt.figure()
    plt.plot(
        x_vals,
        y_vals,
    )
    plt.ylim(0, 1.05)
    plt.legend()
    plt.title(game_id)
    plt.xlabel("Explanation Order")
    plt.ylabel("Shapley-weighted R2")
    plt.savefig("plots/r2/r2_"+game_id+".png")
    plt.show()



if __name__ == "__main__":

    # UNIVERSAL
    hpo_game_universal, _, names = setup_game(
        game_type="universal",
        benchmark_name="lcbench",
        normalize_loaded=True,
        instance_index=1,
        n_configs=10_000,
    )

    GAME_LIST = [hpo_game_universal]


    # GLOBAL
    for instance_index in range(34):
        hpo_game_global, _, names = setup_game(
            game_type="global",
            benchmark_name="lcbench",
            normalize_loaded=True,
            instance_index=instance_index,
            n_configs=10_000,
        )
        GAME_LIST.append(hpo_game_global)


    r2_scores = {}

    for game in GAME_LIST:
        r2_scores[game.game_id] = evaluate_game(game=game, indices=["FSII"])
        plot_r2(results=r2_scores[game.game_id]["FSII"],game_id=game.game_id)


