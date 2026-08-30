"""Tests for the value-of-information engine.

The claim under test is that the variance ranking and the decision ranking are
different orderings, so the load-bearing fixture is built to separate them
completely: one parameter enters *both* options identically — moving the payoff
enormously and the difference between options not at all — and another enters
them with opposite signs on a tenth of the spread. The first must dominate the
variance ranking and buy almost none of the decision value; the second must do
the reverse.

The second claim is about EVSI, and it is tested at its two limits rather than
at one arbitrary sample size, because the limits are what make it the right
construction: zero observations must be worth exactly nothing, and a very large
study must converge on EVPPI without exceeding it.

Several relationships here hold by theory rather than by estimation —
``0 <= EVSI <= EVPPI <= EVPI`` — and are asserted as invariants. The estimator
is biased upward, which is why the "cannot change the choice" tests are written
against a share of EVPI and against the bias *shrinking with draws*, rather than
against an exact zero that would never appear in real output.
"""

import math
import os
import sqlite3
import tempfile

import pytest

from src.utils import value_of_information
from src.utils.value_of_information import (
    DEFAULT_BINS,
    ENGINE_VERSION,
    LOUD_VARIANCE_SHARE,
    MIN_DRAWS,
    MIN_OPTIONS,
    NEGLIGIBLE_SHARE_OF_EVPI,
    VOIError,
    analyse,
    baseline_decision,
    build_option,
    build_parameter,
    compare_rankings,
    delete_analysis,
    demo_abatement_decision,
    demo_decision,
    evpi,
    evppi,
    evppi_ranking,
    evsi,
    expected_net_benefit_of_sampling,
    get_analyses,
    get_voi_notes,
    save_analysis,
    simulate,
    summarise,
    variance_ranking,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_db(monkeypatch):
    handle, path = tempfile.mkstemp(suffix=".db")
    os.close(handle)
    monkeypatch.setattr(value_of_information, "DB_NAME", path)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


def _certain_decision():
    """No uncertainty that matters: one option always wins by a mile."""
    parameters = [build_parameter("noise", "normal", mean=0.0, sd=1.0)]
    options = [
        build_option("Good", lambda state: 1000.0 + state["noise"]),
        build_option("Bad", lambda state: 10.0 + state["noise"]),
    ]
    return options, parameters


def _coin_flip_decision():
    """A single parameter whose sign decides everything."""
    parameters = [build_parameter("swing", "normal", mean=0.0, sd=100.0)]
    options = [
        build_option("Up", lambda state: state["swing"]),
        build_option("Down", lambda state: -state["swing"]),
    ]
    return options, parameters


# ---------------------------------------------------------------------------
# build_parameter
# ---------------------------------------------------------------------------


def test_a_normal_parameter_is_updatable():
    assert build_parameter("a", "normal", mean=1.0, sd=2.0)["updatable"] is True


def test_a_uniform_parameter_is_not_updatable():
    assert build_parameter("a", "uniform", low=0.0, high=1.0)["updatable"] is False


def test_a_triangular_mode_defaults_to_the_midpoint():
    parameter = build_parameter("a", "triangular", low=0.0, high=10.0)
    assert parameter["mode"] == pytest.approx(5.0)


def test_an_unknown_distribution_is_refused():
    with pytest.raises(VOIError, match="Distribution must be"):
        build_parameter("a", "beta")


def test_a_negative_sd_is_refused():
    with pytest.raises(VOIError, match="non-negative sd"):
        build_parameter("a", "normal", sd=-1.0)


def test_a_uniform_without_bounds_is_refused():
    with pytest.raises(VOIError, match="low < high"):
        build_parameter("a", "uniform")


def test_an_inverted_uniform_is_refused():
    with pytest.raises(VOIError, match="low < high"):
        build_parameter("a", "uniform", low=5.0, high=1.0)


def test_a_mode_outside_the_bounds_is_refused():
    with pytest.raises(VOIError, match="low <= mode <= high"):
        build_parameter("a", "triangular", low=0.0, high=10.0, mode=20.0)


# ---------------------------------------------------------------------------
# build_option
# ---------------------------------------------------------------------------


def test_an_option_needs_a_callable_payoff():
    with pytest.raises(VOIError, match="callable payoff"):
        build_option("a", 42.0)


def test_an_option_cost_is_subtracted_from_the_payoff():
    parameters = [build_parameter("x", "normal", mean=0.0, sd=1.0)]
    options = [
        build_option("free", lambda state: 100.0),
        build_option("costly", lambda state: 100.0, cost=30.0),
    ]
    simulation = simulate(options, parameters, draws=MIN_DRAWS)
    decision = baseline_decision(simulation)
    assert decision["expected_payoffs"][1] == pytest.approx(70.0)


def test_a_non_numeric_cost_is_refused():
    with pytest.raises(VOIError, match="non-numeric cost"):
        build_option("a", lambda state: 1.0, cost="free")


# ---------------------------------------------------------------------------
# Validation — the refusals
# ---------------------------------------------------------------------------


def test_a_single_option_is_not_a_decision():
    parameters = [build_parameter("x", "normal", sd=1.0)]
    options = [build_option("only", lambda state: 1.0)]
    with pytest.raises(VOIError, match="undefined without a decision"):
        simulate(options, parameters)


def test_the_refusal_explains_why_one_option_is_not_a_choice():
    parameters = [build_parameter("x", "normal", sd=1.0)]
    options = [build_option("only", lambda state: 1.0)]
    with pytest.raises(VOIError, match="nothing information could change"):
        simulate(options, parameters)


def test_no_parameters_is_refused():
    options, _ = _coin_flip_decision()
    with pytest.raises(VOIError, match="At least one uncertain parameter"):
        simulate(options, [])


def test_duplicate_option_names_are_refused():
    parameters = [build_parameter("x", "normal", sd=1.0)]
    options = [
        build_option("same", lambda state: 1.0),
        build_option("same", lambda state: 2.0),
    ]
    with pytest.raises(VOIError, match="appears twice"):
        simulate(options, parameters)


def test_duplicate_parameter_names_are_refused():
    options, _ = _coin_flip_decision()
    parameters = [
        build_parameter("x", "normal", sd=1.0),
        build_parameter("x", "normal", sd=2.0),
    ]
    with pytest.raises(VOIError, match="appears twice"):
        simulate(options, parameters)


def test_too_few_draws_is_refused():
    options, parameters = _coin_flip_decision()
    with pytest.raises(VOIError, match="At least %d draws" % MIN_DRAWS):
        simulate(options, parameters, draws=10)


def test_raw_option_mappings_are_refused():
    _, parameters = _coin_flip_decision()
    with pytest.raises(VOIError, match="build_option"):
        simulate([{"name": "a"}, {"name": "b"}], parameters)


def test_raw_parameter_mappings_are_refused():
    options, _ = _coin_flip_decision()
    with pytest.raises(VOIError, match="build_parameter"):
        simulate(options, [{"name": "x"}])


def test_a_non_numeric_payoff_is_refused():
    parameters = [build_parameter("x", "normal", sd=1.0)]
    options = [
        build_option("bad", lambda state: float("nan")),
        build_option("fine", lambda state: 1.0),
    ]
    with pytest.raises(VOIError, match="non-numeric payoff"):
        simulate(options, parameters, draws=MIN_DRAWS)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def test_every_option_sees_the_same_draws():
    """The comparison between two options must hold the world fixed."""
    parameters = [build_parameter("x", "normal", mean=0.0, sd=10.0)]
    options = [
        build_option("a", lambda state: state["x"]),
        build_option("b", lambda state: state["x"]),
    ]
    simulation = simulate(options, parameters, draws=500)
    for row in simulation["payoffs"]:
        assert row[0] == pytest.approx(row[1])


def test_simulation_is_deterministic_for_a_seed():
    options, parameters = _coin_flip_decision()
    first = simulate(options, parameters, draws=500, seed=7)
    second = simulate(options, parameters, draws=500, seed=7)
    assert first["payoffs"] == second["payoffs"]


def test_simulation_records_every_parameter_sample():
    options, parameters = demo_abatement_decision()
    simulation = simulate(options, parameters, draws=500)
    for name in simulation["parameters"]:
        assert len(simulation["samples"][name]) == 500


def test_a_uniform_parameter_stays_inside_its_bounds():
    parameters = [build_parameter("x", "uniform", low=2.0, high=5.0)]
    options = [
        build_option("a", lambda state: state["x"]),
        build_option("b", lambda state: -state["x"]),
    ]
    simulation = simulate(options, parameters, draws=1000)
    assert all(2.0 <= value <= 5.0 for value in simulation["samples"]["x"])


def test_a_lognormal_parameter_is_positive():
    parameters = [build_parameter("x", "lognormal", mean=0.0, sd=1.0)]
    options = [
        build_option("a", lambda state: state["x"]),
        build_option("b", lambda state: -state["x"]),
    ]
    simulation = simulate(options, parameters, draws=1000)
    assert all(value > 0 for value in simulation["samples"]["x"])


# ---------------------------------------------------------------------------
# The baseline decision
# ---------------------------------------------------------------------------


def test_the_dominant_option_is_recommended():
    options, parameters = _certain_decision()
    decision = baseline_decision(simulate(options, parameters, draws=1000))
    assert decision["recommended"] == "Good"
    assert decision["probability_recommended_is_best"] == pytest.approx(1.0)


def test_a_certain_decision_has_no_opportunity_loss():
    options, parameters = _certain_decision()
    decision = baseline_decision(simulate(options, parameters, draws=1000))
    assert decision["expected_opportunity_loss"] == pytest.approx(0.0)


def test_a_coin_flip_decision_is_right_about_half_the_time():
    options, parameters = _coin_flip_decision()
    decision = baseline_decision(simulate(options, parameters, draws=8000))
    assert decision["probability_recommended_is_best"] == pytest.approx(0.5, abs=0.05)


def test_the_probabilities_of_being_best_sum_to_one():
    options, parameters = demo_abatement_decision()
    decision = baseline_decision(simulate(options, parameters, draws=2000))
    assert sum(decision["probability_best"]) == pytest.approx(1.0)


def test_probability_wrong_is_the_complement():
    options, parameters = demo_abatement_decision()
    decision = baseline_decision(simulate(options, parameters, draws=2000))
    assert decision["probability_wrong"] == pytest.approx(
        1.0 - decision["probability_recommended_is_best"]
    )


def test_probability_and_magnitude_of_being_wrong_can_disagree():
    """Second-best by a rounding error, most of the time, is not a real risk."""
    parameters = [build_parameter("x", "normal", mean=0.0, sd=1.0)]
    options = [
        build_option("steady", lambda state: 100.0),
        build_option("jittery", lambda state: 100.0 + state["x"]),
    ]
    decision = baseline_decision(simulate(options, parameters, draws=8000))
    assert decision["probability_wrong"] > 0.4
    assert decision["expected_opportunity_loss"] < 1.0


# ---------------------------------------------------------------------------
# EVPI
# ---------------------------------------------------------------------------


def test_a_certain_decision_has_zero_evpi():
    options, parameters = _certain_decision()
    assert evpi(simulate(options, parameters, draws=2000))["evpi"] == pytest.approx(0.0)


def test_evpi_equals_the_expected_opportunity_loss():
    """Not a coincidence: they are the same quantity written two ways."""
    options, parameters = demo_abatement_decision()
    simulation = simulate(options, parameters, draws=4000)
    assert evpi(simulation)["evpi"] == pytest.approx(
        baseline_decision(simulation)["expected_opportunity_loss"], rel=1e-9
    )


def test_evpi_is_never_negative():
    for factory in (_certain_decision, _coin_flip_decision, demo_abatement_decision):
        options, parameters = factory()
        assert evpi(simulate(options, parameters, draws=1000))["evpi"] >= 0.0


def test_evpi_grows_as_the_decision_gets_closer():
    far = demo_decision(gap=400.0)
    close = demo_decision(gap=0.0)
    far_value = evpi(simulate(*far, draws=4000))["evpi"]
    close_value = evpi(simulate(*close, draws=4000))["evpi"]
    assert close_value > far_value


def test_perfect_information_never_makes_the_decision_worse():
    options, parameters = demo_abatement_decision()
    result = evpi(simulate(options, parameters, draws=2000))
    assert (
        result["expected_payoff_with_perfect_information"]
        >= result["expected_payoff_now"]
    )


# ---------------------------------------------------------------------------
# EVPPI — the load-bearing contrast
# ---------------------------------------------------------------------------


def test_the_parameter_that_decides_the_choice_carries_the_decision_value():
    options, parameters = demo_decision(loud_spread=400.0, decisive_spread=30.0)
    simulation = simulate(options, parameters, draws=8000)
    decisive = evppi(simulation, "heat_pump_performance")
    assert decisive["share_of_evpi"] > 0.9


def test_the_parameter_that_drives_the_variance_carries_almost_none_of_it():
    """It enters both options identically, so it cannot change the choice."""
    options, parameters = demo_decision(loud_spread=400.0, decisive_spread=30.0)
    simulation = simulate(options, parameters, draws=8000)
    loud = evppi(simulation, "grid_intensity")
    variance = variance_ranking(simulation)
    assert variance["shares"]["grid_intensity"] > 0.9
    assert loud["share_of_evpi"] < NEGLIGIBLE_SHARE_OF_EVPI


def test_the_upward_bias_shrinks_as_the_draws_grow():
    """The true EVPPI here is zero; what is left is the estimator's own noise."""
    options, parameters = demo_decision()
    small = evppi(simulate(options, parameters, draws=2000), "grid_intensity")["evppi"]
    large = evppi(simulate(options, parameters, draws=40000), "grid_intensity")["evppi"]
    assert large < small


def test_evppi_never_exceeds_evpi():
    options, parameters = demo_abatement_decision()
    simulation = simulate(options, parameters, draws=4000)
    ceiling = evpi(simulation)["evpi"]
    for name in simulation["parameters"]:
        entry = evppi(simulation, name)
        assert entry["evppi"] <= ceiling * 1.05
        assert entry["above_evpi"] is False


def test_evppi_is_never_negative():
    options, parameters = demo_abatement_decision()
    simulation = simulate(options, parameters, draws=4000)
    for name in simulation["parameters"]:
        assert evppi(simulation, name)["evppi"] >= 0.0


def test_a_single_parameter_decision_has_evppi_equal_to_evpi():
    """With one uncertainty, resolving it *is* resolving everything."""
    options, parameters = _coin_flip_decision()
    simulation = simulate(options, parameters, draws=20000)
    ceiling = evpi(simulation)["evpi"]
    assert evppi(simulation, "swing")["evppi"] == pytest.approx(ceiling, rel=0.05)


def test_evppi_of_an_unknown_parameter_is_refused():
    options, parameters = _coin_flip_decision()
    simulation = simulate(options, parameters, draws=1000)
    with pytest.raises(VOIError, match="not in this simulation"):
        evppi(simulation, "nonexistent")


def test_the_ranking_covers_every_parameter():
    options, parameters = demo_abatement_decision()
    simulation = simulate(options, parameters, draws=2000)
    ranking = evppi_ranking(simulation)
    assert set(ranking["order"]) == set(simulation["parameters"])


def test_the_ranking_is_ordered_by_value():
    options, parameters = demo_abatement_decision()
    ranking = evppi_ranking(simulate(options, parameters, draws=4000))
    values = [entry["evppi"] for entry in ranking["entries"]]
    assert values == sorted(values, reverse=True)


# ---------------------------------------------------------------------------
# The two rankings
# ---------------------------------------------------------------------------


def test_the_rankings_disagree_on_the_contrast_fixture():
    options, parameters = demo_decision()
    comparison = compare_rankings(simulate(options, parameters, draws=8000))
    assert comparison["agree"] is False
    assert "grid_intensity" in comparison["wasted_measurements"]


def test_a_wasted_measurement_is_loud_and_worthless():
    options, parameters = demo_decision()
    comparison = compare_rankings(simulate(options, parameters, draws=8000))
    row = next(
        item for item in comparison["rows"] if item["parameter"] == "grid_intensity"
    )
    assert row["variance_share"] > LOUD_VARIANCE_SHARE
    assert row["share_of_evpi"] < NEGLIGIBLE_SHARE_OF_EVPI


def test_the_rankings_also_disagree_on_the_abatement_decision():
    options, parameters = demo_abatement_decision()
    comparison = compare_rankings(simulate(options, parameters, draws=8000))
    assert comparison["largest_move"] > 0


def test_every_parameter_appears_in_both_rankings():
    options, parameters = demo_abatement_decision()
    comparison = compare_rankings(simulate(options, parameters, draws=2000))
    assert len(comparison["rows"]) == len(parameters)
    for row in comparison["rows"]:
        assert row["variance_rank"] >= 1
        assert row["decision_rank"] >= 1


def test_variance_shares_lie_between_zero_and_one():
    options, parameters = demo_abatement_decision()
    shares = variance_ranking(simulate(options, parameters, draws=2000))["shares"]
    assert all(0.0 <= value <= 1.0 for value in shares.values())


# ---------------------------------------------------------------------------
# EVSI — tested at its limits
# ---------------------------------------------------------------------------


def test_a_study_of_zero_observations_is_worth_nothing():
    options, parameters = demo_decision()
    simulation = simulate(options, parameters, draws=4000)
    result = evsi(simulation, parameters, "heat_pump_performance", 0, 60.0)
    assert result["evsi"] == pytest.approx(0.0)


def test_a_very_large_study_converges_on_evppi():
    options, parameters = demo_decision()
    simulation = simulate(options, parameters, draws=8000)
    ceiling = evppi(simulation, "heat_pump_performance")["evppi"]
    result = evsi(simulation, parameters, "heat_pump_performance", 100000, 60.0)
    assert result["evsi"] == pytest.approx(ceiling, rel=0.05)


def test_evsi_never_exceeds_evppi():
    options, parameters = demo_decision()
    simulation = simulate(options, parameters, draws=8000)
    for size in (1, 5, 25, 200, 5000):
        result = evsi(simulation, parameters, "heat_pump_performance", size, 60.0)
        assert result["above_evppi"] is False


def test_evsi_rises_with_the_sample_size():
    options, parameters = demo_decision()
    simulation = simulate(options, parameters, draws=8000)
    values = [
        evsi(simulation, parameters, "heat_pump_performance", size, 60.0)["evsi"]
        for size in (1, 5, 25, 200)
    ]
    assert values == sorted(values)


def test_a_noisier_measurement_is_worth_less():
    options, parameters = demo_decision()
    simulation = simulate(options, parameters, draws=8000)
    precise = evsi(simulation, parameters, "heat_pump_performance", 20, 20.0)["evsi"]
    vague = evsi(simulation, parameters, "heat_pump_performance", 20, 400.0)["evsi"]
    assert precise > vague


def test_evsi_refuses_a_non_conjugate_prior():
    options, parameters = demo_abatement_decision()
    simulation = simulate(options, parameters, draws=1000)
    with pytest.raises(VOIError, match="only a normal prior is implemented"):
        evsi(simulation, parameters, "occupancy", 10, 0.2)


def test_evsi_refuses_a_perfect_measurement():
    options, parameters = demo_decision()
    simulation = simulate(options, parameters, draws=1000)
    with pytest.raises(VOIError, match="use EVPPI"):
        evsi(simulation, parameters, "heat_pump_performance", 10, 0.0)


def test_evsi_refuses_a_negative_sample_size():
    options, parameters = demo_decision()
    simulation = simulate(options, parameters, draws=1000)
    with pytest.raises(VOIError, match="cannot be negative"):
        evsi(simulation, parameters, "heat_pump_performance", -1, 60.0)


def test_evsi_refuses_a_parameter_with_no_prior_uncertainty():
    parameters = [
        build_parameter("fixed", "normal", mean=1.0, sd=0.0),
        build_parameter("real", "normal", mean=0.0, sd=10.0),
    ]
    options = [
        build_option("a", lambda state: state["real"] + state["fixed"]),
        build_option("b", lambda state: -state["real"] + state["fixed"]),
    ]
    simulation = simulate(options, parameters, draws=1000)
    with pytest.raises(VOIError, match="nothing a study could resolve"):
        evsi(simulation, parameters, "fixed", 10, 1.0)


def test_evsi_refuses_an_unknown_parameter():
    options, parameters = demo_decision()
    simulation = simulate(options, parameters, draws=1000)
    with pytest.raises(VOIError, match="not in this decision"):
        evsi(simulation, parameters, "nonexistent", 10, 1.0)


# ---------------------------------------------------------------------------
# ENBS
# ---------------------------------------------------------------------------


def test_a_free_study_is_always_worth_running():
    options, parameters = demo_decision()
    simulation = simulate(options, parameters, draws=4000)
    result = expected_net_benefit_of_sampling(
        simulation,
        parameters,
        "heat_pump_performance",
        measurement_sd=60.0,
        sample_sizes=[0, 10, 100],
    )
    assert result["worthwhile"] is True
    assert result["optimum"]["sample_size"] > 0


def test_an_expensive_study_does_not_pay_for_itself():
    options, parameters = demo_decision()
    simulation = simulate(options, parameters, draws=4000)
    result = expected_net_benefit_of_sampling(
        simulation,
        parameters,
        "heat_pump_performance",
        measurement_sd=60.0,
        sample_sizes=[0, 10, 100],
        fixed_cost=1e6,
    )
    assert result["worthwhile"] is False
    assert "Act on the data you have" in result["headline"]


def test_a_larger_population_makes_a_study_worthwhile():
    """The same study, spread over more households that share the decision."""
    options, parameters = demo_decision()
    simulation = simulate(options, parameters, draws=4000)

    def run(population):
        return expected_net_benefit_of_sampling(
            simulation,
            parameters,
            "heat_pump_performance",
            measurement_sd=60.0,
            sample_sizes=[0, 20, 100],
            fixed_cost=500.0,
            population=population,
        )

    assert run(1.0)["worthwhile"] is False
    assert run(5000.0)["worthwhile"] is True


def test_the_net_benefit_is_value_minus_cost():
    options, parameters = demo_decision()
    simulation = simulate(options, parameters, draws=2000)
    result = expected_net_benefit_of_sampling(
        simulation,
        parameters,
        "heat_pump_performance",
        measurement_sd=60.0,
        sample_sizes=[0, 10],
        fixed_cost=50.0,
        cost_per_observation=2.0,
        population=10.0,
    )
    for row in result["rows"]:
        assert row["net_benefit"] == pytest.approx(row["population_evsi"] - row["cost"])


def test_a_zero_sample_costs_nothing():
    options, parameters = demo_decision()
    simulation = simulate(options, parameters, draws=2000)
    result = expected_net_benefit_of_sampling(
        simulation,
        parameters,
        "heat_pump_performance",
        measurement_sd=60.0,
        sample_sizes=[0, 10],
        fixed_cost=500.0,
    )
    assert result["rows"][0]["cost"] == 0.0
    assert result["rows"][0]["net_benefit"] == pytest.approx(0.0)


def test_enbs_refuses_negative_costs():
    options, parameters = demo_decision()
    simulation = simulate(options, parameters, draws=1000)
    with pytest.raises(VOIError, match="Costs cannot be negative"):
        expected_net_benefit_of_sampling(
            simulation, parameters, "heat_pump_performance", 60.0, [10], fixed_cost=-1.0
        )


def test_enbs_refuses_a_non_positive_population():
    options, parameters = demo_decision()
    simulation = simulate(options, parameters, draws=1000)
    with pytest.raises(VOIError, match="Population must be positive"):
        expected_net_benefit_of_sampling(
            simulation, parameters, "heat_pump_performance", 60.0, [10], population=0.0
        )


def test_enbs_refuses_an_empty_sample_size_list():
    options, parameters = demo_decision()
    simulation = simulate(options, parameters, draws=1000)
    with pytest.raises(VOIError, match="At least one sample size"):
        expected_net_benefit_of_sampling(
            simulation, parameters, "heat_pump_performance", 60.0, []
        )


def test_enbs_refuses_a_negative_sample_size():
    options, parameters = demo_decision()
    simulation = simulate(options, parameters, draws=1000)
    with pytest.raises(VOIError, match="cannot be negative"):
        expected_net_benefit_of_sampling(
            simulation, parameters, "heat_pump_performance", 60.0, [10, -5]
        )


# ---------------------------------------------------------------------------
# analyse
# ---------------------------------------------------------------------------


def test_analyse_recommends_acting_when_evpi_is_below_the_cheapest_measurement():
    options, parameters = _certain_decision()
    result = analyse(options, parameters, draws=1000, cheapest_measurement=1.0)
    assert result["act_on_what_you_have"] is True
    assert "Act on what you have" in result["headline"]


def test_analyse_does_not_recommend_acting_when_information_is_valuable():
    options, parameters = demo_decision()
    result = analyse(options, parameters, draws=4000, cheapest_measurement=0.01)
    assert result["act_on_what_you_have"] is False


def test_analyse_carries_the_engine_version():
    options, parameters = _coin_flip_decision()
    assert analyse(options, parameters, draws=1000)["engine_version"] == ENGINE_VERSION


def test_analyse_reports_the_decision_the_ceiling_and_the_ranking():
    options, parameters = demo_abatement_decision()
    result = analyse(options, parameters, draws=2000)
    assert result["decision"]["recommended"] in result["options"]
    assert result["evpi"]["evpi"] >= 0.0
    assert result["evppi"]["order"]


def test_analyse_records_the_draws_and_bins_used():
    options, parameters = _coin_flip_decision()
    result = analyse(options, parameters, draws=1000, bins=12)
    assert result["draws"] == 1000
    assert result["bins"] == 12


def test_analyse_refuses_a_single_option():
    parameters = [build_parameter("x", "normal", sd=1.0)]
    with pytest.raises(VOIError, match="undefined without a decision"):
        analyse([build_option("only", lambda state: 1.0)], parameters)


# ---------------------------------------------------------------------------
# Notes and summaries
# ---------------------------------------------------------------------------


def test_notes_lead_with_the_headline():
    options, parameters = demo_decision()
    result = analyse(options, parameters, draws=2000)
    assert get_voi_notes(result)[0] == result["headline"]


def test_notes_report_how_often_and_how_much_the_choice_is_wrong():
    options, parameters = demo_abatement_decision()
    notes = get_voi_notes(analyse(options, parameters, draws=2000))
    assert any("displayed identically today" in note for note in notes)


def test_notes_flag_a_wasted_measurement():
    options, parameters = demo_decision()
    notes = get_voi_notes(analyse(options, parameters, draws=8000))
    assert any("cannot change the choice" in note for note in notes)


def test_notes_say_when_no_measurement_can_help():
    options, parameters = _certain_decision()
    notes = get_voi_notes(analyse(options, parameters, draws=1000, cheapest_measurement=1.0))
    assert any("currently cannot produce" in note for note in notes)


def test_summary_is_one_line():
    options, parameters = demo_abatement_decision()
    summary = summarise(analyse(options, parameters, draws=2000))
    assert "\n" not in summary
    assert "EVPI" in summary


# ---------------------------------------------------------------------------
# Demo decisions
# ---------------------------------------------------------------------------


def test_the_contrast_demo_puts_the_loud_parameter_in_both_options():
    options, parameters = demo_decision(loud_spread=500.0, decisive_spread=1.0)
    simulation = simulate(options, parameters, draws=2000)
    differences = [row[0] - row[1] for row in simulation["payoffs"]]
    # The difference between the options depends only on the decisive parameter,
    # so its spread must be small even when grid_intensity is very wide.
    assert math.sqrt(sum(d * d for d in differences) / len(differences)) < 10.0


def test_a_large_gap_removes_the_decision_uncertainty():
    options, parameters = demo_decision(gap=5000.0)
    decision = baseline_decision(simulate(options, parameters, draws=2000))
    assert decision["probability_recommended_is_best"] == pytest.approx(1.0)


def test_the_abatement_demo_has_three_measures_and_four_parameters():
    options, parameters = demo_abatement_decision()
    assert len(options) == 3
    assert len(parameters) == 4


def test_the_abatement_demo_is_deterministic():
    options, parameters = demo_abatement_decision()
    first = simulate(options, parameters, draws=500, seed=3)
    second = simulate(options, parameters, draws=500, seed=3)
    assert first["payoffs"] == second["payoffs"]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_and_read_back_an_analysis(temp_db):
    options, parameters = demo_abatement_decision()
    result = analyse(options, parameters, draws=1000)
    analysis_id = save_analysis("user-1", result, label="Retrofit")
    assert analysis_id is not None

    analyses = get_analyses("user-1")
    assert len(analyses) == 1
    assert analyses[0]["label"] == "Retrofit"
    assert analyses[0]["recommended"] == result["decision"]["recommended"]
    assert analyses[0]["payload"]["engine_version"] == ENGINE_VERSION


def test_an_act_now_analysis_is_stored_as_such(temp_db):
    options, parameters = _certain_decision()
    result = analyse(options, parameters, draws=1000, cheapest_measurement=1.0)
    save_analysis("user-1", result)
    assert get_analyses("user-1")[0]["act_now"] is True


def test_analyses_are_scoped_to_their_user(temp_db):
    options, parameters = _coin_flip_decision()
    save_analysis("user-1", analyse(options, parameters, draws=1000))
    assert get_analyses("user-2") == []


def test_saving_without_a_user_is_a_no_op(temp_db):
    options, parameters = _coin_flip_decision()
    assert save_analysis("", analyse(options, parameters, draws=1000)) is None


def test_saving_a_non_result_is_a_no_op(temp_db):
    assert save_analysis("user-1", {}) is None


def test_delete_removes_only_the_named_analysis(temp_db):
    options, parameters = _coin_flip_decision()
    result = analyse(options, parameters, draws=1000)
    first = save_analysis("user-1", result, label="one")
    save_analysis("user-1", result, label="two")
    assert delete_analysis("user-1", first) is True
    remaining = get_analyses("user-1")
    assert len(remaining) == 1
    assert remaining[0]["label"] == "two"


def test_delete_refuses_another_users_analysis(temp_db):
    options, parameters = _coin_flip_decision()
    analysis_id = save_analysis("user-1", analyse(options, parameters, draws=1000))
    assert delete_analysis("user-2", analysis_id) is False


def test_delete_without_a_user_is_false(temp_db):
    assert delete_analysis("", 1) is False


def test_reads_without_a_user_are_empty(temp_db):
    assert get_analyses(None) == []


def test_analyses_come_back_newest_first(temp_db):
    options, parameters = _coin_flip_decision()
    result = analyse(options, parameters, draws=1000)
    save_analysis("user-1", result, label="older")
    save_analysis("user-1", result, label="newer")
    assert [entry["label"] for entry in get_analyses("user-1")] == ["newer", "older"]


def test_storage_failure_is_swallowed_not_raised(monkeypatch):
    """A dashboard must render when the database is unavailable."""

    def explode(*_args, **_kwargs):
        raise sqlite3.Error("disk is on fire")

    monkeypatch.setattr(value_of_information, "_connect", explode)
    options, parameters = _coin_flip_decision()
    result = analyse(options, parameters, draws=1000)
    assert save_analysis("user-1", result) is None
    assert get_analyses("user-1") == []
    assert delete_analysis("user-1", 1) is False


def test_a_corrupt_payload_reads_back_as_empty(temp_db):
    options, parameters = _coin_flip_decision()
    save_analysis("user-1", analyse(options, parameters, draws=1000))
    with sqlite3.connect(temp_db) as conn:
        conn.execute("UPDATE value_of_information_analyses SET payload = 'not json'")
    assert get_analyses("user-1")[0]["payload"] == {}
