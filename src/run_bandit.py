"""Run multi-armed bandit model selection experiment.

Fits BPR and ItemGraphPropagation models on the training split, freezes
them, then runs an offline replay simulation where a bandit strategy
learns which model serves better recommendations per user.  Results are
exported to a timestamped run directory under ``./runs/``.

Usage examples
--------------
Epsilon-greedy on validation split (default):
    python -m src.run_bandit --split val --epsilon 0.1

Epsilon-greedy on test split with custom BPR epochs:
    python -m src.run_bandit --split test --epsilon 0.05 --bpr-n-epochs 30

Shuffled user order (ablation):
    python -m src.run_bandit --process-order shuffled --epsilon 0.2

See all options:
    python -m src.run_bandit --help
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from data.dataframes import movies, test, train, users, val
from src.eval.eval import evaluate as evaluate_basic
from src.models.bandit.bandit_model_selector import BanditModelSelector
from src.models.bandit.simulation import BanditSimulationReport, run_bandit_simulation
from src.models.bandit.strategy import (
    ArmSelectionStrategy,
    EpsilonGreedyStrategy,
    ThompsonSamplingStrategy,
)
from src.models.bpr import BPRRecommender
from src.models.graph import ItemGraphPropagationRanker

logger = logging.getLogger(__name__)


# ─── CLI argument parsing ────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-armed bandit model selection experiment.",
    )

    # ── Evaluation ────────────────────────────────────────────────────
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Evaluation split; child models are always fit on train (default: val).",
    )
    parser.add_argument(
        "--k", type=int, default=10,
        help="Top-K cutoff for recommendations and NDCG (default: 10).",
    )
    parser.add_argument(
        "--threshold", type=float, default=4.0,
        help="Relevance threshold for binary metrics (default: 4.0).",
    )
    parser.add_argument(
        "--process-order",
        type=str,
        default="temporal",
        choices=["temporal", "shuffled"],
        help="User processing order in simulation (default: temporal).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    # ── Strategy ──────────────────────────────────────────────────────
    parser.add_argument(
        "--strategy",
        type=str,
        default="epsilon_greedy",
        choices=["epsilon_greedy", "thompson"],
        help="Bandit strategy to use (default: epsilon_greedy).",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.1,
        help="Exploration probability for epsilon-greedy (default: 0.1).",
    )

    # ── Thompson Sampling parameters ─────────────────────────────────
    parser.add_argument(
        "--reward-threshold", type=float, default=0.0,
        help=(
            "Thompson Sampling: continuous rewards >= this are successes. "
            "Must be in [0, 1] (default: 0.0, i.e. any positive NDCG is a success)."
        ),
    )
    parser.add_argument(
        "--prior-alpha", type=float, default=1.0,
        help="Thompson Sampling: initial Beta prior alpha (default: 1.0).",
    )
    parser.add_argument(
        "--prior-beta", type=float, default=1.0,
        help="Thompson Sampling: initial Beta prior beta (default: 1.0).",
    )

    # ── BPR model parameters ─────────────────────────────────────────
    parser.add_argument(
        "--bpr-n-factors", type=int, default=64,
        help="BPR latent factor dimension (default: 64).",
    )
    parser.add_argument(
        "--bpr-n-epochs", type=int, default=20,
        help="BPR training epochs (default: 20).",
    )
    parser.add_argument(
        "--bpr-lr", type=float, default=0.01,
        help="BPR learning rate (default: 0.01).",
    )
    parser.add_argument(
        "--bpr-regularization", type=float, default=0.01,
        help="BPR L2 regularization coefficient (default: 0.01).",
    )

    # ── ItemGraph model parameters ───────────────────────────────────
    parser.add_argument(
        "--graph-alpha", type=float, default=0.85,
        help="ItemGraph propagation damping factor (default: 0.85).",
    )
    parser.add_argument(
        "--graph-steps", type=int, default=2,
        help="ItemGraph propagation steps (default: 2).",
    )
    parser.add_argument(
        "--graph-threshold", type=float, default=4.0,
        help="ItemGraph relevance threshold for edge construction (default: 4.0).",
    )

    # ── Run directory ─────────────────────────────────────────────────
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Custom run directory name (default: auto-generated timestamp).",
    )

    return parser.parse_args()


# ─── Strategy factory ────────────────────────────────────────────────────


def build_strategy(args: argparse.Namespace) -> ArmSelectionStrategy:
    """Create and return the arm selection strategy from CLI arguments.

    Supported strategies:

    - ``epsilon_greedy``: epsilon-greedy exploration/exploitation.
    - ``thompson``: Thompson Sampling with a Beta-Bernoulli reward model.

    New strategies can be added here without modifying any other code.
    """
    if args.strategy == "thompson":
        return ThompsonSamplingStrategy(
            reward_threshold=args.reward_threshold,
            prior_alpha=args.prior_alpha,
            prior_beta=args.prior_beta,
            random_state=args.seed,
        )
    # Default: epsilon-greedy
    return EpsilonGreedyStrategy(epsilon=args.epsilon, random_state=args.seed)


# ─── Report generation ───────────────────────────────────────────────────


def generate_report(
    report: BanditSimulationReport,
    individual_arm_metrics: Dict,
    converged_ndcg: float,
    converged_precision: float,
    converged_recall: float,
    config: Dict,
    report_path: Path,
) -> None:
    """Write a Markdown simulation report to *report_path*.

    Parameters
    ----------
    report : BanditSimulationReport
        Result of ``run_bandit_simulation()``.
    individual_arm_metrics : dict
        Mapping from arm name to ``Metrics`` (from standalone evaluation).
    converged_ndcg, converged_precision, converged_recall : float
        Metrics from evaluating the converged policy via ``evaluate()``.
    config : dict
        All hyperparameters and settings.
    report_path : Path
        Output file path.
    """
    k = config.get("k", 10)
    lines: List[str] = []
    lines.append("# Bandit Model Selector — Simulation Report\n")
    lines.append(f"Generated: {datetime.now().isoformat()}\n")

    # ── Configuration ─────────────────────────────────────────────────
    lines.append("## Configuration\n")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    for key, value in sorted(config.items()):
        lines.append(f"| {key} | {value} |")
    lines.append("")

    # ── Individual arm baselines (pre-bandit) ─────────────────────────
    lines.append("## Individual Arm Baselines (standalone, pre-bandit)\n")
    lines.append(f"| Arm | NDCG@{k} | Precision@{k} | Recall@{k} |")
    lines.append("|-----|---------|-------------|-----------|")
    for arm_name, arm_metrics in individual_arm_metrics.items():
        lines.append(
            f"| {arm_name} "
            f"| {arm_metrics.ndcg:.5f} "
            f"| {arm_metrics.precision:.5f} "
            f"| {arm_metrics.recall:.5f} |"
        )
    lines.append("")

    # ── Per-arm results ───────────────────────────────────────────────
    lines.append("## Per-Arm Bandit Results\n")
    lines.append("| Arm | Selections | Selection % | Mean NDCG@{k} |".format(
        k=config.get("k", 10)
    ))
    lines.append("|-----|-----------|-------------|--------------|")
    for arm_name in report.arm_names:
        count = report.per_arm_selection_count[arm_name]
        fraction = report.per_arm_selection_fraction[arm_name]
        mean_reward = report.per_arm_mean_reward[arm_name]
        lines.append(
            f"| {arm_name} | {count} | {fraction * 100:.1f}% | {mean_reward:.5f} |"
        )
    lines.append("")

    # ── Converged policy evaluation ───────────────────────────────────
    lines.append("## Converged Policy Evaluation\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| NDCG@{config.get('k', 10)} | {converged_ndcg:.5f} |")
    lines.append(f"| Precision@{config.get('k', 10)} | {converged_precision:.5f} |")
    lines.append(f"| Recall@{config.get('k', 10)} | {converged_recall:.5f} |")
    lines.append("")

    # ── Summary ───────────────────────────────────────────────────────
    lines.append("## Summary\n")
    lines.append(f"- **Total users processed:** {report.total_users_processed}")
    lines.append(
        f"- **Users skipped (no relevant items):** "
        f"{report.users_skipped_no_ground_truth}"
    )
    lines.append(f"- **Overall mean reward (NDCG):** {report.mean_reward:.5f}")
    lines.append("")

    # ── Final arm statistics ──────────────────────────────────────────
    lines.append("## Final Arm Statistics\n")
    lines.append("| Arm | Pull Count | Reward Sum | Mean Reward |")
    lines.append("|-----|-----------|-----------|-------------|")
    for arm_stat in report.final_arm_statistics:
        arm_name = report.arm_names[arm_stat.arm_index]
        lines.append(
            f"| {arm_name} | {arm_stat.pull_count} "
            f"| {arm_stat.reward_sum:.4f} "
            f"| {arm_stat.mean_reward:.5f} |"
        )
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


# ─── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    # ── Run directory ─────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"bandit_{timestamp}"
    run_directory = Path("runs") / run_name
    run_directory.mkdir(parents=True, exist_ok=True)

    # ── Dual logging (console + file) ─────────────────────────────────
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    log_format = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(run_directory / "simulation.log")
    file_handler.setFormatter(log_format)
    root_logger.addHandler(file_handler)

    logger.info("Run directory: %s", run_directory)

    evaluation_ratings = val if args.split == "val" else test

    # ── Phase 1: Fit child models on train (freeze before bandit) ─────
    logger.info("=" * 60)
    logger.info("Phase 1: Fitting child models on train split")
    logger.info("=" * 60)

    logger.info("Fitting BPR arm (n_factors=%d, n_epochs=%d, lr=%s, reg=%s)...",
                args.bpr_n_factors, args.bpr_n_epochs, args.bpr_lr, args.bpr_regularization)
    bpr_model = BPRRecommender(
        n_factors=args.bpr_n_factors,
        n_epochs=args.bpr_n_epochs,
        lr=args.bpr_lr,
        regularization=args.bpr_regularization,
        random_state=args.seed,
    )
    bpr_model.fit(train, users=users, movies=movies)
    logger.info("BPR arm fitted and frozen.")

    logger.info("Fitting ItemGraph arm (alpha=%s, steps=%d, threshold=%s)...",
                args.graph_alpha, args.graph_steps, args.graph_threshold)
    item_graph_model = ItemGraphPropagationRanker(
        relevance_threshold=args.graph_threshold,
        alpha=args.graph_alpha,
        n_steps=args.graph_steps,
    )
    item_graph_model.fit(train, users=users, movies=movies)
    logger.info("ItemGraph arm fitted and frozen.")

    # ── Phase 1b: Evaluate each arm independently (sanity check) ─────
    logger.info("=" * 60)
    logger.info("Phase 1b: Evaluating individual arms on %s split", args.split)
    logger.info("=" * 60)

    arm_models = {"BPR": bpr_model, "ItemGraph": item_graph_model}
    individual_arm_metrics = {}
    for arm_name, arm_model in arm_models.items():
        arm_metrics = evaluate_basic(
            model=arm_model,
            train_ratings=train,
            test_ratings=evaluation_ratings,
            users=users,
            movies=movies,
            k=args.k,
            threshold=args.threshold,
        )
        individual_arm_metrics[arm_name] = arm_metrics
        logger.info(
            "  %s standalone — NDCG@%d: %.5f, Precision@%d: %.5f, Recall@%d: %.5f",
            arm_name, args.k, arm_metrics.ndcg,
            args.k, arm_metrics.precision,
            args.k, arm_metrics.recall,
        )

    # ── Phase 2: Build strategy ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Phase 2: Building strategy '%s'", args.strategy)
    logger.info("=" * 60)
    strategy = build_strategy(args)

    # ── Phase 3: Run bandit simulation ────────────────────────────────
    logger.info("=" * 60)
    logger.info("Phase 3: Running bandit simulation on %s split", args.split)
    logger.info("=" * 60)

    simulation_report = run_bandit_simulation(
        arms=[bpr_model, item_graph_model],
        arm_names=["BPR", "ItemGraph"],
        strategy=strategy,
        train_ratings=train,
        evaluation_ratings=evaluation_ratings,
        users=users,
        movies=movies,
        k=args.k,
        relevance_threshold=args.threshold,
        process_order=args.process_order,
        random_state=args.seed,
    )

    logger.info("=== Bandit Simulation Results ===")
    logger.info("Users processed: %d", simulation_report.total_users_processed)
    logger.info(
        "Users skipped (no ground truth): %d",
        simulation_report.users_skipped_no_ground_truth,
    )
    logger.info("Overall mean NDCG@%d: %.5f", args.k, simulation_report.mean_reward)

    for arm_name in simulation_report.arm_names:
        logger.info(
            "  Arm '%s': selected %d times (%.1f%%), mean NDCG@%d = %.5f",
            arm_name,
            simulation_report.per_arm_selection_count[arm_name],
            simulation_report.per_arm_selection_fraction[arm_name] * 100,
            args.k,
            simulation_report.per_arm_mean_reward[arm_name],
        )

    # ── Phase 4: Evaluate converged policy ────────────────────────────
    logger.info("=" * 60)
    logger.info("Phase 4: Evaluating converged bandit policy")
    logger.info("=" * 60)

    # The strategy retains its learned state from the simulation.
    # We wrap it in a BanditModelSelector (which does NOT call
    # strategy.initialize()) to produce predictions via the existing
    # evaluation framework.  The exploration rate is kept as-is.
    bandit_selector = BanditModelSelector(
        arms=[bpr_model, item_graph_model],
        arm_names=["BPR", "ItemGraph"],
        strategy=strategy,
    )

    converged_metrics = evaluate_basic(
        model=bandit_selector,
        train_ratings=train,
        test_ratings=evaluation_ratings,
        users=users,
        movies=movies,
        k=args.k,
        threshold=args.threshold,
    )
    logger.info("=== Converged Policy Evaluation ===")
    logger.info("NDCG@%d:      %.5f", args.k, converged_metrics.ndcg)
    logger.info("Precision@%d: %.5f", args.k, converged_metrics.precision)
    logger.info("Recall@%d:    %.5f", args.k, converged_metrics.recall)

    # ── Phase 5: Export report ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Phase 5: Exporting report to %s", run_directory / "report.md")
    logger.info("=" * 60)

    config = {
        "strategy": args.strategy,
        "epsilon": args.epsilon,
        "reward_threshold": args.reward_threshold,
        "prior_alpha": args.prior_alpha,
        "prior_beta": args.prior_beta,
        "split": args.split,
        "k": args.k,
        "relevance_threshold": args.threshold,
        "process_order": args.process_order,
        "seed": args.seed,
        "bpr_n_factors": args.bpr_n_factors,
        "bpr_n_epochs": args.bpr_n_epochs,
        "bpr_lr": args.bpr_lr,
        "bpr_regularization": args.bpr_regularization,
        "graph_alpha": args.graph_alpha,
        "graph_steps": args.graph_steps,
        "graph_threshold": args.graph_threshold,
    }

    generate_report(
        report=simulation_report,
        individual_arm_metrics=individual_arm_metrics,
        converged_ndcg=converged_metrics.ndcg,
        converged_precision=converged_metrics.precision,
        converged_recall=converged_metrics.recall,
        config=config,
        report_path=run_directory / "report.md",
    )

    logger.info("Done. Run directory: %s", run_directory)


if __name__ == "__main__":
    main()
