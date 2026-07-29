"""runnable file! Trains the n-tuple network with TD(0) afterstate learning.

    python train_ntuple.py --games 100000

Runs against utils/bitboard.py rather than the gymnasium env, which is what
makes tens of millions of moves practical (verify_fast_env.py checks the two
agree). Progress prints every --report-every games and weights are checkpointed,
so a run can be stopped with Ctrl-C and picked back up with --resume.

The tile reach-rates say more about the agent than the score column does.
"""

import argparse
import os
import signal
import sys
import time

from models.NTupleNetwork import NTupleNetwork
from models.NTupleTrainer import NTupleTrainer, TrainingStats

DEFAULT_OUTPUT_DIR = os.path.join("models", "ntuple_files")
REPORTED_TILES = (512, 1024, 2048, 4096, 8192)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100_000, help="training games")
    parser.add_argument("--learning-rate", type=float, default=1.0, help="alpha")
    parser.add_argument(
        "--no-temporal-coherence",
        action="store_true",
        help="use one fixed learning rate for every weight (then try alpha ~0.1)",
    )
    parser.add_argument(
        "--optimistic-init",
        type=float,
        default=0.0,
        help="start every board valued this high, to explore without acting "
        "randomly. Only meaningful on a fresh run, not with --resume. Guei used "
        "roughly half the score the agent ends up averaging",
    )
    parser.add_argument(
        "--tc-fraction",
        type=float,
        default=1.0,
        help="fraction of training spent in the temporal-coherence phase, "
        "counted from the end. 1.0 uses TC throughout, Guei's OTD+TC uses 0.1",
    )
    parser.add_argument("--report-every", type=int, default=1_000)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume", default=None, help="weights .npz to continue from")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--eval-games", type=int, default=300, help="greedy games to play at the end"
    )
    return parser.parse_args()


def format_report(block: TrainingStats, total_games: int, elapsed: float) -> str:
    reach = "  ".join(
        f"{tile}:{block.reach_rate(tile):5.1%}" for tile in REPORTED_TILES
    )
    return (
        f"{total_games:>8,} games | {elapsed / 60:6.1f} min | "
        f"mean {block.mean_score:>8,.0f} | best {block.best_score:>7,} | "
        f"tile {block.best_tile:>5} | {reach} | "
        f"{block.moves_per_second():>6,.0f} mv/s"
    )


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.resume:
        network = NTupleNetwork.load(args.resume)
        print(f"resumed from {args.resume}")
        optimizer_path = args.resume.replace(".npz", ".opt.npz")
        if args.optimistic_init:
            print("ignoring --optimistic-init, it only applies to a fresh network")
    else:
        network = NTupleNetwork()
        if args.optimistic_init:
            network.optimistic_init(args.optimistic_init)
            print(f"optimistic init: every board starts valued at {args.optimistic_init:,.0f}")

    # With OTD+TC the run opens at a fixed rate and only switches to TC for the
    # last tc-fraction of games. TC from the start is the plain schedule.
    tc_start_game = int(args.games * (1.0 - args.tc_fraction))
    use_tc_now = not args.no_temporal_coherence and tc_start_game <= 0

    trainer = NTupleTrainer(
        network=network,
        learning_rate=args.learning_rate,
        temporal_coherence=use_tc_now,
        seed=args.seed,
    )

    # Picking the TC counters back up keeps a resume from re-learning rates it
    # already tuned. Without them the first few thousand games score lower.
    if args.resume and os.path.exists(optimizer_path):
        trainer.load_optimizer(optimizer_path)
        print(f"resumed learning rates from {optimizer_path}")

    print(
        f"patterns: {len(network.patterns)} x {network.tuple_length}-tuple, "
        f"{network.num_lookups} lookups/eval, "
        f"{network.weights.size:,} weights ({network.memory_bytes() / 1e6:.0f} MB)"
    )
    if args.no_temporal_coherence:
        print(f"alpha {args.learning_rate}, temporal coherence off")
    elif tc_start_game > 0:
        print(
            f"alpha {args.learning_rate}, temporal coherence from game "
            f"{tc_start_game:,} onward (last {args.tc_fraction:.0%})"
        )
    else:
        print(f"alpha {args.learning_rate}, temporal coherence on")
    print()

    # Ctrl-C should save the run, not throw it away
    interrupted = {"flag": False}

    def handle_interrupt(signum, frame):
        if interrupted["flag"]:
            sys.exit(1)  # second Ctrl-C, bail out now
        interrupted["flag"] = True
        print("\ninterrupt received, finishing the current block and saving...")

    signal.signal(signal.SIGINT, handle_interrupt)

    started = time.time()
    games_done = 0
    best_tile_overall = 0
    best_score_overall = 0

    while games_done < args.games and not interrupted["flag"]:
        if not trainer.temporal_coherence and not args.no_temporal_coherence:
            if games_done >= tc_start_game:
                trainer.enable_temporal_coherence()
                print(f"    switching to temporal coherence at game {games_done:,}")

        block_size = min(args.report_every, args.games - games_done)
        block = trainer.train(block_size)
        games_done += block_size
        best_tile_overall = max(best_tile_overall, block.best_tile)
        best_score_overall = max(best_score_overall, block.best_score)

        print(format_report(block, games_done, time.time() - started))

        if args.checkpoint_every and games_done % args.checkpoint_every == 0:
            path = os.path.join(args.output_dir, f"ntuple_{games_done}.npz")
            network.save(path)
            print(f"    checkpoint -> {path}  (weights touched: {network.coverage():.2%})")

    final_path = os.path.join(args.output_dir, "ntuple_final.npz")
    network.save(final_path)
    trainer.save_optimizer(final_path.replace(".npz", ".opt.npz"))
    print(f"\nsaved {final_path}")
    print(
        f"training peak: tile {best_tile_overall}, score {best_score_overall:,} "
        f"over {games_done:,} games in {(time.time() - started) / 60:.1f} min"
    )

    if args.eval_games:
        print(f"\nevaluating {args.eval_games} greedy games (no learning)...")
        evaluation = trainer.evaluate(args.eval_games)
        print(
            f"  mean score {evaluation.mean_score:,.0f} | "
            f"best score {evaluation.best_score:,} | best tile {evaluation.best_tile}"
        )
        for tile in REPORTED_TILES:
            print(f"  reached {tile:>5}: {evaluation.reach_rate(tile):6.1%}")
        print("\nExpectimax on top of these weights scores higher:")
        print("  python play_ntuple.py --games 50 --budget 2000")


if __name__ == "__main__":
    main()
