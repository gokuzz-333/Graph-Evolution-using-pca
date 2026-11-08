"""Single entry point:  python run_project.py   (see README.md for options)."""
import argparse
import sys

from src.pipeline import RunConfig, run_pipeline
from src.project_config import (SNAPSHOT_COUNT, SNAPSHOT_MODE, GRAPH_REPRESENTATION,
                                EDGE_MODE, QUANTUM_SAMPLES, RANDOM_SEED)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--interval", choices=["equal", "month", "week"], default=SNAPSHOT_MODE,
                   help="equal time ranges (default), fixed 30-day months, or optional 7-day weeks")
    p.add_argument("--snapshots", type=int, default=SNAPSHOT_COUNT,
                   help="number of chronological equal snapshots")
    p.add_argument("--snapshot-boundaries", default=None,
                   help="optional comma-separated day boundaries relative to first event, e.g. 0,100,300,804")
    p.add_argument("--edge-mode", choices=["binary", "count"], default=EDGE_MODE)
    p.add_argument("--data-file", default=None, help="local SNAP file (default: download to data/)")
    p.add_argument("--output-dir", default=None, help="where results/ and figures/ go (default: project root)")
    p.add_argument("--skip-quantum", action="store_true")
    p.add_argument("--quantum-samples", "--max-quantum-samples", dest="quantum_samples",
                   type=int, default=QUANTUM_SAMPLES, help="nodes in the sample-matched comparison")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--graph-representation", choices=["adjacency", "laplacian", "combined"],
                   default=GRAPH_REPRESENTATION,
                   help="which of the three sparse graph-matrix embeddings is primary")
    p.add_argument("--no-log-transform", action="store_true",
                   help="skip log1p on the two comparison features")
    args = p.parse_args(argv)
    boundaries = ([float(value) for value in args.snapshot_boundaries.split(",")]
                  if args.snapshot_boundaries else None)
    cfg = RunConfig(interval=args.interval, snapshot_count=args.snapshots,
                    snapshot_boundaries_days=boundaries,
                    graph_representation=args.graph_representation,
                    edge_mode=args.edge_mode, data_file=args.data_file,
                    skip_quantum=args.skip_quantum, quantum_samples=args.quantum_samples,
                    seed=args.seed, log_transform=not args.no_log_transform)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    run_pipeline(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
