from eval import _run

import time
import argparse

if __name__ == "__main__":

    def parse_args():

        parser = argparse.ArgumentParser(
            description="Produce agreement scatterplot for experiment"
        )

        parser.add_argument(
            "--experiment",
            type=str,
            default=None,
            required=True,
            help="Path to the experiment",
        )
        #parser.add_argument(
        #    "--test_oracle",
        #    type=str,
        #    default=None,
        #    required=True,
        #    help="Path to test oracle"
        #)
        parser.add_argument(
            "--savedir", type=str, default="/mnt/home/viz/design_bench_internal/tmp"
        )
        parser.add_argument("--datadir", type=str, default="/mnt/public/datasets")
        parser.add_argument("--use_ema", action='store_true')
        parser.add_argument("--model", type=str, default=None)
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--method", type=str, default="plot")
        parser.add_argument("--method_kwargs", type=eval, default="{}")
        parser.add_argument("--sample_kwargs", type=eval, default="{}")
        args = parser.parse_args()
        return args

    args = parse_args()

    _run(args.experiment, 
         args.savedir, 
         method=args.method,
         method_kwargs=args.method_kwargs,
         seed=args.seed, 
         args=args)
