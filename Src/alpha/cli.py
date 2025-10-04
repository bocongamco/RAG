# Src/alpha/cli.py
import argparse, yaml
from .train import fit_and_eval

def main():
    p = argparse.ArgumentParser("Train/Eval α (hybrid)")
    p.add_argument("--config", default="Config/eval.yaml")
    p.add_argument("--val-frac",  type=float, default=0.20, help="Validation fraction (reporting split)")
    p.add_argument("--test-frac", type=float, default=0.15, help="Held-out test fraction")
    p.add_argument("--out",       type=str,   default="Outputs/alpha_eval", help="Artifacts folder")

    # expose CV/selection knobs (flags override YAML)
    p.add_argument("--select", default=None, help='Selection: "cv" or "val"')
    p.add_argument("--cv-folds", type=int, default=None, help="Number of CV folds (when select=cv)")
    p.add_argument("--one-std-err", type=int, default=None,
                   help="1 = enable one-std-err rule, 0 = strict argmax")
    p.add_argument("--tie-break", default=None, help='Tie-breaker: "min_alpha" or "max_train"')

    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))

    # prefer flags when provided, else fall back to YAML, else train.py defaults
    select      = args.select if args.select is not None else cfg.get("select")
    cv_folds    = args.cv_folds if args.cv_folds is not None else cfg.get("cv_folds")
    one_std_err = (bool(args.one_std_err) if args.one_std_err is not None
                   else bool(cfg.get("one_std_err")) if "one_std_err" in cfg else None)
    tie_break   = args.tie_break if args.tie_break is not None else cfg.get("tie_break")

    out = fit_and_eval(
        qrels_path=cfg.get("qrels_path", "Data/qrels.csv"),
        k=cfg.get("k", 5),
        seed=cfg.get("seed", 42),
        grid=cfg.get("grid"),
        k_each=cfg.get("k_each", 25),
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        out_dir=args.out,
        # pass-through (only pass if not None so train.py can keep its defaults)
        **({ "select": select } if select is not None else {}),
        **({ "cv_folds": cv_folds } if cv_folds is not None else {}),
        **({ "one_std_err": one_std_err } if one_std_err is not None else {}),
        **({ "tie_break": tie_break } if tie_break is not None else {}),
    )

    msg = (f"α={out['alpha']:.2f} | train nDCG@k={out['train_ndcg']:.3f} "
           f"| val nDCG@k={out['val_ndcg']:.3f}")
    if "test_ndcg" in out:
        msg += f" | test nDCG@k={out['test_ndcg']:.3f}"
    print(msg)
    print(f"[alpha] artifacts saved to: {args.out}")

if __name__ == "__main__":
    main()
