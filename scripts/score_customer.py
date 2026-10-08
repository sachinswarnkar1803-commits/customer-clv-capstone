"""Score one existing customer from the fitted project models.

Usage:
    python scripts/score_customer.py --customer-id 12345
"""
import argparse
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.config import load_config
from src.models.purchase_model import PurchaseModelBGNBD
from src.models.monetary_model import MonetaryModelGammaGamma
from src.models.inactivity_model import InactivitySurvivalModel
from src.clv.clv_calculator import ProbabilisticCLVCalculator
from src.segmentation.segmenter import ActionSegmenter
from src.nba.nba_engine import NBAEngine


def read_customers():
    path = ROOT / "data" / "processed" / "customer_clv_segments.csv"
    if not path.exists():
        raise FileNotFoundError("Run `python scripts/run_pipeline.py` before scoring customers.")
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--customer-id", required=True)
    args = parser.parse_args()
    cfg = load_config()
    customers = read_customers()
    row = customers[customers.customer_id.astype(str) == str(args.customer_id)].copy()
    if row.empty:
        raise SystemExit(f"Customer {args.customer_id} was not found in customer_clv_segments.csv")

    bgf = PurchaseModelBGNBD(cfg); bgf.load_model()
    ggf = MonetaryModelGammaGamma(cfg); ggf.load_model()
    survival = InactivitySurvivalModel(cfg); survival.load_model()
    clv = ProbabilisticCLVCalculator(cfg)
    row["p_alive"] = bgf.predict_p_alive(row).values
    row["exp_avg_monetary"] = ggf.predict_expected_average_spend(row).values
    row = pd.concat([row, survival.predict_inactivity_probabilities(row)], axis=1)
    row = clv.compute_clv(row, bgf, ggf, n_simulation_samples=500)
    population = customers.copy()
    idx = population.index[population.customer_id.astype(str) == str(args.customer_id)][0]
    for col in row.columns:
        if col in population.columns:
            population.loc[idx, col] = row.iloc[0][col]
    population = ActionSegmenter(cfg).segment_customers(population)
    scored = population[population.customer_id.astype(str) == str(args.customer_id)].copy()
    scored = NBAEngine(cfg).generate_recommendations(scored)
    r = scored.iloc[0]
    horizon = cfg.clv.default_horizon_days
    print(f"Customer: {args.customer_id}")
    print(f"P(Alive): {r['p_alive']:.4f}")
    print(f"Expected purchases ({horizon}d): {r[f'exp_purchases_{horizon}d']:.3f}")
    print(f"Expected CLV ({horizon}d): {r[f'clv_expected_{horizon}d']:.2f}")
    print(f"Inactivity risk (90d): {r['inactivity_prob_90d']:.2%}")
    print(f"Action segment: {r['action_segment']}")
    print(f"Next-best-action: {r['nba_recommended_action']}")
    print(f"Channel: {r['nba_channel']}")

if __name__ == "__main__":
    main()
