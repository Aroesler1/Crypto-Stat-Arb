"""
Phase 3: Cost-aware execution experiments.

Phases 1-2 established the honest headline: the clustering signal has real
gross structure (positive gross Sharpe in every configuration) but nothing
survives realistic crypto taker costs. The natural follow-up question is
whether SLOWER TRADING can keep enough of the alpha while cutting enough of
the cost (the classic aim-in-front-of-the-target logic of Garleanu &
Pedersen, "Dynamic Trading with Predictable Returns and Transaction Costs").

Two execution controls are swept on the best net configuration from Phase 2
(Cluster Deviation, SPONGE k=3):

1. No-trade band: a token's weight only moves when the target differs from
   the held weight by more than `band` (absolute weight units). Small signal
   wiggles stop generating round trips.
2. Rebalance frequency: weights update every k-th day and are held in
   between.

Every (band, frequency) cell is a trial; the leaderboard reports net Sharpe
at 25/50 bps, break-even cost, turnover, and the Deflated Sharpe Ratio with
the full Phase-3 grid as the multiple-testing pool.
"""
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from stat_arb.data.loader import DataLoader
from stat_arb.data.universe import UniverseManager
from stat_arb.pca.market_mode import MarketModeExtractor
from stat_arb.graphs.knn_graph import KNNGraphBuilder
from stat_arb.clustering.sponge import SPONGEClustering
from stat_arb.signals.cluster_deviation import ClusterDeviationStrategy
from stat_arb.backtest.engine import BacktestEngine
from stat_arb.backtest.walk_forward import WalkForwardBacktest, WalkForwardConfig
from stat_arb.backtest.costs import TransactionCostModel
from stat_arb.backtest.statistics import deflated_sharpe_ratio, per_period_sharpe


def make_execution_portfolio_func(prev_weights_holder, weight_band=0.0,
                                  trade_frequency_days=1, max_turnover_per_day=0.15):
    """Portfolio construction with a no-trade band and a rebalance frequency.

    Behaves exactly like the phase-1/2 turnover-capped construction when
    weight_band=0 and trade_frequency_days=1.
    """

    def portfolio_func(signals, returns, **kwargs):
        weights = signals.copy()

        for i, date in enumerate(weights.index):
            if i == 0:
                prior = prev_weights_holder[0] if prev_weights_holder[0] is not None \
                    else pd.Series(0.0, index=weights.columns)
            else:
                prior = weights.iloc[i - 1]
            prior = prior.reindex(weights.columns).fillna(0.0)

            target = weights.loc[date].fillna(0.0)

            # rebalance frequency: hold between scheduled trade days
            if trade_frequency_days > 1 and (i % trade_frequency_days) != 0:
                weights.loc[date] = prior
                continue

            # no-trade band: ignore sub-band target moves
            if weight_band > 0:
                delta = target - prior
                hold = delta.abs() < weight_band
                target = target.where(~hold, prior)

            # daily turnover cap (same as phases 1-2)
            delta = target - prior
            turnover = delta.abs().sum()
            if turnover > max_turnover_per_day:
                target = prior + delta * (max_turnover_per_day / turnover)

            weights.loc[date] = target

        if not weights.empty:
            prev_weights_holder[0] = weights.iloc[-1].copy()
        return weights

    return portfolio_func


def default_clusterer(adj, n_clusters):
    """The published clusterer: SPONGE on the signed k-NN graph."""
    sponge = SPONGEClustering(n_clusters=n_clusters, random_state=42)
    sponge.fit(adj)
    return sponge.labels_


def run_phase3_config(excess_returns, universe_mask, H=5, L=20,
                      weight_band=0.0, trade_frequency_days=1,
                      n_pca_components=1, clusterer=None, n_clusters=3,
                      diagnostics=None):
    """One walk-forward run of Cluster Deviation with the given controls.

    Defaults reproduce phase 2 strategy 2 exactly (one PCA component removed,
    SPONGE k=3), so every published number still comes out of this function.
    The parameters exist so later steps can vary one thing at a time against
    the same signal path rather than forking it:

    ``n_pca_components``  how many principal components to remove on top of
                          whatever excess is already in `excess_returns`. 0
                          removes none, which is what the ETH-excess-only and
                          BTC-excess-only arms of the residualization ablation
                          need.
    ``clusterer``         callable (adjacency, n_clusters) -> labels. Lets the
                          clustering-method comparison swap in SSSNET, Pivot,
                          hierarchical or k-means without touching the signal.
    ``diagnostics``       optional list; one dict is appended per rebalance with
                          the variance removed, the signed-graph density and the
                          cluster labels, which is what the ablation reports and
                          what cluster stability (ARI) is computed from.
    """
    if clusterer is None:
        clusterer = default_clusterer
    return_cols_to_tokens = {col: col.replace('_returns', '') for col in excess_returns.columns}
    prev_weights = [None]

    def signal_func(train_returns, test_dates, full_returns, **kwargs):
        rebalance_date = test_dates[0]

        if rebalance_date in universe_mask.index:
            current_univ = universe_mask.loc[rebalance_date]
        else:
            prior_dates = universe_mask.index[universe_mask.index <= rebalance_date]
            current_univ = universe_mask.loc[prior_dates[-1]] if len(prior_dates) > 0 else universe_mask.iloc[0]

        valid_return_cols = [col for col in train_returns.columns
                             if return_cols_to_tokens.get(col, '') in current_univ.index
                             and current_univ.get(return_cols_to_tokens.get(col, ''), False)]
        if len(valid_return_cols) == 0:
            valid_return_cols = train_returns.columns.tolist()

        train_subset = train_returns[valid_return_cols].dropna(
            axis=1, thresh=len(train_returns) * 0.8).fillna(0)
        if train_subset.shape[1] < 10:
            return pd.DataFrame(0, index=test_dates, columns=full_returns.columns)

        if n_pca_components > 0:
            pca = MarketModeExtractor(n_components=n_pca_components)
            pca.fit(train_subset)
            train_residuals = pca.residualize(train_subset)
            var_removed = float(np.sum(pca.explained_variance_ratio_))
        else:
            # No PCA step. The returns keep whatever excess they arrived with,
            # which is the ETH-excess-only and BTC-excess-only arms.
            pca = None
            train_residuals = train_subset
            var_removed = 0.0

        knn = KNNGraphBuilder(k=10)
        corr = knn.compute_correlation_matrix(train_residuals)
        adj = knn.build_weighted_knn(corr)

        try:
            labels = clusterer(adj, n_clusters)
        except Exception:
            return pd.DataFrame(0, index=test_dates, columns=full_returns.columns)
        labels = np.asarray(labels)

        unique_labels = np.unique(labels)
        cluster_cohesion = {}
        assets_list = list(train_subset.columns)
        for c in unique_labels:
            members = [assets_list[i] for i in range(len(labels)) if labels[i] == c]
            if len(members) < 2:
                cluster_cohesion[c] = -1.0
            else:
                c_corr = corr.loc[members, members]
                cluster_cohesion[c] = (c_corr.sum().sum() - len(members)) / (len(members) * (len(members) - 1))
        noisy_cluster = min(cluster_cohesion, key=cluster_cohesion.get)
        valid_clusters = [c for c in unique_labels if c != noisy_cluster]

        context_start = test_dates[0] - pd.Timedelta(days=L + 30)
        context_slice = full_returns.loc[context_start:test_dates[-1], assets_list].fillna(0)
        context_residuals = (pca.residualize(context_slice) if pca is not None
                             else context_slice)

        if diagnostics is not None:
            off_diag = adj.to_numpy().copy()
            np.fill_diagonal(off_diag, 0.0)
            n = off_diag.shape[0]
            n_pairs = n * (n - 1)
            diagnostics.append({
                'rebalance_date': rebalance_date,
                'n_assets': n,
                'variance_removed': var_removed,
                # share of possible ordered pairs carrying an edge
                'graph_density': float((off_diag != 0).sum() / n_pairs) if n_pairs else np.nan,
                'negative_edge_share': (float((off_diag < 0).sum() / max((off_diag != 0).sum(), 1))),
                'labels': labels.copy(),
                'assets': list(assets_list),
                'n_clusters_found': int(len(np.unique(labels))),
            })

        cluster_labels_array = np.array([labels[assets_list.index(col)] for col in assets_list])
        strategy = ClusterDeviationStrategy(
            lookback=H, zscore_window=L, entry_threshold=1.5, composite_type='equal')

        test_signals = strategy.compute_signals(
            context_residuals, cluster_labels_array,
            [col.replace('_returns', '') for col in assets_list], lag=1)
        test_signals = test_signals.reindex(test_dates)
        if test_signals.empty or test_signals.isna().all().all():
            return pd.DataFrame(0, index=test_dates, columns=full_returns.columns)

        weights = strategy.generate_target_weights(
            signals=test_signals,
            cluster_labels=cluster_labels_array,
            tokens=[col.replace('_returns', '') for col in assets_list],
            clusters_to_trade=valid_clusters,
        )
        weights = weights.reindex(columns=full_returns.columns, fill_value=0.0)

        target_leverage = 1.5
        gross = weights.abs().sum(axis=1)
        scale = (target_leverage / gross.replace(0, np.nan)).fillna(0.0)
        return weights.mul(scale, axis=0)

    portfolio_func = make_execution_portfolio_func(
        prev_weights, weight_band=weight_band, trade_frequency_days=trade_frequency_days)

    wf = WalkForwardBacktest(WalkForwardConfig(
        train_window=365, test_window=28, refit_frequency=28, min_train_history=365))
    weights, fold_info = wf.run_backtest(
        returns=excess_returns, signal_func=signal_func, portfolio_func=portfolio_func)

    if weights.empty or weights.abs().sum().sum() == 0:
        return None

    engine = BacktestEngine()
    turnover = engine.compute_turnover(weights)
    gross_ret = engine.compute_gross_returns(weights, excess_returns)

    out = {'weights': weights, 'gross_returns': gross_ret, 'turnover': turnover}
    for cost_bps in (25, 50):
        net, _, _ = engine.compute_net_returns(weights, excess_returns, cost_bps=cost_bps)
        out[f'net_{cost_bps}'] = net
    out['breakeven'] = TransactionCostModel().find_breakeven_cost(gross_ret, turnover)
    return out


def annualized_sharpe(r):
    return (r.mean() * 365) / (r.std() * np.sqrt(365) + 1e-8)


def run_phase3(bands=(0.0, 0.02, 0.05), frequencies=(1, 2, 3, 5), verbose=True):
    data_dir = Path(__file__).parent.parent / 'data'
    loader = DataLoader(str(data_dir))
    excess_returns, prices, volumes, eth_data = loader.get_aligned_data()

    univ_manager = UniverseManager(
        mcap_percentile_low=0.0, mcap_percentile_high=1.0,
        min_volume_usd=50_000, min_history_days=60)
    universe_mask = univ_manager.get_universe_membership(prices, volumes, eth_data, excess_returns)

    rows = []
    net50_series = {}
    total = len(bands) * len(frequencies)
    idx = 0
    for band in bands:
        for freq in frequencies:
            idx += 1
            if verbose:
                print(f"[{idx}/{total}] band={band:.2f}, freq={freq}d...", flush=True)
            result = run_phase3_config(
                excess_returns, universe_mask, weight_band=band, trade_frequency_days=freq)
            if result is None:
                continue
            row = {
                'weight_band': band,
                'trade_frequency_days': freq,
                'gross_sharpe': annualized_sharpe(result['gross_returns']),
                'net_sharpe_25': annualized_sharpe(result['net_25']),
                'net_sharpe_50': annualized_sharpe(result['net_50']),
                'avg_turnover': result['turnover'].mean(),
                'breakeven_bps': result['breakeven'],
            }
            rows.append(row)
            net50_series[(band, freq)] = result['net_50']
            if verbose:
                print(f"    gross={row['gross_sharpe']:.2f} net@50={row['net_sharpe_50']:.2f} "
                      f"turnover={row['avg_turnover']*100:.1f}% BE={row['breakeven_bps']:.1f}bps", flush=True)

    results = pd.DataFrame(rows)

    # DSR with the whole phase-3 grid as the trial pool (net @ 50 bps)
    trial_srs = [per_period_sharpe(s) for s in net50_series.values()]
    results['dsr_net_50'] = [
        deflated_sharpe_ratio(net50_series[(row.weight_band, row.trade_frequency_days)],
                              n_trials=len(trial_srs), trial_sharpes=trial_srs)['dsr']
        for row in results.itertuples(index=False)
    ]

    results = results.sort_values('net_sharpe_50', ascending=False).reset_index(drop=True)

    results_dir = Path(__file__).parent / 'reporting' / 'phase3'
    results_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_dir / 'execution_experiments.csv', index=False)

    if verbose:
        print("\nPHASE 3: EXECUTION EXPERIMENTS (Cluster Dev, SPONGE k=3)")
        print(results.to_string(index=False,
                                float_format=lambda value: f"{value:0.3f}"))
    return results


if __name__ == '__main__':
    run_phase3()
