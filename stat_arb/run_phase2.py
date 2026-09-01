"""
Phase 2: Alternative Clustering Methods and Strategy Comparison

Tests:
- SPONGE with k = {2,3,4,5,6,8,10}
- BNC with same k grid
- Signed Spectral with same k grid
- Strategy 1: Z-score mean reversion (baseline)
- Strategy 2: Cluster composite deviation
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from stat_arb.data.loader import DataLoader
from stat_arb.data.universe import UniverseManager
from stat_arb.pca.market_mode import MarketModeExtractor
from stat_arb.graphs.knn_graph import KNNGraphBuilder
from stat_arb.clustering.sponge import SPONGEClustering
from stat_arb.clustering.bnc import BNCClustering
from stat_arb.clustering.signed_spectral import SignedSpectralClustering
from stat_arb.signals.zscore_strategy import ZScoreStrategy
from stat_arb.signals.cluster_deviation import ClusterDeviationStrategy
from stat_arb.backtest.engine import BacktestEngine
from stat_arb.backtest.walk_forward import WalkForwardBacktest, WalkForwardConfig
from stat_arb.backtest.costs import TransactionCostModel


def run_phase2_clustering_sweep(k_grid=None, H=5, L=20, cost_bps=50, verbose=True):
    """
    Run clustering method comparison across k values.

    Parameters
    ----------
    k_grid : list
        Grid of k values to test
    H, L : int
        Signal parameters
    cost_bps : float
        Transaction cost

    Returns
    -------
    pd.DataFrame
        Results for each clustering method and k
    """
    if k_grid is None:
        k_grid = [2, 3, 4, 5, 6, 8, 10]

    clustering_methods = {
        'SPONGE': SPONGEClustering,
        'BNC': BNCClustering,
        'SignedSpectral': SignedSpectralClustering,
    }

    # Load data once
    data_dir = Path(__file__).parent.parent / 'data'
    loader = DataLoader(str(data_dir))
    excess_returns, prices, volumes, eth_data = loader.get_aligned_data()

    # Universe
    univ_manager = UniverseManager(
        mcap_percentile_low=0.0, mcap_percentile_high=1.0,
        min_volume_usd=50_000, min_history_days=60
    )
    universe_mask = univ_manager.get_universe_membership(prices, volumes, eth_data, excess_returns)

    results_list = []
    total_runs = len(clustering_methods) * len(k_grid)
    run_idx = 0

    for method_name, ClusterClass in clustering_methods.items():
        for k in k_grid:
            run_idx += 1
            if verbose:
                print(f"[{run_idx}/{total_runs}] {method_name} k={k}...")

            try:
                result = _run_single_config(
                    excess_returns, prices, volumes, eth_data, universe_mask,
                    ClusterClass, k, H, L, cost_bps
                )

                if result is not None:
                    results_list.append({
                        'method': method_name,
                        'k': k,
                        **result
                    })
                    if verbose:
                        print(f"  Gross Sharpe: {result['gross_sharpe']:.2f}, "
                              f"Net Sharpe @ {cost_bps}bps: {result['net_sharpe']:.2f}")
                else:
                    results_list.append({
                        'method': method_name, 'k': k,
                        'gross_sharpe': np.nan, 'net_sharpe': np.nan
                    })

            except Exception as e:
                if verbose:
                    print(f"  Error: {e}")
                results_list.append({
                    'method': method_name, 'k': k,
                    'gross_sharpe': np.nan, 'net_sharpe': np.nan
                })

    return pd.DataFrame(results_list)


def _run_single_config(excess_returns, prices, volumes, eth_data, universe_mask,
                        ClusterClass, n_clusters, H, L, cost_bps):
    """Run a single clustering configuration."""
    return_cols_to_tokens = {col: col.replace('_returns', '') for col in excess_returns.columns}
    prev_weights = [None]

    def signal_func(train_returns, test_dates, full_returns, **kwargs):
        rebalance_date = test_dates[0]

        # Universe filtering
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

        train_subset = train_returns[valid_return_cols].dropna(axis=1, thresh=len(train_returns)*0.8).fillna(0)

        if train_subset.shape[1] < max(10, n_clusters * 3):
            return pd.DataFrame(0, index=test_dates, columns=full_returns.columns)

        # PCA
        pca = MarketModeExtractor(n_components=1)
        pca.fit(train_subset)
        train_residuals = pca.residualize(train_subset)

        # Graph and clustering
        knn = KNNGraphBuilder(k=10)
        corr = knn.compute_correlation_matrix(train_residuals)
        adj = knn.build_weighted_knn(corr)

        clusterer = ClusterClass(n_clusters=n_clusters, random_state=42)
        try:
            clusterer.fit(adj)
            labels = clusterer.labels_
        except:
            return pd.DataFrame(0, index=test_dates, columns=full_returns.columns)

        # Drop noisy cluster
        unique_labels = np.unique(labels)
        cluster_cohesion = {}
        assets_list = list(train_subset.columns)

        for c in unique_labels:
            members = [assets_list[i] for i in range(len(labels)) if labels[i] == c]
            if len(members) < 2:
                cluster_cohesion[c] = -1.0
            else:
                c_corr = corr.loc[members, members]
                avg_corr = (c_corr.sum().sum() - len(members)) / (len(members) * (len(members) - 1))
                cluster_cohesion[c] = avg_corr

        noisy_cluster = min(cluster_cohesion, key=cluster_cohesion.get)
        valid_clusters = [c for c in unique_labels if c != noisy_cluster]

        # Signal generation
        context_start = test_dates[0] - pd.Timedelta(days=L + 30)
        context_slice = full_returns.loc[context_start:test_dates[-1], assets_list].fillna(0)
        context_residuals = pca.residualize(context_slice)

        strategy = ZScoreStrategy(H=H, L=L, quantile=0.2)
        raw_signals = strategy.compute_signals(context_residuals, lag=1)
        test_signals = raw_signals.reindex(test_dates)

        if test_signals.empty or test_signals.isna().all().all():
            return pd.DataFrame(0, index=test_dates, columns=full_returns.columns)

        cluster_labels_array = np.array([labels[assets_list.index(col)] for col in assets_list])

        weights = strategy.generate_target_weights(
            signals=test_signals,
            cluster_labels=cluster_labels_array,
            tokens=[col.replace('_returns', '') for col in assets_list],
            # full context so lagged inverse-vol weights are populated from day 1
            returns=context_slice,
            clusters_to_trade=valid_clusters
        )

        weights = weights.reindex(columns=full_returns.columns, fill_value=0.0)

        # Normalize leverage
        target_leverage = 1.5
        for date in weights.index:
            gross = weights.loc[date].abs().sum()
            if gross > 1e-8:
                weights.loc[date] = weights.loc[date] * (target_leverage / gross)

        return weights

    def portfolio_func(signals, returns, **kwargs):
        weights = signals.copy()
        max_turnover_per_day = 0.15

        if prev_weights[0] is not None:
            for i, date in enumerate(weights.index):
                if i == 0:
                    prior = prev_weights[0]
                else:
                    prior = weights.iloc[i-1]

                current = weights.loc[date]
                common_cols = current.index.intersection(prior.index)

                if len(common_cols) > 0:
                    delta = current.loc[common_cols] - prior.loc[common_cols].fillna(0)
                    turnover = delta.abs().sum()

                    if turnover > max_turnover_per_day:
                        scale = max_turnover_per_day / turnover
                        new_weights = prior.loc[common_cols].fillna(0) + delta * scale
                        weights.loc[date, common_cols] = new_weights

        if not weights.empty:
            prev_weights[0] = weights.iloc[-1].copy()

        return weights

    # Run backtest
    wf = WalkForwardBacktest(WalkForwardConfig(
        train_window=365, test_window=28, refit_frequency=28, min_train_history=365
    ))

    weights, fold_info = wf.run_backtest(
        returns=excess_returns,
        signal_func=signal_func,
        portfolio_func=portfolio_func
    )

    if weights.empty or weights.abs().sum().sum() == 0:
        return None

    # Compute metrics
    engine = BacktestEngine(cost_bps=cost_bps)
    net_ret, gross_ret, costs = engine.compute_net_returns(weights, excess_returns, cost_bps=cost_bps)
    turnover = engine.compute_turnover(weights)

    n_days = len(net_ret)
    gross_sharpe = (gross_ret.mean() * 365) / (gross_ret.std() * np.sqrt(365) + 1e-8)
    net_sharpe = (net_ret.mean() * 365) / (net_ret.std() * np.sqrt(365) + 1e-8)
    gross_total = gross_ret.sum()
    net_total = net_ret.sum()
    avg_turnover = turnover.mean()

    # Find break-even
    cost_model = TransactionCostModel()
    breakeven = cost_model.find_breakeven_cost(gross_ret, turnover, target_sharpe=0.0)

    return {
        'gross_sharpe': gross_sharpe,
        'net_sharpe': net_sharpe,
        'gross_return': gross_total,
        'net_return': net_total,
        'avg_turnover': avg_turnover,
        'breakeven_bps': breakeven,
        'n_days': n_days,
        # kept for DSR computation in the leaderboard; dropped before CSV export
        'net_returns': net_ret,
    }


def run_strategy2_test(H=5, L=20, cost_bps=50, verbose=True):
    """
    Test Strategy 2: Cluster composite deviation.
    """
    if verbose:
        print("\n" + "="*70)
        print("Strategy 2: Cluster Composite Deviation")
        print("="*70)

    data_dir = Path(__file__).parent.parent / 'data'
    loader = DataLoader(str(data_dir))
    excess_returns, prices, volumes, eth_data = loader.get_aligned_data()

    univ_manager = UniverseManager(
        mcap_percentile_low=0.0, mcap_percentile_high=1.0,
        min_volume_usd=50_000, min_history_days=60
    )
    universe_mask = univ_manager.get_universe_membership(prices, volumes, eth_data, excess_returns)

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

        train_subset = train_returns[valid_return_cols].dropna(axis=1, thresh=len(train_returns)*0.8).fillna(0)

        if train_subset.shape[1] < 10:
            return pd.DataFrame(0, index=test_dates, columns=full_returns.columns)

        # PCA
        pca = MarketModeExtractor(n_components=1)
        pca.fit(train_subset)
        train_residuals = pca.residualize(train_subset)

        # Clustering (SPONGE k=3)
        knn = KNNGraphBuilder(k=10)
        corr = knn.compute_correlation_matrix(train_residuals)
        adj = knn.build_weighted_knn(corr)

        sponge = SPONGEClustering(n_clusters=3, random_state=42)
        try:
            sponge.fit(adj)
            labels = sponge.labels_
        except:
            return pd.DataFrame(0, index=test_dates, columns=full_returns.columns)

        # Drop noisy cluster
        unique_labels = np.unique(labels)
        cluster_cohesion = {}
        assets_list = list(train_subset.columns)

        for c in unique_labels:
            members = [assets_list[i] for i in range(len(labels)) if labels[i] == c]
            if len(members) < 2:
                cluster_cohesion[c] = -1.0
            else:
                c_corr = corr.loc[members, members]
                avg_corr = (c_corr.sum().sum() - len(members)) / (len(members) * (len(members) - 1))
                cluster_cohesion[c] = avg_corr

        noisy_cluster = min(cluster_cohesion, key=cluster_cohesion.get)
        valid_clusters = [c for c in unique_labels if c != noisy_cluster]

        # Signal generation using Strategy 2
        context_start = test_dates[0] - pd.Timedelta(days=L + 30)
        context_slice = full_returns.loc[context_start:test_dates[-1], assets_list].fillna(0)
        context_residuals = pca.residualize(context_slice)

        cluster_labels_array = np.array([labels[assets_list.index(col)] for col in assets_list])

        # Use ClusterDeviationStrategy
        strategy = ClusterDeviationStrategy(
            lookback=H,
            zscore_window=L,
            entry_threshold=1.5,  # More aggressive entry
            composite_type='equal'
        )

        test_signals = strategy.compute_signals(
            context_residuals, cluster_labels_array,
            [col.replace('_returns', '') for col in assets_list], lag=1
        )
        test_signals = test_signals.reindex(test_dates)

        if test_signals.empty or test_signals.isna().all().all():
            return pd.DataFrame(0, index=test_dates, columns=full_returns.columns)

        weights = strategy.generate_target_weights(
            signals=test_signals,
            cluster_labels=cluster_labels_array,
            tokens=[col.replace('_returns', '') for col in assets_list],
            # full context so lagged inverse-vol weights are populated from day 1
            returns=context_slice,
            clusters_to_trade=valid_clusters
        )

        weights = weights.reindex(columns=full_returns.columns, fill_value=0.0)

        # Normalize
        target_leverage = 1.5
        for date in weights.index:
            gross = weights.loc[date].abs().sum()
            if gross > 1e-8:
                weights.loc[date] = weights.loc[date] * (target_leverage / gross)

        return weights

    def portfolio_func(signals, returns, **kwargs):
        weights = signals.copy()
        max_turnover_per_day = 0.15

        if prev_weights[0] is not None:
            for i, date in enumerate(weights.index):
                if i == 0:
                    prior = prev_weights[0]
                else:
                    prior = weights.iloc[i-1]

                current = weights.loc[date]
                common_cols = current.index.intersection(prior.index)

                if len(common_cols) > 0:
                    delta = current.loc[common_cols] - prior.loc[common_cols].fillna(0)
                    turnover = delta.abs().sum()

                    if turnover > max_turnover_per_day:
                        scale = max_turnover_per_day / turnover
                        new_weights = prior.loc[common_cols].fillna(0) + delta * scale
                        weights.loc[date, common_cols] = new_weights

        if not weights.empty:
            prev_weights[0] = weights.iloc[-1].copy()

        return weights

    # Run backtest
    wf = WalkForwardBacktest(WalkForwardConfig(
        train_window=365, test_window=28, refit_frequency=28, min_train_history=365
    ))

    weights, fold_info = wf.run_backtest(
        returns=excess_returns,
        signal_func=signal_func,
        portfolio_func=portfolio_func
    )

    if weights.empty or weights.abs().sum().sum() == 0:
        if verbose:
            print("Strategy 2 failed - no valid weights generated")
        return None

    # Compute metrics
    engine = BacktestEngine(cost_bps=cost_bps)
    net_ret, gross_ret, costs = engine.compute_net_returns(weights, excess_returns, cost_bps=cost_bps)
    turnover = engine.compute_turnover(weights)

    gross_sharpe = (gross_ret.mean() * 365) / (gross_ret.std() * np.sqrt(365) + 1e-8)
    net_sharpe = (net_ret.mean() * 365) / (net_ret.std() * np.sqrt(365) + 1e-8)

    cost_model = TransactionCostModel()
    breakeven = cost_model.find_breakeven_cost(gross_ret, turnover, target_sharpe=0.0)

    if verbose:
        print(f"Gross Sharpe: {gross_sharpe:.2f}")
        print(f"Net Sharpe @ {cost_bps}bps: {net_sharpe:.2f}")
        print(f"Avg Turnover: {turnover.mean()*100:.1f}%")
        print(f"Break-even cost: {breakeven:.1f} bps")

    return {
        'gross_sharpe': gross_sharpe,
        'net_sharpe': net_sharpe,
        'gross_return': gross_ret.sum(),
        'net_return': net_ret.sum(),
        'avg_turnover': turnover.mean(),
        'breakeven_bps': breakeven,
        'net_returns': net_ret,
        'weights': weights,
    }


def generate_phase2_leaderboard(clustering_results, strategy2_results=None, cost_bps=50):
    """
    Generate final Phase 2 leaderboard.
    """
    from stat_arb.backtest.statistics import deflated_sharpe_ratio, per_period_sharpe

    results_dir = Path(__file__).parent / 'reporting' / 'phase2'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Collect per-config net return series (not exported to CSV)
    net_return_series = {}
    if 'net_returns' in clustering_results.columns:
        for _, row in clustering_results.iterrows():
            if isinstance(row.get('net_returns'), pd.Series):
                net_return_series[(row['method'], row['k'], 'Z-Score MR')] = row['net_returns']
        clustering_results = clustering_results.drop(columns=['net_returns'])

    # Save clustering results
    clustering_results.to_csv(results_dir / 'clustering_sweep_results.csv', index=False)

    # Create leaderboard
    leaderboard = clustering_results.copy()
    leaderboard['strategy'] = 'Z-Score MR'

    if strategy2_results is not None:
        s2_row = pd.DataFrame([{
            'method': 'SPONGE',
            'k': 3,
            'strategy': 'Cluster Dev',
            'gross_sharpe': strategy2_results['gross_sharpe'],
            'net_sharpe': strategy2_results['net_sharpe'],
            'gross_return': strategy2_results['gross_return'],
            'net_return': strategy2_results['net_return'],
            'avg_turnover': strategy2_results['avg_turnover'],
            'breakeven_bps': strategy2_results['breakeven_bps'],
        }])
        leaderboard = pd.concat([leaderboard, s2_row], ignore_index=True)
        if isinstance(strategy2_results.get('net_returns'), pd.Series):
            net_return_series[('SPONGE', 3, 'Cluster Dev')] = strategy2_results['net_returns']

    # Sort by gross sharpe
    leaderboard = leaderboard.sort_values('gross_sharpe', ascending=False)

    # Deflated Sharpe for the SELECTED (top net-Sharpe) configuration.
    # The leaderboard itself is the multiple-testing pool: every configuration
    # tried counts as a trial, and the cross-trial Sharpe dispersion comes
    # from the sweep's own per-period Sharpes.
    leaderboard['dsr_net'] = np.nan
    if net_return_series:
        trial_srs = [per_period_sharpe(s) for s in net_return_series.values()]
        n_trials = len(trial_srs)
        for idx, row in leaderboard.iterrows():
            key = (row['method'], row['k'], row['strategy'])
            if key in net_return_series:
                dsr_info = deflated_sharpe_ratio(
                    net_return_series[key], n_trials=n_trials, trial_sharpes=trial_srs
                )
                leaderboard.loc[idx, 'dsr_net'] = dsr_info['dsr']

    leaderboard.to_csv(results_dir / 'leaderboard.csv', index=False)

    # Print leaderboard
    print("\n" + "="*80)
    print("PHASE 2 LEADERBOARD (sorted by Gross Sharpe)")
    print("="*80)
    print(f"{'Rank':<5} {'Method':<15} {'k':<4} {'Strategy':<12} {'Gross SR':<10} "
          f"{'Net SR':<10} {'DSR':<7} {'Turnover':<10} {'BE bps':<8}")
    print("-"*80)

    for i, (_, row) in enumerate(leaderboard.iterrows()):
        if pd.notna(row['gross_sharpe']):
            strategy = row.get('strategy', 'Z-Score MR')
            dsr_txt = f"{row['dsr_net']:.2f}" if pd.notna(row.get('dsr_net')) else "n/a"
            print(f"{i+1:<5} {row['method']:<15} {int(row['k']):<4} {strategy:<12} "
                  f"{row['gross_sharpe']:<10.2f} {row['net_sharpe']:<10.2f} {dsr_txt:<7} "
                  f"{row['avg_turnover']*100:<10.1f}% {row['breakeven_bps']:<8.1f}")

    print("="*80)

    # Best configuration
    best = leaderboard.iloc[0]
    print(f"\nBest Configuration: {best['method']} k={int(best['k'])} "
          f"({best.get('strategy', 'Z-Score MR')})")
    print(f"  Gross Sharpe: {best['gross_sharpe']:.2f}")
    print(f"  Net Sharpe @ {cost_bps}bps: {best['net_sharpe']:.2f}")
    print(f"  Break-even cost: {best['breakeven_bps']:.1f} bps")

    return leaderboard


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent.parent)

    print("\n" + "="*70)
    print("STAT-ARB BACKTEST: PHASE 2")
    print("Alternative Clustering Methods & Strategies")
    print("="*70)

    # Run clustering sweep
    print("\n[1/3] Running clustering method sweep...")
    clustering_results = run_phase2_clustering_sweep(
        k_grid=[2, 3, 4, 5, 6],
        H=5, L=20, cost_bps=50, verbose=True
    )

    # Run Strategy 2
    print("\n[2/3] Testing Strategy 2 (Cluster Deviation)...")
    strategy2_results = run_strategy2_test(H=5, L=20, cost_bps=50, verbose=True)

    # Generate leaderboard
    print("\n[3/3] Generating Phase 2 Leaderboard...")
    leaderboard = generate_phase2_leaderboard(clustering_results, strategy2_results)

    print("\nPhase 2 complete!")
