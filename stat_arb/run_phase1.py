"""
Phase 1: Fixed SPONGE k=3 Baseline Backtest

Baseline specification:
- SPONGE clustering with k=3
- Drop noisy cluster (lowest within-cluster cohesion)
- Return-based clustered z-score mean reversion
- Daily walk-forward OOS with 365-day train, 28-day test
- No lookahead: signals use data through t-1
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import local modules
from stat_arb.data.loader import DataLoader
from stat_arb.data.universe import UniverseManager
from stat_arb.pca.market_mode import MarketModeExtractor
from stat_arb.graphs.knn_graph import KNNGraphBuilder
from stat_arb.clustering.sponge import SPONGEClustering
from stat_arb.signals.zscore_strategy import ZScoreStrategy
from stat_arb.backtest.engine import BacktestEngine
from stat_arb.backtest.walk_forward import WalkForwardBacktest, WalkForwardConfig
from stat_arb.backtest.costs import TransactionCostModel
from stat_arb.backtest.statistics import probabilistic_sharpe_ratio, deflated_sharpe_ratio


def run_phase1(H=5, L=60, cost_bps=50, verbose=True):
    """
    Run Phase 1 baseline backtest.

    Parameters
    ----------
    H : int
        Momentum lookback window for rolling sum
    L : int
        Z-score normalization window
    cost_bps : float
        Transaction cost in bps per side
    verbose : bool
        Print progress messages

    Returns
    -------
    dict
        Results dictionary with weights, returns, metrics, fold_info
    """
    if verbose:
        print("="*70)
        print("Phase 1: SPONGE k=3 Baseline Backtest")
        print("="*70)
        print(f"Parameters: H={H}, L={L}, cost_bps={cost_bps}")

    # 1. Load Data
    data_dir = Path(__file__).parent.parent / 'data'
    loader = DataLoader(str(data_dir))
    if verbose:
        print("\n[1/6] Loading data...")
    excess_returns, prices, volumes, eth_data = loader.get_aligned_data()

    if verbose:
        print(f"  Data range: {excess_returns.index[0].date()} to {excess_returns.index[-1].date()}")
        print(f"  Total tokens (excl stablecoins): {len(excess_returns.columns)}")
        print(f"  Total trading days: {len(excess_returns)}")

    # 2. Universe Management
    if verbose:
        print("\n[2/6] Computing universe membership...")
    univ_manager = UniverseManager(
        mcap_percentile_low=0.0,
        mcap_percentile_high=1.0,  # Include all for now
        min_volume_usd=50_000,
        min_history_days=60
    )
    universe_mask = univ_manager.get_universe_membership(prices, volumes, eth_data, excess_returns)

    # Map return columns to token names (strip _returns suffix)
    return_cols_to_tokens = {col: col.replace('_returns', '') for col in excess_returns.columns}
    tokens_to_return_cols = {v: k for k, v in return_cols_to_tokens.items()}

    # Save universe for inspection
    results_dir = Path(__file__).parent / 'reporting' / 'phase1'
    results_dir.mkdir(parents=True, exist_ok=True)
    universe_mask.to_csv(results_dir / 'universe_mask.csv')

    if verbose:
        avg_univ = universe_mask.sum(axis=1).mean()
        print(f"  Avg universe size: {avg_univ:.1f} assets")

    # 3. Define Signal Generation Function
    def phase1_signal_func(train_returns, test_dates, full_returns, **kwargs):
        """
        Generate signals for test_dates using model fit on train_returns.
        NO LOOKAHEAD: all signals use data up to t-1.
        """
        # Get rebalance date (first day of test period)
        rebalance_date = test_dates[0]

        # Get universe at rebalance date
        if rebalance_date in universe_mask.index:
            current_univ = universe_mask.loc[rebalance_date]
        else:
            # Fallback to closest prior date
            prior_dates = universe_mask.index[universe_mask.index <= rebalance_date]
            if len(prior_dates) > 0:
                current_univ = universe_mask.loc[prior_dates[-1]]
            else:
                current_univ = universe_mask.iloc[0]

        # Get valid columns (in universe)
        valid_return_cols = [col for col in train_returns.columns
                           if return_cols_to_tokens.get(col, '') in current_univ.index
                           and current_univ.get(return_cols_to_tokens.get(col, ''), False)]

        if len(valid_return_cols) == 0:
            # Try matching by return column names directly in universe_mask
            valid_return_cols = [col for col in train_returns.columns
                               if col in current_univ.index and current_univ.get(col, False)]

        if len(valid_return_cols) == 0:
            # Last resort: use all columns with sufficient data
            valid_return_cols = train_returns.columns.tolist()

        # Filter training data
        train_subset = train_returns[valid_return_cols].dropna(axis=1, thresh=len(train_returns)*0.8)
        train_subset = train_subset.fillna(0)

        if train_subset.shape[1] < 10:
            if verbose:
                print(f"  Warning: Too few assets ({train_subset.shape[1]}) at {rebalance_date.date()}")
            return pd.DataFrame(0, index=test_dates, columns=full_returns.columns)

        # A. PCA Market Mode Removal (fit on training data only)
        pca = MarketModeExtractor(n_components=1)
        pca.fit(train_subset)

        # B. Residualize training data
        train_residuals = pca.residualize(train_subset)

        # C. Build k-NN graph and cluster (SPONGE k=3)
        knn = KNNGraphBuilder(k=10)
        corr = knn.compute_correlation_matrix(train_residuals)
        adj = knn.build_weighted_knn(corr)

        sponge = SPONGEClustering(n_clusters=3, random_state=42)
        try:
            sponge.fit(adj)
            labels = sponge.labels_
        except Exception as e:
            if verbose:
                print(f"  Clustering failed at {rebalance_date.date()}: {e}")
            return pd.DataFrame(0, index=test_dates, columns=full_returns.columns)

        # D. Identify and drop noisy cluster (lowest cohesion)
        unique_labels = np.unique(labels)
        cluster_cohesion = {}
        cluster_members = {}
        assets_list = list(train_subset.columns)

        for c in unique_labels:
            members = [assets_list[i] for i in range(len(labels)) if labels[i] == c]
            cluster_members[c] = members
            if len(members) < 2:
                cluster_cohesion[c] = -1.0
            else:
                # Average pairwise correlation within cluster
                c_corr = corr.loc[members, members]
                avg_corr = (c_corr.sum().sum() - len(members)) / (len(members) * (len(members) - 1))
                cluster_cohesion[c] = avg_corr

        # Drop cluster with lowest cohesion
        noisy_cluster = min(cluster_cohesion, key=cluster_cohesion.get)
        valid_clusters = [c for c in unique_labels if c != noisy_cluster]

        # E. Prepare context data for signal computation
        # Need lookback for z-score (L days) plus some buffer
        context_start = test_dates[0] - pd.Timedelta(days=L + 30)
        context_end = test_dates[-1]

        context_slice = full_returns.loc[context_start:context_end, assets_list].fillna(0)

        # Residualize context using training-fitted PCA
        context_residuals = pca.residualize(context_slice)

        # F. Compute z-score signals
        strategy = ZScoreStrategy(H=H, L=L, quantile=0.2)
        raw_signals = strategy.compute_signals(context_residuals, lag=1)  # lag=1 for no lookahead

        # Get signals for test dates only
        test_signals = raw_signals.reindex(test_dates)

        if test_signals.empty or test_signals.isna().all().all():
            return pd.DataFrame(0, index=test_dates, columns=full_returns.columns)

        # G. Generate target weights
        # Create cluster labels array aligned with assets_list
        cluster_labels_array = np.array([labels[assets_list.index(col)] for col in assets_list])

        weights = strategy.generate_target_weights(
            signals=test_signals,
            cluster_labels=cluster_labels_array,
            tokens=[col.replace('_returns', '') for col in assets_list],  # Strategy expects token names
            # Full context (starts L+30d before the test window) so the lagged
            # inverse-vol weights have real estimates from the first test day
            returns=context_slice,
            clusters_to_trade=valid_clusters
        )

        # H. Reindex weights to full universe
        weights = weights.reindex(columns=full_returns.columns, fill_value=0.0)

        # I. Normalize to target gross leverage
        target_leverage = 1.5
        for date in weights.index:
            gross = weights.loc[date].abs().sum()
            if gross > 1e-8:
                weights.loc[date] = weights.loc[date] * (target_leverage / gross)

        return weights

    # 4. Define Portfolio Function with turnover control
    prev_weights = [None]  # Use list to allow modification in nested function

    def phase1_portfolio_func(signals, returns, **kwargs):
        """
        Apply turnover control to signals.
        Limits daily weight change to avoid excessive trading.
        """
        weights = signals.copy()
        max_turnover_per_day = 0.15  # Max 15% turnover per day

        if prev_weights[0] is not None:
            for i, date in enumerate(weights.index):
                if i == 0:
                    # For first day of fold, compare to last known weights
                    prior = prev_weights[0]
                else:
                    prior = weights.iloc[i-1]

                # Align columns
                current = weights.loc[date]
                common_cols = current.index.intersection(prior.index)

                if len(common_cols) > 0:
                    delta = current.loc[common_cols] - prior.loc[common_cols].fillna(0)
                    turnover = delta.abs().sum()

                    if turnover > max_turnover_per_day:
                        scale = max_turnover_per_day / turnover
                        new_weights = prior.loc[common_cols].fillna(0) + delta * scale
                        weights.loc[date, common_cols] = new_weights

        # Store last weights for next fold
        if not weights.empty:
            prev_weights[0] = weights.iloc[-1].copy()

        return weights

    # 5. Run Walk-Forward Backtest
    if verbose:
        print("\n[3/6] Running walk-forward backtest...")
        print("  Train window: 365 days")
        print("  Test window: 28 days")
        print("  Refit frequency: 28 days")

    wf_config = WalkForwardConfig(
        train_window=365,
        test_window=28,
        refit_frequency=28,
        min_train_history=365
    )
    wf = WalkForwardBacktest(wf_config)

    weights, fold_info = wf.run_backtest(
        returns=excess_returns,
        signal_func=phase1_signal_func,
        portfolio_func=phase1_portfolio_func
    )

    # Check for successful folds
    successful_folds = [f for f in fold_info if f.get('status') == 'success']
    failed_folds = [f for f in fold_info if f.get('status') == 'error']

    if verbose:
        print(f"  Completed: {len(successful_folds)}/{len(fold_info)} folds successful")
        if failed_folds:
            print(f"  Failed folds: {len(failed_folds)}")

    if weights.empty or weights.abs().sum().sum() == 0:
        print("\nERROR: No valid weights generated. Check data alignment.")
        return None

    # 6. Compute Performance
    if verbose:
        print("\n[4/6] Computing performance metrics...")

    engine = BacktestEngine(cost_bps=cost_bps)
    net_ret, gross_ret, costs = engine.compute_net_returns(weights, excess_returns, cost_bps=cost_bps)

    # Compute metrics
    turnover = engine.compute_turnover(weights)
    cum_ret = engine.compute_cumulative_returns(net_ret)
    drawdown = engine.compute_drawdown(cum_ret)
    exposure_stats = engine.compute_exposure_stats(weights)

    # Annualized metrics
    n_days = len(net_ret)
    ann_ret = net_ret.mean() * 365
    ann_vol = net_ret.std() * np.sqrt(365)
    sharpe = ann_ret / (ann_vol + 1e-8)
    max_dd = drawdown.min()
    avg_turnover = turnover.mean()

    # Total return (log)
    total_ret = net_ret.sum()

    # CAGR
    years = n_days / 365
    cagr = (np.exp(total_ret) ** (1/years) - 1) * 100 if years > 0 else 0

    # Multiple-testing-aware statistics (Bailey & Lopez de Prado).
    # n_trials = 9 reflects the 3x3 H/L grid this configuration was drawn from;
    # if you widen the search, raise it accordingly.
    psr = probabilistic_sharpe_ratio(net_ret)
    dsr_info = deflated_sharpe_ratio(net_ret, n_trials=9)

    # 7. Save Results
    if verbose:
        print("\n[5/6] Saving results...")

    net_ret.to_csv(results_dir / 'net_returns.csv')
    gross_ret.to_csv(results_dir / 'gross_returns.csv')
    weights.to_csv(results_dir / 'weights.csv')
    turnover.to_csv(results_dir / 'turnover.csv')
    pd.DataFrame(fold_info).to_csv(results_dir / 'folds.csv')

    # 8. Print Summary
    # Gross performance (pre-cost)
    gross_sharpe = (gross_ret.mean() * 365) / (gross_ret.std() * np.sqrt(365) + 1e-8)
    gross_total = gross_ret.sum()
    cost_drag = costs.sum()

    if verbose:
        print("\n[6/6] Phase 1 Results Summary")
        print("="*70)
        print(f"Phase 1 Baseline: SPONGE k=3, dropped noisy cluster (lowest cohesion)")
        print(f"Parameters: H={H}, L={L}")
        print("-"*70)
        print(f"Period: {net_ret.index[0].date()} to {net_ret.index[-1].date()}")
        print(f"Trading days: {n_days}")
        print("-"*70)
        print("GROSS (pre-cost):")
        print(f"  Sharpe Ratio:    {gross_sharpe:.2f}")
        print(f"  Total Return:    {gross_total*100:.2f}%")
        print(f"NET (post-cost @ {cost_bps} bps):")
        print(f"  Sharpe Ratio:    {sharpe:.2f}")
        print(f"  CAGR:            {cagr:.2f}%")
        print(f"  Ann. Return:     {ann_ret*100:.2f}%")
        print(f"  Ann. Volatility: {ann_vol*100:.2f}%")
        print(f"  Max Drawdown:    {max_dd*100:.2f}%")
        print(f"  Total Return:    {total_ret*100:.2f}%")
        print(f"STATISTICAL SIGNIFICANCE:")
        print(f"  PSR (P[SR>0]):          {psr:.3f}")
        print(f"  DSR (vs {dsr_info['n_trials']}-trial max):  {dsr_info['dsr']:.3f}")
        print(f"COST ANALYSIS:")
        print(f"  Total Cost Drag: {cost_drag*100:.2f}%")
        print(f"  Avg Daily Turnover: {avg_turnover*100:.2f}%")
        print("-"*70)
        print(f"Avg Gross Exposure: {exposure_stats['gross_exposure'].mean():.2f}")
        print(f"Avg Net Exposure:   {exposure_stats['net_exposure'].mean():.4f}")
        print(f"Avg # Longs:        {exposure_stats['n_longs'].mean():.1f}")
        print(f"Avg # Shorts:       {exposure_stats['n_shorts'].mean():.1f}")
        print("="*70)

    return {
        'weights': weights,
        'net_returns': net_ret,
        'gross_returns': gross_ret,
        'costs': costs,
        'turnover': turnover,
        'cum_returns': cum_ret,
        'drawdown': drawdown,
        'exposure_stats': exposure_stats,
        'fold_info': fold_info,
        'metrics': {
            'sharpe': sharpe,
            'psr': psr,
            'dsr': dsr_info['dsr'],
            'gross_sharpe': gross_sharpe,
            'cagr': cagr,
            'ann_return': ann_ret,
            'ann_vol': ann_vol,
            'max_drawdown': max_dd,
            'avg_turnover': avg_turnover,
            'total_return': total_ret,
            'gross_return': gross_total,
            'cost_drag': cost_drag,
            'n_days': n_days,
        },
        'params': {'H': H, 'L': L, 'cost_bps': cost_bps}
    }


def run_cost_sensitivity(results, cost_grid=None, verbose=True):
    """
    Run cost sensitivity analysis on existing results.

    Parameters
    ----------
    results : dict
        Results from run_phase1
    cost_grid : list
        List of cost levels in bps
    verbose : bool
        Print results

    Returns
    -------
    pd.DataFrame
        Sensitivity results
    """
    if cost_grid is None:
        cost_grid = [10, 25, 50, 100, 200]

    if results is None:
        print("No results to analyze")
        return None

    weights = results['weights']
    gross_ret = results['gross_returns']
    turnover = results['turnover']

    cost_model = TransactionCostModel(cost_grid=cost_grid)
    sensitivity = cost_model.sensitivity_analysis(gross_ret, turnover, cost_grid)

    # Find break-even cost
    breakeven = cost_model.find_breakeven_cost(gross_ret, turnover, target_sharpe=0.0)

    if verbose:
        print("\n" + "="*70)
        print("Cost Sensitivity Analysis")
        print("="*70)
        print(f"{'Cost (bps)':<12} {'Sharpe':<10} {'Ann Ret':<12} {'Max DD':<12}")
        print("-"*70)
        for _, row in sensitivity.iterrows():
            print(f"{int(row['cost_bps']):<12} {row['sharpe']:<10.2f} {row['ann_return']*100:<12.2f}% {row['max_drawdown']*100:<12.2f}%")
        print("-"*70)
        print(f"Break-even cost (Sharpe=0): {breakeven:.1f} bps")
        print("="*70)

    return sensitivity, breakeven


def run_carry_sensitivity(results, carry_grid=None, verbose=True):
    """
    Stress net performance for financing carry (perp funding / borrow drag)
    applied to gross exposure, on top of the base transaction cost.

    Per-token funding series are not in the dataset, so this is a uniform
    stress knob, not a funding model; see BacktestEngine.carry_bps_daily.
    """
    if carry_grid is None:
        carry_grid = [0, 3, 6, 10]  # bps/day on gross exposure (~0-36% ann at 1x)

    if results is None:
        print("No results to analyze")
        return None

    weights = results['weights']
    gross_ret = results['gross_returns']
    turnover = results['turnover']
    base_costs = results['costs']
    gross_exposure = weights.abs().sum(axis=1)

    rows = []
    for carry in carry_grid:
        carry_costs = gross_exposure.reindex(gross_ret.index).fillna(0.0) * carry / 10000
        net = gross_ret - base_costs.reindex(gross_ret.index).fillna(0.0) - carry_costs
        ann_ret = net.mean() * 365
        ann_vol = net.std() * np.sqrt(365)
        rows.append({
            'carry_bps_daily': carry,
            'sharpe': ann_ret / (ann_vol + 1e-8),
            'ann_return': ann_ret,
        })
    sens = pd.DataFrame(rows)

    if verbose:
        print("\n" + "="*70)
        print("Financing Carry Sensitivity (bps/day on gross exposure)")
        print("="*70)
        for _, row in sens.iterrows():
            print(f"  carry={int(row['carry_bps_daily']):>3} bps/day: "
                  f"Sharpe {row['sharpe']:>6.2f}, Ann Ret {row['ann_return']*100:>7.2f}%")
        print("="*70)

    return sens


def run_parameter_sensitivity(H_grid=None, L_grid=None, cost_bps=50, verbose=True):
    """
    Run H/L parameter sensitivity grid.

    Parameters
    ----------
    H_grid : list
        Grid of H values
    L_grid : list
        Grid of L values
    cost_bps : float
        Transaction cost
    verbose : bool
        Print progress

    Returns
    -------
    pd.DataFrame
        Results for each H/L combination
    """
    if H_grid is None:
        H_grid = [1, 5, 10]
    if L_grid is None:
        L_grid = [20, 60, 120]

    results_list = []

    total = len(H_grid) * len(L_grid)
    idx = 0

    if verbose:
        print("\n" + "="*70)
        print("H/L Parameter Sensitivity Grid")
        print("="*70)

    for H in H_grid:
        for L in L_grid:
            idx += 1
            if verbose:
                print(f"\n[{idx}/{total}] Running H={H}, L={L}...")

            result = run_phase1(H=H, L=L, cost_bps=cost_bps, verbose=False)

            if result is not None:
                results_list.append({
                    'H': H,
                    'L': L,
                    'sharpe': result['metrics']['sharpe'],
                    'cagr': result['metrics']['cagr'],
                    'max_drawdown': result['metrics']['max_drawdown'],
                    'avg_turnover': result['metrics']['avg_turnover'],
                    'total_return': result['metrics']['total_return'],
                })
                if verbose:
                    print(f"  Sharpe: {result['metrics']['sharpe']:.2f}, CAGR: {result['metrics']['cagr']:.2f}%")
            else:
                results_list.append({
                    'H': H, 'L': L, 'sharpe': np.nan, 'cagr': np.nan,
                    'max_drawdown': np.nan, 'avg_turnover': np.nan, 'total_return': np.nan
                })

    df = pd.DataFrame(results_list)

    if verbose:
        print("\n" + "="*70)
        print("Parameter Sensitivity Summary")
        print("="*70)
        print(df.to_string(index=False))
        best = df.loc[df['sharpe'].idxmax()]
        print(f"\nBest: H={int(best['H'])}, L={int(best['L'])} -> Sharpe={best['sharpe']:.2f}")
        print("="*70)

    return df


def generate_phase1_report(results, cost_sensitivity=None, param_sensitivity=None):
    """
    Generate comprehensive Phase 1 report with plots.
    """
    if results is None:
        print("No results to report")
        return

    results_dir = Path(__file__).parent / 'reporting' / 'phase1'
    results_dir.mkdir(parents=True, exist_ok=True)

    net_ret = results['net_returns']
    cum_ret = results['cum_returns']
    drawdown = results['drawdown']
    turnover = results['turnover']
    metrics = results['metrics']
    params = results['params']

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Equity Curve
    ax1 = axes[0, 0]
    cum_ret.plot(ax=ax1, linewidth=1.5)
    ax1.set_title('Cumulative Returns (Log)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Return')
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax1.grid(True, alpha=0.3)

    # 2. Drawdown
    ax2 = axes[0, 1]
    drawdown.plot(ax=ax2, linewidth=1.5, color='red')
    ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
    ax2.set_title('Drawdown')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown')
    ax2.grid(True, alpha=0.3)

    # 3. Rolling Sharpe (60-day)
    ax3 = axes[1, 0]
    rolling_sharpe = (net_ret.rolling(60).mean() / net_ret.rolling(60).std()) * np.sqrt(365)
    rolling_sharpe.plot(ax=ax3, linewidth=1.5)
    ax3.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax3.axhline(metrics['sharpe'], color='green', linestyle='--', linewidth=1, label=f'Full period: {metrics["sharpe"]:.2f}')
    ax3.set_title('Rolling 60-Day Sharpe Ratio')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Daily Turnover
    ax4 = axes[1, 1]
    turnover.plot(ax=ax4, linewidth=0.8, alpha=0.7)
    ax4.axhline(metrics['avg_turnover'], color='red', linestyle='--', linewidth=1, label=f'Avg: {metrics["avg_turnover"]*100:.1f}%')
    ax4.set_title('Daily Turnover')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Turnover')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'Phase 1 Baseline: SPONGE k=3, H={params["H"]}, L={params["L"]}, {params["cost_bps"]} bps',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'phase1_diagnostics.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save metrics summary
    summary = pd.DataFrame({
        'Metric': ['Sharpe Ratio', 'CAGR (%)', 'Ann. Return (%)', 'Ann. Volatility (%)',
                   'Max Drawdown (%)', 'Avg Turnover (%)', 'Total Return (log %)', 'Trading Days'],
        'Value': [f"{metrics['sharpe']:.2f}", f"{metrics['cagr']:.2f}", f"{metrics['ann_return']*100:.2f}",
                  f"{metrics['ann_vol']*100:.2f}", f"{metrics['max_drawdown']*100:.2f}",
                  f"{metrics['avg_turnover']*100:.2f}", f"{metrics['total_return']*100:.2f}",
                  f"{metrics['n_days']}"]
    })
    summary.to_csv(results_dir / 'metrics_summary.csv', index=False)

    print(f"\nReport saved to {results_dir}")
    print("  - phase1_diagnostics.png")
    print("  - metrics_summary.csv")


if __name__ == "__main__":
    import sys

    # Change to project directory
    import os
    os.chdir(Path(__file__).parent.parent)

    print("\n" + "="*70)
    print("STAT-ARB BACKTEST: PHASE 1 BASELINE")
    print("="*70)

    # Run baseline
    results = run_phase1(H=5, L=60, cost_bps=50, verbose=True)

    if results is not None:
        # Cost sensitivity
        cost_sens, breakeven = run_cost_sensitivity(results, verbose=True)

        # Financing carry stress
        run_carry_sensitivity(results, verbose=True)

        # Generate report
        generate_phase1_report(results)

        # Parameter sensitivity (optional - takes longer)
        run_param_sens = input("\nRun H/L parameter sensitivity grid? (y/n): ").strip().lower() == 'y'
        if run_param_sens:
            param_sens = run_parameter_sensitivity(verbose=True)
    else:
        print("\nBacktest failed. Please check error messages above.")
