"""
Report generation for stat-arb backtest.
Generates comprehensive Phase 1 and Phase 2 reports.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from stat_arb.data.loader import DataLoader
from stat_arb.pca.market_mode import MarketModeExtractor


def compute_factor_betas(portfolio_returns, factor_returns, window=60):
    """
    Compute rolling beta of portfolio returns to a factor.

    Returns:
        rolling_beta: pd.Series of rolling betas
        full_period_beta: float of full period beta
    """
    # Align
    common_idx = portfolio_returns.index.intersection(factor_returns.index)
    port = portfolio_returns.loc[common_idx].dropna()
    factor = factor_returns.loc[common_idx].dropna()

    common_idx = port.index.intersection(factor.index)
    port = port.loc[common_idx]
    factor = factor.loc[common_idx]

    # Full period beta
    cov = np.cov(port.values, factor.values)[0, 1]
    var = np.var(factor.values)
    full_period_beta = cov / (var + 1e-8)

    # Rolling beta
    rolling_cov = port.rolling(window).cov(factor)
    rolling_var = factor.rolling(window).var()
    rolling_beta = rolling_cov / (rolling_var + 1e-8)

    return rolling_beta, full_period_beta


def generate_phase1_report(results, data_dir='data'):
    """
    Generate comprehensive Phase 1 report with all required diagnostics.
    """
    results_dir = Path(__file__).parent / 'phase1'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Extract data
    net_ret = results['net_returns']
    gross_ret = results['gross_returns']
    weights = results['weights']
    turnover = results['turnover']
    cum_ret = results['cum_returns']
    drawdown = results['drawdown']
    metrics = results['metrics']
    params = results['params']

    # Load ETH data for beta computation
    loader = DataLoader(data_dir)
    eth_data = loader.load_eth_data()
    eth_returns = np.log(eth_data['close']).diff()

    # Compute ETH beta
    eth_beta_rolling, eth_beta_full = compute_factor_betas(net_ret, eth_returns)

    # Compute PC1 (market mode) using PCA on excess returns
    excess_returns = loader.load_excess_returns()
    common_dates = net_ret.index.intersection(excess_returns.index)

    # Fit PCA on available data
    pca = MarketModeExtractor(n_components=1)
    pca.fit(excess_returns.loc[:net_ret.index[0]].dropna(how='all').fillna(0))
    pc1_returns = pca.get_factor_returns(excess_returns.loc[common_dates].fillna(0))['PC1']

    # Compute PC1 beta
    pc1_beta_rolling, pc1_beta_full = compute_factor_betas(net_ret, pc1_returns)

    # Save diagnostics
    diagnostics = pd.DataFrame({
        'net_returns': net_ret,
        'gross_returns': gross_ret,
        'turnover': turnover,
        'eth_beta_rolling': eth_beta_rolling,
        'pc1_beta_rolling': pc1_beta_rolling,
    })
    diagnostics.to_csv(results_dir / 'diagnostics.csv')

    # Create comprehensive report figure
    fig = plt.figure(figsize=(16, 14))

    # 1. Equity Curve
    ax1 = fig.add_subplot(3, 3, 1)
    cum_ret.plot(ax=ax1, linewidth=1.5, label='Strategy')
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax1.set_title('Cumulative Returns (Log)')
    ax1.set_xlabel('')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Drawdown
    ax2 = fig.add_subplot(3, 3, 2)
    drawdown.plot(ax=ax2, linewidth=1.5, color='red')
    ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
    ax2.set_title(f'Drawdown (Max: {metrics["max_drawdown"]*100:.1f}%)')
    ax2.set_xlabel('')
    ax2.grid(True, alpha=0.3)

    # 3. Rolling Sharpe
    ax3 = fig.add_subplot(3, 3, 3)
    rolling_sharpe = (net_ret.rolling(60).mean() / net_ret.rolling(60).std()) * np.sqrt(365)
    rolling_sharpe.plot(ax=ax3, linewidth=1.5)
    ax3.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax3.axhline(metrics['sharpe'], color='green', linestyle='--', alpha=0.7,
                label=f'Full: {metrics["sharpe"]:.2f}')
    ax3.set_title('Rolling 60-Day Sharpe')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Daily Turnover
    ax4 = fig.add_subplot(3, 3, 4)
    turnover.plot(ax=ax4, linewidth=0.8, alpha=0.7)
    ax4.axhline(metrics['avg_turnover'], color='red', linestyle='--',
                label=f'Avg: {metrics["avg_turnover"]*100:.1f}%')
    ax4.set_title('Daily Turnover')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. ETH Beta
    ax5 = fig.add_subplot(3, 3, 5)
    eth_beta_rolling.plot(ax=ax5, linewidth=1.5)
    ax5.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax5.axhline(eth_beta_full, color='green', linestyle='--', alpha=0.7,
                label=f'Full: {eth_beta_full:.3f}')
    ax5.set_title('Rolling ETH Beta')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. PC1 Beta
    ax6 = fig.add_subplot(3, 3, 6)
    pc1_beta_rolling.plot(ax=ax6, linewidth=1.5)
    ax6.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax6.axhline(pc1_beta_full, color='green', linestyle='--', alpha=0.7,
                label=f'Full: {pc1_beta_full:.3f}')
    ax6.set_title('Rolling PC1 (Market Mode) Beta')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 7. Monthly Returns
    ax7 = fig.add_subplot(3, 3, 7)
    monthly_ret = net_ret.resample('ME').sum() * 100
    colors = ['green' if x > 0 else 'red' for x in monthly_ret.values]
    monthly_ret.plot(kind='bar', ax=ax7, color=colors, alpha=0.7)
    ax7.axhline(0, color='black', linewidth=0.5)
    ax7.set_title('Monthly Returns (%)')
    ax7.set_xticklabels([d.strftime('%Y-%m') for d in monthly_ret.index], rotation=45, ha='right')
    ax7.grid(True, alpha=0.3, axis='y')

    # 8. Gross vs Net Returns Distribution
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.hist(gross_ret.values * 100, bins=50, alpha=0.5, label=f'Gross (mean={gross_ret.mean()*100:.2f}%)', color='blue')
    ax8.hist(net_ret.values * 100, bins=50, alpha=0.5, label=f'Net (mean={net_ret.mean()*100:.2f}%)', color='red')
    ax8.axvline(0, color='black', linestyle='--')
    ax8.set_title('Daily Returns Distribution (%)')
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)

    # 9. Summary Statistics Text
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')
    summary_text = f"""
PHASE 1 BASELINE RESULTS
========================
Configuration: SPONGE k=3
Parameters: H={params['H']}, L={params['L']}
Dropped noisy cluster (lowest cohesion)

Period: {net_ret.index[0].date()} to {net_ret.index[-1].date()}
Trading Days: {metrics['n_days']}

GROSS PERFORMANCE:
  Sharpe Ratio: {metrics['gross_sharpe']:.2f}
  Total Return: {metrics['gross_return']*100:.2f}%

NET PERFORMANCE (@ {params['cost_bps']} bps):
  Sharpe Ratio: {metrics['sharpe']:.2f}
  CAGR: {metrics['cagr']:.2f}%
  Ann. Return: {metrics['ann_return']*100:.2f}%
  Ann. Volatility: {metrics['ann_vol']*100:.2f}%
  Max Drawdown: {metrics['max_drawdown']*100:.2f}%

RISK ANALYSIS:
  Avg Turnover: {metrics['avg_turnover']*100:.1f}%
  Cost Drag: {metrics['cost_drag']*100:.2f}%
  ETH Beta: {eth_beta_full:.4f}
  PC1 Beta: {pc1_beta_full:.4f}
"""
    ax9.text(0.1, 0.95, summary_text, transform=ax9.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Phase 1: SPONGE k=3 Baseline Backtest Report', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'phase1_full_report.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save summary to CSV
    summary_df = pd.DataFrame({
        'Metric': [
            'Gross Sharpe', 'Net Sharpe', 'CAGR (%)', 'Ann. Return (%)',
            'Ann. Volatility (%)', 'Max Drawdown (%)', 'Avg Turnover (%)',
            'Cost Drag (%)', 'ETH Beta', 'PC1 Beta', 'Trading Days'
        ],
        'Value': [
            f"{metrics['gross_sharpe']:.2f}", f"{metrics['sharpe']:.2f}",
            f"{metrics['cagr']:.2f}", f"{metrics['ann_return']*100:.2f}",
            f"{metrics['ann_vol']*100:.2f}", f"{metrics['max_drawdown']*100:.2f}",
            f"{metrics['avg_turnover']*100:.1f}", f"{metrics['cost_drag']*100:.2f}",
            f"{eth_beta_full:.4f}", f"{pc1_beta_full:.4f}", f"{metrics['n_days']}"
        ]
    })
    summary_df.to_csv(results_dir / 'phase1_summary.csv', index=False)

    print(f"\nPhase 1 Report generated at {results_dir}")
    print("  - phase1_full_report.png")
    print("  - phase1_summary.csv")
    print("  - diagnostics.csv")

    return {
        'eth_beta': eth_beta_full,
        'pc1_beta': pc1_beta_full,
        'eth_beta_rolling': eth_beta_rolling,
        'pc1_beta_rolling': pc1_beta_rolling,
    }


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent.parent.parent)

    from stat_arb.run_phase1 import run_phase1

    # Same configuration as the run_phase1 default, so every checked-in
    # phase1 artifact reflects one config
    results = run_phase1(H=5, L=60, cost_bps=50, verbose=True)

    if results:
        diagnostics = generate_phase1_report(results)
        print(f"\nETH Beta: {diagnostics['eth_beta']:.4f}")
        print(f"PC1 Beta: {diagnostics['pc1_beta']:.4f}")
