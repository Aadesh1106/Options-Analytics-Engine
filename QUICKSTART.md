# Quick Start Guide

## Installation

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Verify Installation**:
```bash
python main_demo.py
```

## Running Examples

### Example 1: Black-Scholes Pricing
```bash
python examples/example_1_black_scholes.py
```
Demonstrates:
- Call and put option pricing
- Put-call parity verification
- Greeks calculation
- Vectorized pricing across strikes

### Example 2: Monte Carlo Pricing
```bash
python examples/example_2_monte_carlo.py
```
Demonstrates:
- Monte Carlo simulation with 100,000 paths
- Comparison with Black-Scholes
- Convergence analysis
- Variance reduction techniques

### Example 3: Implied Volatility
```bash
python examples/example_3_implied_vol.py
```
Demonstrates:
- Newton-Raphson implied volatility solver
- Convergence diagnostics
- Volatility smile generation
- 3D volatility surface visualization

### Example 4: Delta Hedging
```bash
python examples/example_4_delta_hedging.py
```
Demonstrates:
- Dynamic delta hedging simulation
- PnL tracking
- Hedging error analysis
- Rebalancing frequency comparison

### Example 5: Volatility Arbitrage
```bash
python examples/example_5_vol_arbitrage.py
```
Demonstrates:
- Mispricing detection
- Portfolio construction
- Backtesting framework
- Performance reporting

### Example 6: Portfolio Risk
```bash
python examples/example_6_portfolio_risk.py
```
Demonstrates:
- Portfolio Greeks aggregation
- Price stress testing
- Volatility stress testing
- Risk reporting

## Project Structure

```
OPTIONS PRICING & VOLATILITY TRADING ENGINE/
├── src/
│   ├── models/              # Pricing models
│   │   ├── black_scholes.py # Black-Scholes analytical pricing
│   │   └── monte_carlo.py   # Monte Carlo simulation
│   ├── greeks/              # Greeks calculation
│   │   └── greeks_calculator.py
│   ├── volatility/          # Volatility analysis
│   │   ├── implied_vol.py   # Implied volatility solver
│   │   └── vol_surface.py   # Volatility surface generation
│   ├── strategies/          # Trading strategies
│   │   ├── delta_hedging.py # Delta hedging simulator
│   │   └── vol_arbitrage.py # Volatility arbitrage
│   ├── simulation/          # Simulation engines
│   │   └── path_simulator.py
│   ├── risk/               # Risk management
│   │   └── portfolio_risk.py
│   └── utils/              # Utilities
│       └── helpers.py
├── examples/               # Usage examples
├── outputs/                # Generated outputs
├── main_demo.py           # Comprehensive demo
├── requirements.txt       # Dependencies
└── README.md             # Documentation
```

## Quick Usage

### Price an Option
```python
from src.models.black_scholes import BlackScholesModel

bs = BlackScholesModel(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
call_price = bs.call_price()
put_price = bs.put_price()
```

### Calculate Greeks
```python
from src.greeks.greeks_calculator import GreeksCalculator

greeks = GreeksCalculator(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
delta = greeks.delta('call')
gamma = greeks.gamma()
vega = greeks.vega()
```

### Monte Carlo Pricing
```python
from src.models.monte_carlo import MonteCarloEngine

mc = MonteCarloEngine(S=100, K=100, T=1.0, r=0.05, sigma=0.20, n_simulations=100000)
price, ci = mc.price_call()
```

### Solve for Implied Volatility
```python
from src.volatility.implied_vol import ImpliedVolatilitySolver

solver = ImpliedVolatilitySolver(S=100, K=100, T=1.0, r=0.05)
implied_vol = solver.solve(market_price=10.45, option_type='call')
```

## Features

✅ **Black-Scholes Pricing**: Vectorized analytical pricing  
✅ **Greeks Calculation**: All first and second-order Greeks  
✅ **Monte Carlo Engine**: 100,000+ path simulations  
✅ **Implied Volatility**: Newton-Raphson solver  
✅ **Volatility Surface**: 3D visualization  
✅ **Delta Hedging**: Dynamic hedging simulation  
✅ **Volatility Arbitrage**: Mispricing detection and backtesting  
✅ **Portfolio Risk**: Greeks aggregation and stress testing  

## Output Files

All visualizations are saved to the `outputs/` directory:
- `monte_carlo_convergence.png`
- `volatility_smile.png`
- `volatility_surface_3d.png`
- `volatility_surface_heatmap.png`
- `delta_hedging_simulation.png`
- `vol_arbitrage_backtest.png`
- `portfolio_greeks_profile.png`

## Mathematical Rigor

The implementation includes:
- Numerical stability for extreme parameters
- Overflow/underflow protection
- Convergence monitoring
- Confidence interval estimation
- Variance reduction techniques

## Performance

- Vectorized NumPy operations
- Efficient batch pricing
- Optimized Greek calculations
- Fast Monte Carlo simulation

## Next Steps

1. Run `main_demo.py` to see all features
2. Explore individual examples in `examples/`
3. Review the comprehensive README.md
4. Modify parameters and experiment
5. Build your own strategies!

## Support

For issues or questions, refer to:
- README.md for detailed documentation
- Example scripts for usage patterns
- Inline code documentation
