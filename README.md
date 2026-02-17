# Options Pricing & Volatility Trading Engine

A production-level derivatives pricing and volatility trading system implementing industry-standard models with mathematical rigor and numerical stability.

## 🎯 Overview

This engine provides comprehensive tools for:
- **Options Pricing**: Black-Scholes analytical pricing and Monte Carlo simulation
- **Greeks Calculation**: Delta, Gamma, Vega, Theta, Rho with vectorized implementations
- **Volatility Analysis**: Implied volatility solving, volatility surface generation
- **Trading Strategies**: Delta hedging simulation, volatility arbitrage
- **Risk Management**: Portfolio Greeks aggregation, stress testing

## 📐 Mathematical Foundation

### Black-Scholes Model

The Black-Scholes formula for European options:

**Call Option:**
```
C = S₀N(d₁) - Ke^(-rT)N(d₂)
```

**Put Option:**
```
P = Ke^(-rT)N(-d₂) - S₀N(-d₁)
```

Where:
```
d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

- S₀: Current stock price
- K: Strike price
- r: Risk-free rate
- T: Time to maturity
- σ: Volatility
- N(·): Cumulative standard normal distribution

### Greeks

- **Delta (Δ)**: ∂V/∂S - Rate of change of option value with respect to underlying price
- **Gamma (Γ)**: ∂²V/∂S² - Rate of change of delta with respect to underlying price
- **Vega (ν)**: ∂V/∂σ - Sensitivity to volatility changes
- **Theta (Θ)**: ∂V/∂t - Time decay of option value
- **Rho (ρ)**: ∂V/∂r - Sensitivity to interest rate changes

### Monte Carlo Pricing

Simulates risk-neutral price paths:
```
S(t) = S₀ exp[(r - σ²/2)t + σ√t·Z]
```
where Z ~ N(0,1)

Option price = e^(-rT) · E[max(S_T - K, 0)]

## 🏗️ Architecture

### Modular Design

```
src/
├── models/              # Pricing models
│   ├── black_scholes.py
│   └── monte_carlo.py
├── greeks/              # Greeks calculation
│   └── greeks_calculator.py
├── volatility/          # Volatility analysis
│   ├── implied_vol.py
│   └── vol_surface.py
├── strategies/          # Trading strategies
│   ├── delta_hedging.py
│   └── vol_arbitrage.py
├── simulation/          # Simulation engines
│   └── path_simulator.py
├── risk/               # Risk management
│   └── portfolio_risk.py
└── utils/              # Utilities
    └── helpers.py
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from src.models.black_scholes import BlackScholesModel
from src.greeks.greeks_calculator import GreeksCalculator

# Price an option
bs = BlackScholesModel(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
call_price = bs.call_price()
put_price = bs.put_price()

# Calculate Greeks
greeks = GreeksCalculator(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
delta = greeks.delta('call')
gamma = greeks.gamma()
vega = greeks.vega()
```

### Monte Carlo Pricing

```python
from src.models.monte_carlo import MonteCarloEngine

mc = MonteCarloEngine(S=100, K=100, T=1.0, r=0.05, sigma=0.2, n_simulations=100000)
call_price, ci = mc.price_call()
print(f"Call Price: {call_price:.4f} ± {ci:.4f}")
```

### Implied Volatility

```python
from src.volatility.implied_vol import ImpliedVolatilitySolver

solver = ImpliedVolatilitySolver(S=100, K=100, T=1.0, r=0.05)
implied_vol = solver.solve(market_price=10.45, option_type='call')
```

### Delta Hedging Simulation

```python
from src.strategies.delta_hedging import DeltaHedgingSimulator

simulator = DeltaHedgingSimulator(
    S0=100, K=100, T=1.0, r=0.05, sigma=0.2,
    rebalance_frequency='daily'
)
results = simulator.run_simulation()
simulator.plot_results()
```

## 📊 Features

### 1. Black-Scholes Pricing
- Vectorized implementation for efficient batch pricing
- Analytical formulas for calls and puts
- Numerical stability for edge cases

### 2. Greeks Calculation
- All first and second-order Greeks
- Vectorized computation
- Portfolio-level aggregation

### 3. Monte Carlo Engine
- 100,000+ path simulations
- Variance reduction techniques
- Confidence interval estimation
- Convergence analysis

### 4. Implied Volatility Solver
- Newton-Raphson method with numerical stability
- Convergence diagnostics
- Handles edge cases (deep ITM/OTM)

### 5. Volatility Surface
- Implied volatility smile generation
- 3D surface visualization
- Strike and maturity interpolation

### 6. Delta Hedging
- Dynamic hedging simulation
- Discrete vs continuous rebalancing comparison
- PnL tracking and analysis
- Hedging error quantification

### 7. Volatility Arbitrage
- Mispricing detection
- Long/short volatility strategies
- Backtesting framework
- Performance attribution

### 8. Risk Analysis
- Portfolio Greeks aggregation
- Scenario analysis
- Volatility shock stress testing
- VaR and CVaR calculation

## 📈 Outputs

The engine generates:
- **Hedging PnL curves**: Track profit/loss from delta hedging
- **Greeks evolution charts**: Visualize Greeks over time
- **Volatility surfaces**: 3D visualization of implied volatility
- **Strategy performance reports**: Comprehensive backtesting results
- **Risk reports**: Portfolio risk metrics and stress test results

## 🔬 Numerical Stability

The implementation includes:
- Careful handling of extreme parameter values
- Numerical precision checks
- Overflow/underflow protection
- Convergence monitoring

## ⚠️ Assumptions & Limitations

### Assumptions:
1. **Frictionless markets**: No transaction costs or taxes
2. **Continuous trading**: Can hedge at any time
3. **No dividends**: Unless explicitly modeled
4. **Constant volatility**: For Black-Scholes (relaxed in vol surface)
5. **Log-normal returns**: Underlying follows geometric Brownian motion
6. **European options**: Early exercise not considered

### Limitations:
1. **Model risk**: Black-Scholes assumes constant volatility
2. **Discrete hedging**: Real-world hedging is discrete, not continuous
3. **Liquidity**: Assumes infinite liquidity
4. **Jump risk**: Does not model discontinuous price movements
5. **Volatility smile**: Black-Scholes doesn't explain volatility smile

## 🔮 Future Improvements

1. **Advanced Models**:
   - Heston stochastic volatility model
   - Jump-diffusion models (Merton, Kou)
   - Local volatility models

2. **American Options**:
   - Binomial tree pricing
   - Least-squares Monte Carlo (LSM)

3. **Exotic Options**:
   - Barrier options
   - Asian options
   - Lookback options

4. **Machine Learning**:
   - Neural network pricing
   - Volatility forecasting
   - Feature engineering for option pricing

5. **Real Data Integration**:
   - Live market data feeds
   - Historical data analysis
   - Calibration to market prices

6. **Performance**:
   - GPU acceleration (CuPy)
   - Parallel processing
   - Cython optimization

## 📚 References

1. Black, F., & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities"
2. Hull, J. C. (2018). "Options, Futures, and Other Derivatives"
3. Gatheral, J. (2006). "The Volatility Surface"
4. Glasserman, P. (2003). "Monte Carlo Methods in Financial Engineering"

## 📝 License

MIT License

## 👤 Author

Aadesh - Quantitative Finance Engineer

---

**Note**: This is an educational and research tool. Use at your own risk for actual trading.
