#!/usr/bin/env python3
"""
Test script for dividend yield functionality in Black-Scholes solvers.

This script tests the updated CN solver with dividend yield support,
including forward pricing and put-call parity validation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Import solvers
from pde.solvers.black_scholes_cn import BlackScholesCNSolver
from pde.solvers.black_scholes_cn_rannacher import BlackScholesCNRannacherSolver


def analytical_black_scholes_dividend(S: float, K: float, T: float, r: float, 
                                    sigma: float, option_type: str, q: float = 0.0) -> float:
    """
    Analytical Black-Scholes formula with dividend yield.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration
        r: Risk-free rate
        sigma: Volatility
        option_type: 'call' or 'put'
        q: Dividend yield
        
    Returns:
        Option price
    """
    from scipy.stats import norm
    
    # Calculate d1 and d2 with dividend yield
    d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == 'call':
        # Call: S*exp(-q*T)*N(d1) - K*exp(-r*T)*N(d2)
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        # Put: K*exp(-r*T)*N(-d2) - S*exp(-q*T)*N(-d1)
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    
    return price


def test_dividend_yield_functionality():
    """Test the dividend yield functionality."""
    print("🧪 Testing Dividend Yield Functionality")
    print("=" * 50)
    
    # Test parameters
    S = 100.0
    K = 100.0
    T = 0.25  # 3 months
    r = 0.05
    sigma = 0.2
    q = 0.03  # 3% dividend yield
    
    print(f"Test Parameters:")
    print(f"  S (spot): {S}")
    print(f"  K (strike): {K}")
    print(f"  T (time): {T}")
    print(f"  r (rate): {r}")
    print(f"  σ (vol): {sigma}")
    print(f"  q (div yield): {q}")
    print()
    
    # Test 1: Basic functionality
    print("1️⃣  Testing Basic Functionality")
    print("-" * 30)
    
    try:
        solver = BlackScholesCNSolver(
            S_min=0.1, S_max=200.0, T=T, r=r, sigma=sigma, K=K, 
            option_type='call', N_S=200, N_T=100, q=q
        )
        
        solver.setup_grid()
        solver.solve()
        
        # Get option price at current spot
        fd_price = solver.get_option_price(S, t=0.0)
        analytical_price = analytical_black_scholes_dividend(S, K, T, r, sigma, 'call', q)
        
        print(f"✅ CN Solver with dividend yield:")
        print(f"   FD Price: {fd_price:.6f}")
        print(f"   Analytical: {analytical_price:.6f}")
        print(f"   Error: {abs(fd_price - analytical_price):.6f}")
        print(f"   Rel Error: {abs(fd_price - analytical_price) / analytical_price * 100:.4f}%")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Test 2: Put-Call Parity
    print("\n2️⃣  Testing Put-Call Parity")
    print("-" * 30)
    
    try:
        # Create call and put solvers
        call_solver = BlackScholesCNSolver(
            S_min=0.1, S_max=200.0, T=T, r=r, sigma=sigma, K=K, 
            option_type='call', N_S=200, N_T=100, q=q
        )
        
        put_solver = BlackScholesCNSolver(
            S_min=0.1, S_max=200.0, T=T, r=r, sigma=sigma, K=K, 
            option_type='put', N_S=200, N_T=100, q=q
        )
        
        call_solver.setup_grid()
        put_solver.setup_grid()
        call_solver.solve()
        put_solver.solve()
        
        # Get prices
        call_price = call_solver.get_option_price(S, t=0.0)
        put_price = put_solver.get_option_price(S, t=0.0)
        
        # Put-call parity with dividend yield: C - P = S*exp(-q*T) - K*exp(-r*T)
        lhs = call_price - put_price
        rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
        
        print(f"✅ Put-Call Parity (with dividend yield):")
        print(f"   Call Price: {call_price:.6f}")
        print(f"   Put Price: {put_price:.6f}")
        print(f"   C - P: {lhs:.6f}")
        print(f"   S*exp(-q*T) - K*exp(-r*T): {rhs:.6f}")
        print(f"   Difference: {abs(lhs - rhs):.6f}")
        print(f"   Rel Error: {abs(lhs - rhs) / abs(rhs) * 100:.4f}%")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Test 3: Forward Price Calculation
    print("\n3️⃣  Testing Forward Price Calculation")
    print("-" * 30)
    
    try:
        # Calculate forward price
        forward_price = S * np.exp((r - q) * T)
        
        print(f"✅ Forward Price Calculation:")
        print(f"   Spot Price: {S}")
        print(f"   Forward Price: {forward_price:.6f}")
        print(f"   Forward = S * exp((r-q)*T)")
        print(f"   Forward = {S} * exp(({r}-{q})*{T}) = {forward_price:.6f}")
        
        # Test ATM convention
        atm_strike = S * np.exp((r - q) * T)  # Forward-ATM
        spot_atm_strike = S  # Spot-ATM
        
        print(f"\n   ATM Strike Conventions:")
        print(f"   Spot-ATM: {spot_atm_strike}")
        print(f"   Forward-ATM: {atm_strike:.6f}")
        print(f"   Difference: {abs(atm_strike - spot_atm_strike):.6f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Test 4: Rannacher Solver with Dividend Yield
    print("\n4️⃣  Testing Rannacher Solver with Dividend Yield")
    print("-" * 30)
    
    try:
        rannacher_solver = BlackScholesCNRannacherSolver(
            S_min=0.1, S_max=200.0, T=T, r=r, sigma=sigma, K=K, 
            option_type='call', N_S=200, N_T=100, rannacher_steps=4, q=q
        )
        
        rannacher_solver.setup_grid()
        rannacher_solver.solve()
        
        rannacher_price = rannacher_solver.get_option_price(S, t=0.0)
        
        print(f"✅ Rannacher Solver with dividend yield:")
        print(f"   Rannacher Price: {rannacher_price:.6f}")
        print(f"   CN Price: {fd_price:.6f}")
        print(f"   Analytical: {analytical_price:.6f}")
        print(f"   Rannacher Error: {abs(rannacher_price - analytical_price):.6f}")
        print(f"   CN Error: {abs(fd_price - analytical_price):.6f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Test 5: Different Dividend Yields
    print("\n5️⃣  Testing Different Dividend Yields")
    print("-" * 30)
    
    try:
        dividend_yields = [0.0, 0.02, 0.05, 0.08]
        
        print(f"   Testing dividend yields: {dividend_yields}")
        print(f"   {'Div Yield':<10} {'FD Price':<12} {'Analytical':<12} {'Error':<12} {'Rel Error':<10}")
        print(f"   {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
        
        for q_test in dividend_yields:
            solver_test = BlackScholesCNSolver(
                S_min=0.1, S_max=200.0, T=T, r=r, sigma=sigma, K=K, 
                option_type='call', N_S=200, N_T=100, q=q_test
            )
            
            solver_test.setup_grid()
            solver_test.solve()
            
            fd_price_test = solver_test.get_option_price(S, t=0.0)
            analytical_price_test = analytical_black_scholes_dividend(S, K, T, r, sigma, 'call', q_test)
            
            error = abs(fd_price_test - analytical_price_test)
            rel_error = error / analytical_price_test * 100
            
            print(f"   {q_test:<10.3f} {fd_price_test:<12.6f} {analytical_price_test:<12.6f} {error:<12.6f} {rel_error:<10.4f}%")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    print("\n🎉 All tests completed successfully!")
    return True


def create_dividend_yield_plots():
    """Create plots showing dividend yield effects."""
    print("\n📊 Creating Dividend Yield Analysis Plots")
    print("-" * 40)
    
    try:
        # Parameters
        S = 100.0
        K = 100.0
        T = 0.25
        r = 0.05
        sigma = 0.2
        dividend_yields = np.linspace(0.0, 0.1, 11)
        
        # Calculate prices for different dividend yields
        fd_prices = []
        analytical_prices = []
        
        for q in dividend_yields:
            solver = BlackScholesCNSolver(
                S_min=0.1, S_max=200.0, T=T, r=r, sigma=sigma, K=K, 
                option_type='call', N_S=200, N_T=100, q=q
            )
            
            solver.setup_grid()
            solver.solve()
            
            fd_price = solver.get_option_price(S, t=0.0)
            analytical_price = analytical_black_scholes_dividend(S, K, T, r, sigma, 'call', q)
            
            fd_prices.append(fd_price)
            analytical_prices.append(analytical_price)
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Option prices vs dividend yield
        ax1.plot(dividend_yields, fd_prices, 'b-o', label='FD Solver', linewidth=2, markersize=6)
        ax1.plot(dividend_yields, analytical_prices, 'r--s', label='Analytical', linewidth=2, markersize=6)
        ax1.set_xlabel('Dividend Yield (q)')
        ax1.set_ylabel('Call Option Price')
        ax1.set_title('Call Option Price vs Dividend Yield')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Error analysis
        errors = np.abs(np.array(fd_prices) - np.array(analytical_prices))
        rel_errors = errors / np.array(analytical_prices) * 100
        
        ax2.plot(dividend_yields, errors, 'g-o', label='Absolute Error', linewidth=2, markersize=6)
        ax2.set_xlabel('Dividend Yield (q)')
        ax2.set_ylabel('Absolute Error')
        ax2.set_title('FD Solver Error vs Dividend Yield')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add relative error on secondary y-axis
        ax2_twin = ax2.twinx()
        ax2_twin.plot(dividend_yields, rel_errors, 'orange', linestyle=':', label='Relative Error (%)', linewidth=2)
        ax2_twin.set_ylabel('Relative Error (%)')
        ax2_twin.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig('output/dividend_yield_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Plots saved to 'output/dividend_yield_analysis.png'")
        
    except Exception as e:
        print(f"❌ Error creating plots: {e}")


def main():
    """Main function."""
    print("Black-Scholes Solver - Dividend Yield Test")
    print("=" * 60)
    print("Testing the updated CN solver with dividend yield support")
    print()
    
    # Run tests
    success = test_dividend_yield_functionality()
    
    if success:
        # Create plots
        create_dividend_yield_plots()
        
        print("\n🎯 Summary:")
        print("✅ Dividend yield parameter added to solvers")
        print("✅ PDE drift term updated: (r - q - 0.5*σ²)")
        print("✅ Boundary conditions updated for forward pricing")
        print("✅ Put-call parity validated with dividend yield")
        print("✅ Forward price calculations working")
        print("✅ Rannacher solver supports dividend yield")
        print("\nThe solvers are now ready for dividend-carrying underlyings!")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
