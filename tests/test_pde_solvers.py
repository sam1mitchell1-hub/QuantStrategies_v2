"""
Unit tests for PDE solvers.

This module contains unit tests for the finite difference PDE solvers,
designed to run in CI without matplotlib dependencies.
"""

import pytest
import numpy as np
from pde import BlackScholesCNSolver
from pde.utils.matrix_utils import thomas_algorithm


class TestBlackScholesCNSolver:
    """Test cases for Black-Scholes Crank-Nicolson solver."""
    
    def test_initialization(self):
        """Test solver initialization with default parameters."""
        solver = BlackScholesCNSolver()
        
        assert solver.K == 100.0
        assert solver.T == 1.0
        assert solver.r == 0.05
        assert solver.sigma == 0.2
        assert solver.option_type == 'call'
        assert solver.N_S == 100
        assert solver.N_T == 100
        assert solver.S_max == 400.0  # 4 * K
        
    def test_initialization_with_parameters(self):
        """Test solver initialization with custom parameters."""
        solver = BlackScholesCNSolver(
            S_min=1.0,
            S_max=200.0,
            T=0.5,
            r=0.03,
            sigma=0.25,
            K=50.0,
            option_type='put',
            N_S=50,
            N_T=50
        )
        
        assert solver.S_min == 1.0
        assert solver.S_max == 200.0
        assert solver.T == 0.5
        assert solver.r == 0.03
        assert solver.sigma == 0.25
        assert solver.K == 50.0
        assert solver.option_type == 'put'
        assert solver.N_S == 50
        assert solver.N_T == 50
        
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test negative S_min
        with pytest.raises(ValueError, match="S_min must be non-negative"):
            BlackScholesCNSolver(S_min=-1.0)
            
        # Test S_max <= S_min
        with pytest.raises(ValueError, match="S_max must be greater than S_min"):
            BlackScholesCNSolver(S_min=100.0, S_max=50.0)
            
        # Test negative T
        with pytest.raises(ValueError, match="T must be positive"):
            BlackScholesCNSolver(T=-1.0)
            
        # Test negative sigma
        with pytest.raises(ValueError, match="sigma must be positive"):
            BlackScholesCNSolver(sigma=-0.1)
            
        # Test negative K (this will trigger S_max validation first)
        with pytest.raises(ValueError, match="S_max must be greater than S_min"):
            BlackScholesCNSolver(K=-50.0)
            
        # Test invalid option type
        with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
            BlackScholesCNSolver(option_type='invalid')
            
        # Test insufficient grid points
        with pytest.raises(ValueError, match="N_S must be at least 2"):
            BlackScholesCNSolver(N_S=1)
            
        with pytest.raises(ValueError, match="N_T must be at least 1"):
            BlackScholesCNSolver(N_T=0)
    
    def test_setup_grid(self):
        """Test grid setup."""
        solver = BlackScholesCNSolver(N_S=50, N_T=25)
        solver.setup_grid()
        
        assert solver.S_grid is not None
        assert solver.t_grid is not None
        assert solver.x_grid is not None
        assert len(solver.S_grid) == 50
        assert len(solver.t_grid) == 26  # N_T + 1
        assert len(solver.x_grid) == 50
        assert solver.solution.shape == (26, 50)
        
    def test_payoff_function(self):
        """Test payoff function calculation."""
        solver = BlackScholesCNSolver(K=100.0, option_type='call')
        
        # Test call option payoff
        S_test = np.array([80, 100, 120])
        payoff = solver.get_payoff(S_test)
        expected = np.array([0, 0, 20])  # max(S-K, 0)
        np.testing.assert_array_equal(payoff, expected)
        
        # Test put option payoff
        solver.option_type = 'put'
        payoff = solver.get_payoff(S_test)
        expected = np.array([20, 0, 0])  # max(K-S, 0)
        np.testing.assert_array_equal(payoff, expected)
        
    def test_boundary_conditions(self):
        """Test boundary condition calculations."""
        solver = BlackScholesCNSolver(K=100.0, T=1.0, r=0.05, option_type='call')
        
        # Test call option boundary at S=0
        bc_zero = solver.get_boundary_condition_at_zero(0.5)
        assert bc_zero == 0.0
        
        # Test call option boundary at S→∞
        bc_inf = solver.get_boundary_condition_at_infinity(0.5)
        expected = solver.S_max - solver.K * np.exp(-solver.r * (solver.T - 0.5))
        assert abs(bc_inf - expected) < 1e-10
        
        # Test put option boundary conditions
        solver.option_type = 'put'
        bc_zero = solver.get_boundary_condition_at_zero(0.5)
        expected = solver.K * np.exp(-solver.r * (solver.T - 0.5))
        assert abs(bc_zero - expected) < 1e-10
        
        bc_inf = solver.get_boundary_condition_at_infinity(0.5)
        assert bc_inf == 0.0
        
    def test_solve_basic(self):
        """Test basic solving functionality."""
        solver = BlackScholesCNSolver(N_S=20, N_T=10)
        solver.setup_grid()
        solution = solver.solve()
        
        assert solver.solved
        assert solution.shape == (11, 20)  # (N_T+1, N_S)
        assert np.all(np.isfinite(solution))
        
    def test_solution_properties(self):
        """Test basic solution properties."""
        solver = BlackScholesCNSolver(K=100.0, option_type='call', N_S=30, N_T=15)
        solver.setup_grid()
        solution = solver.solve()
        
        # Solution should be non-negative for call option
        assert np.all(solution >= 0)
        
        # Solution at expiration should equal payoff
        payoff = solver.get_payoff(solver.S_grid)
        np.testing.assert_array_almost_equal(solution[-1, :], payoff, decimal=10)
        
        # Solution should be monotonic in S for call option
        for t_idx in range(solution.shape[0]):
            S_values = solution[t_idx, :]
            # Check that solution is non-decreasing in S for call option
            assert np.all(np.diff(S_values) >= -1e-10)  # Allow small numerical errors
        
    def test_option_pricing(self):
        """Test option price calculation."""
        solver = BlackScholesCNSolver(K=100.0, option_type='call', N_S=50, N_T=25)
        solver.setup_grid()
        solver.solve()
        
        # Test pricing at various points
        S_test = [80, 100, 120]
        for S in S_test:
            price = solver.get_option_price(S, t=0.0)
            assert price >= 0
            assert np.isfinite(price)
            
    def test_greeks_calculation(self):
        """Test Greeks calculation."""
        solver = BlackScholesCNSolver(K=100.0, option_type='call', N_S=50, N_T=25)
        solver.setup_grid()
        solver.solve()
        
        greeks = solver.get_greeks(S=100.0, t=0.0)
        
        assert 'delta' in greeks
        assert 'gamma' in greeks
        assert 'theta' in greeks
        
        # Basic sanity checks
        assert np.isfinite(greeks['delta'])
        assert np.isfinite(greeks['gamma'])
        assert np.isfinite(greeks['theta'])
        
        # For call option, delta should be between 0 and 1
        assert 0 <= greeks['delta'] <= 1
        
        # Gamma should be non-negative
        assert greeks['gamma'] >= 0
        
    def test_solution_surface(self):
        """Test solution surface retrieval."""
        solver = BlackScholesCNSolver(N_S=20, N_T=10)
        solver.setup_grid()
        solver.solve()
        
        surface = solver.get_solution_surface()
        
        assert 'S_grid' in surface
        assert 't_grid' in surface
        assert 'solution' in surface
        
        assert len(surface['S_grid']) == 20
        assert len(surface['t_grid']) == 11
        assert surface['solution'].shape == (11, 20)
        
    def test_put_call_parity(self):
        """Test put-call parity relationship."""
        # Create call and put solvers with same parameters
        call_solver = BlackScholesCNSolver(
            K=100.0, T=1.0, r=0.05, sigma=0.2, option_type='call', N_S=50, N_T=25
        )
        put_solver = BlackScholesCNSolver(
            K=100.0, T=1.0, r=0.05, sigma=0.2, option_type='put', N_S=50, N_T=25
        )
        
        call_solver.setup_grid()
        put_solver.setup_grid()
        call_solver.solve()
        put_solver.solve()
        
        # Test put-call parity: C - P = S - K*exp(-r*T)
        S_test = 100.0
        call_price = call_solver.get_option_price(S_test, t=0.0)
        put_price = put_solver.get_option_price(S_test, t=0.0)
        
        lhs = call_price - put_price
        rhs = S_test - call_solver.K * np.exp(-call_solver.r * call_solver.T)
        
        # Should be approximately equal (within numerical precision)
        assert abs(lhs - rhs) < 0.1  # Allow for finite difference errors


class TestThomasAlgorithm:
    """Test cases for Thomas algorithm implementation."""
    
    def test_simple_tridiagonal(self):
        """Test Thomas algorithm with simple tridiagonal system."""
        # Simple 3x3 system: x = [1.25, 2.5, 3.0]
        # System: [2 1 0] [x1]   [5]
        #         [1 2 1] [x2] = [8]
        #         [0 1 2] [x3]   [7]
        a = np.array([0, 1])      # Lower diagonal
        b = np.array([2, 2, 2])   # Main diagonal
        c = np.array([1, 1])      # Upper diagonal
        d = np.array([5, 8, 7])   # RHS
        
        x = thomas_algorithm(a, b, c, d)
        expected = np.array([1.25, 2.5, 3.0])
        
        np.testing.assert_array_almost_equal(x, expected)
        
    def test_identity_matrix(self):
        """Test with identity matrix."""
        n = 5
        a = np.zeros(n-1)
        b = np.ones(n)
        c = np.zeros(n-1)
        d = np.array([1, 2, 3, 4, 5])
        
        x = thomas_algorithm(a, b, c, d)
        expected = d
        
        np.testing.assert_array_almost_equal(x, expected)
        
    def test_singular_matrix(self):
        """Test error handling for singular matrix."""
        a = np.array([0, 0])
        b = np.array([0, 1, 1])  # Zero pivot
        c = np.array([1, 0])
        d = np.array([1, 2, 3])
        
        with pytest.raises(ValueError, match="Singular matrix"):
            thomas_algorithm(a, b, c, d)
            
    def test_array_length_validation(self):
        """Test validation of array lengths."""
        a = np.array([1, 2])
        b = np.array([1, 2, 3])
        c = np.array([1, 2])
        d = np.array([1, 2, 3, 4])  # Wrong length
        
        with pytest.raises(ValueError, match="Array lengths must be consistent"):
            thomas_algorithm(a, b, c, d)


if __name__ == "__main__":
    pytest.main([__file__])
