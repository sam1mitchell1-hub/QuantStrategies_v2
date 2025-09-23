"""
Matrix operation utilities for PDE solvers.
"""

import numpy as np
from typing import Tuple


def thomas_algorithm(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Solve a tridiagonal system using the Thomas algorithm.
    
    Solves the system:
    b[0] * x[0] + c[0] * x[1] = d[0]
    a[i] * x[i-1] + b[i] * x[i] + c[i] * x[i+1] = d[i]  for i = 1, ..., n-2
    a[n-1] * x[n-2] + b[n-1] * x[n-1] = d[n-1]
    
    Args:
        a: Lower diagonal (length n-1, a[0] is not used)
        b: Main diagonal (length n)
        c: Upper diagonal (length n-1, c[n-1] is not used)
        d: Right-hand side vector (length n)
        
    Returns:
        Solution vector x (length n)
    """
    n = len(d)
    if len(a) != n - 1 or len(b) != n or len(c) != n - 1:
        raise ValueError("Array lengths must be consistent")
    
    # Make copies to avoid modifying input arrays
    a = a.copy()
    b = b.copy()
    c = c.copy()
    d = d.copy()
    
    # Forward elimination
    for i in range(1, n):
        if abs(b[i-1]) < 1e-15:
            raise ValueError("Singular matrix: zero pivot encountered")
        
        factor = a[i-1] / b[i-1]
        b[i] -= factor * c[i-1]
        d[i] -= factor * d[i-1]
    
    # Back substitution
    x = np.zeros(n)
    x[n-1] = d[n-1] / b[n-1]
    
    for i in range(n-2, -1, -1):
        x[i] = (d[i] - c[i] * x[i+1]) / b[i]
    
    return x


def create_tridiagonal_matrix(a: np.ndarray, b: np.ndarray, c: np.ndarray, n: int) -> np.ndarray:
    """
    Create a tridiagonal matrix from its diagonals.
    
    Args:
        a: Lower diagonal (length n-1)
        b: Main diagonal (length n)
        c: Upper diagonal (length n-1)
        n: Matrix size
        
    Returns:
        Tridiagonal matrix (n x n)
    """
    matrix = np.zeros((n, n))
    
    # Main diagonal
    np.fill_diagonal(matrix, b)
    
    # Upper diagonal
    for i in range(n-1):
        matrix[i, i+1] = c[i]
    
    # Lower diagonal
    for i in range(1, n):
        matrix[i, i-1] = a[i-1]
    
    return matrix


def validate_tridiagonal_system(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    """
    Validate that a tridiagonal system is well-posed.
    
    Args:
        a: Lower diagonal
        b: Main diagonal
        c: Upper diagonal
        d: Right-hand side
        
    Returns:
        True if system is valid, False otherwise
    """
    n = len(d)
    
    # Check array lengths
    if len(a) != n - 1 or len(b) != n or len(c) != n - 1:
        return False
    
    # Check for zero pivots
    for i in range(n):
        if abs(b[i]) < 1e-15:
            return False
    
    return True
