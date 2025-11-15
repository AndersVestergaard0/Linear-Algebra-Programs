# -*- coding: utf-8 -*-
"""
@Project: LinalgDat2022
@File: GaussExtensions.py

@Description: Project B Gauss extensions

"""

import math

from sys import path
path.append('../Core')
from Vector import Vector
from Matrix import Matrix


def AugmentRight(A: Matrix, v: Vector) -> Matrix:
    """
    Forms the augmented matrix (A|v) by placing vector v to the right of matrix A.

    Parameters:
        A (Matrix): A matrix with M rows.
        v (Vector): A vector of length M.

    Returns:
        Matrix: The augmented matrix (A|v).
    """
    M = A.M_Rows
    N = A.N_Cols
    if v.size() != M:
        raise ValueError("number of rows of A and length of v differ.")

    B = Matrix(M, N + 1)
    for i in range(M):
        for j in range(N):
            B[i, j] = A[i, j]
        B[i, N] = v[i]
    return B


def ElementaryRowReplacement(A: Matrix, i: int, m: float, j: int) -> Matrix:
    """
    Performs an elementary row operation: Row_i <- Row_i + m * Row_j.

    Parameters:
        A (Matrix): The matrix to modify.
        i (int): The target row index.
        m (float): The multiplier for row j.
        j (int): The source row index.

    Returns:
        Matrix: The modified matrix.
    """
    N = A.N_Cols
    for col in range(N):
        A[i, col] += A[j, col] * m
    return A


def ElementaryRowInterchange(A: Matrix, i: int, j: int) -> Matrix:
    """
    Swaps two rows in a matrix.

    Parameters:
        A (Matrix): The matrix to modify.
        i (int): The first row index.
        j (int): The second row index.

    Returns:
        Matrix: The modified matrix.
    """
    N = A.N_Cols
    for col in range(N):
        A[i, col], A[j, col] = A[j, col], A[i, col]
    return A


def ElementaryRowScaling(A: Matrix, i: int, c: float) -> Matrix:
    """
    Scales a row in a matrix by a constant factor.

    Parameters:
        A (Matrix): The matrix to modify.
        i (int): The row index.
        c (float): The scaling factor.

    Returns:
        Matrix: The modified matrix.
    """
    N = A.N_Cols
    for col in range(N):
        A[i, col] *= c
    return A


def ForwardReduction(A: Matrix) -> Matrix:
    """
    Performs forward reduction (Gaussian elimination without scaling) to convert
    a matrix into row-echelon form.

    Parameters:
        A (Matrix): The matrix to reduce.

    Returns:
        Matrix: The row-echelon form of the matrix.
    """
    M = A.M_Rows
    N = A.N_Cols

    pivot_row = 0

    for col in range(N):
        pivot_found = False
        for row in range(pivot_row, M):
            if A[row, col] != 0:
                ElementaryRowInterchange(A, row, pivot_row)
                pivot_found = True
                break
        if pivot_found != True:
            continue

        for row in range(pivot_row + 1, M):
            factor = -A[row, col] / A[pivot_row, col]
            ElementaryRowReplacement(A, row, factor, pivot_row)

        pivot_row += 1

        if pivot_row >= M:
            break

    return A


def BackwardReduction(A: Matrix) -> Matrix:
    """
    Performs backward reduction to convert a matrix in row-echelon form into reduced
    row-echelon form.

    Parameters:
        A (Matrix): The matrix to reduce.

    Returns:
        Matrix: The reduced row-echelon form of the matrix.
    """
    M = A.M_Rows
    N = A.N_Cols

    for pivot_row in range(M - 1, -1, -1):
        # Find pivot column (first non-zero entry)
        pivot_col = -1
        for col in range(N):
            if (A[pivot_row, col] * A[pivot_row, col]) >= 1e-16:
                pivot_col = col
                break

        if pivot_col == -1:
            continue  # Entire row is zero, skip

        # Scale pivot row to make pivot = 1
        factor = 1.0 / A[pivot_row, pivot_col]
        ElementaryRowScaling(A, pivot_row, factor)

        # Eliminate entries above pivot
        for row in range(pivot_row - 1, -1, -1):
            factor = -A[row, pivot_col]
            ElementaryRowReplacement(A, row, factor, pivot_row)

    return A


def GaussElimination(A: Matrix, v: Vector) -> Vector:
    """
    Solves a system of linear equations Ax = v using Gaussian elimination.

    Parameters:
        A (Matrix): A square coefficient matrix.
        v (Vector): The right-hand side vector.

    Returns:
        Vector: The solution vector x.
    """
    M = A.M_Rows
    N = A.N_Cols

    if M != N:
        raise ValueError("Matrix A must be square for Gauss elimination.")

    if v.size() != M:
        raise ValueError("Length of vector v must match number of rows in A.")

    # Step 1: Form augmented matrix (A|v)
    Aug = AugmentRight(A, v)

    # Step 2: Forward reduction to row-echelon form
    ForwardReduction(Aug)

    # Step 3: Backward reduction to reduced row-echelon form
    BackwardReduction(Aug)

    # Step 4: Extract solution from last column
    x = Vector(M)
    for i in range(M):
        x[i] = Aug[i, N]

    return x
