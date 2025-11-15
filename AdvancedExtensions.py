# -*- coding: utf-8 -*-
"""
@Project: LinalgDat2022
@File: AdvancedExtensions.py

@Description: Project C Determinant and Gram-Schmidt extensions.

"""

import math
from sys import path
path.append('../Core')
from Matrix import Matrix
from Vector import Vector


Tolerance = 1e-6


def SquareSubMatrix(A: Matrix, i: int, j: int) -> Matrix:
    """
    This function creates the square submatrix given a square matrix as
    well as row and column indices to remove from it.

    Remarks:
        See page 246-247 in "Linear Algebra for Engineers and Scientists"
        by K. Hardy.

    Parameters:
        A:  N-by-N matrix
        i: int. The index of the row to remove.
        j: int. The index of the column to remove.

    Return:
        The resulting (N - 1)-by-(N - 1) submatrix.
    """

    if A.M_Rows != A.N_Cols:
        raise Exception("Matrix is not Square")
    
    N = A.N_Cols
    Result = Matrix(N - 1, N - 1)

    result_row = 0
    for original_row in range(N):
        if original_row == i:
            continue
        
        result_col = 0

        for original_col in range(N):
            if original_col == j:
                continue
            
            Result[result_row, result_col] = A[original_row, original_col]
            result_col += 1
        result_row += 1

    return Result

def Determinant(A: Matrix) -> float:
    """
    This function computes the determinant of a given square matrix.

    Remarks:
        * See page 247 in "Linear Algebra for Engineers and Scientists"
        by K. Hardy.
        * Hint: Use SquareSubMatrix.

    Parameter:
        A: N-by-N matrix.

    Return:
        The determinant of the matrix.
    """
    n = A.N_Cols
    if A.M_Rows != n:
        raise Exception("Matrix is not Square")

    # Base case for a 1x1 matrix
    if n == 1:
        return A[0, 0]

    determinant = 0
    for j in range(n):
        sub_matrix = SquareSubMatrix(A, 0, j)
        sign = (-1)**j
        determinant += sign * A[0, j] * Determinant(sub_matrix)

    return determinant

    

def VectorNorm(v: Vector) -> float:
    """
    This function computes the Euclidean norm of a Vector. This has been implemented
    in Project A and is provided here for convenience

    Parameter:
         v: Vector

    Return:
         Euclidean norm, i.e. (\sum v[i]^2)^0.5
    """
    nv = 0.0
    for i in range(len(v)):
        nv += v[i]**2
    return math.sqrt(nv)


def SetColumn(A: Matrix, v: Vector, j: int) -> Matrix:
    """
    This function copies Vector 'v' as a column of Matrix 'A'
    at column position j.

    Parameters:
        A: M-by-N Matrix.
        v: size M vector
        j: int. Column number.

    Return:
        Matrix A  after modification.

    Raise:
        ValueError if j is out of range or if len(v) != A.M_Rows.
    """

    if len(v) != A.M_Rows:
        raise Exception("Length of v must be equal to rows of A")

    if j < 0 or j >= A.N_Cols:
        raise Exception("Column index j is out of range")

    for i in range(A.M_Rows):
        A[i,j] = v[i]

    return A


def IsZeroVector(v: Vector) -> bool:
    for i in range(len(v)):
        if (v[i] != 0):
            return False
    return True

   
def GramSchmidt(A: Matrix) -> tuple:
    """
    This function computes the Gram-Schmidt process on a given matrix.

    Remarks:
        See page 229 in "Linear Algebra for Engineers and Scientists"
        by K. Hardy.

    Parameter:
        A: M-by-N matrix. All columns are implicitly assumed linear independent.

    Return:
        tuple (Q, R) where Q is a M-by-N orthonormal matrix and R is an
        N-by-N upper triangular matrix.
    """

    Q = Matrix(A.M_Rows, A.N_Cols)
    R = Matrix(A.N_Cols, A.N_Cols)

    for j in range(A.N_Cols):
        uj = A.Column(j)
        Q = SetColumn(Q, uj, j)

        for i in range(j):
            R[i, j] = Q.Column(i).__matmul__(uj)
            Q = SetColumn(Q, Q.Column(j) - (R[i, j] * Q.Column(i)), j)

        if IsZeroVector(Q.Column(j)):
            continue

        R[j, j] = VectorNorm(Q.Column(j))
        Q = SetColumn(Q, (round(1.0 / R[j, j], 8)) * Q.Column(j), j)

    return (Q, R)
