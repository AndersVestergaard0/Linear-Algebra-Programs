module ProjectA

open System
open LinAlgDat.Core

type BasicOps = class
  /// <summary>
  /// This function creates an augmented Matrix given a Matrix A, and a
  /// right-hand side Vector v.
  /// </summary>
  ///
  /// <remarks>
  /// See page 12 in "Linear Algebra for Engineers and Scientists"
  /// by K. Hardy.
  /// This implementation is provided for you.
  /// </remarks>
  ///
  /// <param name="A">An M-by-N Matrix.</param>
  /// <param name="v">An M-size Vector.</param>
  ///
  /// <returns>An M-by-(N+1) augmented Matrix [A | v].</returns>
  static member AugmentRight (A : Matrix) (v : Vector) : Matrix =
    let m_rows = A.M_Rows
    let n_cols = A.N_Cols

    let retval = Array2D.zeroCreate m_rows (n_cols + 1)

    for i in 0..m_rows-1 do
        for j in 0..n_cols-1 do
            retval.[i, j] <- A.[i, j]
        retval.[i, n_cols] <- v.[i]
    Matrix retval

  /// <summary>
  /// This function computes the Matrix-Vector product of a Matrix A,
  /// and a column Vector v.
  /// </summary>
  ///
  /// <remarks>
  /// See page 68 in "Linear Algebra for Engineers and Scientists"
  /// by K. Hardy.
  /// </remarks>
  ///
  /// <param name="A">An M-by-N Matrix.</param>
  /// <param name="v">An N-size Vector.</param>
  ///
  /// <returns>An M-size Vector b such that b = A * v.</returns>
  static member MatVecProduct (A : Matrix) (v : Vector) : Vector =
    let m_rows = A.M_Rows
    let n_cols = A.N_Cols

    let mutable b =  Vector(m_rows)
    if A.N_Cols <> v.Size then

     let errorMsg = 
      sprintf "Matrix column count (%d) must match Vector size (%d) for MatVecProduct." A.N_Cols v.Size
     raise (ArgumentException(errorMsg, "v"))

    for i in 0..m_rows-1 do
      let mutable sum = 0.0
      for j in 0..n_cols-1 do
        sum <- sum + (A.[i,j] * v.[j])
      b.[i] <- sum

    b

  /// <summary>
  /// This function computes the Matrix product of two given matrices
  /// A and B.
  /// </summary>
  ///
  /// <remarks>
  /// See page 58 in "Linear Algebra for Engineers and Scientists"
  /// by K. Hardy.
  /// </remarls>
  ///
  /// <param name="A">An M-by-N Matrix.</param>
  /// <param name="B">An N-by-P Matrix.</param>
  ///
  /// <returns>The M-by-P Matrix A * B.</returns>
  
  static member MatrixProduct (A : Matrix) (B : Matrix) : Matrix =
   if A.N_Cols <> B.M_Rows then
     let errorMsg =
       sprintf "Matrix A columns (%d) must match Matrix B rows (%d) for MatrixProduct." A.N_Cols B.M_Rows
     raise (ArgumentException(errorMsg, "B"))

   let m_rows_A = A.M_Rows 
   let n_inner_dim = A.N_Cols 
   let p_cols_B = B.N_Cols 

   let mutable C = Matrix(m_rows_A, p_cols_B)

   for i in 0 .. m_rows_A - 1 do
     for j in 0 .. p_cols_B - 1 do
       let mutable sum = 0.0
       for k in 0 .. n_inner_dim - 1 do
         sum <- sum + (A.[i, k] * B.[k, j])

       C.[i, j] <- sum

   C
  
  /// <summary>
  /// This function computes the transpose of a given Matrix.
  /// </summary>
  ///
  /// <remarks>
  /// See page 69 in "Linear Algebra for Engineers and Scientists"
  /// by K. Hardy.
  /// </remarks>
  ///
  /// <param name="A">An M-by-N Matrix.</param>
  ///
  /// <returns>The N-by-M Matrix B such that B = A^T.</returns>
  static member Transpose (A : Matrix) : Matrix =
    let m_rows = A.M_Rows
    let n_cols = A.N_Cols

    let mutable B =  Matrix(A.N_Cols, A.M_Rows)

    for i in 0..m_rows-1 do
      for j in 0..n_cols-1 do
        B.[j,i] <- A.[i,j]

    B

  /// <summary>
  /// This function computes the Euclidean Vector norm of a given
  /// Vector.
  /// </summary>
  ///
  /// <remarks>
  /// See page 197 in "Linear Algebra for Engineers and Scientists"
  /// by K. Hardy.
  /// </remarks>
  ///
  /// <param name="v">An N-dimensional Vector.</param>
  ///
  /// <returns>The Euclidean norm of the Vector.</returns>
  static member VectorNorm (v : Vector) : float =
    let mutable sum = 0.0
    
    for i in 0 .. v.Size - 1 do
      sum <- sum + (v.[i] * v.[i])

    sqrt sum


end
