from question2 import Matrix, Vector

def matrix_multiply(A, B):
    """Compute matrix multiplication AB."""
    return A * B

def matrix_vector_multiply(A, p):
    """Compute matrix-vector multiplication Ap."""
    return A * p

def matrix_transpose_vector_multiply(A, p):
    """Compute A^T * p (transpose-vector multiplication)."""
    return A.T * p

def matrix_inverse_vector_multiply(A, p):
    """Compute A^(-1) * p (inverse-vector multiplication)."""
    return A.inverse * p

def matrix_inverse(A):
    """Compute matrix inverse A^(-1)."""
    return A.inverse

def matrix_transpose(A):
    """Compute matrix transpose A^T."""
    return A.T

def vector_cross(p1, p2):
    """Compute cross product p1 × p2."""
    if p1.height != 3 or p2.height != 3:
        raise Exception("Cross product only defined for 3D vectors")
    
    # Direct formula for cross product
    result = Vector(3)
    result[0, 0] = p1[1, 0] * p2[2, 0] - p1[2, 0] * p2[1, 0]  # i component
    result[1, 0] = p1[2, 0] * p2[0, 0] - p1[0, 0] * p2[2, 0]  # j component
    result[2, 0] = p1[0, 0] * p2[1, 0] - p1[1, 0] * p2[0, 0]  # k component
    return result

def vector_dot(p1, p2):
    """Compute dot product p1^T * p2."""
    if p1.height != p2.height:
        raise Exception(f"Dot product requires same size vectors: p1 has size {p1.height}, p2 has size {p2.height}")
    
    # Use matrix multiplication: p1^T * p2 gives 1x1 matrix, extract scalar
    result_matrix = p1.T * p2
    return result_matrix[0, 0]

def matrix_chain_multiply(*matrices):
    """Compute chain multiplication A1 * A2 * A3 * ... * A7."""
    if len(matrices) == 0:
        raise Exception("At least one matrix required")
    
    result = matrices[0]
    for matrix in matrices[1:]:
        result = result * matrix
    return result

# Example tests
if __name__ == "__main__":
    # Create test matrices and vectors
    A = Matrix(3, 3, [1, 0, 0, 0, 1, 0, 0, 0, 1])  # Identity matrix
    B = Matrix(3, 3, [0, -1, 0, 1, 0, 0, 0, 0, 1])  # Rotation matrix
    C = Matrix(3, 3, [2, 0, 0, 0, 2, 0, 0, 0, 2])   # Scale matrix
    
    p1 = Vector(3, [1, 2, 3])
    p2 = Vector(3, [4, 5, 6])
    
    print("=== Test Matrices ===")
    print("A (Identity):")
    A.display()
    print("\nB (Rotation):")
    B.display()
    print(f"\np1 = {p1.content}")
    print(f"p2 = {p2.content}")
    
    print("\n=== Required Operations ===")
    
    # 1. AB - Matrix multiplication
    print("1. AB (Matrix multiplication):")
    result_AB = matrix_multiply(A, B)
    result_AB.display()
    
    # 2. Ap - Matrix-vector multiplication
    print("\n2. Ap (Matrix-vector multiplication):")
    result_Ap = matrix_vector_multiply(A, p1)
    print(f"   Result: {result_Ap.content}")
    
    # 3. A^T p - Transpose-vector multiplication
    print("\n3. A^T p (Transpose-vector multiplication):")
    result_ATp = matrix_transpose_vector_multiply(A, p1)
    print(f"   Result: {result_ATp.content}")
    
    # 4. A^(-1) p - Inverse-vector multiplication
    print("\n4. A^(-1) p (Inverse-vector multiplication):")
    try:
        result_Ainv_p = matrix_inverse_vector_multiply(A, p1)
        print(f"   Result: {result_Ainv_p.content}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 5. p1 × p2 - Vector cross product
    print("\n5. p1 × p2 (Vector cross product):")
    try:
        result_cross = vector_cross(p1, p2)
        print(f"   Result: {result_cross.content}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 6. p1^T p2 - Vector dot product
    print("\n6. p1^T p2 (Vector dot product):")
    try:
        result_dot = vector_dot(p1, p2)
        print(f"   Result: {result_dot}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 7. A1 A2 A3 ... A7 - Matrix chain multiplication
    print("\n7. A * B * C (Matrix chain multiplication):")
    try:
        result_chain = matrix_chain_multiply(A, B, C)
        result_chain.display()
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n=== Additional Tests ===")
    
    # Test matrix inverse
    print("\n8. Matrix Inverse A^(-1):")
    try:
        A_inv = matrix_inverse(A)
        A_inv.display()
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test matrix transpose
    print("\n9. Matrix Transpose B^T:")
    B_T = matrix_transpose(B)
    B_T.display()
    
    print("\n=== All Tests Complete ===")
