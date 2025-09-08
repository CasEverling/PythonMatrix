from matrix import Matrix, Vector

def test_matrix_operations_pass_fail():
    passed = 0
    failed = 0

    def check(description, condition):
        nonlocal passed, failed
        if condition:
            print(f"PASS: {description}")
            passed += 1
        else:
            print(f"FAIL: {description}")
            failed += 1

    # Matrices
    A = Matrix(2, 2, [1, 2, 3, 4])
    B = Matrix(2, 2, [2, 0, 1, 2])
    C = Matrix(2, 2, [0, 1, -1, 0])

    # Vectors
    p = Vector(2, [5, 6])
    q = Vector(2, [7, 8])
    v1 = Vector(3, [1, 2, 3])
    v2 = Vector(3, [4, 5, 6])

    # Matrix-Matrix multiplication AB
    AB_expected = Matrix(2, 2, [4, 4, 10, 8])
    AB_result = A * B
    check("Matrix AB multiplication", AB_result.content == AB_expected.content)

    # Matrix-Vector multiplication Ap
    Ap_expected = Vector(2, [17, 39])
    Ap_result = A * p
    check("Matrix A * vector p", Ap_result.content == Ap_expected.content)

    # Transpose-Vector multiplication A^T p
    ATp_expected = Vector(2, [23, 34])
    ATp_result = A.T * p
    check("Transpose A^T * vector p", ATp_result.content == ATp_expected.content)

    # Inverse-Vector multiplication A^-1 p
    # Inverse of A = [[-2,1],[1.5,-0.5]]; multiply by p = [5,6]
    Ainvp_expected = Vector(2, [-4, 4.5])
    Ainvp_result = A.inverse * p
    check("Inverse A^-1 * vector p", all(abs(a-b) < 1e-6 for a,b in zip(Ainvp_result.content, Ainvp_expected.content)))

    # Dot product p1^T p2 (1x1 matrix)
    dot_expected = 32  # 1*4 + 2*5 + 3*6 = 32
    dot_result = (v1.T * v2)[0, 0]
    check("Dot product v1^T * v2", abs(dot_result - dot_expected) < 1e-6)

    # Cross product v1 x v2
    cross_expected = Vector(3, [-3, 6, -3])
    cross_result = Vector(3, [
        v1[1,0]*v2[2,0] - v1[2,0]*v2[1,0],
        v1[2,0]*v2[0,0] - v1[0,0]*v2[2,0],
        v1[0,0]*v2[1,0] - v1[1,0]*v2[0,0]
    ])
    check("Cross product v1 x v2", cross_result.content == cross_expected.content)

    # Chained matrix multiplication A*B*C
    ABC_expected = Matrix(2, 2, [4, 4, 10, 8]) * C  # compute expected manually
    ABC_result = A * B * C
    check("Chained matrix multiplication A*B*C", ABC_result.content == ABC_expected.content)

    # Dimension mismatch check
    dimension_error_caught = False
    try:
        _ = A * Vector(3, [1,2,3])
    except Exception:
        dimension_error_caught = True
    check("Dimension mismatch raises exception", dimension_error_caught)

    # ===== EXTRA TESTS =====
    
    # Identity matrix tests
    I = Matrix(3, 3, [1, 0, 0, 0, 1, 0, 0, 0, 1])  # 3x3 Identity matrix
    rotation_matrix = Matrix(3, 3, [0, -1, 0, 1, 0, 0, 0, 0, 1])  # Rotation matrix
    
    # Matrix multiplication with identity
    IR_expected = Matrix(3, 3, [0, -1, 0, 1, 0, 0, 0, 0, 1])
    IR_result = I * rotation_matrix
    check("Identity * Rotation matrix", IR_result.content == IR_expected.content)
    
    # Matrix inverse of identity
    I_inv_expected = Matrix(3, 3, [1, 0, 0, 0, 1, 0, 0, 0, 1])
    I_inv_result = I.inverse
    check("Identity matrix inverse", I_inv_result.content == I_inv_expected.content)
    
    # Matrix transpose of identity
    I_T_expected = Matrix(3, 3, [1, 0, 0, 0, 1, 0, 0, 0, 1])
    I_T_result = I.T
    check("Identity matrix transpose", I_T_result.content == I_T_expected.content)
    
    # Vector cross product test (same as existing but more explicit)
    p1 = Vector(3, [1, 2, 3])
    p2 = Vector(3, [4, 5, 6])
    cross_expected_extra = Vector(3, [-3, 6, -3])  # [2*6-3*5, 3*4-1*6, 1*5-2*4] = [-3, 6, -3]
    cross_result_extra = Vector(3, [
        p1[1,0]*p2[2,0] - p1[2,0]*p2[1,0],
        p1[2,0]*p2[0,0] - p1[0,0]*p2[2,0],
        p1[0,0]*p2[1,0] - p1[1,0]*p2[0,0]
    ])
    check("Vector cross product p1 x p2", cross_result_extra.content == cross_expected_extra.content)
    
    # Vector dot product test (same calculation as before but more explicit)
    dot_expected_extra = 32  # 1*4 + 2*5 + 3*6 = 32
    dot_result_extra = (p1.T * p2)[0, 0]
    check("Vector dot product p1^T * p2", abs(dot_result_extra - dot_expected_extra) < 1e-6)
    
    # Additional test: Rotation matrix properties
    rotation_transpose = rotation_matrix.T
    should_be_identity = rotation_matrix * rotation_transpose
    identity_check = all(abs(should_be_identity[i,j] - (1 if i==j else 0)) < 1e-6 
                        for i in range(3) for j in range(3))
    check("Rotation matrix * Rotation^T = Identity", identity_check)

    # Summary
    print(f"\nTests passed: {passed}")
    print(f"Tests failed: {failed}")
    print(f"Total tests: {passed + failed}")

if __name__ == "__main__":
    test_matrix_operations_pass_fail()
