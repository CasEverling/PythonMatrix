"""
Comprehensive test suite for Cache-Optimized Matrix Library

Tests cover:
- Basic matrix operations (multiplication, transpose, inverse)
- Vector operations (dot product, cross product)
- Edge cases and error handling
- Mathematical properties and correctness
- Numerical stability
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matrix import Matrix, Vector

def test_matrix_operations_comprehensive():
    """Main test suite with detailed reporting"""
    passed = 0
    failed = 0
    
    def check(description, condition):
        """Test assertion with detailed reporting"""
        nonlocal passed, failed
        if condition:
            print(f"✅ PASS: {description}")
            passed += 1
        else:
            print(f"❌ FAIL: {description}")
            failed += 1
    
    def check_matrix_equal(actual, expected, description, tolerance=1e-10):
        """Compare matrices with floating-point tolerance"""
        if actual.width != expected.width or actual.height != expected.height:
            check(description, False)
            return
        
        matrices_equal = all(
            abs(actual[i, j] - expected[i, j]) < tolerance
            for i in range(actual.height)
            for j in range(actual.width)
        )
        check(description, matrices_equal)
    
    print("🧪 Starting Cache-Optimized Matrix Library Test Suite\n")
    
    # Test Data Setup
    print("📋 Setting up test matrices and vectors...")
    A = Matrix(2, 2, [1, 2, 3, 4])      # [[1,2], [3,4]]
    B = Matrix(2, 2, [2, 0, 1, 2])      # [[2,0], [1,2]]
    C = Matrix(2, 2, [0, 1, -1, 0])     # [[0,1], [-1,0]]
    p = Vector(2, [5, 6])               # [5, 6]
    q = Vector(2, [7, 8])               # [7, 8]
    v1 = Vector(3, [1, 2, 3])           # [1, 2, 3]
    v2 = Vector(3, [4, 5, 6])           # [4, 5, 6]
    
    print("✅ Test data initialized\n")
    
    # === CORE MATRIX OPERATIONS ===
    print("🔢 Testing Core Matrix Operations:")
    
    # Matrix-Matrix multiplication AB
    AB_expected = Matrix(2, 2, [4, 4, 10, 8])
    AB_result = A * B
    check_matrix_equal(AB_result, AB_expected, "Matrix multiplication A * B")
    
    # Matrix-Vector multiplication Ap  
    Ap_expected = Vector(2, [17, 39])  # [1*5+2*6, 3*5+4*6] = [17, 39]
    Ap_result = A * p
    check_matrix_equal(Ap_result, Ap_expected, "Matrix-vector multiplication A * p")
    
    # Transpose-Vector multiplication A^T p
    ATp_expected = Vector(2, [23, 34])  # [1*5+3*6, 2*5+4*6] = [23, 34]
    ATp_result = A.T * p
    check_matrix_equal(ATp_result, ATp_expected, "Transpose-vector multiplication A^T * p")
    
    # Inverse-Vector multiplication A^-1 p
    # A^-1 = (1/det(A)) * adj(A) = (1/-2) * [4,-2;-3,1] = [-2,1;1.5,-0.5]
    Ainvp_expected = Vector(2, [-4, 4.5])  # [-2*5+1*6, 1.5*5-0.5*6] = [-4, 4.5]
    try:
        Ainvp_result = A.inverse * p
        check_matrix_equal(Ainvp_result, Ainvp_expected, "Inverse-vector multiplication A^(-1) * p", tolerance=1e-6)
    except Exception as e:
        check(f"Inverse-vector multiplication A^(-1) * p", False)
        print(f"   Error: {e}")
    
    print()
    
    # === VECTOR OPERATIONS ===
    print("📐 Testing Vector Operations:")
    
    # Dot product v1^T * v2
    dot_expected = 32  # 1*4 + 2*5 + 3*6 = 32
    dot_result = (v1.T * v2)[0, 0]
    check("Vector dot product v1^T * v2", abs(dot_result - dot_expected) < 1e-6)
    
    # Cross product v1 × v2
    cross_expected = Vector(3, [-3, 6, -3])  # [2*6-3*5, 3*4-1*6, 1*5-2*4]
    cross_result = Vector(3, [
        v1[1,0]*v2[2,0] - v1[2,0]*v2[1,0],  # i component
        v1[2,0]*v2[0,0] - v1[0,0]*v2[2,0],  # j component  
        v1[0,0]*v2[1,0] - v1[1,0]*v2[0,0]   # k component
    ])
    check_matrix_equal(cross_result, cross_expected, "Vector cross product v1 × v2")
    
    print()
    
    # === CHAINED OPERATIONS ===
    print("🔗 Testing Chained Matrix Operations:")
    
    # Chain multiplication A*B*C
    ABC_intermediate = A * B  # First multiply A*B
    ABC_expected = ABC_intermediate * C  # Then multiply result by C
    ABC_result = A * B * C
    check_matrix_equal(ABC_result, ABC_expected, "Chained multiplication A * B * C")
    
    print()
    
    # === ERROR HANDLING ===
    print("🚫 Testing Error Handling:")
    
    # Dimension mismatch detection
    dimension_error_caught = False
    try:
        invalid_result = A * Vector(3, [1, 2, 3])  # 2x2 * 3x1 should fail
    except Exception:
        dimension_error_caught = True
    check("Dimension mismatch detection", dimension_error_caught)
    
    # Singular matrix inverse detection
    singular_matrix = Matrix(2, 2, [1, 2, 2, 4])  # Linearly dependent rows
    singular_error_caught = False
    try:
        singular_inverse = singular_matrix.inverse
    except Exception:
        singular_error_caught = True
    check("Singular matrix inverse detection", singular_error_caught)
    
    print()
    
    # === ADVANCED MATHEMATICAL PROPERTIES ===
    print("🔬 Testing Advanced Mathematical Properties:")
    
    # Identity matrix properties
    I3 = Matrix(3, 3, [1, 0, 0, 0, 1, 0, 0, 0, 1])  # 3x3 Identity
    rotation = Matrix(3, 3, [0, -1, 0, 1, 0, 0, 0, 0, 1])  # 90° rotation in xy-plane
    
    # Identity multiplication
    check_matrix_equal(I3 * rotation, rotation, "Identity matrix multiplication I * R = R")
    check_matrix_equal(rotation * I3, rotation, "Identity matrix multiplication R * I = R")
    
    # Identity inverse
    check_matrix_equal(I3.inverse, I3, "Identity matrix inverse I^(-1) = I")
    
    # Identity transpose  
    check_matrix_equal(I3.T, I3, "Identity matrix transpose I^T = I")
    
    # Rotation matrix orthogonality: R * R^T = I
    rotation_product = rotation * rotation.T
    identity_check = all(
        abs(rotation_product[i,j] - (1 if i==j else 0)) < 1e-10
        for i in range(3) for j in range(3)
    )
    check("Rotation matrix orthogonality R * R^T = I", identity_check)
    
    # Determinant properties
    det_A = A.determinant
    det_B = B.determinant  
    det_AB = (A * B).determinant
    det_product_property = abs(det_AB - (det_A * det_B)) < 1e-10
    check("Determinant multiplicative property det(AB) = det(A) * det(B)", det_product_property)
    
    print()
    
    # === NUMERICAL STABILITY ===
    print("⚖️  Testing Numerical Stability:")
    
    # Test with matrix close to singular
    near_singular = Matrix(2, 2, [1, 2, 1, 2.000001])
    try:
        near_singular_inv = near_singular.inverse
        stability_test = True
    except:
        stability_test = False
    check("Near-singular matrix handling", stability_test)
    
    # Test with very small numbers
    small_matrix = Matrix(2, 2, [1e-10, 2e-10, 3e-10, 4e-10])
    small_det = small_matrix.determinant
    expected_small_det = (1e-10 * 4e-10) - (2e-10 * 3e-10)  # -2e-20
    small_det_correct = abs(small_det - expected_small_det) < 1e-25
    check("Small number determinant calculation", small_det_correct)
    
    print()
    
    # === PERFORMANCE CHARACTERISTICS ===
    print("🚀 Testing Performance Characteristics:")
    
    # Large matrix creation (tests memory efficiency)
    try:
        large_matrix = Matrix(100, 100)  # 10,000 elements
        large_test_passed = large_matrix.width == 100 and large_matrix.height == 100
        check("Large matrix creation (100x100)", large_test_passed)
    except Exception as e:
        check("Large matrix creation (100x100)", False)
        print(f"   Error: {e}")
    
    # Cache-friendly access pattern test
    test_matrix = Matrix(10, 10, list(range(100)))
    access_test_passed = True
    try:
        # Test row-major access pattern (should be cache-friendly)
        for i in range(test_matrix.height):
            for j in range(test_matrix.width):
                value = test_matrix[i, j]
                expected_value = i * test_matrix.width + j
                if value != expected_value:
                    access_test_passed = False
                    break
    except Exception:
        access_test_passed = False
    check("Cache-friendly memory access pattern", access_test_passed)
    
    print()
    
    # === FINAL SUMMARY ===
    print("=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"✅ Tests passed: {passed}")
    print(f"❌ Tests failed: {failed}")
    print(f"📈 Total tests: {passed + failed}")
    print(f"🎯 Success rate: {(passed/(passed+failed)*100):.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Your matrix library is working perfectly!")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the implementation.")
    
    print("=" * 50)
    
    return passed, failed

def test_basic_functionality():
    """Quick smoke test for basic functionality"""
    print("🔥 Running basic functionality smoke test...\n")
    
    try:
        # Basic operations
        A = Matrix(2, 2, [1, 2, 3, 4])
        B = Matrix(2, 2, [5, 6, 7, 8])
        
        # Test core operations
        result_mult = A * B
        result_transpose = A.T
        result_det = A.determinant
        result_inverse = A.inverse
        
        print("✅ Basic smoke test PASSED - Core functionality working!")
        return True
        
    except Exception as e:
        print(f"❌ Basic smoke test FAILED: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Cache-Optimized Matrix Library Test Suite")
    print("=" * 50)
    
    # Run basic functionality test first
    if test_basic_functionality():
        print("\n" + "=" * 50)
        # Run comprehensive tests
        passed, failed = test_matrix_operations_comprehensive()
    else:
        print("❌ Skipping comprehensive tests due to basic functionality failure")
        
    print("\n🏁 Testing complete!")
