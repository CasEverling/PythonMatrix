# PythonMatrix
Cache-Optimized Matrix Library
A high-performance, educational matrix library implemented in Python with cache locality optimization and mathematical rigor.

# 🚀 Key Features
Cache-Friendly Design: Contiguous memory layout using single array instead of list-of-lists
Mathematical Accuracy: Gaussian elimination with partial pivoting for numerical stability
Pythonic Interface: Natural mathematical notation with operator overloading
Comprehensive Operations: Full linear algebra suite including determinants, inverses, and decompositions
Robust Error Handling: Detailed dimension checking and meaningful error messages

# 🎯 Design Philosophy
This library prioritizes performance and mathematical correctness over convenience. Built from scratch to demonstrate:

Understanding of memory layout and cache optimization
Implementation of fundamental linear algebra algorithms
Clean object-oriented design with proper abstraction
Production-quality error handling and validation

# 🔧 Core Architecture
Memory Optimization
python
## Traditional approach (poor cache locality)
matrix = [[1, 2], [3, 4]]  # Scattered memory locations

## Our approach (cache-friendly)
matrix = Matrix(2, 2, [1, 2, 3, 4])  # Contiguous array: [1, 2, 3, 4]
Smart Indexing
python
## Access element at row i, column j
value = matrix[i, j]  # Internally: content[i * width + j]
📚 Usage Examples
Basic Operations
python
from matrix import Matrix, Vector

## Create matrices
A = Matrix(2, 2, [1, 2, 3, 4])  # [[1, 2], [3, 4]]
B = Matrix(2, 2, [5, 6, 7, 8])  # [[5, 6], [7, 8]]

## Matrix multiplication - clean syntax!
C = A * B

## Transpose
A_T = A.T

## Inverse (uses adjugate method)
A_inv = A.inverse

## Determinant (Gaussian elimination with partial pivoting)
det_A = A.determinant
Vector Operations
python
## Vectors are special matrices (width=1)
v1 = Vector(3, [1, 2, 3])
v2 = Vector(3, [4, 5, 6])

## Matrix-vector multiplication
result = A * v1

## Dot product (mathematical notation)
dot_product = (v1.T * v2)[0, 0]

## Cross product
cross_result = v1.cross(v2)  # Custom method for 3D vectors
Advanced Operations
python
## Chain multiplication
result = A * B * C * D  # Automatic left-to-right evaluation

## Adjugate matrix (cofactor transpose)
adj_A = A.adj

## Minor matrices
minor_01 = A.minor(0, 1)  # Remove row 0, column 1
🧮 Mathematical Implementation
Determinant Calculation
Algorithm: Gaussian elimination with partial pivoting
Stability: Chooses largest pivot for numerical accuracy
Efficiency: O(n³) time complexity
Robustness: Handles singular matrices gracefully
Matrix Inverse
Method: Adjugate matrix approach
Formula: A⁻¹ = (1/det(A)) × adj(A)
Validation: Automatic singularity detection
Precision: Maintains numerical stability
Memory Layout
Matrix(3, 2, [a, b, c, d, e, f])

Logical view:     Memory layout:
[a  b]           [a, b, c, d, e, f]
[c  d]            ↑  Better cache performance!
[e  f]
🎨 API Design
Operator Overloading
python
## Mathematical operations feel natural
result = A * B + C * D
transpose = A.T
inverse = A.inverse
Property-Based Interface
python
## Properties for mathematical concepts
det = matrix.determinant    # Not determinant()
adj = matrix.adj           # Not adjugate()
transpose = matrix.T       # Not transpose()
Generator-Based Iteration
python
## Memory-efficient row/column access
for row in matrix.rows:
    for element in row:
        process(element)

for col in matrix.cols:
    for element in col:
        process(element)
# 🔍 Error Handling
Comprehensive validation with educational error messages:

python
## Dimension mismatch
>>> A = Matrix(2, 3, [...])
>>> B = Matrix(4, 2, [...])
>>> C = A * B
Exception: Matrix multiplication is only allowed if matrix A has the same number of columns as matrix B has of rows

## Singular matrix
>>> A = Matrix(2, 2, [1, 2, 2, 4])  # Linearly dependent rows
>>> A_inv = A.inverse
Exception: Matrix does not have an inverse. Matrix: 
1.00 | 2.00
2.00 | 4.00

## Type safety
>>> matrix[1.5, 2]
Exception: Row and col values must be integers not (<class 'float'>, <class 'int'>)
# 🚀 Performance Features
Cache Optimization
Contiguous memory: Single array storage
Predictable access patterns: Row-major ordering
Reduced memory fragmentation: Better than nested lists
Algorithmic Efficiency
Partial pivoting: Numerical stability without full pivoting overhead
In-place operations: Memory-efficient transformations
Early termination: Singularity detection shortcuts computation
Memory Efficiency
Minimal overhead: Direct array storage
Type consistency: Uniform float storage
Reference optimization: Shared content arrays where possible
# 🧪 Testing
The library includes comprehensive test suites covering:

Basic operations: Addition, multiplication, transpose
Advanced operations: Determinant, inverse, adjugate
Edge cases: Singular matrices, dimension mismatches
Numerical stability: Pivot selection, precision handling
Performance validation: Cache locality benefits
# 🎓 Educational Value
This implementation demonstrates:

Data Structure Design: Cache-friendly memory layouts
Algorithm Implementation: Gaussian elimination, cofactor expansion
Numerical Methods: Partial pivoting, stability considerations
Software Engineering: Clean APIs, error handling, documentation
Performance Optimization: Memory access patterns, computational efficiency
# 📋 Requirements
Python 3.7+
No external dependencies (pure Python implementation)
# 🏗️ Installation
bash
git clone https://github.com/yourusername/cache-optimized-matrix-library
cd cache-optimized-matrix-library
python -m pytest tests/  # Run tests
# 🤝 Contributing
This is primarily an educational project showcasing fundamental computer science concepts. Contributions that maintain the educational focus and performance characteristics are welcome.

📄 License
MIT License - Feel free to use this for educational purposes and learning!

Built with ❤️ for performance, education, and clean code

