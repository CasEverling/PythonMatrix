from typing import Generator

class Matrix:
    def __init__(self, width:int, height:int, content:list[float] = None):
        if (width * height == 0):
            raise Exception("Matrix cannot have a dimension of size zero")
        
        if (type(width) != int or type(height) != int):
            raise Exception(f"Width and height must be intergers, not {type(width), type(height)}")

        if not content:
            content = [0 for _ in range(width * height)]

        if len(content) != (width * height):
            raise Exception("Size paramentes do not match with the mateix size")

        self.width, self.height = width, height
        self.content = content

    #region Static Ops
    @staticmethod
    def __matrix_mult(m1, m2):
        if m1.width != m2.height:
            raise Exception("Matrix multiplication is only allowed if matrix A has the same number of columns as matrix B has of rows")
        
        m3: Matrix = Matrix(m2.width, m1.height)

        for row_number in range(m3.height):
            for col_number in range(m3.width):
                m3[row_number, col_number] = sum(
                    [v1 * v2 for v1, v2 in zip(
                        [v1 for v1 in m1.row(row_number)],
                        [v2 for v2 in m2.col(col_number)]
                )])
        
        return m3
    
    def __scalar_mult(m1, scalar):
        return Matrix(m1.width, m1.height, [v1 * scalar for v1 in m1.content])

    #endregion

    #region Non-Static Ops
    def __mul__ (self, other) -> None:
        if type(other) in [Matrix, Vector]:
            return Matrix.__matrix_mult(self, other)
        elif type(other) in [int, float]:
            return Matrix.__scalar_mult(self, other)
        else:
            raise Exception(f"Invalid operation * between Matrix and {type(other)}")

    @property
    def T(self) -> "Matrix":
        resulting_matrix: Matrix = Matrix(self.height, self.width)
        for row in range(self.height):
            for col in range(self.width):
                resulting_matrix[col, row] = self[row, col]
        return resulting_matrix

    @property
    def determinant(self) -> float:
        
        if self.width != self.height:
            raise Exception(f"Only Square matrixes have determinants, matrix {self.width} by {self.height} is not square")
        
        multiplier: float = 1

        replacement_matrix: Matrix = Matrix(self.width, self.height, self.content.copy())
        
        # Remove all necessary zeros from the diagonal
        row_number: int = 0
        while row_number < replacement_matrix.height:
            if replacement_matrix[row_number, row_number] != 0:
                row_number += 1
            elif row_number + 1 == replacement_matrix.height:
                return 0.0 # Impossible to remove zero from the main diagonal
            else:
                replacement_matrix.swap_row(row_number, row_number + 1)
                multiplier *= -1
                
        
        # performs structured gaussian elimination
        for row_number in range(replacement_matrix.height):

            # Checks if there is a zero in the main diagonal and try changing it
            if replacement_matrix[row_number, row_number] == 0:
                for offset in range(1, replacement_matrix.height - row_number):
                    # Looks for the next row that can be switched
                    if replacement_matrix[row_number + offset, row_number] != 0:
                        replacement_matrix.swap_row(row_number, row_number + offset)
                        multiplier *= -1
                        break # Managed to remove zero from main diagonal
                else:
                    return 0 # Number zero must exist on the diagonal

            # Eliminates all other first row values
            for second_row in range(row_number + 1, replacement_matrix.height):
                if replacement_matrix[second_row, row_number] != 0:  # Skip if already zero
                    row_multiplier = (replacement_matrix[second_row, row_number]) / (replacement_matrix[row_number, row_number])
                    for col in range(row_number, replacement_matrix.width):
                        replacement_matrix[second_row, col] -= row_multiplier * replacement_matrix[row_number, col]

        # Calculate the main diagonal
        diagonal: float = 1
        for i in range(replacement_matrix.height):
            diagonal *= replacement_matrix[i, i]

        return multiplier * diagonal
        
    def minor(self, row: int, col: int) -> "Matrix":
        if self.width != self.height:
            raise Exception("Minors are only defined for square matrices")
        
        content = []
        for i in range(self.height):
            if i == row:
                continue
            for j in range(self.width):
                if j == col:
                    continue
                content.append(self[i, j])
        
        return Matrix(self.width - 1, self.height - 1, content)

    @property
    def adj(self) -> "Matrix":
        if self.width != self.height:
            raise Exception("Adjugate is only defined for square matrices")

        cofactors = []
        for i in range(self.height):
            for j in range(self.width):
                minor_det = self.minor(i, j).determinant
                sign = (-1) ** (i + j)
                cofactors.append(sign * minor_det)

        # Build cofactor matrix
        cofactor_matrix = Matrix(self.width, self.height, cofactors)

        # Adjugate is transpose of cofactor matrix
        return cofactor_matrix.T


    
    @property
    def inverse(self) -> None:
        determinant: float = self.determinant
        if determinant == 0:
            raise Exception(f"Matrix does not have a inverse. Matris: \n {self.display()}")
        return self.adj * (1/determinant)
    
    #endregion

    def swap_row(self, x: int, y: int) -> None:
        if type(x) != int or type(y) != int:
            raise Exception("Inputs of swapt_row must be of type int")
        
        if x >= self.height or y >= self.height:
            raise Exception(f"Rows outside of the boundrie [0,{self.height-1}]")
        
        begin_x: int = x * self.height
        begin_y: int = y * self.height

        row_x: list[float] = self.content[begin_x : begin_x + self.height]

        # Replace row x with row y info
        for i in range(self.width):
            self.content[ begin_x + i ] = self.content[ begin_y + i ]

        # Replace row y with row x info
        for i, val in enumerate(row_x):
            self.content[ begin_y + i ] = val, 
    
    # region Iter Tools
    def row(self, i: int) -> Generator[float, None, None]:
        for j in range(self.width):
            yield self[i, j]

    def col(self, i: int) -> Generator[float, None, None]:
        for j in range(self.height):
            yield self[j, i]
    
    @property
    def rows(self) -> Generator[Generator[float, None, None], None, None]:
        for i in range(self.height):
            yield self.row(i)
    
    @property
    def cols(self) -> Generator[Generator[float, None, None], None, None]:
        for i in range(self.width):
            yield self.col(i)

    #endregion

    #region Index Access
    def __getitem__(self, index) -> float:
        if type(index) != tuple:
            raise Exception("A (row, col) touple must be provided when accessing index")
        if len(index) != 2:
            raise Exception("Number of tuple parameter for position must be 2")
        if type(index[0]) != int or type(index[1]) != int:
            raise Exception(f"Row and col values must be integers not ({type(index[0])}, {type(index[1])})")
        
        try:
            return self.content[index[0] * self.width + index[1]]
        except IndexError:
            raise IndexError(f"Index ({index[0], index[1]}) is out of boundy of matrix of size ({self.height},{self.width})")

    def __setitem__(self, index, value: float) -> None:
        if type(index) != tuple:
            raise Exception("A (row, col) touple must be provided when accessing index")
        if len(index) != 2:
            raise Exception("Number of tuple parameter for position must be 2")
        if type(index[0]) != int or type(index[1]) != int:
            raise Exception(f"Row and col values must be integers not ({type(index[0])}, {type(index[1])})")
        
        try:
            self.content[index[0] * self.width + index[1]] = value
        except IndexError:
            raise IndexError(f"Index ({index[0], index[1]}) is out of boundy of matrix of size ({self.height},{self.width})")
    #endregion

    def display(self) -> None:
        for row in self.rows:
            print(" | ".join([f"{x:.2f}" for x in row]))

class Vector(Matrix):
    def __init__(self, size, content = None):
        super().__init__(1, size, content)
    

        
