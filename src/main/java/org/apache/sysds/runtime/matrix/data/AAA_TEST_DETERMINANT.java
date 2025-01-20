package org.apache.sysds.runtime.matrix.data;

// det(A) = d
// input: Square dimension Matrix
// output determinant value as double value (in systemds will be represented as 1x1 matrixblock)

// if det = 0 it indicates linear dependence of rows or cols
// it means matrix is singular
// It indicates that the dimensions of the transformation get squashed to a lower dimension.
// singular matrix = square matrix that does not have an inverse

 /** BareissDeterminant
 * + Exact for int matrices (no floating-point errors).
 * + Efficient for dense matrices
 * - Less common and more complex to implement than LU
 */

/** LU Decomposition:
* 1. Decompose matrix into  L  and  U .
*  2. Compute the determinant as the product of the diagonal elements of  U  
* + widely implemented in numerical libraries
* - can have rounding errors in floating point matrices
*/

/** Laplace Expansion:
 * 1. Expand the determinant recursively along a row or column.
 * + Intuitive and easy to understand, especially for small matrices.
 * + Useful for symbolic computation and educational purposes.
 * - high-dimension matrices: factorial time complexity ( O(n!) ) 
 */

/** Gaussian Elimination:
 * 1. Reduce the matrix to upper triangular form using row operations
 * 2. Compute the determinant as the product of the diagonal elements
 * + Efficient ( O(n^3) for numerical determinant computation.
 * - can lead to floating-point errors without pivoting 
 * - can require significant memory for storing intermediate matrices with pivoting
 */
public class AAA_TEST_DETERMINANT {

    public static void printMatrix(double[][] matrix) {
        for (double[] row : matrix) {
            for (double element : row) {
                System.out.printf("%.2f ", element);
            }
            System.out.println();
        }
        System.out.println();
    }

    /** Gauss */
    public static double calculateGaussDeterminant(double[][] matrix) {
        int size = matrix.length;

        // Process each column to create an upper triangular matrix
        for (int pivotCol = 0; pivotCol < size; pivotCol++) {
            System.out.println("Processing Column: " + pivotCol);
            printMatrix(matrix);

            // Find a non-zero pivot in the current column
            boolean nonZeroPivotFound = false;
            for (int pivotRow = pivotCol; pivotRow < size; pivotRow++) {
                if (Math.abs(matrix[pivotRow][pivotCol]) > 1e-9) { // Use a small epsilon for floating-point comparison
                    System.out.println("Found non-zero pivot at row " + pivotRow + ", column " + pivotCol);

                    // Swap rows if necessary to move pivot to the diagonal position
                    if (pivotRow != pivotCol) {
                        System.out.println("Swapping rows " + pivotCol + " and " + pivotRow);
                        double[] tempRow = matrix[pivotRow];
                        matrix[pivotRow] = matrix[pivotCol];
                        matrix[pivotCol] = tempRow;
                        printMatrix(matrix);
                    }
                    nonZeroPivotFound = true;
                    break;
                }
            }

            // If no valid pivot is found, determinant is zero
            if (!nonZeroPivotFound) {
                System.out.println("No non-zero pivot found in column " + pivotCol);
                return 0.0;
            }

            // Eliminate entries below the pivot
            for (int row = pivotCol + 1; row < size; row++) {
                double factor = matrix[row][pivotCol] / matrix[pivotCol][pivotCol];
                System.out.printf("Elimination factor for row %d: %.2f\n", row, factor);

                // Update the row using the elimination factor
                for (int col = pivotCol; col < size; col++) {
                    matrix[row][col] -= factor * matrix[pivotCol][col];
                }
            }
        }

        printMatrix(matrix);

        // Calculate determinant as the product of diagonal elements
        double determinant = 1.0;
        for (int i = 0; i < size; i++) {
            determinant *= matrix[i][i];
        }

        return determinant;
    }

    /** Laplace recursive */
    public static double laplaceDeterminant(double[][] matrix) {
        int lengthOfMatrix = matrix.length;

        // Create copy, compute not inplace, keep original matrix
        double[][] tempMatrix = new double[lengthOfMatrix][lengthOfMatrix];
        for (int i = 0; i < lengthOfMatrix; i++) {
            System.arraycopy(matrix[i], 0, tempMatrix[i], 0, lengthOfMatrix);
        }

        double prevPivot = 1.0;  // Start with 1 for first pivot

        for (int k = 0; k < lengthOfMatrix - 1; k++) {
            // If the pivot is zero it means the determinant is zero
            if (tempMatrix[k][k] == 0.0) {
                return 0.0;
            }

            for (int i = k + 1; i < lengthOfMatrix; i++) {
                for (int j = k + 1; j < lengthOfMatrix; j++) {
                    tempMatrix[i][j] = (tempMatrix[i][j] * tempMatrix[k][k] - tempMatrix[i][k] * tempMatrix[k][j]) / prevPivot;
                }
            }
            prevPivot = tempMatrix[k][k];  // Update pivot
        }

        return tempMatrix[lengthOfMatrix - 1][lengthOfMatrix - 1];  // The determinant is the last pivot
    }

    public static void main(String[] args) {
        double[][] matrix = {
            {4, 3, 2},
            {3, 1, 5},
            {2, 5, 7}
        };
        // double[][] matrix = {
        //     {1, 2},
        //     {3, 4}
        // };

        System.out.println("Determinant: " + laplaceDeterminant(matrix)); 
        System.out.println("Determinant: " + calculateGaussDeterminant(matrix)); 

    }
}
