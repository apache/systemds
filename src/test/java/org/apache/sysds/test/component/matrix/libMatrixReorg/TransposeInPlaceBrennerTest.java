/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.test.component.matrix.libMatrixReorg;

import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

public class TransposeInPlaceBrennerTest {

	
	@Test
	public void transposeInPlaceDenseBrennerOnePrime() {
		// 3*4-1 = 11
		testTransposeInPlaceDense(3, 4, 1);
	}

	@Test
	public void transposeInPlaceDenseBrennerTwoPrimes() {
		// 4*9-1 = 5*7
		testTransposeInPlaceDense(4, 9, 0.96);
	}

	@Test
	public void transposeInPlaceDenseBrennerThreePrimes() {
		// 2*53-1 = 3*5*7
		testTransposeInPlaceDense(2, 53, 0.52);
	}

	@Test
	public void transposeInPlaceDenseBrennerThreePrimesOneExpo() {
		// 1151*2999-1 = (2**3)*3*143827
		testTransposeInPlaceDense(1151, 2999, 0.82);
	}

	@Test
	public void transposeInPlaceDenseBrennerThreePrimesAllExpos() {
		// 9*10889-1 = (2**4)*(5**3)*(7**2)
		testTransposeInPlaceDense(9, 10889, 0.74);
	}

	@Test
	public void transposeInPlaceDenseBrennerFourPrimesOneExpo() {
		// 53*4421-1 = (2**3)*3*13*751
		testTransposeInPlaceDense(53, 4421, 0.75);
	}

	@Test
	public void transposeInPlaceDenseBrennerFivePrimes() {
		// 3*3337-1 = 2*5*7*11*13
		testTransposeInPlaceDense(3, 3337, 0.68);
	}

	@Test
	public void transposeInPlaceDenseBrennerSixPrimesOneExpo() {
		// 53*7177-1 = (2**2)*5*7*11*13*19
		testTransposeInPlaceDense(53, 7177, 0.78);
	}

	@Test
	public void transposeInPlaceDenseBrennerSevenPrimesThreeExpos() {
		// 2087*17123-1 = (2**2)*3*(5**2)*(7**2)*11*13*17
		testTransposeInPlaceDense(2087, 17123, 0.79);
	}

	@Test
	public void transposeInPlaceDenseBrennerEightPrimes() {
		// 347*27953-1 = 2*3*5*7*11*13*17*19
		testTransposeInPlaceDense(347, 27953, 0.86);
	} 

	@Test
	public void transposeInPlaceDenseBrennerNinePrimes() {
		// 317*703763-1 = 2*3*5*7*11*13*17*19*23
		MatrixBlock X = MatrixBlock.randOperations(317, 703763, 0.52);

		Exception exception = assertThrows(RuntimeException.class,
			() -> LibMatrixReorg.transposeInPlaceDenseBrenner(X, 1));
		assertTrue(exception.getMessage().contains("Not enough space, need to expand input arrays."));
	}

	private void testTransposeInPlaceDense(int rows, int cols, double sparsity) {
		MatrixBlock X = MatrixBlock.randOperations(rows, cols, sparsity);
		MatrixBlock tX = LibMatrixReorg.transpose(X);
        LibMatrixReorg.transposeInPlaceDenseBrenner(X, 1);
		
		
		TestUtils.compareMatrices(X, tX, 0);
	}
    
	// Tests for tensor permutations
	@Test
    public void testTensorPermuteSplit_3D() {
        int[] shape = {50,2,10};
        int[] perm = {1,2,0}; 
        testTransposeInPlaceTensor(shape, perm);
    } 

	@Test
    public void testTensorPermuteSplit_8D() {
        int[] shape = {3,2,1,3,2,3,1,2};
        int[] perm = {4,5,6,7,0,1,2,3};
        testTransposeInPlaceTensor(shape, perm);
    } 

	@Test
    public void testTensorPermuteSplit_4D() {
        int[] shape = {3,2,5,3};
        int[] perm = {2,3,0,1}; 
        testTransposeInPlaceTensor(shape, perm);
    } 

    @Test
    public void testTensorPermuteSplit_2D_21() {
        int[] shape = {4, 10};
        int[] perm = {1,0}; 
        testTransposeInPlaceTensor(shape, perm);
    } 

	//Test for primitives 
	@Test
    public void testTensorPermute_3D_213() {
        int[] shape = {4, 2, 7};
        int[] perm = {1,0,2}; 
        testTransposeInPlaceTensor(shape, perm);
    }
 
	@Test
    public void testTensorPermute_3D_132() {
        int[] shape = {3, 4, 2};
        int[] perm = {0, 2, 1}; 
        testTransposeInPlaceTensor(shape, perm);
    }
    
	@Test
    public void testTensorPermute_4D_1324() {
        int[] shape = {3, 2, 2, 3};
        int[] perm = {0, 2, 1, 3}; 
        testTransposeInPlaceTensor(shape, perm);
    }

	@Test
    public void testTensorPermute_4Db_1324() {
        int[] shape = {3, 4, 5, 6};
        int[] perm = {0, 2, 1, 3}; 
        testTransposeInPlaceTensor(shape, perm);
    }

     @Test
    public void testTensorPermuteSplit_5D() {
        int[] shape = {2, 3, 4, 5, 6};
        int[] perm = {2, 3, 4, 0, 1};
        testTransposeInPlaceTensor(shape, perm);
    }

	@Test
    public void testTensorPermuteSplit_6D() {
    int[] shape = {4, 3, 2, 5, 8, 2};
    int[] perm = {3, 4, 5, 0, 1, 2}; 
    testTransposeInPlaceTensor(shape, perm);
    }

	@Test
     public void testTensorPermuteSplit_5D_MiddleSwap() {
    int[] shape = {2, 6, 2, 4, 5};
    int[] perm = {4, 3, 2, 1, 0}; 
    testTransposeInPlaceTensor(shape, perm);
}

   @Test
     public void testTensorPermute_5D_MiddleSwap_Complex() {
    int[] shape = {2, 2, 3, 4, 2};
    int[] perm = {0, 2, 1, 3, 4}; 
    testTransposeInPlaceTensor(shape, perm);
}

@Test
     public void testTensorPermute_7Db() {
    int[] shape = {20, 30, 15, 5, 2, 5, 2};
    int[] perm = {0, 6, 1, 5, 4, 2, 3}; 
    testTransposeInPlaceTensor(shape, perm);
}

@Test
     public void testTensorPermute_7D() {
    int[] shape = {2, 3, 5, 5, 2, 3, 2};
    int[] perm = {0, 6, 1, 5, 4, 2, 3}; 
    testTransposeInPlaceTensor(shape, perm);
}
 
	@Test
    public void testTensorPermuteSplit_Max2() {
    int[] shape = {1000, 300, 100};
    int[] perm = {2, 0, 1}; 
    testTransposeInPlaceTensor(shape, perm);
    }

	@Test
    public void testTensorPermuteSplit_Max3() {
    int[] shape = {8000, 4000, 2}; 
    int[] perm = {2, 0, 1}; 
    testTransposeInPlaceTensor(shape, perm);
}  

    @Test
    public void testTensorPermute_3D_allCases() {
    int[] shape = {2, 3, 2}; 
    int[] perm1 = {0,1, 2}; 
	int[] perm2 = {0,2, 1}; 
	int[] perm3 = {1,0,2};
	int[] perm4 = {1,2,0}; 
	int[] perm5 = {2,0,1}; 
	int[] perm6 = {2,1,0}; 
    testTransposeInPlaceTensor(shape, perm1);
	testTransposeInPlaceTensor(shape, perm2);
	testTransposeInPlaceTensor(shape, perm3);
	testTransposeInPlaceTensor(shape, perm4);
	testTransposeInPlaceTensor(shape, perm5);
	testTransposeInPlaceTensor(shape, perm6);

} 
	@Test
    public void testTensorPermuteSplit_4Db_213() {
    int[] shape = {2, 3, 4};
    int[] perm  = {1, 0, 2}; 
    testTransposeInPlaceTensor(shape, perm); 
}
     @Test
    public void testTensorPermuteSplit_4Db_132() {
    int[] shape = {2, 3, 4};
    int[] perm  = {0, 2, 1}; 
    testTransposeInPlaceTensor(shape, perm); 
}

	// Edge case tests
	
	// 1. Square matrices
	@Test
	public void transposeInPlaceDenseSquare5x5() {
		testTransposeInPlaceDense(5, 5, 0.8);
	}

	@Test
	public void transposeInPlaceDenseSquare100x100() {
		testTransposeInPlaceDense(100, 100, 0.7);
	}

	@Test
	public void testTensorPermute_3D_SquareDims() {
		int[] shape = {4, 4, 4};
		int[] perm = {2, 0, 1};
		testTransposeInPlaceTensor(shape, perm);
	}

	// 2. Vectors (1×N and N×1)
	@Test
	public void transposeInPlaceDenseRowVector() {
		testTransposeInPlaceDense(1, 50, 0.9);
	}

	@Test
	public void transposeInPlaceDenseColVector() {
		testTransposeInPlaceDense(50, 1, 0.9);
	}

	@Test
	public void testTensorPermute_VectorLike() {
		int[] shape = {1, 20};
		int[] perm = {1, 0};
		testTransposeInPlaceTensor(shape, perm);
	}

	// 3. Single element
	@Test
	public void transposeInPlaceDenseSingleElement() {
		testTransposeInPlaceDense(1, 1, 1.0);
	}

	@Test
	public void testTensorPermute_SingleElement() {
		int[] shape = {1, 1, 1};
		int[] perm = {2, 1, 0};
		testTransposeInPlaceTensor(shape, perm);
	}

	// 4. Prime dimensions
	@Test
	public void transposeInPlaceDensePrime7x11() {
		testTransposeInPlaceDense(7, 11, 0.75);
	}

	@Test
	public void transposeInPlaceDensePrime13x17() {
		testTransposeInPlaceDense(13, 17, 0.82);
	}

	@Test
	public void testTensorPermute_AllPrimeDims() {
		int[] shape = {3, 5, 7};
		int[] perm = {1, 2, 0};
		testTransposeInPlaceTensor(shape, perm);
	}

	// 5. Power of 2 dimensions (common in computing, just to be sure)
	@Test
	public void transposeInPlaceDensePowerOf2_64x128() {
		testTransposeInPlaceDense(64, 128, 0.6);
	}

	@Test
	public void transposeInPlaceDensePowerOf2_32x64() {
		testTransposeInPlaceDense(32, 64, 0.85);
	}

	@Test
	public void testTensorPermute_PowerOf2Dims() {
		int[] shape = {8, 16, 4};
		int[] perm = {2, 1, 0};
		testTransposeInPlaceTensor(shape, perm);
	}

	// 7. Consecutive transpose (should return to original)
	@Test
	public void transposeInPlaceDenseConsecutiveTwice() {
		MatrixBlock X = MatrixBlock.randOperations(7, 13, 0.75);
		MatrixBlock original = new MatrixBlock(X);
		
		LibMatrixReorg.transposeInPlaceDenseBrenner(X, 1);
		LibMatrixReorg.transposeInPlaceDenseBrenner(X, 1);
		
		TestUtils.compareMatrices(X, original, 0);
	}

	@Test
	public void testTensorPermute_ConsecutiveTwice() {
		int[] shape = {3, 4, 5};
		int[] perm = {1, 2, 0};
		
		MatrixBlock matrix = createDenseTensor(shape);
		MatrixBlock original = new MatrixBlock(matrix);
		
		LibMatrixReorg.transposeInPlaceTensor(matrix, shape, perm);
		// Apply reverse permutation to get back
		int[] reversePerm = new int[perm.length];
		for (int i = 0; i < perm.length; i++) {
			reversePerm[perm[i]] = i;
		}
		int[] newShape = new int[shape.length];
		for (int i = 0; i < perm.length; i++) {
			newShape[i] = shape[perm[i]];
		}
		LibMatrixReorg.transposeInPlaceTensor(matrix, newShape, reversePerm);
		
		TestUtils.compareMatrices(matrix, original, 0);
	}

	// 8.tensors with dimension=1
	@Test
	public void testTensorPermute_WithDim1_case1() {
		int[] shape = {1, 5, 3};
		int[] perm = {2, 0, 1};
		testTransposeInPlaceTensor(shape, perm);
	}

	@Test
	public void testTensorPermute_WithDim1_case2() {
		int[] shape = {4, 1, 2, 1};
		int[] perm = {2, 3, 0, 1};
		testTransposeInPlaceTensor(shape, perm);
	}

	@Test
	public void testTensorPermute_WithDim1_case3() {
		int[] shape = {3, 1, 4};
		int[] perm = {1, 2, 0};
		testTransposeInPlaceTensor(shape, perm);
	}

	// 9. Invalid permutations (negative tests)
	// NOTE: more detailed error handling can be added in the future, currently these are just checking for exceptions
	@Test
	public void testTensorPermute_InvalidPerm_OutOfRange() {
		int[] shape = {2, 3, 4};
		int[] perm = {0, 1, 3}; // 3 is out of range for 3D tensor
		
		MatrixBlock matrix = createDenseTensor(shape);
		
		assertThrows(Exception.class,
			() -> LibMatrixReorg.transposeInPlaceTensor(matrix, shape, perm));
	}


	@Test
	public void testTensorPermute_InvalidPerm_WrongLength() {
		int[] shape = {2, 3, 4};
		int[] perm = {0, 1}; // only 2 elements but 3d tensor
		
		MatrixBlock matrix = createDenseTensor(shape);
		
		assertThrows(Exception.class,
			() -> LibMatrixReorg.transposeInPlaceTensor(matrix, shape, perm));
	}

	@Test
	public void testTensorPermute_InvalidPerm_Negative() {
		int[] shape = {2, 3, 4};
		int[] perm = {-1, 1, 2}; // negtive index
		
		MatrixBlock matrix = createDenseTensor(shape);
		
		assertThrows(Exception.class,
			() -> LibMatrixReorg.transposeInPlaceTensor(matrix, shape, perm));
	}

	// 10. Null/empty inputs
	@Test
	public void testTensorPermute_EmptyShape() {
		int[] shape = {};
		int[] perm = {};
		
		assertThrows(Exception.class,
			() -> createDenseTensor(shape));
	}

	@Test
	public void testTensorPermute_NullMatrix() {
		int[] shape = {2, 3};
		int[] perm = {1, 0};
		
		assertThrows(Exception.class,
			() -> LibMatrixReorg.transposeInPlaceTensor(null, shape, perm));
	}


    //Filling matrices 
    private static MatrixBlock createDenseTensor(int[] shape) {
    long size = 1;
    for (int s : shape)
        size *= s;

    if (size > Integer.MAX_VALUE)
        throw new IllegalArgumentException("Tensor too large: " + size);

    int rows = shape[0];
    long colsL = size / rows;
    int cols = (int) colsL;

    MatrixBlock matrix = new MatrixBlock(rows, cols, false);
    matrix.allocateDenseBlock();

    double[] values = matrix.getDenseBlockValues();
    for (int i = 0; i < values.length; i++)
        values[i] = i;

    if (matrix.getDenseBlock() != null)
        matrix.getDenseBlock().setDims(shape);

    return matrix;
} 


    private void testTransposeInPlaceTensor(int[] shape, int[] perm) {
        
        MatrixBlock matrix =createDenseTensor(shape);
        MatrixBlock expected = permutationOutOfPlace(matrix, shape, perm);
        LibMatrixReorg.transposeInPlaceTensor(matrix, shape, perm);
        TestUtils.compareMatrices(matrix, expected, 0);
        TestUtils.compareTensorValues(matrix, expected, 0);
		
    }
    
	//returns the expected matrix (found out-of-place) for comparision
    private MatrixBlock permutationOutOfPlace(MatrixBlock in, int[] shape, int[] perm) {
        int[] newShape = new int[shape.length];
        for(int i=0; i<perm.length; i++){
			newShape[i] = shape[perm[i]];
		} 
        
        int newRows = newShape[0];
        long newCols = 1;
        for(int i = 1; i < newShape.length; i++) {
           newCols *= newShape[i];
    }

         MatrixBlock out = new MatrixBlock(newRows, (int)newCols, false);
         out.allocateDenseBlock();
        
        double[] inVal = in.getDenseBlockValues();
        double[] outVal = out.getDenseBlockValues();

		int[] originalCoords = new int[shape.length];
        int[] permCoords = new int[shape.length];

        for(int i = 0; i < inVal.length; i++) {
          getCoords(i, shape, originalCoords); 
          for(int j = 0; j < perm.length; j++) {
            permCoords[j] = originalCoords[perm[j]];
           }
          int outIdx = getIndex(permCoords, newShape);
          outVal[outIdx] = inVal[i];
      }
        
	    out.setNumRows(newShape[0]);
        long cols = 1;
        for(int i=1; i<newShape.length; i++){
			cols *= newShape[i];
		}
        out.setNumColumns((int) cols);
        out.getDenseBlock().setDims(newShape);
        
        return out;
    }

    private void getCoords(int index, int[] shape, int[] originalCoords) {
    for (int i = shape.length - 1; i >= 0; i--) {
        originalCoords[i] = index % shape[i];
        index /= shape[i];
    }
    }

    private int getIndex(int[] coords, int[] shape) {
        int index = 0;
        int multiplier = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            index += coords[i] * multiplier;
            multiplier *= shape[i];
        }
        return index;
    }

    //Test for correct meta-data after permutation
	@Test
    public void testTensorPermuteSplitShape_6D() {
    int[] shape = {2, 3, 4, 5, 6, 7};
    int[] perm = {1, 2, 3, 4, 5, 0}; 
    
    long size = 1;
    for(int s : shape) {
		size *= s; 
	}
    
    MatrixBlock X = new MatrixBlock((int) size, 1, false);
    X.allocateDenseBlock();
    LibMatrixReorg.transposeInPlaceTensor(X, shape, perm);
    testTransposeInPlaceTensorShape(X, shape, perm);
}

    @Test
    public void testTensorPermuteSplitShape_6D_Max() {
    int[] shape = {1000, 500, 20, 2, 2, 2};
    int[] perm = {1, 2, 3, 4, 5, 0}; 
    
    long size = 1;
    for(int s : shape) {
		size *= s; 
	}
    
    MatrixBlock X = new MatrixBlock((int) size, 1, false);
    X.allocateDenseBlock();
    LibMatrixReorg.transposeInPlaceTensor(X, shape, perm);
    testTransposeInPlaceTensorShape(X, shape, perm);
}

    @Test
    public void testTensorPermuteSplitShape_4D() {
    int[] shape = {100, 22, 70, 90};
    int[] perm = {1, 2, 3, 0}; 
    
    long size = 1;
    for(int s : shape) {
		size *= s; 
	}
    
    MatrixBlock X = new MatrixBlock((int) size, 1, false);
    X.allocateDenseBlock();
    LibMatrixReorg.transposeInPlaceTensor(X, shape, perm);
    testTransposeInPlaceTensorShape(X, shape, perm);
}


    @Test
    public void testTensorPermuteSplitShape_8D() {
    int[] shape = {10, 22, 7, 9, 30, 6, 4, 7};
    int[] perm = { 3, 4, 5, 6, 7, 0, 1, 2}; 
    
    long size = 1;
    for(int s : shape) {
		size *= s; 
	}
    
    MatrixBlock X = new MatrixBlock((int) size, 1, false);
    X.allocateDenseBlock();
    LibMatrixReorg.transposeInPlaceTensor(X, shape, perm);
    testTransposeInPlaceTensorShape(X, shape, perm);
}

@Test
    public void testTensorPermuteSplitShape_5D_middle() {
    int[] shape = {10, 8, 5, 4, 2};
    int[] perm = {0, 2, 1, 3, 4}; 
    
    long size = 1;
    for(int s : shape) {
		size *= s; 
	}
    
    MatrixBlock X = new MatrixBlock((int) size, 1, false);
    X.allocateDenseBlock();
    LibMatrixReorg.transposeInPlaceTensor(X, shape, perm);
    testTransposeInPlaceTensorShape(X, shape, perm);
}

  @Test
    public void testTensorPermuteSplitShape_5D() {
    int[] shape = {2,3,5,2,8}; 
    int[] perm = {3,4,0,1,2}; 
    
    long size = 1;
    for(int s : shape) {
		size *= s; }

	MatrixBlock X = new MatrixBlock((int) size, 1, false);
    X.allocateDenseBlock();
    LibMatrixReorg.transposeInPlaceTensor(X, shape, perm);
    testTransposeInPlaceTensorShape(X, shape, perm);
	}
    
	@Test
    public void testTensorPermuteSplitShape_2D() {
    int[] shape = {2,3}; 
    int[] perm = {1,0}; 
    
    long size = 1;
    for(int s : shape) {
		size *= s; 
	}
    
    MatrixBlock X = new MatrixBlock((int) size, 1, false);
    X.allocateDenseBlock();
    LibMatrixReorg.transposeInPlaceTensor(X, shape, perm);
    testTransposeInPlaceTensorShape(X, shape, perm);
}
	private void testTransposeInPlaceTensorShape(MatrixBlock transposed_X, int[] originalShape, int[] perm){
        int[] expectedShape = new int[originalShape.length];
        for(int i = 0; i < perm.length; i++) {
        expectedShape[i] = originalShape[perm[i]];
       }
	   int expectedRows = expectedShape[0];
       long expectedCols = 1;
       for(int i = 1; i < expectedShape.length; i++) {
        expectedCols *= expectedShape[i];
       }
       
	   // MatrixBlock shape-match
	   assertEquals("Matrix Rows mismatch", expectedRows, transposed_X.getNumRows());
       assertEquals("Matrix Columns mismatch", (int)expectedCols, transposed_X.getNumColumns());
       
	   // DenseBlock shape-match
	   int[] transposedShape = new int[originalShape.length];
	   DenseBlock dense_X = transposed_X.getDenseBlock();
	   if(dense_X != null){
		  //Comparison of each dimension
	      for (int i = 0; i < expectedShape.length; i++) {
			transposedShape[i] = dense_X.getDim(i);
            assertEquals("Dimension " + i + " mismatch", expectedShape[i], dense_X.getDim(i));
        }
         int currentExpectedSuffix = expectedShape[expectedShape.length - 1]; 
		  //Comparison of suffixes
         for (int i = expectedShape.length - 1; i >= 1; i--) {
            assertEquals("Suffix product at dim " + i + " mismatch", currentExpectedSuffix, dense_X.getCumODims(i - 1));
         if(i > 1) {
            currentExpectedSuffix *= expectedShape[i - 1];
             }
        }
	}
	   

}
}
