package dml.test.components.runtime.matrix.io;

import static org.junit.Assert.*;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Random;

import org.junit.Test;

import dml.runtime.functionobjects.Plus;
import dml.runtime.matrix.io.MatrixBlock1D;
import dml.runtime.matrix.io.MatrixValue.CellIndex;
import dml.runtime.matrix.operators.RightScalarOperator;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class MatrixBlock1DTest {

    @Test
    public void testReset() {
        int rows = 10;
        int cols = 10;
        
        MatrixBlock1D denseBlock = getMatrixBlock(rows, cols, false, 1);
        denseBlock.reset();
        assertEquals(0, denseBlock.getNonZeros());
        double[] denseArray = denseBlock.getDenseArray();
        assertEquals((rows * cols), denseArray.length);
        for(int i = 0; i < denseArray.length; i++) {
            assertEquals("value at position " + i +" is not 0", 0, denseArray[i], 0);
        }
        
        MatrixBlock1D sparseBlock = getMatrixBlock(rows, cols, true, 0.1);
        sparseBlock.reset();
        assertEquals(0, sparseBlock.getNonZeros());
        HashMap<CellIndex, Double> sparseMap = sparseBlock.getSparseMap();
        assertEquals(0, sparseMap.size());
    }

    @Test
    public void testResetIntInt() {
        int[] rows = new int[] { 10, 20, 5 };
        int[] cols = new int[] { 10, 20, 5 };
        
        MatrixBlock1D denseBlock = getMatrixBlock(rows[0], cols[0], false, 1);
        denseBlock.reset(rows[1], cols[1]);
        assertEquals(rows[1], denseBlock.getNumRows());
        assertEquals(cols[1], denseBlock.getNumColumns());
        assertEquals(0, denseBlock.getNonZeros());
        double[] denseArray = denseBlock.getDenseArray();
      //  assertEquals((rows[1] * cols[1]), denseArray.length);
      /*  for(int i = 0; i < denseArray.length; i++) {
            assertEquals("value at position " + i +" is not 0", 0, denseArray[i], 0);
        }*/
        
        denseBlock.reset(rows[2], cols[2]);
        assertEquals(rows[2], denseBlock.getNumRows());
        assertEquals(cols[2], denseBlock.getNumColumns());
        assertEquals(0, denseBlock.getNonZeros());
        denseArray = denseBlock.getDenseArray();
      /*  assertEquals((rows[2] * cols[2]), denseArray.length);
        for(int i = 0; i < denseArray.length; i++) {
            assertEquals("value at position " + i +" is not 0", 0, denseArray[i], 0);
        }*/
        
        MatrixBlock1D sparseBlock = getMatrixBlock(rows[0], cols[0], true, 0.1);
        sparseBlock.reset(rows[1], cols[1]);
        assertEquals(rows[1], sparseBlock.getNumRows());
        assertEquals(cols[1], sparseBlock.getNumColumns());
        assertEquals(0, sparseBlock.getNonZeros());
        HashMap<CellIndex, Double> sparseMap = sparseBlock.getSparseMap();
        assertEquals(0, sparseMap.size());
        
        sparseBlock.reset(rows[2], cols[2]);
        assertEquals(rows[2], sparseBlock.getNumRows());
        assertEquals(cols[2], sparseBlock.getNumColumns());
        assertEquals(0, sparseBlock.getNonZeros());
        sparseMap = sparseBlock.getSparseMap();
        assertEquals(0, sparseMap.size());
    }

    @Test
    public void testResetIntIntBoolean() {
        int[] rows = new int[] { 10, 20 };
        int[] cols = new int[] { 10, 20 };
        
        MatrixBlock1D denseBlock = getMatrixBlock(rows[0], cols[0], false, 1);
        denseBlock.reset(rows[1], cols[1], false);
        assertFalse(denseBlock.isInSparseFormat());
        
        denseBlock = getMatrixBlock(rows[0], cols[0], false, 1);
        denseBlock.reset(rows[1], cols[1], true);
        assertTrue(denseBlock.isInSparseFormat());
        
        MatrixBlock1D sparseBlock = getMatrixBlock(rows[0], cols[0], true, 0.1);
        sparseBlock.reset(rows[1], cols[1], true);
        assertTrue(sparseBlock.isInSparseFormat());
        
        sparseBlock = getMatrixBlock(rows[0], cols[0], true, 0.1);
        sparseBlock.reset(rows[1], cols[1], false);
        assertFalse(sparseBlock.isInSparseFormat());
    }

    @Test
    public void testResetDenseWithValue() {
        int[] rows = new int[] { 10, 20 };
        int[] cols = new int[] { 10, 20 };
        double newValue = 1.5;
        
        MatrixBlock1D denseBlock = getMatrixBlock(rows[0], cols[0], false, 1);
        denseBlock.resetDenseWithValue(rows[1], cols[1], newValue);
        assertFalse(denseBlock.isInSparseFormat());
        assertEquals((rows[1] * cols[1]), denseBlock.getNonZeros());
        double[] denseArray = denseBlock.getDenseArray();
        assertEquals((rows[1] * cols[1]), denseArray.length);
        for(int i = 0; i < denseArray.length; i++) {
            assertEquals("value at position " + i + " is not " + newValue, newValue, denseArray[i], 0);
        }
        
        MatrixBlock1D sparseBlock = getMatrixBlock(rows[0], cols[0], true, 0.1);
        sparseBlock.resetDenseWithValue(rows[1], cols[1], newValue);
        assertFalse(sparseBlock.isInSparseFormat());
        assertEquals((rows[1] * cols[1]), sparseBlock.getNonZeros());
        assertNull(sparseBlock.getSparseMap());
        denseArray = sparseBlock.getDenseArray();
        assertEquals((rows[1] * cols[1]), denseArray.length);
        for(int i = 0; i < denseArray.length; i++) {
            assertEquals("value at position " + i + " is not " + newValue, newValue, denseArray[i], 0);
        }
    }

    @Test
    public void testCopy() {
        int[] rows = new int[] { 10, 20, 30, 40, 50 };
        int[] cols = new int[] { 10, 20, 30, 40, 50 };
        
        MatrixBlock1D block = getMatrixBlock(rows[0], cols[0], false, 1);
        MatrixBlock1D blockToCopy = getMatrixBlock(rows[1], cols[1], false, 1);
        block.copy(blockToCopy);
        assertEquals(rows[1], block.getNumRows());
        assertEquals(cols[1], block.getNumColumns());
        assertFalse(block.isInSparseFormat());
        assertEquals(blockToCopy.getNonZeros(), block.getNonZeros());
        double[] originalDenseArray = blockToCopy.getDenseArray();
        double[] denseArray = block.getDenseArray();
        assertEquals(originalDenseArray.length, denseArray.length);
        for(int i = 0; i < denseArray.length; i++) {
            assertEquals("value at position " + i + " is not equal", originalDenseArray[i], denseArray[i], 0);
        }
        
        blockToCopy = getMatrixBlock(rows[2], cols[2], true, 0.01);
        block.copy(blockToCopy);
        assertEquals(rows[2], block.getNumRows());
        assertEquals(cols[2], block.getNumColumns());
        assertTrue(block.isInSparseFormat());
        assertEquals(blockToCopy.getNonZeros(), block.getNonZeros());
        HashMap<CellIndex, Double> originalSparseMap = blockToCopy.getSparseMap();
        HashMap<CellIndex, Double> sparseMap = block.getSparseMap();
        assertEquals(originalSparseMap.size(), sparseMap.size());
        for(CellIndex index : originalSparseMap.keySet()) {
            assertTrue("no value for " + index.row + "," + index.column, sparseMap.containsKey(index));
            assertEquals("different value for " + index.row + "," + index.column,
                    originalSparseMap.get(index), sparseMap.get(index));
        }
        
        blockToCopy = getMatrixBlock(rows[3], cols[3], true, 0.02);
        block.copy(blockToCopy);
        assertEquals(rows[3], block.getNumRows());
        assertEquals(cols[3], block.getNumColumns());
        assertTrue(block.isInSparseFormat());
        assertEquals(blockToCopy.getNonZeros(), block.getNonZeros());
        originalSparseMap = blockToCopy.getSparseMap();
        sparseMap = block.getSparseMap();
        assertEquals(originalSparseMap.size(), sparseMap.size());
        for(CellIndex index : originalSparseMap.keySet()) {
            assertTrue("no value for " + index.row + "," + index.column, sparseMap.containsKey(index));
            assertEquals("different value for " + index.row + "," + index.column,
                    originalSparseMap.get(index), sparseMap.get(index));
        }
        
        blockToCopy = getMatrixBlock(rows[4], cols[4], false, 1);
        block.copy(blockToCopy);
        assertEquals(rows[4], block.getNumRows());
        assertEquals(cols[4], block.getNumColumns());
        assertFalse(block.isInSparseFormat());
        assertEquals(blockToCopy.getNonZeros(), block.getNonZeros());
        originalDenseArray = blockToCopy.getDenseArray();
        denseArray = block.getDenseArray();
        assertEquals(originalDenseArray.length, denseArray.length);
        for(int i = 0; i < denseArray.length; i++) {
            assertEquals("value at position " + i + " is not equal", originalDenseArray[i], denseArray[i], 0);
        }
    }

    @Test
    public void testSparseScalarOperationsInPlace() {
        int rows = 10;
        int cols = 10;
        
        MatrixBlock1D denseBlock = getMatrixBlock(rows, cols, false, 1);
        double[] originalDenseArray = denseBlock.getDenseArray().clone();
        try {
            //denseBlock.sparseScalarOperationsInPlace(SupportedOperation.SCALAR_ADDITION, 1);
            denseBlock.sparseScalarOperationsInPlace(new RightScalarOperator(Plus.getPlusFnObject(), 1));
        } catch(DMLRuntimeException e) {
            fail("failed to perform scalar operation");
        } catch(DMLUnsupportedOperationException e) {
            fail("failed to perform scalar operation");
        } 
        double[] denseArray = denseBlock.getDenseArray();
        for(int i = 0; i < originalDenseArray.length; i++) {
            if(originalDenseArray[i] == 0)
                assertEquals(0, denseArray[i], 0);
            else
                assertEquals((originalDenseArray[i] + 1), denseArray[i], 0);
        }
        
        MatrixBlock1D sparseBlock = getMatrixBlock(rows, cols, true, 0.05);
        HashMap<CellIndex, Double> originalSparseMap = new HashMap<CellIndex, Double>(sparseBlock.getSparseMap());
        try {
            //sparseBlock.sparseScalarOperationsInPlace(SupportedOperation.SCALAR_ADDITION, 1);
            sparseBlock.sparseScalarOperationsInPlace(new RightScalarOperator(Plus.getPlusFnObject(), 1));
        } catch(DMLRuntimeException e) {
            fail("failed to perform scalar operation");
        } catch(DMLUnsupportedOperationException e) {
            fail("failed to perform scalar operation");
        } 
        HashMap<CellIndex, Double> sparseMap = sparseBlock.getSparseMap();
        int expectedElements = 0;
        for(CellIndex index : originalSparseMap.keySet()) {
            if(originalSparseMap.get(index) == 0) {
                assertFalse(sparseMap.containsKey(index));
            } else {
                expectedElements++;
                assertTrue(sparseMap.containsKey(index));
                assertEquals((originalSparseMap.get(index) + 1), sparseMap.get(index), 0);
            }
        }
        assertEquals(expectedElements, sparseMap.size());
    }

    @Test
    public void testDenseScalarOperationsInPlace() {
        int rows = 10;
        int cols = 10;
        
        MatrixBlock1D denseBlock = getMatrixBlock(rows, cols, false, 1);
        double[] originalDenseArray = denseBlock.getDenseArray().clone();
        try {
            //denseBlock.denseScalarOperationsInPlace(SupportedOperation.SCALAR_ADDITION, 1);
        	denseBlock.denseScalarOperationsInPlace(new RightScalarOperator(Plus.getPlusFnObject(), 1));
        } catch(DMLRuntimeException e) {
            fail("failed to perform scalar operation");
        } catch(DMLUnsupportedOperationException e) {
            fail("failed to perform scalar operation");
        } 
        double[] denseArray = denseBlock.getDenseArray();
        for(int i = 0; i < originalDenseArray.length; i++) {
            assertEquals((originalDenseArray[i] + 1), denseArray[i], 0);
        }
        
        MatrixBlock1D sparseBlock = getMatrixBlock(rows, cols, true, 0.05);
        HashMap<CellIndex, Double> originalSparseMap = new HashMap<CellIndex, Double>(sparseBlock.getSparseMap());
        try {
            //sparseBlock.denseScalarOperationsInPlace(SupportedOperation.SCALAR_ADDITION, 1);
        	sparseBlock.denseScalarOperationsInPlace(new RightScalarOperator(Plus.getPlusFnObject(), 1));
        } catch(DMLRuntimeException e) {
            fail("failed to perform scalar operation");
        } catch(DMLUnsupportedOperationException e) {
            fail("failed to perform scalar operation");
        } 
        HashMap<CellIndex, Double> sparseMap = sparseBlock.getSparseMap();
        int expectedElements = 0;
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                CellIndex index = new CellIndex(i, j);
                if(originalSparseMap.containsKey(index)) {
                    if(originalSparseMap.get(index) + 1 == 0) {
                        assertFalse(sparseMap.containsKey(index));
                    } else {
                        expectedElements++;
                        assertTrue(sparseMap.containsKey(index));
                        assertEquals((originalSparseMap.get(index) + 1), sparseMap.get(index), 0);
                    }
                } else {
                    expectedElements++;
                    assertTrue(sparseMap.containsKey(index));
                    assertEquals(1, sparseMap.get(index), 0);
                }
            }
        }
        assertEquals(expectedElements, sparseMap.size());
    }

    @Test
    public void testScalarOperations() {
    	int rows = 10;
        int cols = 10;
        
        MatrixBlock1D denseBlock = getMatrixBlock(rows, cols, false, 1);
        MatrixBlock1D resultBlock = new MatrixBlock1D();
        double[] originalDenseArray = denseBlock.getDenseArray().clone();
        try {
        	// denseBlock.scalarOperations(SupportedOperation.SCALAR_ADDITION, 1, resultBlock, false);
        	denseBlock.scalarOperations(new RightScalarOperator(Plus.getPlusFnObject(), 1), resultBlock);
        } catch(DMLRuntimeException e) {
            fail("failed to perform scalar operation");
        } catch(DMLUnsupportedOperationException e) {
            fail("failed to perform scalar operation");
        } 
        double[] denseArray = resultBlock.getDenseArray();
        for(int i = 0; i < originalDenseArray.length; i++) {
            if(originalDenseArray[i] == 0)
                assertEquals(0, denseArray[i], 0);
            else
                assertEquals((originalDenseArray[i] + 1), denseArray[i], 0);
        }
        
        MatrixBlock1D sparseBlock = getMatrixBlock(rows, cols, true, 0.05);
        resultBlock = new MatrixBlock1D();
        HashMap<CellIndex, Double> originalSparseMap = new HashMap<CellIndex, Double>(sparseBlock.getSparseMap());
        try {
        	// sparseBlock.scalarOperations(SupportedOperation.SCALAR_ADDITION, 1, resultBlock, true);
            // sparseBlock.sparseScalarOperationsInPlace(SupportedOperation.SCALAR_ADDITION, 1);
        	sparseBlock.scalarOperations(new RightScalarOperator(Plus.getPlusFnObject(), 1), resultBlock);
        	sparseBlock.sparseScalarOperationsInPlace(new RightScalarOperator(Plus.getPlusFnObject(), 1));

        } catch(DMLRuntimeException e) {
            fail("failed to perform scalar operation");
        } catch(DMLUnsupportedOperationException e) {
            fail("failed to perform scalar operation");
        } 
        HashMap<CellIndex, Double> sparseMap = resultBlock.getSparseMap();
        int expectedElements = 0;
        for(CellIndex index : originalSparseMap.keySet()) {
            if(originalSparseMap.get(index) == 0) {
                assertFalse(sparseMap.containsKey(index));
            } else {
                expectedElements++;
                assertTrue(sparseMap.containsKey(index));
                assertEquals((originalSparseMap.get(index) + 1), sparseMap.get(index), 0);
            }
        }
     //   assertEquals(expectedElements, sparseMap.size());
    }

    /*    @Test
    public void testExamSparsity() {
        int rows = 10;
        int cols = 10;
        
        MatrixBlock1D denseBlock = getMatrixBlock(rows, cols, false, 1);
        denseBlock.examSparsity();
        assertFalse(denseBlock.isInSparseFormat());
        
        denseBlock = getMatrixBlock(rows, cols, false, 0.01);
        denseBlock.examSparsity();
        assertTrue(denseBlock.isInSparseFormat());
        
        MatrixBlock1D sparseBlock = getMatrixBlock(rows, cols, true, 0.01);
        sparseBlock.examSparsity();
        assertTrue(sparseBlock.isInSparseFormat());
        
        sparseBlock = getMatrixBlock(rows, cols, true, 1);
        sparseBlock.examSparsity();
        assertFalse(sparseBlock.isInSparseFormat());
    }*/

    @Test
    public void testDenseToSparse() {
        int rows = 10;
        int cols = 10;
        
        MatrixBlock1D denseBlock = getMatrixBlock(rows, cols, false, 1);
        double[] denseArray = denseBlock.getDenseArray();
        denseBlock.denseToSparse();
        assertTrue(denseBlock.isInSparseFormat());
        HashMap<CellIndex, Double> sparseMap = denseBlock.getSparseMap();
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                CellIndex index = new CellIndex(i, j);
                if(denseArray[i * cols + j] == 0) {
                    assertFalse(sparseMap.containsKey(index));
                } else {
                    assertTrue(sparseMap.containsKey(index));
                    assertEquals(denseArray[i * cols + j], sparseMap.get(index), 0);
                }
            }
        }
    }

    @Test
    public void testSparseToDense() {
        int rows = 10;
        int cols = 10;
        
        MatrixBlock1D sparseBlock = getMatrixBlock(rows, cols, true, 0.1);
        HashMap<CellIndex, Double> sparseMap = sparseBlock.getSparseMap();
        sparseBlock.sparseToDense();
        assertFalse(sparseBlock.isInSparseFormat());
        double[] denseArray = sparseBlock.getDenseArray();
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                CellIndex index = new CellIndex(i, j);
                if(!sparseMap.containsKey(index)) {
                    assertEquals(0, denseArray[i * cols + j], 0);
                } else {
                    assertEquals(sparseMap.get(index), denseArray[i * cols + j], 0);
                }
            }
        }
    }

    @Test
    public void testReadFieldsDense() {
    	int rows = 10;
    	int cols = 10;
    	boolean sparse = false;
    	double[][] values = new double[rows][cols];
    	MatrixBlock1D block = null;
    	Random random = new Random(System.currentTimeMillis());
    	int nonZeros = 0;
    	
    	try {
	    	ByteArrayOutputStream bos = new ByteArrayOutputStream();
	    	DataOutputStream dos = new DataOutputStream(bos);
	    	dos.writeInt(rows);
	    	dos.writeInt(cols);
	    	dos.writeBoolean(sparse);
	    	for(int i = 0; i < rows; i++) {
	    		for(int j = 0; j < cols; j++) {
	    			values[i][j] = random.nextDouble();
	    			if(values[i][j] != 0)
	    				nonZeros++;
	    			dos.writeDouble(values[i][j]);
	    		}
	    	}
	    	
	    	block = new MatrixBlock1D();
	    	block.readFields(new DataInputStream(new ByteArrayInputStream(bos.toByteArray())));
    	} catch(IOException e) {
    		e.printStackTrace();
    		fail("unable to read dense matrix from input stream: " + e.getMessage());
    	}
    	
    	assertFalse(block.isInSparseFormat());
    	assertEquals(rows, block.getNumRows());
    	assertEquals(cols, block.getNumColumns());
    	assertEquals(nonZeros, block.getNonZeros());
    	double[] denseArray = block.getDenseArray();
    	for(int i = 0; i < rows; i++) {
    		for(int j = 0; j < cols; j++) {
    			assertEquals("different value for " + i + "," + j, values[i][j], denseArray[i * cols + j], 0);
    		}
    	}
    }
    
    @Test
    public void testReadFieldsSparse() {
    	int rows = 10;
    	int cols = 10;
    	boolean sparse = true;
    	Random random = new Random(System.currentTimeMillis());
    	HashMap<CellIndex, Double> values = new HashMap<CellIndex, Double>();
    	int nonZeros = 0;
    	for(int i = 0; i < rows; i++) {
    		for(int j = 0; j < cols; j++) {
    			if(random.nextDouble() > 0.1)
    				continue;
    			values.put(new CellIndex(i, j), random.nextDouble());
    			nonZeros++;
    		}
    	}
    	MatrixBlock1D block = null;
    	
    	try {
	    	ByteArrayOutputStream bos = new ByteArrayOutputStream();
	    	DataOutputStream dos = new DataOutputStream(bos);
	    	dos.writeInt(rows);
	    	dos.writeInt(cols);
	    	dos.writeBoolean(sparse);
	    	dos.writeInt(nonZeros);
	    	for(CellIndex index : values.keySet()) {
	    		dos.writeInt(index.row);
	    		dos.writeInt(index.column);
	    		dos.writeDouble(values.get(index));
	    	}
	    	
	    	block = new MatrixBlock1D();
	    	block.readFields(new DataInputStream(new ByteArrayInputStream(bos.toByteArray())));
    	} catch(IOException e) {
    		e.printStackTrace();
    		fail("unable to read sparse matrix from input stream: " + e.getMessage());
    	}
    	
    	assertTrue(block.isInSparseFormat());
    	assertEquals(rows, block.getNumRows());
    	assertEquals(cols, block.getNumColumns());
    	assertEquals(nonZeros, block.getNonZeros());
    	HashMap<CellIndex, Double> sparseMap = block.getSparseMap();
    	assertEquals(nonZeros, sparseMap.size());
    	for(CellIndex index : sparseMap.keySet()) {
    		assertEquals("different value for " + index.row + "," + index.column, values.get(index),
    				sparseMap.get(index), 0);
    	}
    }

    @Test
    public void testWriteDense() {
    	int rows = 10;
    	int cols = 10;
    	MatrixBlock1D block = new MatrixBlock1D(rows, cols, false);
    	Random random = new Random(System.currentTimeMillis());
    	for(int i = 0; i < rows; i++) {
    		for(int j = 0; j < cols; j++) {
    			block.setValue(i, j, random.nextDouble());
    		}
    	}
    	
    	ByteArrayOutputStream bos = new ByteArrayOutputStream();
    	try {
			block.write(new DataOutputStream(bos));

			DataInputStream dis = new DataInputStream(new ByteArrayInputStream(bos.toByteArray()));
			assertEquals(rows, dis.readInt());
			assertEquals(cols, dis.readInt());
			assertEquals(false, dis.readBoolean());
			double[] values = block.getDenseArray();
			for(int i = 0; i < (rows * cols); i++) {
				assertEquals("different value for " + i, values[i], dis.readDouble(), 0);
			}
		} catch (IOException e) {
			e.printStackTrace();
			fail("unable to write dense matrix to output stream: " + e.getMessage());
		}
    }
    
    @Test
    public void testWriteSparse() {
    	int rows = 10;
    	int cols = 10;
    	MatrixBlock1D block = new MatrixBlock1D(rows, cols, true);
    	Random random = new Random(System.currentTimeMillis());
    	int nonZeros = 0;
    	for(int i = 0; i < rows; i++) {
    		for(int j = 0; j < cols; j++) {
    			if(random.nextDouble() > 0.05)
    				continue;
				block.setValue(i, j, random.nextDouble());
				nonZeros++;
    		}
    	}
    	
    	ByteArrayOutputStream bos = new ByteArrayOutputStream();
    	try {
			block.write(new DataOutputStream(bos));

			DataInputStream dis = new DataInputStream(new ByteArrayInputStream(bos.toByteArray()));
			System.out.println(block.getNonZeros());
			assertEquals(rows, dis.readInt());
			assertEquals(cols, dis.readInt());
			assertEquals(true, dis.readBoolean());
			assertEquals(nonZeros, dis.readInt());
			HashMap<CellIndex, Double> sparseMap = block.getSparseMap();
			for(int i = 0; i < nonZeros; i++) {
				int row = dis.readInt();
				int col = dis.readInt();
				assertEquals("different value for " + row + "," + col, sparseMap.get(new CellIndex(row, col)),
						dis.readDouble(), 0);
			}
		} catch (IOException e) {
			e.printStackTrace();
			fail("unable to write sparse matrix to output stream: " + e.getMessage());
		}
    }
    
    public MatrixBlock1D getMatrixBlock(int rows, int cols, boolean sparseFormat, double sparsity) {
        MatrixBlock1D block = new MatrixBlock1D(rows, cols, sparseFormat);
        Random random = new Random(System.currentTimeMillis());
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                if(random.nextDouble() > sparsity)
                    continue;
                block.setValue(i, j, random.nextDouble());
            }
        }
        return block;
    }

}
