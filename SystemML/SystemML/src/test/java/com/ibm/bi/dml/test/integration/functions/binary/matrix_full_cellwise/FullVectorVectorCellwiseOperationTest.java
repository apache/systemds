/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.binary.matrix_full_cellwise;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

/**
 * TODO cleanup outer(X,Y,z) definition to take two column vectors instead of column and row vector.
 * 
 */
public class FullVectorVectorCellwiseOperationTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME = "FullVectorVectorCellwiseOperation";
	private final static String TEST_DIR = "functions/binary/matrix_full_cellwise/";
	private final static double eps = 1e-10;
	
	private final static int rows1 = 1001;
	private final static int rows2 = 1009;
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.01;
	
	private enum OpType{
		ADDITION,
		SUBSTRACTION,
		MULTIPLICATION,
		LESS_THAN,
	}
	
	private enum SparsityType{
		DENSE,
		SPARSE,
		EMPTY
	}
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_DIR, TEST_NAME,new String[]{"C"}));
	}
	
	@Test
	public void testAdditionDenseDenseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.DENSE, SparsityType.DENSE, ExecType.SPARK);
	}
	
	@Test
	public void testAdditionDenseSparseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.SPARK);
	}
	
	@Test
	public void testAdditionDenseEmptySP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.SPARK);
	}
	
	@Test
	public void testAdditionSparseDenseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.SPARK);
	}
	
	@Test
	public void testAdditionSparseSparseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.SPARK);
	}
	
	@Test
	public void testAdditionSparseEmptySP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.SPARK);
	}
	
	@Test
	public void testAdditionEmptyDenseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.SPARK);
	}
	
	@Test
	public void testAdditionEmptySparseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.SPARK);
	}
	
	@Test
	public void testAdditionEmptyEmptySP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.SPARK);
	}
	
	@Test
	public void testSubstractionDenseDenseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.DENSE, SparsityType.DENSE, ExecType.SPARK);
	}
	
	@Test
	public void testSubstractionDenseSparseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.SPARK);
	}
	
	@Test
	public void testSubstractionDenseEmptySP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.SPARK);
	}
	
	@Test
	public void testSubstractionSparseDenseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.SPARK);
	}
	
	@Test
	public void testSubstractionSparseSparseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.SPARK);
	}
	
	@Test
	public void testSubstractionSparseEmptySP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.SPARK);
	}
	
	@Test
	public void testSubstractionEmptyDenseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.SPARK);
	}
	
	@Test
	public void testSubstractionEmptySparseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.SPARK);
	}
	
	@Test
	public void testSubstractionEmptyEmptySP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.SPARK);
	}
	
	@Test
	public void testMultiplicationDenseDenseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.DENSE, SparsityType.DENSE, ExecType.SPARK);
	}
	
	@Test
	public void testMultiplicationDenseSparseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.SPARK);
	}
	
	@Test
	public void testMultiplicationDenseEmptySP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.SPARK);
	}
	
	@Test
	public void testMultiplicationSparseDenseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.SPARK);
	}
	
	@Test
	public void testMultiplicationSparseSparseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.SPARK);
	}
	
	@Test
	public void testMultiplicationSparseEmptySP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.SPARK);
	}
	
	@Test
	public void testMultiplicationEmptyDenseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.SPARK);
	}
	
	@Test
	public void testMultiplicationEmptySparseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.SPARK);
	}
	
	@Test
	public void testMultiplicationEmptyEmptySP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.SPARK);
	}
	
	@Test
	public void testLessThanDenseDenseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.DENSE, SparsityType.DENSE, ExecType.SPARK);
	}
	
	@Test
	public void testLessThanDenseSparseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.DENSE, SparsityType.SPARSE, ExecType.SPARK);
	}
	
	@Test
	public void testLessThanDenseEmptySP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.DENSE, SparsityType.EMPTY, ExecType.SPARK);
	}
	
	@Test
	public void testLessThanSparseDenseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.SPARSE, SparsityType.DENSE, ExecType.SPARK);
	}
	
	@Test
	public void testLessThanSparseSparseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.SPARK);
	}
	
	@Test
	public void testLessThanSparseEmptySP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.SPARK);
	}
	
	@Test
	public void testLessThanEmptyDenseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.EMPTY, SparsityType.DENSE, ExecType.SPARK);
	}
	
	@Test
	public void testLessThanEmptySparseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.SPARK);
	}
	
	@Test
	public void testLessThanEmptyEmptySP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.SPARK);
	}
	
	// ---------------------------------------------
	@Test
	public void testAdditionDenseDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.DENSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testAdditionDenseSparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testAdditionDenseEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testAdditionSparseDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testAdditionSparseSparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testAdditionSparseEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testAdditionEmptyDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testAdditionEmptySparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testAdditionEmptyEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testAdditionDenseDenseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.DENSE, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testAdditionDenseSparseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testAdditionDenseEmptyMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.MR);
	}
	
	@Test
	public void testAdditionSparseDenseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testAdditionSparseSparseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testAdditionSparseEmptyMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.MR);
	}
	
	@Test
	public void testAdditionEmptyDenseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testAdditionEmptySparseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testAdditionEmptyEmptyMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.MR);
	}	
	
	@Test
	public void testSubstractionDenseDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.DENSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testSubstractionDenseSparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testSubstractionDenseEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testSubstractionSparseDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testSubstractionSparseSparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testSubstractionSparseEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testSubstractionEmptyDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testSubstractionEmptySparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testSubstractionEmptyEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testSubstractionDenseDenseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.DENSE, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testSubstractionDenseSparseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testSubstractionDenseEmptyMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.MR);
	}
	
	@Test
	public void testSubstractionSparseDenseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testSubstractionSparseSparseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testSubstractionSparseEmptyMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.MR);
	}
	
	@Test
	public void testSubstractionEmptyDenseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testSubstractionEmptySparseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testSubstractionEmptyEmptyMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.MR);
	}
	
	@Test
	public void testMultiplicationDenseDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.DENSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationDenseSparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationDenseEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationSparseDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationSparseSparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationSparseEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationEmptyDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationEmptySparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationEmptyEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationDenseDenseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.DENSE, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testMultiplicationDenseSparseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testMultiplicationDenseEmptyMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.MR);
	}
	
	@Test
	public void testMultiplicationSparseDenseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testMultiplicationSparseSparseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testMultiplicationSparseEmptyMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.MR);
	}
	
	@Test
	public void testMultiplicationEmptyDenseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testMultiplicationEmptySparseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testMultiplicationEmptyEmptyMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.MR);
	}
	
	@Test
	public void testLessThanDenseDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.DENSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testLessThanDenseSparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.DENSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testLessThanDenseEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.DENSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testLessThanSparseDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.SPARSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testLessThanSparseSparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testLessThanSparseEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testLessThanEmptyDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.EMPTY, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testLessThanEmptySparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testLessThanEmptyEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testLessThanDenseDenseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.DENSE, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testLessThanDenseSparseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.DENSE, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testLessThanDenseEmptyMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.DENSE, SparsityType.EMPTY, ExecType.MR);
	}
	
	@Test
	public void testLessThanSparseDenseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.SPARSE, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testLessThanSparseSparseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testLessThanSparseEmptyMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.MR);
	}
	
	@Test
	public void testLessThanEmptyDenseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.EMPTY, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testLessThanEmptySparseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testLessThanEmptyEmptyMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.MR);
	}

	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runMatrixVectorCellwiseOperationTest( OpType type, SparsityType sparseM1, SparsityType sparseM2, ExecType instType)
	{
		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		switch( instType ){
		case MR: rtplatform = RUNTIME_PLATFORM.HADOOP; break;
			case SPARK: rtplatform = RUNTIME_PLATFORM.SPARK; break;
			default: rtplatform = RUNTIME_PLATFORM.HYBRID; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
	
		try
		{
			String opcode = null;
			String opcoder = null;
			switch( type )
			{
				case ADDITION: opcode = "+"; opcoder="+";  break;
				case SUBSTRACTION: opcode = "-";  opcoder="-"; break;
				case MULTIPLICATION: opcode="*";  opcoder="mult"; break;
				case LESS_THAN: opcode="<"; opcoder="lt"; break;
			}
			
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain","recompile_runtime","-args", 
											HOME + INPUT_DIR + "A",
					                        HOME + INPUT_DIR + "B",
					                        opcode,
					                        HOME + OUTPUT_DIR + "C"    };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + opcoder + " " + HOME + EXPECTED_DIR;
			
			loadTestConfiguration(config);
	
			//get sparsity
			double lsparsity1 = 1.0, lsparsity2 = 1.0;
			switch( sparseM1 ){
				case DENSE: lsparsity1 = sparsity1; break;
				case SPARSE: lsparsity1 = sparsity2; break;
				case EMPTY: lsparsity1 = 0.0; break;
			}
			switch( sparseM2 ){
				case DENSE: lsparsity2 = sparsity1; break;
				case SPARSE: lsparsity2 = sparsity2; break;
				case EMPTY: lsparsity2 = 0.0; break;
			}
			
			//generate actual dataset
			double[][] A = getRandomMatrix(rows1, 1, 0, (lsparsity1==0)?0:1, lsparsity1, 7); 
			writeInputMatrixWithMTD("A", A, true);
			double[][] B = getRandomMatrix(1, rows2, 0, (lsparsity2==0)?0:1, lsparsity2, 3); 
			writeInputMatrixWithMTD("B", B, true);
	
			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("C");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("C");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			checkDMLMetaDataFile("C", new MatrixCharacteristics(rows1,rows2,1,1));
		}
		finally
		{
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}		
}