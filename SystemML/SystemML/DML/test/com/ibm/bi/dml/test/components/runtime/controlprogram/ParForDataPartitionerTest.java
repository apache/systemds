/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.components.runtime.controlprogram;

import java.io.IOException;

import org.junit.Assert;
import org.junit.Test;

import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitioner;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.parfor.DataPartitioner;
import com.ibm.bi.dml.runtime.controlprogram.parfor.DataPartitionerLocal;
import com.ibm.bi.dml.runtime.controlprogram.parfor.DataPartitionerRemoteMR;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.util.DataConverter;
import com.ibm.bi.dml.runtime.util.LocalFileUtils;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.test.utils.TestUtils;


public class ParForDataPartitionerTest 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private int _brlen = 1000;
	private int _bclen = 1000;
	private int _rows = 1200;
	private int _cols = 1500;
	private int _rows2 = 3000; //for test opt BLOCKWISE_N
	private int _cols2 = 4000; //for test opt BLOCKWISE_N
	private int _n = 2000;
	
	private double _sparsity1 = 0.7d;
	private double _sparsity2 = 0.07d;
	private String _fname = "./scratch_space/A3";
	
	//internal switches
	private boolean _runLocal = true;
	private boolean _runRemote = true;
		
	@Test
	public void testLocalRowWisePartitioningTextCellDense() 
	{
		if( _runLocal )
			testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.ROW_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testLocalColWisePartitioningTextCellDense() 
	{
		if( _runLocal )
			testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.COLUMN_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testLocalRowBlockWisePartitioningTextCellDense() 
	{
		if( _runLocal )
			testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.ROW_BLOCK_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testLocalColBlockWisePartitioningTextCellDense() 
	{
		if( _runLocal )
			testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.COLUMN_BLOCK_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testLocalRowWisePartitioningBinaryCellDense() 
	{
		if( _runLocal )
			testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.ROW_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testLocalColWisePartitioningBinaryCellDense() 
	{
		if( _runLocal )
			testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.COLUMN_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testLocalRowBlockWisePartitioningBinaryCellDense() 
	{
		if( _runLocal )
			testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.ROW_BLOCK_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testLocalColBlockWisePartitioningBinaryCellDense() 
	{
		if( _runLocal )
			testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.COLUMN_BLOCK_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, _sparsity1 );
	}

	
	@Test
	public void testLocalRowWisePartitioningBinaryBlockDense() 
	{
		if( _runLocal )
			testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.ROW_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testLocalColWisePartitioningBinaryBlockDense() 
	{
		if( _runLocal )
			testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.COLUMN_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testLocalRowBlockWisePartitioningBinaryBlockDense() 
	{
		if( _runLocal )
			testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.ROW_BLOCK_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testLocalColBlockWisePartitioningBinaryBlockDense() 
	{ 
		if( _runLocal )
			testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.COLUMN_BLOCK_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testRemoteRowWisePartitioningTextCellDense() 
	{
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testRemoteColWisePartitioningTextCellDense() 
	{
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testRemoteRowBlockWisePartitioningTextCellDense() 
	{
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_BLOCK_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testRemoteColBlockWisePartitioningTextCellDense() 
	{
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_BLOCK_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, _sparsity1 );
	}
	
	
	@Test
	public void testRemoteRowWisePartitioningBinaryCellDense() 
	{
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testRemoteColWisePartitioningBinaryCellDense() 
	{
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testRemoteRowBlockWisePartitioningBinaryCellDense() 
	{
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_BLOCK_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testRemoteColBlockWisePartitioningBinaryCellDense() 
	{
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_BLOCK_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, _sparsity1 );
	}

	@Test
	public void testRemoteRowWisePartitioningBinaryBlockDense() 
	{
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testRemoteColWisePartitioningBinaryBlockDense() 
	{
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testRemoteRowBlockWisePartitioningBinaryBlockDense() 
	{
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_BLOCK_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testRemoteColBlockWisePartitioningBinaryBlockDense() 
	{ 
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_BLOCK_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity1 );
	}

	@Test
	public void testLocalRowWisePartitioningTextCellSparse() 
	{
		if( _runLocal )
			testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.ROW_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testLocalColWisePartitioningTextCellSparse() 
	{
		if( _runLocal )
			testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.COLUMN_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testLocalRowBlockWisePartitioningTextCellSparse() 
	{
		if( _runLocal )
			testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.ROW_BLOCK_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testLocalColBlockWisePartitioningTextCellSparse() 
	{
		if( _runLocal )
			testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.COLUMN_BLOCK_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testLocalRowWisePartitioningBinaryCellSparse() 
	{
		if( _runLocal )
			testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.ROW_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testLocalColWisePartitioningBinaryCellSparse() 
	{
		if( _runLocal )
			testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.COLUMN_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testLocalRowBlockWisePartitioningBinaryCellSparse() 
	{
		if( _runLocal )
			testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.ROW_BLOCK_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testLocalColBlockWisePartitioningBinaryCellSparse() 
	{
		if( _runLocal )
			testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.COLUMN_BLOCK_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, _sparsity2 );
	}

	
	@Test
	public void testLocalRowWisePartitioningBinaryBlockSparse() 
	{
		if( _runLocal )
			testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.ROW_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testLocalColWisePartitioningBinaryBlockSparse() 
	{
		if( _runLocal )
			testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.COLUMN_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testLocalRowBlockWisePartitioningBinaryBlockSparse() 
	{
		if( _runLocal )
			testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.ROW_BLOCK_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testLocalColBlockWisePartitioningBinaryBlockSparse() 
	{ 
		if( _runLocal )
			testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.COLUMN_BLOCK_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testRemoteRowWisePartitioningTextCellSparse() 
	{
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testRemoteColWisePartitioningTextCellSparse() 
	{
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testRemoteRowBlockWisePartitioningTextCellSparse() 
	{
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_BLOCK_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testRemoteColBlockWisePartitioningTextCellSparse() 
	{
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_BLOCK_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, _sparsity2 );
	}
	
	
	@Test
	public void testRemoteRowWisePartitioningBinaryCellSparse() 
	{
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testRemoteColWisePartitioningBinaryCellSparse() 
	{
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testRemoteRowBlockWisePartitioningBinaryCellSparse() 
	{
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_BLOCK_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testRemoteColBlockWisePartitioningBinaryCellSparse() 
	{
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_BLOCK_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, _sparsity2 );
	}

	@Test
	public void testRemoteRowWisePartitioningBinaryBlockSparse() 
	{
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testRemoteColWisePartitioningBinaryBlockSparse() 
	{
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testRemoteRowBlockWisePartitioningBinaryBlockSparse() 
	{
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_BLOCK_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testRemoteColBlockWisePartitioningBinaryBlockSparse() 
	{ 
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_BLOCK_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity2 );
	}

	@Test
	public void testRemoteRowBlockWiseNPartitioningTextCellDense() 
	{
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_BLOCK_WISE_N, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testRemoteColBlockWiseNPartitioningTextCellDense() 
	{
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_BLOCK_WISE_N, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testRemoteRowBlockWiseNPartitioningBinaryCellDense() 
	{
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_BLOCK_WISE_N, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testRemoteColBlockWiseNPartitioningBinaryCellDense() 
	{
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_BLOCK_WISE_N, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, _sparsity1 );
	}

	@Test
	public void testRemoteRowBlockWiseNPartitioningBinaryBlockDense() 
	{
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_BLOCK_WISE_N, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testRemoteColBlockWiseNPartitioningBinaryBlockDense() 
	{ 
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_BLOCK_WISE_N, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity1 );
	}

	@Test
	public void testRemoteRowBlockWiseNPartitioningBinaryBlockDenseAligned() 
	{
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_BLOCK_WISE_N, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity1, true );
	}
	
	@Test
	public void testRemoteColBlockWiseNPartitioningBinaryBlockDenseAligned() 
	{ 
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_BLOCK_WISE_N, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity1, true );
	}
	

	@Test
	public void testRemoteRowBlockWiseNPartitioningTextCellSparse() 
	{
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_BLOCK_WISE_N, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testRemoteColBlockWiseNPartitioningTextCellSparse() 
	{
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_BLOCK_WISE_N, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testRemoteRowBlockWiseNPartitioningBinaryCellSparse() 
	{
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_BLOCK_WISE_N, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testRemoteColBlockWiseNPartitioningBinaryCellSparse() 
	{
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_BLOCK_WISE_N, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, _sparsity2 );
	}

	@Test
	public void testRemoteRowBlockWiseNPartitioningBinaryBlockSparse() 
	{
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_BLOCK_WISE_N, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testRemoteColBlockWiseNPartitioningBinaryBlockSparse() 
	{ 
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_BLOCK_WISE_N, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity2 );
	}

	@Test
	public void testRemoteRowBlockWiseNPartitioningBinaryBlockSparseAligned() 
	{
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_BLOCK_WISE_N, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity2, true );
	}
	
	@Test
	public void testRemoteColBlockWiseNPartitioningBinaryBlockSparseAligned() 
	{ 
		if( _runRemote )
			testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_BLOCK_WISE_N, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity2, true );
	}
	
	
	
	private void testMatrixDataPartitioning( PDataPartitioner dp, PDataPartitionFormat format, InputInfo ii, OutputInfo oi, double sparsity )
	{
		testMatrixDataPartitioning( dp, format, ii, oi, sparsity, false );
	}	
	
	private void testMatrixDataPartitioning( PDataPartitioner dp, PDataPartitionFormat format, InputInfo ii, OutputInfo oi, double sparsity, boolean testAligned )
	{
		int rows = -1, cols = -1;
		if( testAligned ) {
			rows = _rows2;
			cols = _cols2;
		}
		else {
			rows = _rows;
			cols = _cols;
		}
		
		double[][] matrix = TestUtils.generateTestMatrix(rows, cols, 0, 1, sparsity, 7);
		double[][] matrix2 = null;
		
		try 
		{
			//create and write input
			MatrixBlock mb1 = DataConverter.convertToMatrixBlock(matrix);
			
			MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, _brlen, _bclen);
			MatrixFormatMetaData meta = new MatrixFormatMetaData(mc, oi, ii);
			DataConverter.writeMatrixToHDFS(mb1, _fname, oi, mc);		
			MatrixObject mo1 = new MatrixObject(ValueType.DOUBLE,_fname);
			mo1.setMetaData(meta);
			
			DataPartitioner dpart = null;
			if( dp == PDataPartitioner.LOCAL )
				dpart = new DataPartitionerLocal(format, _n, 4);
			else if( dp == PDataPartitioner.REMOTE_MR )
				dpart = new DataPartitionerRemoteMR(format, _n, 4, 4, 3, 1, false, false);
			
			MatrixObject mo2 = dpart.createPartitionedMatrixObject(mo1, _fname+"_dp", true);	
			ii = ((MatrixFormatMetaData)mo2.getMetaData()).getInputInfo();
			matrix2 = readPartitionedMatrix(format, mo2.getFileName(),ii, rows, cols, _brlen, _bclen, _n);
			
			//cleanup
			MapReduceTool.deleteFileIfExistOnHDFS(_fname);
			MapReduceTool.deleteFileIfExistOnHDFS(mo2.getFileName());
			LocalFileUtils.cleanupWorkingDirectory();
		} 
		catch (Exception e) 
		{
			e.printStackTrace();
		}

		//compare
		for( int i=0; i<rows; i++ )
			for( int j=0; j<cols; j++ )
				if( matrix[i][j]!=matrix2[i][j] )
					Assert.fail("Wrong value i="+i+", j="+j+", value1="+matrix[i][j]+", value2="+matrix2[i][j]);
	}
	
	private double[][] readPartitionedMatrix(PDataPartitionFormat dpf, String fname, InputInfo ii, int rows, int cols, int brlen, int bclen, int n) 
		throws IOException
	{
		double[][] matrix = new double[rows][cols];
		
		switch( dpf )
		{
			case ROW_WISE:
				for( int i=0; i<rows; i++ )
				{
					MatrixBlock mb2 = DataConverter.readMatrixFromHDFS(fname+"/"+(i+1), ii, 1, cols, brlen, bclen);
					double[][] tmp = DataConverter.convertToDoubleMatrix(mb2);
					for( int j=0; j<cols; j++ )
						matrix[i][j] = tmp[0][j];
				}
				break;
			case ROW_BLOCK_WISE:
				for( int i=0; i<rows; i+=brlen )
				{
					MatrixBlock mb2 = DataConverter.readMatrixFromHDFS(fname+"/"+(i/brlen+1), ii, brlen, cols, brlen, bclen);
					double[][] tmp = DataConverter.convertToDoubleMatrix(mb2);
					for( int k=0; k<brlen && i+k<rows; k++ )
						for( int j=0; j<cols; j++ )
							matrix[i+k][j] = tmp[k][j];
				}
				break;	
			case ROW_BLOCK_WISE_N:
				for( int i=0; i<rows; i+=n )
				{
					MatrixBlock mb2 = DataConverter.readMatrixFromHDFS(fname+"/"+(i/n+1), ii, n, cols, brlen, bclen);
					double[][] tmp = DataConverter.convertToDoubleMatrix(mb2);
					for( int k=0; k<n && i+k<rows; k++ )
						for( int j=0; j<cols; j++ )
							matrix[i+k][j] = tmp[k][j];
				}
				break;		
			case COLUMN_WISE:
				for( int j=0; j<cols; j++ )
				{
					MatrixBlock mb2 = DataConverter.readMatrixFromHDFS(fname+"/"+(j+1), ii, rows, 1, brlen, bclen);
					double[][] tmp = DataConverter.convertToDoubleMatrix(mb2);
					for( int i=0; i<rows; i++ )
						matrix[i][j] = tmp[i][0];
				}
				break;
			case COLUMN_BLOCK_WISE:
				for( int j=0; j<cols; j+=bclen )
				{
					MatrixBlock mb2 = DataConverter.readMatrixFromHDFS(fname+"/"+(j/bclen+1), ii, rows, bclen, brlen, bclen);
					double[][] tmp = DataConverter.convertToDoubleMatrix(mb2);
					for( int k=0; k<bclen && j+k<cols; k++ )
						for( int i=0; i<rows; i++ )
							matrix[i][j+k] = tmp[i][k];		
				}
				break;	
			case COLUMN_BLOCK_WISE_N:
				for( int j=0; j<cols; j+=n )
				{
					MatrixBlock mb2 = DataConverter.readMatrixFromHDFS(fname+"/"+(j/n+1), ii, rows, n, brlen, bclen);
					double[][] tmp = DataConverter.convertToDoubleMatrix(mb2);
					for( int k=0; k<n && j+k<cols; k++ )
						for( int i=0; i<rows; i++ )
							matrix[i][j+k] = tmp[i][k];		
				}
				break;	
				
		}
		
		return matrix;
	}
}
