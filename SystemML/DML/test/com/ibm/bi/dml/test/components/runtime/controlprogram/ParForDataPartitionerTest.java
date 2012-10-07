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
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.test.utils.TestUtils;


public class ParForDataPartitionerTest 
{
	private int _brlen = 1000;
	private int _bclen = 1000;
	private int _rows = 1200;
	private int _cols = 1500;
	private double _sparsity1 = 0.7d;
	private double _sparsity2 = 0.07d;
	private String _fname = "./scratch_space/A3";
	
	
	@Test
	public void testLocalRowWisePartitioningTextCellDense() 
	{
		testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.ROW_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testLocalColWisePartitioningTextCellDense() 
	{
		testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.COLUMN_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testLocalRowBlockWisePartitioningTextCellDense() 
	{
		testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.ROW_BLOCK_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testLocalColBlockWisePartitioningTextCellDense() 
	{
		testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.COLUMN_BLOCK_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testLocalRowWisePartitioningBinaryCellDense() 
	{
		testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.ROW_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testLocalColWisePartitioningBinaryCellDense() 
	{
		testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.COLUMN_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testLocalRowBlockWisePartitioningBinaryCellDense() 
	{
		testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.ROW_BLOCK_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testLocalColBlockWisePartitioningBinaryCellDense() 
	{
		testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.COLUMN_BLOCK_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, _sparsity1 );
	}

	
	@Test
	public void testLocalRowWisePartitioningBinaryBlockDense() 
	{
		testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.ROW_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testLocalColWisePartitioningBinaryBlockDense() 
	{
		testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.COLUMN_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testLocalRowBlockWisePartitioningBinaryBlockDense() 
	{
		testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.ROW_BLOCK_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testLocalColBlockWisePartitioningBinaryBlockDense() 
	{ 
		testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.COLUMN_BLOCK_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testRemoteRowWisePartitioningTextCellDense() 
	{
		testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testRemoteColWisePartitioningTextCellDense() 
	{
		testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testRemoteRowBlockWisePartitioningTextCellDense() 
	{
		testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_BLOCK_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testRemoteColBlockWisePartitioningTextCellDense() 
	{
		testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_BLOCK_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, _sparsity1 );
	}
	
	
	@Test
	public void testRemoteRowWisePartitioningBinaryCellDense() 
	{
		testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testRemoteColWisePartitioningBinaryCellDense() 
	{
		testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testRemoteRowBlockWisePartitioningBinaryCellDense() 
	{
		testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_BLOCK_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testRemoteColBlockWisePartitioningBinaryCellDense() 
	{
		testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_BLOCK_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, _sparsity1 );
	}

	@Test
	public void testRemoteRowWisePartitioningBinaryBlockDense() 
	{
		testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testRemoteColWisePartitioningBinaryBlockDense() 
	{
		testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testRemoteRowBlockWisePartitioningBinaryBlockDense() 
	{
		testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_BLOCK_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity1 );
	}
	
	@Test
	public void testRemoteColBlockWisePartitioningBinaryBlockDense() 
	{ 
		testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_BLOCK_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity1 );
	}

	@Test
	public void testLocalRowWisePartitioningTextCellSparse() 
	{
		testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.ROW_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testLocalColWisePartitioningTextCellSparse() 
	{
		testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.COLUMN_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testLocalRowBlockWisePartitioningTextCellSparse() 
	{
		testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.ROW_BLOCK_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testLocalColBlockWisePartitioningTextCellSparse() 
	{
		testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.COLUMN_BLOCK_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testLocalRowWisePartitioningBinaryCellSparse() 
	{
		testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.ROW_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testLocalColWisePartitioningBinaryCellSparse() 
	{
		testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.COLUMN_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testLocalRowBlockWisePartitioningBinaryCellSparse() 
	{
		testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.ROW_BLOCK_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testLocalColBlockWisePartitioningBinaryCellSparse() 
	{
		testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.COLUMN_BLOCK_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, _sparsity2 );
	}

	
	@Test
	public void testLocalRowWisePartitioningBinaryBlockSparse() 
	{
		testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.ROW_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testLocalColWisePartitioningBinaryBlockSparse() 
	{
		testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.COLUMN_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testLocalRowBlockWisePartitioningBinaryBlockSparse() 
	{
		testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.ROW_BLOCK_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testLocalColBlockWisePartitioningBinaryBlockSparse() 
	{ 
		testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.COLUMN_BLOCK_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testRemoteRowWisePartitioningTextCellSparse() 
	{
		testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testRemoteColWisePartitioningTextCellSparse() 
	{
		testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testRemoteRowBlockWisePartitioningTextCellSparse() 
	{
		testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_BLOCK_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testRemoteColBlockWisePartitioningTextCellSparse() 
	{
		testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_BLOCK_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, _sparsity2 );
	}
	
	
	@Test
	public void testRemoteRowWisePartitioningBinaryCellSparse() 
	{
		testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testRemoteColWisePartitioningBinaryCellSparse() 
	{
		testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testRemoteRowBlockWisePartitioningBinaryCellSparse() 
	{
		testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_BLOCK_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testRemoteColBlockWisePartitioningBinaryCellSparse() 
	{
		testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_BLOCK_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, _sparsity2 );
	}

	@Test
	public void testRemoteRowWisePartitioningBinaryBlockSparse() 
	{
		testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testRemoteColWisePartitioningBinaryBlockSparse() 
	{
		testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testRemoteRowBlockWisePartitioningBinaryBlockSparse() 
	{
		testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_BLOCK_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity2 );
	}
	
	@Test
	public void testRemoteColBlockWisePartitioningBinaryBlockSparse() 
	{ 
		testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_BLOCK_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, _sparsity2 );
	}
	
	
	private void testMatrixDataPartitioning( PDataPartitioner dp, PDataPartitionFormat format, InputInfo ii, OutputInfo oi, double sparsity )
	{
		double[][] matrix = TestUtils.generateTestMatrix(_rows, _cols, 0, 1, sparsity, 7);
		double[][] matrix2 = null;
		
		try 
		{
			//create and write input
			MatrixBlock mb1 = DataConverter.convertToMatrixBlock(matrix);
			
			MatrixCharacteristics mc = new MatrixCharacteristics(_rows, _cols, _brlen, _bclen);
			MatrixFormatMetaData meta = new MatrixFormatMetaData(mc, oi, ii);
			DataConverter.writeMatrixToHDFS(mb1, _fname, oi, _rows, _cols, _brlen, _bclen);		
			MatrixObject mo1 = new MatrixObject(ValueType.DOUBLE,_fname);
			mo1.setMetaData(meta);
			
			DataPartitioner dpart = null;
			if( dp == PDataPartitioner.LOCAL )
				dpart = new DataPartitionerLocal(format);
			else if( dp == PDataPartitioner.REMOTE_MR )
				dpart = new DataPartitionerRemoteMR(format, 7, 4, 4, 3, 1, true);
			
			MatrixObject mo2 = dpart.createPartitionedMatrixObject(mo1, true);	
			ii = ((MatrixFormatMetaData)mo2.getMetaData()).getInputInfo();
			matrix2 = readPartitionedMatrix(format, mo2.getFileName(),ii, _rows, _cols, _brlen, _bclen);
			
			//cleanup
			MapReduceTool.deleteFileIfExistOnHDFS(_fname);
			MapReduceTool.deleteFileIfExistOnHDFS(mo2.getFileName());
			DataPartitionerLocal.cleanupWorkingDirectory(true);
		} 
		catch (Exception e) 
		{
			e.printStackTrace();
		}

		//compare
		for( int i=0; i<_rows; i++ )
			for( int j=0; j<_cols; j++ )
				if( matrix[i][j]!=matrix2[i][j] )
					Assert.fail("Wrong value i="+i+", j="+j+", value1="+matrix[i][j]+", value2="+matrix2[i][j]);
	}
	
	private double[][] readPartitionedMatrix(PDataPartitionFormat dpf, String fname, InputInfo ii, int rows, int cols, int brlen, int bclen) 
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
				
		}
		
		return matrix;
	}
}
