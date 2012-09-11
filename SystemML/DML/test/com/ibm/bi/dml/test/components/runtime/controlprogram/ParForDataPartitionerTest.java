package com.ibm.bi.dml.test.components.runtime.controlprogram;

import java.io.IOException;

import org.junit.Assert;
import org.junit.Test;

import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitioner;
import com.ibm.bi.dml.runtime.controlprogram.parfor.DataPartitioner;
import com.ibm.bi.dml.runtime.controlprogram.parfor.DataPartitionerLocal;
import com.ibm.bi.dml.runtime.controlprogram.parfor.DataPartitionerRemoteMR;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.MatrixObjectNew;
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
	private double _sparsity = 0.7d;
	private String _fname = "./scratch_space/A3";
	
	
	@Test
	public void testLocalRowWisePartitioningTextCell() 
	{
		testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.ROW_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo );
	}
	
	@Test
	public void testLocalColWisePartitioningTextCell() 
	{
		testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.COLUMN_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo );
	}
	
	@Test
	public void testLocalRowBlockWisePartitioningTextCell() 
	{
		testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.ROW_BLOCK_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo );
	}
	
	@Test
	public void testLocalColBlockWisePartitioningTextCell() 
	{
		testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.COLUMN_BLOCK_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo );
	}
	
	@Test
	public void testLocalRowWisePartitioningBinaryCell() 
	{
		testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.ROW_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo );
	}
	
	@Test
	public void testLocalColWisePartitioningBinaryCell() 
	{
		testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.COLUMN_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo );
	}
	
	@Test
	public void testLocalRowBlockWisePartitioningBinaryCell() 
	{
		testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.ROW_BLOCK_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo );
	}
	
	@Test
	public void testLocalColBlockWisePartitioningBinaryCell() 
	{
		testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.COLUMN_BLOCK_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo );
	}

	
	@Test
	public void testLocalRowWisePartitioningBinaryBlock() 
	{
		testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.ROW_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo );
	}
	
	@Test
	public void testLocalColWisePartitioningBinaryBlock() 
	{
		testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.COLUMN_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo );
	}
	
	@Test
	public void testLocalRowBlockWisePartitioningBinaryBlock() 
	{
		testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.ROW_BLOCK_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo );
	}
	
	@Test
	public void testLocalColBlockWisePartitioningBinaryBlock() 
	{ 
		testMatrixDataPartitioning( PDataPartitioner.LOCAL, PDataPartitionFormat.COLUMN_BLOCK_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo );
	}
	
	@Test
	public void testRemoteRowWisePartitioningTextCell() 
	{
		testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo );
	}
	
	@Test
	public void testRemoteColWisePartitioningTextCell() 
	{
		testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo );
	}
	
	@Test
	public void testRemoteRowBlockWisePartitioningTextCell() 
	{
		testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_BLOCK_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo );
	}
	
	@Test
	public void testRemoteColBlockWisePartitioningTextCell() 
	{
		testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_BLOCK_WISE, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo );
	}
	
	
	@Test
	public void testRemoteRowWisePartitioningBinaryCell() 
	{
		testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo );
	}
	
	@Test
	public void testRemoteColWisePartitioningBinaryCell() 
	{
		testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo );
	}
	
	@Test
	public void testRemoteRowBlockWisePartitioningBinaryCell() 
	{
		testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_BLOCK_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo );
	}
	
	@Test
	public void testRemoteColBlockWisePartitioningBinaryCell() 
	{
		testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_BLOCK_WISE, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo );
	}

	@Test
	public void testRemoteRowWisePartitioningBinaryBlock() 
	{
		testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo );
	}
	
	@Test
	public void testRemoteColWisePartitioningBinaryBlock() 
	{
		testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo );
	}
	
	@Test
	public void testRemoteRowBlockWisePartitioningBinaryBlock() 
	{
		testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.ROW_BLOCK_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo );
	}
	
	@Test
	public void testRemoteColBlockWisePartitioningBinaryBlock() 
	{ 
		testMatrixDataPartitioning( PDataPartitioner.REMOTE_MR, PDataPartitionFormat.COLUMN_BLOCK_WISE, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo );
	}
	
	
	private void testMatrixDataPartitioning( PDataPartitioner dp, PDataPartitionFormat format, InputInfo ii, OutputInfo oi )
	{
		double[][] matrix = TestUtils.generateTestMatrix(_rows, _cols, 0, 1, _sparsity, 7);
		double[][] matrix2 = null;
		
		try 
		{
			//create and write input
			MatrixBlock mb1 = DataConverter.convertToMatrixBlock(matrix);
			
			MatrixCharacteristics mc = new MatrixCharacteristics(_rows, _cols, _brlen, _bclen);
			MatrixFormatMetaData meta = new MatrixFormatMetaData(mc, oi, ii);
			DataConverter.writeMatrixToHDFS(mb1, _fname, oi, _rows, _cols, _brlen, _bclen);		
			MatrixObjectNew mo1 = new MatrixObjectNew(ValueType.DOUBLE,_fname);
			mo1.setMetaData(meta);
			
			DataPartitioner dpart = null;
			if( dp == PDataPartitioner.LOCAL )
				dpart = new DataPartitionerLocal(format);
			else if( dp == PDataPartitioner.REMOTE_MR )
				dpart = new DataPartitionerRemoteMR(format, 7, 4, 3, 1, true);
			
			MatrixObjectNew mo2 = dpart.createPartitionedMatrixObject(mo1, true);			
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
