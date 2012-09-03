package com.ibm.bi.dml.test.components.runtime.util;

import java.io.IOException;

import org.junit.Assert;
import org.junit.Test;

import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.util.DataConverter;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.test.utils.TestUtils;


public class DataConverterTest 
{
	private int _brlen = 1000;
	private int _bclen = 1000;
	private int _rows = 2500;
	private int _cols = 1500;
	private double _sparsity = 0.7d;
	private String _fname = "./scratch_space/A";
	
	@Test
	public void testReadWriteTextCellFormat() 
	{
		testReadWriteMatrix( InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo );
	}
	
	@Test
	public void testReadWriteBinaryCellFormat() 
	{
		testReadWriteMatrix( InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo );
	}
	
	@Test
	public void testReadWriteBinaryBlockFormat() 
	{
		testReadWriteMatrix( InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo );
	}
	
	private void testReadWriteMatrix( InputInfo ii, OutputInfo oi )
	{
		double[][] matrix = TestUtils.generateTestMatrix(_rows, _cols, 0, 1, _sparsity, 7);
		double[][] matrix2 = null;
		
		try 
		{
			MatrixBlock mb1 = DataConverter.convertToMatrixBlock(matrix);		
			
			DataConverter.writeMatrixToHDFS(mb1, _fname, oi, _rows, _cols, _brlen, _bclen);		
			MatrixBlock mb2 = DataConverter.readMatrixFromHDFS(_fname, ii, _rows, _cols, _brlen, _bclen);
			
			matrix2 = DataConverter.convertToDoubleMatrix(mb2);
			
			//cleanup
			MapReduceTool.deleteFileIfExistOnHDFS(_fname);
		} 
		catch (IOException e) 
		{
			e.printStackTrace();
		}
		
		//compare
		for( int i=0; i<_rows; i++ )
			for( int j=0; j<_cols; j++ )
				if( matrix[i][j]!=matrix2[i][j] )
					Assert.fail("Wrong value i="+i+", j="+j+", value1="+matrix[i][j]+", value2="+matrix2[i][j]);
	}
}
