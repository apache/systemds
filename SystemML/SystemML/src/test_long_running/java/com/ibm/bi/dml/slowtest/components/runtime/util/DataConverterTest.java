/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.slowtest.components.runtime.util;

import org.junit.Assert;
import org.junit.Test;

import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.runtime.util.DataConverter;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.test.utils.TestUtils;


public class DataConverterTest 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private int _brlen = 1000;
	private int _bclen = 1000;
	private int _rows = 3500;
	private int _cols = 3500;
	private double _sparsity = 0.7d;
	private double _sparsitySkew1 = 0.9d;
	private double _sparsitySkew2 = 0.1d;
	
	private String _fname = "./scratch_space/A";
	
	@Test
	public void testReadWriteTextCellFormat() 
	{
		testReadWriteMatrix( InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo, false );
	}
	
	@Test
	public void testReadWriteBinaryCellFormat() 
	{
		testReadWriteMatrix( InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo, false );
	}
	
	@Test
	public void testReadWriteBinaryBlockFormat() 
	{
		testReadWriteMatrix( InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, false );
	}
	
	@Test
	public void testReadWriteBinaryBlockFormatWithSkew() 
	{
		testReadWriteMatrix( InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo, true );
	}
	
	
	private void testReadWriteMatrix( InputInfo ii, OutputInfo oi, boolean skew )
	{
		double[][] matrix = null;
		double[][] matrix2 = null;
		
		//create initial matrix
		if( skew )
			matrix = generateSkewedTestMatrix();
		else
			matrix = generateUniformTestMatrix();
		
		try 
		{
			MatrixBlock mb1 = DataConverter.convertToMatrixBlock(matrix);		
			
			DataConverter.writeMatrixToHDFS(mb1, _fname, oi, new MatrixCharacteristics(_rows, _cols, _brlen, _bclen));		
			MatrixBlock mb2 = DataConverter.readMatrixFromHDFS(_fname, ii, _rows, _cols, _brlen, _bclen);
			
			matrix2 = DataConverter.convertToDoubleMatrix(mb2);
			
			//cleanup
			MapReduceTool.deleteFileIfExistOnHDFS(_fname);
		} 
		catch(Exception e) 
		{
			e.printStackTrace();
		}
		
		//compare
		for( int i=0; i<_rows; i++ )
			for( int j=0; j<_cols; j++ )
				if( matrix[i][j]!=matrix2[i][j] )
					Assert.fail("Wrong value i="+i+", j="+j+", value1="+matrix[i][j]+", value2="+matrix2[i][j]);
	}
	
	private double[][] generateUniformTestMatrix()
	{
		return TestUtils.generateTestMatrix(_rows, _cols, 0, 1, _sparsity, 7);
	}
	
	private double[][] generateSkewedTestMatrix()
	{
		double[][] mat = new double[_rows][_cols];
		double[][] tmp1 = TestUtils.generateTestMatrix(_rows, _cols/2, 0, 1, _sparsitySkew1, 3);
		double[][] tmp2 = TestUtils.generateTestMatrix(_rows, _cols/2, 0, 1, _sparsitySkew2, 7);
		
		for( int i=0; i<_rows; i++ )
			for( int j=0; j<_cols; j++ )
			{
				if( j<=_cols/2-1 )
					mat[i][j] = tmp1[i][j];
				else
					mat[i][j] = tmp2[i][j-(_cols/2)];
			}
		
		return mat;
	}
}
