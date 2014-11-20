/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.components.runtime.controlprogram;


import java.io.IOException;

import junit.framework.Assert;

import org.junit.Test;

import com.ibm.bi.dml.parser.ParseException;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.instructions.CPFileInstructions.ParameterizedBuiltinCPFileInstruction;
import com.ibm.bi.dml.runtime.instructions.CPFileInstructions.ParameterizedBuiltinCPFileInstruction.RemoveEmpty;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.runtime.util.DataConverter;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.test.utils.TestUtils;

public class RemoveEmptyTest 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//rows and cols must be even 
	private int _brlen = 1000;
	private int _bclen = 1000;
	private int _rows = 5500;
	private int _cols = 3424;
	private double _sparsity1 = 0.7d;
	private double _sparsity2 = 0.07d;
	private String _fname = "./scratch_space/B";

	@Test
	public void testRowsDenseTextCell() 
	{
		testRemoveEmpty( "rows", false, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo );
	}

	@Test
	public void testRowsDenseBinaryCell() 
	{
		testRemoveEmpty( "rows", false, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo );
	}
	
	@Test
	public void testRowsDenseBinaryBlock() 
	{
		testRemoveEmpty( "rows", false, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo );
	}
	
	@Test
	public void testRowsSparseTextCell() 
	{
		testRemoveEmpty( "rows", true, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo );
	}

	@Test
	public void testRowsSparseBinaryCell() 
	{
		testRemoveEmpty( "rows", true, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo );
	}
	
	@Test
	public void testRowsSparseBinaryBlock() 
	{
		testRemoveEmpty( "rows", true, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo );
	}
	
	@Test
	public void testColsDenseTextCell() 
	{
		testRemoveEmpty( "cols", false, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo );
	}

	@Test
	public void testColsDenseBinaryCell() 
	{
		testRemoveEmpty( "cols", false, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo );
	}
	
	@Test
	public void testColsDenseBinaryBlock() 
	{
		testRemoveEmpty( "cols", false, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo );
	}
	
	@Test
	public void testColsSparseTextCell() 
	{
		testRemoveEmpty( "cols", true, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo );
	}

	@Test
	public void testColsSparseBinaryCell() 
	{
		testRemoveEmpty( "cols", true, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo );
	}
	
	@Test
	public void testColsSparseBinaryBlock() 
	{
		testRemoveEmpty( "cols", true, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo );
	}
	
	
	private void testRemoveEmpty( String margin, boolean sparse, InputInfo ii, OutputInfo oi )
	{
		double sparsity = sparse ? _sparsity2 : _sparsity1;		
		double V[][][] = createInputMatrices(margin, _rows, _cols, sparsity, oi);
		double[][] matrixOld = V[0];
		double[][] matrixNew = V[1];
		
		try 
		{
			cleanup();
			
			//create and write original matrix
			MatrixBlock retOld = DataConverter.convertToMatrixBlock(matrixOld);
			retOld.examSparsity();
			MatrixCharacteristics mc = new MatrixCharacteristics(_rows, _cols, _brlen, _bclen, retOld.getNonZeros());
			MatrixFormatMetaData meta = new MatrixFormatMetaData(mc, oi, ii);
			DataConverter.writeMatrixToHDFS(retOld, _fname, oi, mc);		
			
			MatrixObject moIn = new MatrixObject(ValueType.DOUBLE,_fname);
			moIn.setVarName("VarIn");
			moIn.setMetaData(meta);
			
			MatrixObject moOut = new MatrixObject(ValueType.DOUBLE,_fname+"out");
			moOut.setVarName("VarOut");
			moOut.setMetaData(meta);
			
			ParameterizedBuiltinCPFileInstruction pb = new ParameterizedBuiltinCPFileInstruction(null,null,null,null);
			RemoveEmpty rm = pb.new RemoveEmpty(margin, moIn, moOut);
			moOut = rm.execute();
					
			//read matrix
			MatrixBlock ret = DataConverter.readMatrixFromHDFS(_fname+"out", ii, moOut.getNumRows(), moOut.getNumColumns(), _brlen, _bclen);
			double[][] retMat = DataConverter.convertToDoubleMatrix(ret);

			cleanup(); //cleanup files
			
			//compare with the matrix 
			if( matrixNew.length != retMat.length )
				Assert.fail("Wrong number of rows size1="+matrixNew.length+", size2="+retMat.length);
			if( matrixNew[0].length != retMat[0].length )
				Assert.fail("Wrong number of columns size1="+matrixNew[0].length+", size2="+retMat[0].length);
			int rows2 = (int) moOut.getNumRows();
			int cols2 = (int) moOut.getNumColumns();
			for( int i=0; i<rows2; i++ )
				for( int j=0; j<cols2; j++ )
					if( matrixNew[i][j]!=retMat[i][j] )
						Assert.fail("Wrong value i="+i+", j="+j+", value1="+matrixNew[i][j]+", value2="+retMat[i][j]);
		
		} 
		catch (Exception e) 
		{
			e.printStackTrace();
			Assert.fail( e.getMessage() );
		}
	}

	private double[][][] createInputMatrices(String margin, int rows, int cols, double sparsity, OutputInfo oi) 
	{
		double[][][] ret = new double[2][][];
		
		int rowsp = -1, colsp = -1;
		if( margin.equals("rows") ){
			rowsp = rows/2;
			colsp = cols;
		}
		else{
			rowsp = rows;
			colsp = cols/2;
		}
			
		//long seed = System.nanoTime();
        double[][] V = TestUtils.generateTestMatrix(rows, cols, 0, 1, sparsity, 7);
        double[][] Vp = new double[rowsp][colsp];
        
        //clear out every other row/column
        if( margin.equals("rows") )
        {
        	for( int i=0; i<rows; i++ )
        	{
        		boolean clear = i%2!=0;
        		if( clear )
        			for( int j=0; j<cols; j++ )
        				V[i][j] = 0;
        		else
        			for( int j=0; j<cols; j++ )
        				Vp[i/2][j] = V[i][j];
        	}
        }
        else
        {
        	for( int j=0; j<cols; j++ )
        	{
        		boolean clear = j%2!=0;
        		if( clear )
        			for( int i=0; i<rows; i++ )
        				V[i][j] = 0;
        		else
        			for( int i=0; i<rows; i++ )
        				Vp[i][j/2] = V[i][j];
        	}
        }
        
        ret[0] = V;
        ret[1] = Vp;
        
        return ret;
	}
	
	private void cleanup() 
		throws IOException, ParseException
	{
		MapReduceTool.deleteFileIfExistOnHDFS(_fname);
		MapReduceTool.deleteFileIfExistOnHDFS(_fname+"out");
	}
	

}
