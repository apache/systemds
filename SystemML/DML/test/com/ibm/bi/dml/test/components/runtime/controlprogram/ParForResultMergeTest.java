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
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PResultMerge;
import com.ibm.bi.dml.runtime.controlprogram.caching.LazyWriteBuffer;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ResultMerge;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ResultMergeLocalFile;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ResultMergeLocalMemory;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ResultMergeRemoteMR;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.util.DataConverter;
import com.ibm.bi.dml.runtime.util.LocalFileUtils;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.test.utils.TestUtils;

public class ParForResultMergeTest 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private int _brlen = 1000;
	private int _bclen = 1000;
	private int _rows = 3500;
	private int _cols = 2500; 
	private int _par = 4; 
	private double _sparsity1 = 0.3d;
	private double _sparsity2 = 0.7d;
	private String _fname = "./scratch_space/B2";
	private String _fname2 = "./scratch_space/w/B2";

	@Test
	public void testMemSerialDenseEmptyTextCell() 
	{
		testMatrixResultMerge( PResultMerge.LOCAL_MEM, false, false, true, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo );
	}

	@Test
	public void testMemSerialDenseEmptyBinaryCell() 
	{
		testMatrixResultMerge( PResultMerge.LOCAL_MEM, false, false, true, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo );
	}
	
	@Test
	public void testMemSerialDenseEmptyBinaryBlock() 
	{
		testMatrixResultMerge( PResultMerge.LOCAL_MEM, false, false, true, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo );
	}
	
	@Test
	public void testMemSerialSparseEmptyTextCell() 
	{
		testMatrixResultMerge( PResultMerge.LOCAL_MEM, false, true, true, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo );
	}

	@Test
	public void testMemSerialSparseEmptyBinaryCell() 
	{
		testMatrixResultMerge( PResultMerge.LOCAL_MEM, false, true, true, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo );
	}
	
	@Test
	public void testMemSerialSparseEmptyBinaryBlock() 
	{
		testMatrixResultMerge( PResultMerge.LOCAL_MEM, false, true, true, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo );
	}

	@Test
	public void testMemSerialDenseFullTextCell() 
	{
		testMatrixResultMerge( PResultMerge.LOCAL_MEM, false, false, false, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo );
	}

	@Test
	public void testMemSerialDenseFullBinaryCell() 
	{
		testMatrixResultMerge( PResultMerge.LOCAL_MEM, false, false, false, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo );
	}
	
	@Test
	public void testMemSerialDenseFullBinaryBlock() 
	{
		testMatrixResultMerge( PResultMerge.LOCAL_MEM, false, false, false, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo );
	}
	
	@Test
	public void testMemSerialSparseFullTextCell() 
	{
		testMatrixResultMerge( PResultMerge.LOCAL_MEM, false, true, false, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo );
	}

	@Test
	public void testMemSerialSparseFullBinaryCell() 
	{
		testMatrixResultMerge( PResultMerge.LOCAL_MEM, false, true, false, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo );
	}
	
	@Test
	public void testMemSerialSparseFullBinaryBlock() 
	{
		testMatrixResultMerge( PResultMerge.LOCAL_MEM, false, true, false, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo );
	}

	/*
	@Test
	public void testMemParallelDenseEmptyTextCell() 
	{
		testMatrixResultMerge( true, true, false, true, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo );
	}

	@Test
	public void testMemParallelDenseEmptyBinaryCell() 
	{
		testMatrixResultMerge( true, true, false, true, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo );
	}
	
	@Test
	public void testMemParallelDenseEmptyBinaryBlock() 
	{
		testMatrixResultMerge( true, true, false, true, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo );
	}
	
	@Test
	public void testMemParallelSparseEmptyTextCell() 
	{
		testMatrixResultMerge( true, true, true, true, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo );
	}

	@Test
	public void testMemParallelSparseEmptyBinaryCell() 
	{
		testMatrixResultMerge( true, true, true, true, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo );
	}
	
	@Test
	public void testMemParallelSparseEmptyBinaryBlock() 
	{
		testMatrixResultMerge( true, true, true, true, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo );
	}
		
	@Test
	public void testMemParallelDenseFullTextCell() 
	{
		testMatrixResultMerge( true, true, false, false, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo );
	}

	@Test
	public void testMemParallelDenseFullBinaryCell() 
	{
		testMatrixResultMerge( true, true, false, false, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo );
	}
	
	@Test
	public void testMemParallelDenseFullBinaryBlock() 
	{
		testMatrixResultMerge( true, true, false, false, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo );
	}
	
	@Test
	public void testMemParallelSparseFullTextCell() 
	{
		testMatrixResultMerge( true, true, true, false, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo );
	}

	@Test
	public void testMemParallelSparseFullBinaryCell() 
	{
		testMatrixResultMerge( true, true, true, false, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo );
	}
	
	@Test
	public void testMemParallelSparseFullBinaryBlock() 
	{
		testMatrixResultMerge( true, true, true, false, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo );
	}
	 */
	
	@Test
	public void testFileSerialDenseEmptyTextCell() 
	{
		testMatrixResultMerge( PResultMerge.LOCAL_FILE, false, false, true, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo );
	}

	@Test
	public void testFileSerialDenseEmptyBinaryCell() 
	{
		testMatrixResultMerge( PResultMerge.LOCAL_FILE, false, false, true, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo );
	}
	
	@Test
	public void testFileSerialDenseEmptyBinaryBlock() 
	{
		testMatrixResultMerge( PResultMerge.LOCAL_FILE, false, false, true, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo );
	}
	
	
	@Test
	public void testFileSerialSparseEmptyTextCell() 
	{
		testMatrixResultMerge( PResultMerge.LOCAL_FILE, false, true, true, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo );
	}

	@Test
	public void testFileSerialSparseEmptyBinaryCell() 
	{
		testMatrixResultMerge( PResultMerge.LOCAL_FILE, false, true, true, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo );
	}
	
	
	@Test
	public void testFileSerialSparseEmptyBinaryBlock() 
	{
		testMatrixResultMerge( PResultMerge.LOCAL_FILE, false, true, true, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo );
	}
	
	@Test
	public void testFileSerialDenseFullTextCell() 
	{
		testMatrixResultMerge( PResultMerge.LOCAL_FILE, false, false, false, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo );
	}

	@Test
	public void testFileSerialDenseFullBinaryCell() 
	{
		testMatrixResultMerge( PResultMerge.LOCAL_FILE, false, false, false, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo );
	}
	
	@Test
	public void testFileSerialDenseFullBinaryBlock() 
	{
		testMatrixResultMerge( PResultMerge.LOCAL_FILE, false, false, false, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo );
	}
	
	@Test
	public void testFileSerialSparseFullTextCell() 
	{
		testMatrixResultMerge( PResultMerge.LOCAL_FILE, false, true, false, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo );
	}

	@Test
	public void testFileSerialSparseFullBinaryCell() 
	{
		testMatrixResultMerge( PResultMerge.LOCAL_FILE, false, true, false, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo );
	}
	
	@Test
	public void testFileSerialSparseFullBinaryBlock() 
	{
		testMatrixResultMerge( PResultMerge.LOCAL_FILE, false, true, false, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo );
	}
	
	//REMOTE_MR result merge
	
	@Test
	public void testRemoteDenseEmptyTextCell() 
	{
		testMatrixResultMerge( PResultMerge.REMOTE_MR, false, false, true, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo );
	}
	
	@Test
	public void testRemoteDenseEmptyBinaryCell() 
	{
		testMatrixResultMerge( PResultMerge.REMOTE_MR, false, false, true, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo );
	}
	
	@Test
	public void testRemoteDenseEmptyBinaryBlock() 
	{
		testMatrixResultMerge( PResultMerge.REMOTE_MR, false, false, true, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo );
	}
	
	@Test
	public void testRemoteSparseEmptyTextCell() 
	{
		testMatrixResultMerge( PResultMerge.REMOTE_MR, false, true, true, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo );
	}

	@Test
	public void testRemoteSparseEmptyBinaryCell() 
	{
		testMatrixResultMerge( PResultMerge.REMOTE_MR, false, true, true, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo );
	}
	
	
	@Test
	public void testRemoteSparseEmptyBinaryBlock() 
	{
		testMatrixResultMerge( PResultMerge.REMOTE_MR, false, true, true, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo );
	}
	
	@Test
	public void testRemoteDenseFullTextCell() 
	{
		testMatrixResultMerge( PResultMerge.REMOTE_MR, false, false, false, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo );
	}

	@Test
	public void testRemoteDenseFullBinaryCell() 
	{
		testMatrixResultMerge( PResultMerge.REMOTE_MR, false, false, false, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo );
	}
	
	@Test
	public void testRemoteDenseFullBinaryBlock() 
	{
		testMatrixResultMerge( PResultMerge.REMOTE_MR, false, false, false, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo );
	}
	
	@Test
	public void testRemoteSparseFullTextCell() 
	{
		testMatrixResultMerge( PResultMerge.REMOTE_MR, false, true, false, InputInfo.TextCellInputInfo, OutputInfo.TextCellOutputInfo );
	}

	@Test
	public void testRemoteSparseFullBinaryCell() 
	{
		testMatrixResultMerge( PResultMerge.REMOTE_MR, false, true, false, InputInfo.BinaryCellInputInfo, OutputInfo.BinaryCellOutputInfo );
	}
	
	@Test
	public void testRemoteSparseFullBinaryBlock() 
	{
		testMatrixResultMerge( PResultMerge.REMOTE_MR, false, true, false, InputInfo.BinaryBlockInputInfo, OutputInfo.BinaryBlockOutputInfo );
	}
	 
	
	private void testMatrixResultMerge( PResultMerge type , boolean parallel, boolean sparse, boolean initialEmpty, InputInfo ii, OutputInfo oi )
	{
		double sparsity = sparse ? _sparsity1 : _sparsity2;
		
		double[][] matrixOld = TestUtils.generateTestMatrix(_rows, _cols, 0, initialEmpty ? 0 : 1, sparsity, 7);
		double[][] matrixNew = TestUtils.generateTestMatrix(_rows, _cols, 1, 2, sparsity, 7);
		matrixNew[_rows/2][_cols/2] = 0; //to make result merge with compare a bit more challenging	(capture values set to 0)
		
		try 
		{
			cleanup();
			LazyWriteBuffer.init();
			
			
			//create and write original matrix
			MatrixBlock retOld = DataConverter.convertToMatrixBlock(matrixOld);
			retOld.recomputeNonZeros();
			retOld.examSparsity();
			MatrixCharacteristics mc = new MatrixCharacteristics(_rows, _cols, _brlen, _bclen, retOld.getNonZeros());
			MatrixFormatMetaData meta = new MatrixFormatMetaData(mc, oi, ii);
			DataConverter.writeMatrixToHDFS(retOld, _fname, oi, mc);		
			MatrixObject moOut = new MatrixObject(ValueType.DOUBLE,_fname);
			moOut.setVarName("VarOut");
			moOut.setMetaData(meta);
			
			//create inputs 
			MatrixObject[] in = new MatrixObject[ _par ];
			int numCols = _cols/_par;
			for( int k=0; k<_par; k++ ) //for all subresults
			{
				double[][] tmp = new double[_rows][_cols];
				for( int i=0; i<_rows; i++ )
					for( int j=0; j<_cols; j++ )
					{
						if( j/numCols==k )
							tmp[i][j] = matrixNew[i][j];
						else
							tmp[i][j] = matrixOld[i][j];
					}
				
				MatrixBlock tmpMB = DataConverter.convertToMatrixBlock(tmp);
				tmpMB.examSparsity();
				MatrixCharacteristics mc2 = new MatrixCharacteristics(_rows, _cols, _brlen, _bclen, tmpMB.getNonZeros());
				MatrixFormatMetaData meta2 = new MatrixFormatMetaData(mc2, oi, ii);
				DataConverter.writeMatrixToHDFS(tmpMB, _fname2+k, oi, mc2);		
				MatrixObject tmpMo = new MatrixObject(ValueType.DOUBLE,_fname2+k);
				tmpMo.setVarName("Var"+k);
				tmpMo.setMetaData(meta2);
				in[ k ] = tmpMo;
			}
			
			
			ResultMerge rm = null;
			switch( type )
			{
				case LOCAL_MEM:
					rm = new ResultMergeLocalMemory(moOut, in, _fname+"out" );
					break;
				case LOCAL_FILE:
					rm = new ResultMergeLocalFile( moOut, in, _fname+"out" );
					break;
				case REMOTE_MR:
					rm = new ResultMergeRemoteMR( moOut, in, _fname+"out", 7, 4, 4, 3, 1, false );
					break;
			}
			
			//execute result merge
			MatrixObject tmpRet = null;
			if( parallel )
				tmpRet = rm.executeParallelMerge( _par );
			else
				tmpRet = rm.executeSerialMerge();
			tmpRet.exportData();
			
			
			//read matrix
			MatrixBlock ret = DataConverter.readMatrixFromHDFS(_fname+"out", ii, _rows, _cols, _brlen, _bclen);
			double[][] retMat = DataConverter.convertToDoubleMatrix(ret);
	
			//cleanup
			moOut.clearData();
			tmpRet.clearData();
			for( int k=0; k<_par; k++ )
				in[ k ].clearData();			
			cleanup(); //cleanup files
			
			//compare
			for( int i=0; i<_rows; i++ )
				for( int j=0; j<_cols; j++ )
					if( matrixNew[i][j]!=retMat[i][j] )
						Assert.fail("Wrong value i="+i+", j="+j+", value1="+matrixNew[i][j]+", value2="+retMat[i][j]);
		
		} 
		catch (Exception e) 
		{
			e.printStackTrace();
			Assert.fail( e.getMessage() );
		}
	}
	
	
	private void cleanup() 
		throws IOException, ParseException
	{
		MapReduceTool.deleteFileIfExistOnHDFS(_fname);
		for( int k=0; k<_par; k++ )
			MapReduceTool.deleteFileIfExistOnHDFS(_fname+k);
		MapReduceTool.deleteFileIfExistOnHDFS(_fname+"out");

		LocalFileUtils.cleanupWorkingDirectory();
	}
	
	
	/*
	
	private static final int _par = 4;
	private static final int _dim = 10;	
	private IDSequence _seq = new IDSequence();
	  
	@Test
	public void testSerialInMemoryResultMerge() 
		throws DMLRuntimeException, ParserConfigurationException, SAXException, IOException 
	{ 
		runResultMerge( false, true );
	}
	
	@Test
	public void testSerialFileBasedResultMerge() 
		throws DMLRuntimeException, ParserConfigurationException, SAXException, IOException 
	{ 
		runResultMerge( false, false );
	}
	
	@Test
	public void testParallelInMemoryResultMerge() 
		throws DMLRuntimeException, ParserConfigurationException, SAXException, IOException 
	{ 
		runResultMerge( true, true );
	}
	
	@Test
	public void testParallelFileBasedResultMerge() 
		throws DMLRuntimeException, ParserConfigurationException, SAXException, IOException 
	{ 
		runResultMerge( true, false );
	}
	
	private void runResultMerge(boolean parallel, boolean inMem) 
		throws ParserConfigurationException, SAXException, IOException, DMLRuntimeException
	{
		//init cache
		CacheableData.createCacheDir();
		
		//init input, output, comparison obj
		MatrixObjectNew[] in = new MatrixObjectNew[ _par ];
		for( int i=0; i<_par; i++ )
		{
			in[i] = createMatrixObject( _dim, true );
		}
		MatrixObjectNew out = createMatrixObject( _dim, true );
		MatrixObjectNew ref = createMatrixObject( _dim, true );
		
		//populate inputs and comparison object
		generateData( ref, in );			
		if( !inMem )
			for( MatrixObjectNew mo : in )
			{
				//write and delete in-memory data
				if(!inMem)
				{
					mo.exportData();
					mo.clearData();
				}
			}
		
		//run result merge
		ResultMerge rm = new ResultMerge(out, in, "./out", _par);
		if( parallel )
			out = rm.executeParallelMerge();
		else
			out = rm.executeSerialMerge();
		
		//compare result
		if( !checkOutput( ref, out ) )
			Assert.fail("Wrong result matrix.");	
		
		//cleanup
		for( MatrixObjectNew inMO : in )
			inMO.clearData();
		out.clearData();
		ref.clearData();
		
		CacheableData.cleanupCacheDir();
	}

	private MatrixObjectNew createMatrixObject(int dim, boolean withData) 
		throws ParserConfigurationException, SAXException, IOException, CacheException 
	{
		
		DMLConfig conf = null;
		try {
			conf = new DMLConfig(DMLScript.DEFAULT_SYSTEMML_CONFIG_FILEPATH);
		} catch (Exception e){
			System.out.println("ERROR: could not create DMLConfig from config file " + DMLScript.DEFAULT_SYSTEMML_CONFIG_FILEPATH);
		}
		
		String dir = null;
		try {
			dir = conf.getTextValue(DMLConfig.SCRATCH_SPACE);
		} catch (Exception e){
			System.out.println("ERROR: could not retrieve parameter " + DMLConfig.SCRATCH_SPACE + " from DMLConfig");
		}
		
		long id = _seq.getNextID();
		String fname = dir+"/"+String.valueOf(id);
		
		MatrixCharacteristics mc = new MatrixCharacteristics(dim, dim, dim, dim);
		MatrixFormatMetaData md = new MatrixFormatMetaData(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);
		MatrixObjectNew mo = new MatrixObjectNew(ValueType.DOUBLE, fname, md);
		mo.setVarName( String.valueOf(id) );
		
		if( withData )
		{
			MatrixBlock mb = new MatrixBlock(dim,dim,false);
			mb.setValue(0, 0, 7d); // base data to check if div works
			mo.acquireModify(mb);
			mo.release();
		}
	
		return mo;
	}
	
	private void generateData(MatrixObjectNew ref, MatrixObjectNew[] in) 
		throws DMLRuntimeException 
	{
		long rows = ref.getNumRows();
		long cols = ref.getNumColumns();
		int index = 0;
		int subSize = (int) Math.ceil( rows * cols / in.length );
		double value; //dynamically assigned
		
		//set input data
		MatrixBlock refData = ref.acquireModify();
		MatrixBlock inData = in[ index ].acquireModify();
		
		for( int i=0; i<rows; i++ ) 
			for( int j=0; j<cols; j++ )
			{
				value = i*cols+(j+1);
				refData.setValue( i, j, value );				
				inData.setValue(i, j, value);
				if( value % subSize == 0 && index != in.length-1 )
				{
					in[ index ].release();
					index++;
					inData = in[ index ].acquireModify();
				}
			}
		
		ref.release();
		in[ index ].release();
	}
	
	private boolean checkOutput(MatrixObjectNew ref, MatrixObjectNew out) 
		throws CacheException 
	{
		boolean ret = true;
		
		MatrixBlock refMB = ref.acquireRead();
		MatrixBlock outMB = out.acquireRead();
		

		if(    refMB.getNumRows() != outMB.getNumRows() 
			|| refMB.getNumColumns() != outMB.getNumColumns() )
		{
			ret = false; 
		}
		else
		{
			int rows = refMB.getNumRows();
			int cols = refMB.getNumColumns();
			
			for( int i=0; i<rows; i++ )
				for( int j=0; j<cols; j++ )
					if( refMB.getValue(i, j) != outMB.getValue(i, j) )
					{
						System.out.println(refMB.getValue(i, j)+" vs "+outMB.getValue(i, j));
						ret=false;
						i=rows; break;
					}
		}

		
		ref.release();
		out.release();
		
		return ret;
	}
	*/
}
