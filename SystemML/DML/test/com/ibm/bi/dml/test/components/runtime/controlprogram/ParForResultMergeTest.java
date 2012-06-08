package com.ibm.bi.dml.test.components.runtime.controlprogram;

import java.io.IOException;

import javax.xml.parsers.ParserConfigurationException;

import junit.framework.Assert;

import org.junit.Test;
import org.xml.sax.SAXException;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ResultMerge;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDSequence;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.MatrixObject;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.util.DataConverter;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.configuration.DMLConfig;

public class ParForResultMergeTest 
{
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
		//init input, output, comparison obj
		MatrixObject[] in = new MatrixObject[ _par ];
		for( int i=0; i<_par; i++ )
			in[i] = createMatrixObject( _dim, true );
		MatrixObject out = createMatrixObject( _dim, true );
		MatrixObject ref = createMatrixObject( _dim, true );
		
		//propulate inputs and comparison object
		generateData( ref, in );			
		if( !inMem )
			for( MatrixObject mo : in )
			{
				//write and delete in-memory data
				DataConverter.writeMatrixToHDFS(
						mo.getData(), mo.getFileName(), OutputInfo.BinaryBlockOutputInfo, 
                        mo.getNumRows(), mo.getNumColumns(), mo.getNumRows(), mo.getNumColumns());
				mo.setData(null);				
			}
		
		
		//run result merge
		ResultMerge rm = new ResultMerge(out, in, !inMem, !inMem);
		if( parallel )
			out = rm.executeParallelMerge(_par);
		else
			out = rm.executeSerialMerge();
		
		//compare result
		if( !checkOutput( ref, out ) )
			Assert.fail("Wrong result matrix.");	
	}

	private MatrixObject createMatrixObject(int dim, boolean withData) 
		throws ParserConfigurationException, SAXException, IOException 
	{
		DMLConfig conf = new DMLConfig(DMLScript.DEFAULT_SYSTEMML_CONFIG_FILEPATH);
		String dir = conf.getTextValue("scratch");  
		String fname = dir+"/"+String.valueOf(_seq.getNextID());
		
		MatrixCharacteristics mc = new MatrixCharacteristics(dim, dim, dim, dim);
		MatrixFormatMetaData md = new MatrixFormatMetaData(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);
		MatrixObject mo = new MatrixObject(ValueType.DOUBLE, fname, md);
		
		if( withData )
			mo.setData( new MatrixBlock(dim, dim, false) );
	
		return mo;
	}
	
	private void generateData(MatrixObject ref, MatrixObject[] in) 
	{
		int rows = ref.getNumRows();
		int cols = ref.getNumColumns();
		int index = 0;
		int subSize = (int) Math.ceil( rows * cols / in.length );
		double value;
		
		MatrixBlock refData = ref.getData();
		MatrixBlock inData = in[ index ].getData();
		
		for( int i=0; i<rows; i++ )
			for( int j=0; j<cols; j++ )
			{
				value = i*cols+(j+1);
				refData.setValue( i, j, value );				
				inData.setValue(i, j, value);
				if( value % subSize == 0 && index != in.length-1 )
					inData = in[ ++index ].getData();	
			}		
	}
	
	private boolean checkOutput(MatrixObject ref, MatrixObject out) 
	{
		boolean ret = true;
		
		if(    ref.getNumRows() != out.getNumRows() 
			|| ref.getNumColumns() != out.getNumColumns() )
		{
			ret = false; 
		}
		else
		{
			int rows = ref.getNumRows();
			int cols = ref.getNumColumns();
			
			for( int i=0; i<rows; i++ )
				for( int j=0; j<cols; j++ )
					if( ref.getValue(i, j) != out.getValue(i, j) )
					{
						//System.out.println(ref.getValue(i, j)+" vs "+out.getValue(i, j));
						ret=false;
						i=rows; break;
					}
		}
		
		return ret;
	}
}
