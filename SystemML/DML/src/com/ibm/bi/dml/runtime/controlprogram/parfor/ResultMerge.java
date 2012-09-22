package com.ibm.bi.dml.runtime.controlprogram.parfor;

import java.util.ArrayList;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.ConfigurationManager;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.MatrixObjectNew;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlockDSM.IJV;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlockDSM.SparseCellIterator;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.configuration.DMLConfig;

/**
 * Due to independence of all iterations, any result has the following properties:
 * (1) non local var, (2) matrix object, and (3) completely independent.
 * These properties allow us to realize result merging in parallel without any synchronization. 
 * 
 */
public abstract class ResultMerge 
{
	protected static final boolean LDEBUG = DMLScript.DEBUG || false; //local debug flag
	protected static final String NAME_SUFFIX = "_rm";
	
	protected static String STAGING_DIR = null;
	
	//inputs to result merge
	protected MatrixObjectNew   _output      = null;
	protected MatrixObjectNew[] _inputs      = null; 
	protected String            _outputFName = null;
	
	public ResultMerge( MatrixObjectNew out, MatrixObjectNew[] in, String outputFilename )
	{
		_output = out;
		_inputs = in;
		_outputFName = outputFilename;
		
		DMLConfig conf = ConfigurationManager.getConfig();
		if( conf != null )
			STAGING_DIR = conf.getTextValue(DMLConfig.LOCAL_TMP_DIR) + "/resultmerge/";
		else
			STAGING_DIR = "tmp/systemml/resultmerge/";
	}
	
	/**
	 * Merge all given input matrices sequentially into the given output matrix.
	 * The required space in-memory is the size of the output matrix plus the size
	 * of one input matrix at a time.
	 * 
	 * @return output (merged) matrix
	 * @throws DMLRuntimeException
	 */
	public abstract MatrixObjectNew executeSerialMerge() 
		throws DMLRuntimeException;
	
	/**
	 * Merge all given input matrices in parallel into the given output matrix.
	 * The required space in-memory is the size of the output matrix plus the size
	 * of all input matrices.
	 * 
	 * @param par degree of parallelism
	 * @return output (merged) matrix
	 * @throws DMLRuntimeException
	 */
	public abstract MatrixObjectNew executeParallelMerge( int par ) 
		throws DMLRuntimeException;
	
	/**
	 * 
	 * @param out initially empty block
	 * @param in 
	 */
	protected void mergeWithoutComp( MatrixBlock out, MatrixBlock in )
	{
		if( in.isInSparseFormat() ) //sparse input format
		{
			SparseCellIterator iter = in.getSparseCellIterator();
			while( iter.hasNext() )
			{
				IJV cell = iter.next();
				out.setValue(cell.i, cell.j, cell.v);
			}
		}
		else //dense input format
		{
			//for a merge this case will seldom happen, as each input MatrixObject
			//has at most 1/numThreads of all values in it.
			int rows = in.getNumRows();
			int cols = in.getNumColumns();
			if( in.getNonZeros() > 0 )
				for( int i=0; i<rows; i++ )
					for( int j=0; j<cols; j++ )
					{
						double value = in.getValueDenseUnsafe(i,j); //input value
						if( value != 0  ) 					       //for all nnz
						{
							out.setValue( i, j, value );	
						}
					}
		}	
	}

	/**
	 * 
	 * @param out
	 * @param in
	 */
	protected void mergeWithComp( MatrixBlock out, MatrixBlock in, double[][] compare )
	{
		//NOTE: always iterate over entire block in order to compare all values
		//      (using sparse iterator would miss values set to 0)
		
		if( in.isInSparseFormat() ) //sparse input format
		{
			int rows = in.getNumRows();
			int cols = in.getNumColumns();
				for( int i=0; i<rows; i++ )
					for( int j=0; j<cols; j++ )
					{
					    double value = in.getValueSparseUnsafe(i,j);  //input value
						if( value != compare[i][j] )  //for new values only (div)
							out.setValue( i, j, value );	
					}
		}
		else //dense input format
		{
			//for a merge this case will seldom happen, as each input MatrixObject
			//has at most 1/numThreads of all values in it.
			int rows = in.getNumRows();
			int cols = in.getNumColumns();
				for( int i=0; i<rows; i++ )
					for( int j=0; j<cols; j++ )
					{
					    double value = in.getValueDenseUnsafe(i,j);  //input value
						if( value != compare[i][j] )  //for new values only (div)
							out.setValue( i, j, value );	
					}
		}			
	}
	

	protected long computeNonZeros( MatrixObjectNew out, ArrayList<MatrixObjectNew> in )
	{
		MatrixFormatMetaData metadata = (MatrixFormatMetaData) out.getMetaData();
		MatrixCharacteristics mc = metadata.getMatrixCharacteristics();
		long outNNZ = mc.getNonZeros();	
		long ret = outNNZ;
		for( MatrixObjectNew tmp : in )
		{
			MatrixFormatMetaData tmpmetadata = (MatrixFormatMetaData) tmp.getMetaData();
			MatrixCharacteristics tmpmc = tmpmetadata.getMatrixCharacteristics();
			long inNNZ = tmpmc.getNonZeros();
			ret +=  (inNNZ - outNNZ);			
		}
		
		return ret;
	}
}
