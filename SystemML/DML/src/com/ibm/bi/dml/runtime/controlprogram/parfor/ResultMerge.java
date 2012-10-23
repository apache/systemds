package com.ibm.bi.dml.runtime.controlprogram.parfor;

import java.util.ArrayList;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.ConfigurationManager;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlockDSM.IJV;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlockDSM.SparseCellIterator;
import com.ibm.bi.dml.runtime.util.LocalFileUtils;
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
	protected MatrixObject   _output      = null;
	protected MatrixObject[] _inputs      = null; 
	protected String         _outputFName = null;
	
	protected ResultMerge( )
	{
		
	}
	
	public ResultMerge( MatrixObject out, MatrixObject[] in, String outputFilename )
	{
		_output = out;
		_inputs = in;
		_outputFName = outputFilename;
		
		
		//configure staging dir root
		DMLConfig conf = ConfigurationManager.getConfig();
		if( conf != null )
			STAGING_DIR = conf.getTextValue(DMLConfig.LOCAL_TMP_DIR) + "/resultmerge/";
		else
			STAGING_DIR = DMLConfig.getDefaultTextValue(DMLConfig.LOCAL_TMP_DIR) + "/resultmerge/";
		
		//create shared staging dir if not existing
		LocalFileUtils.createLocalFileIfNotExist(STAGING_DIR, DMLConfig.DEFAULT_SHARED_DIR_PERMISSION);
	}
	
	/**
	 * Merge all given input matrices sequentially into the given output matrix.
	 * The required space in-memory is the size of the output matrix plus the size
	 * of one input matrix at a time.
	 * 
	 * @return output (merged) matrix
	 * @throws DMLRuntimeException
	 */
	public abstract MatrixObject executeSerialMerge() 
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
	public abstract MatrixObject executeParallelMerge( int par ) 
		throws DMLRuntimeException;
	
	/**
	 * 
	 * @param out initially empty block
	 * @param in 
	 * @throws DMLRuntimeException 
	 */
	protected void mergeWithoutComp( MatrixBlock out, MatrixBlock in ) 
		throws DMLRuntimeException
	{
		if( in.isInSparseFormat() ) //sparse input format
		{
			SparseCellIterator iter = in.getSparseCellIterator();
			while( iter.hasNext() )
			{
				IJV cell = iter.next();
				out.quickSetValue(cell.i, cell.j, cell.v);
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
							out.quickSetValue( i, j, value );	
					}
		}	
		
		//change sparsity if required
		out.examSparsity(); 
	}

	/**
	 * 
	 * @param out
	 * @param in
	 * @throws DMLRuntimeException 
	 */
	protected void mergeWithComp( MatrixBlock out, MatrixBlock in, double[][] compare ) 
		throws DMLRuntimeException
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
							out.quickSetValue( i, j, value );	
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
					    	out.quickSetValue( i, j, value );	
					}
		}	
		
		//change sparsity if required 
		out.examSparsity(); 
	}
	

	protected long computeNonZeros( MatrixObject out, ArrayList<MatrixObject> in )
	{
		MatrixFormatMetaData metadata = (MatrixFormatMetaData) out.getMetaData();
		MatrixCharacteristics mc = metadata.getMatrixCharacteristics();
		long outNNZ = mc.getNonZeros();	
		long ret = outNNZ;
		for( MatrixObject tmp : in )
		{
			MatrixFormatMetaData tmpmetadata = (MatrixFormatMetaData) tmp.getMetaData();
			MatrixCharacteristics tmpmc = tmpmetadata.getMatrixCharacteristics();
			long inNNZ = tmpmc.getNonZeros();
			ret +=  (inNNZ - outNNZ);			
		}
		
		return ret;
	}
}
