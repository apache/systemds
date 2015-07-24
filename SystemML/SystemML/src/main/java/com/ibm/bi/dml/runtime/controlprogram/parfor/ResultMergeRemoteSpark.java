/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor;


import org.apache.spark.api.java.JavaPairRDD;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.instructions.spark.data.RDDObject;
import com.ibm.bi.dml.runtime.instructions.spark.functions.RDDAggregateUtils;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;

/**
 * MR job class for submitting parfor result merge MR jobs.
 * 
 */
public class ResultMergeRemoteSpark extends ResultMerge
{	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private ExecutionContext _ec = null;
	private int  _numMappers = -1;
	private int  _numReducers = -1;
	
	public ResultMergeRemoteSpark(MatrixObject out, MatrixObject[] in, String outputFilename, ExecutionContext ec, int numMappers, int numReducers) 
	{
		super(out, in, outputFilename);
		
		_ec = ec;
		_numMappers = numMappers;
		_numReducers = numReducers;
	}

	@Override
	public MatrixObject executeSerialMerge() 
		throws DMLRuntimeException 
	{
		//graceful degradation to parallel merge
		return executeParallelMerge( _numMappers );
	}
	
	@Override
	public MatrixObject executeParallelMerge(int par) 
		throws DMLRuntimeException 
	{
		MatrixObject moNew = null; //always create new matrix object (required for nested parallelism)

		LOG.trace("ResultMerge (remote, spark): Execute serial merge for output "+_output.getVarName()+" (fname="+_output.getFileName()+")");

		try
		{
			if( _inputs != null && _inputs.length>0 )
			{
				//prepare compare
				MatrixFormatMetaData metadata = (MatrixFormatMetaData) _output.getMetaData();
				MatrixCharacteristics mcOld = metadata.getMatrixCharacteristics();
				MatrixObject compare = (mcOld.getNonZeros()==0) ? null : _output;
				
				//actual merge
				RDDObject ro = executeMerge(compare, _inputs, _output.getVarName(), mcOld.getRows(), mcOld.getCols(), mcOld.getRowsPerBlock(), mcOld.getColsPerBlock());
				
				//create new output matrix (e.g., to prevent potential export<->read file access conflict
				String varName = _output.getVarName();
				ValueType vt = _output.getValueType();
				moNew = new MatrixObject( vt, _outputFName );
				moNew.setVarName( varName.contains(NAME_SUFFIX) ? varName : varName+NAME_SUFFIX );
				moNew.setDataType( DataType.MATRIX );
				OutputInfo oiOld = metadata.getOutputInfo();
				InputInfo iiOld = metadata.getInputInfo();
				MatrixCharacteristics mc = new MatrixCharacteristics(mcOld.getRows(),mcOld.getCols(),
						                                             mcOld.getRowsPerBlock(),mcOld.getColsPerBlock());
				mc.setNonZeros( computeNonZeros(_output, convertToList(_inputs)) );
				MatrixFormatMetaData meta = new MatrixFormatMetaData(mc,oiOld,iiOld);
				moNew.setMetaData( meta );
				moNew.setRDDHandle( ro );
			}
			else
			{
				moNew = _output; //return old matrix, to prevent copy
			}
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException(ex);
		}
		
		return moNew;		
	}
	
	/**
	 * 
	 * @param fname 	null if no comparison required
	 * @param fnameNew
	 * @param srcFnames
	 * @param ii
	 * @param oi
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @throws DMLRuntimeException
	 */
	@SuppressWarnings("unchecked")
	protected RDDObject executeMerge(MatrixObject compare, MatrixObject[] inputs, String varname, long rlen, long clen, int brlen, int bclen)
		throws DMLRuntimeException 
	{
		SparkExecutionContext sec = (SparkExecutionContext)_ec;
		boolean withCompare = (compare!=null);

		RDDObject ret = null;
		
	    //determine degree of parallelism
		int numRed = (int)determineNumReducers(rlen, clen, brlen, bclen, _numReducers);
	
		//sanity check for empty src files
		if( inputs == null || inputs.length==0  )
			throw new DMLRuntimeException("Execute merge should never be called with no inputs.");
		
		try
		{
		    //Step 1: union over all results
		    JavaPairRDD<MatrixIndexes, MatrixBlock> rdd = (JavaPairRDD<MatrixIndexes, MatrixBlock>) 
		    		sec.getRDDHandleForMatrixObject(_inputs[0], InputInfo.BinaryBlockInputInfo);
		    for( int i=1; i<_inputs.length; i++ ) {
			    JavaPairRDD<MatrixIndexes, MatrixBlock> rdd2 = (JavaPairRDD<MatrixIndexes, MatrixBlock>) 
			    		sec.getRDDHandleForMatrixObject(_inputs[i], InputInfo.BinaryBlockInputInfo);
			    rdd = rdd.union(rdd2);
		    }
		
		    //Step 2a: merge with compare
		    JavaPairRDD<MatrixIndexes, MatrixBlock> out = null;
		    if( withCompare )
		    {
		    	JavaPairRDD<MatrixIndexes, MatrixBlock> compareRdd = (JavaPairRDD<MatrixIndexes, MatrixBlock>) 
			    		sec.getRDDHandleForMatrixObject(compare, InputInfo.BinaryBlockInputInfo);
			    
		    	//merge values which differ from compare values
		    	ResultMergeRemoteSparkWCompare cfun = new ResultMergeRemoteSparkWCompare();
		    	out = rdd.groupByKey(numRed) //group all result blocks per key
		    	         .join(compareRdd)   //join compare block and result blocks 
		    	         .mapToPair(cfun);   //merge result blocks w/ compare
		    }
		    //Step 2b: merge without compare
		    else
		    {
		    	//direct merge in any order (disjointness guaranteed)
		    	out = RDDAggregateUtils.mergeByKey(rdd);
		    }
		    
		    //Step 3: create output rdd handle w/ lineage
		    ret = new RDDObject(out, varname);
		    //TODO need to add lineage information
		}
		catch( Exception ex )
		{
			throw new DMLRuntimeException(ex);
		}	    
		
		return ret;
	}

	/**
	 * 
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @param numRed
	 * @return
	 */
	private int determineNumReducers(long rlen, long clen, int brlen, int bclen, long numRed)
	{
		//set the number of mappers and reducers 
	    long reducerGroups = Math.max(rlen/brlen,1) * Math.max(clen/bclen, 1);
		int ret = (int)Math.min( numRed, reducerGroups );
	    
	    return ret; 	
	}
}
