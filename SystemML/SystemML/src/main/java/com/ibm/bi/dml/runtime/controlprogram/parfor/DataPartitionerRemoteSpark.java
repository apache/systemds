/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor;

import org.apache.spark.api.java.JavaSparkContext;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.runtime.util.MapReduceTool;

/**
 * MR job class for submitting parfor remote partitioning MR jobs.
 *
 */
public class DataPartitionerRemoteSpark extends DataPartitioner
{	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//private boolean _keepIndexes = false;
	private ExecutionContext _ec = null;
	private long _numRed = -1;
	
	public DataPartitionerRemoteSpark(PDataPartitionFormat dpf, int n, ExecutionContext ec, long numRed, boolean keepIndexes) 
	{
		super(dpf, n);
	
		//TODO extend for general purpose partitioning
		//_keepIndexes = keepIndexes;
		
		_ec = ec;
		_numRed = numRed;
	}

	@Override
	@SuppressWarnings("unchecked")
	protected void partitionMatrix(String fname, String fnameNew, InputInfo ii, OutputInfo oi, long rlen, long clen, int brlen, int bclen)
			throws DMLRuntimeException 
	{
		SparkExecutionContext sec = (SparkExecutionContext)_ec;
		JavaSparkContext sc = sec.getSparkContext();

		try
		{
		    //cleanup existing output files
		    MapReduceTool.deleteFileIfExistOnHDFS(fnameNew);
	
		    //determine degree of parallelism
			int numRed = (int)determineNumReducers(rlen, clen, brlen, bclen, _numRed);
	
			//run spark remote data partition job 
			DataPartitionerRemoteSparkMapper dpfun = new DataPartitionerRemoteSparkMapper(rlen, clen, brlen, bclen, ii, oi, _format);
			DataPartitionerRemoteSparkReducer wfun = new DataPartitionerRemoteSparkReducer(fnameNew, oi);
			sc.hadoopFile(fname, ii.inputFormatClass, ii.inputKeyClass, ii.inputValueClass)
			  .flatMapToPair(dpfun) //partition the input blocks
			  .groupByKey(numRed)   //group partition blocks 		          
			  .foreach( wfun );     //write partitions to hdfs 
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException(ex);
		}
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
	private long determineNumReducers(long rlen, long clen, int brlen, int bclen, long numRed)
	{
		//set the number of mappers and reducers 
	    long reducerGroups = -1;
	    switch( _format )
	    {
		    case ROW_WISE: reducerGroups = rlen; break;
		    case COLUMN_WISE: reducerGroups = clen; break;
		    case ROW_BLOCK_WISE: reducerGroups = (rlen/brlen)+((rlen%brlen==0)?0:1); break;
		    case COLUMN_BLOCK_WISE: reducerGroups = (clen/bclen)+((clen%bclen==0)?0:1); break;
		    case ROW_BLOCK_WISE_N: reducerGroups = (rlen/_n)+((rlen%_n==0)?0:1); break;
		    case COLUMN_BLOCK_WISE_N: reducerGroups = (clen/_n)+((clen%_n==0)?0:1); break;
		    default:
				//do nothing
	    }
	    
	    return (int)Math.min( numRed, reducerGroups); 	
	}
}
