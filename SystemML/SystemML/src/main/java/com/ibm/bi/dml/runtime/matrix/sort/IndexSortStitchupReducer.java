/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.matrix.sort;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;

public class IndexSortStitchupReducer extends MapReduceBase 
		implements Reducer<MatrixIndexes, MatrixBlock, MatrixIndexes, MatrixBlock>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private MatrixBlock _tmpBlk = null;
	
	@Override
	public void reduce(MatrixIndexes key, Iterator<MatrixBlock> values, OutputCollector<MatrixIndexes, MatrixBlock> out, Reporter report) 
		 throws IOException 
	{
		try
		{
			//handle first block (to handle dimensions)
			MatrixBlock tmp = values.next();
			_tmpBlk.reset(tmp.getNumRows(), tmp.getNumColumns());
			_tmpBlk.merge(tmp, false);		
			
			//handle remaining blocks
			while( values.hasNext() )
			{
				tmp = values.next();
				_tmpBlk.merge(tmp, false);
			}
		}
		catch(DMLRuntimeException ex)
		{
			throw new IOException(ex);
		}
		
		out.collect(key, _tmpBlk);
	}  
	
	@Override
	public void configure(JobConf job)
	{
		int brlen = MRJobConfiguration.getNumRowsPerBlock(job, (byte) 0);
		_tmpBlk = new MatrixBlock(brlen, 1, false);
	}
}
