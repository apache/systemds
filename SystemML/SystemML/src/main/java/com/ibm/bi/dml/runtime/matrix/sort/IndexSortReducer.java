/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.matrix.sort;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.matrix.SortMR;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.util.MapReduceTool;

public class IndexSortReducer extends MapReduceBase 
    implements Reducer<IndexSortComparable, LongWritable, MatrixIndexes, MatrixBlock>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	
	  private String _taskID = null;
	  private int _brlen = -1;
	  private MatrixIndexes _indexes = null;
	  private MatrixBlock _data = null;
	  private int _pos = 0;
	  
	  private OutputCollector<MatrixIndexes, MatrixBlock> _out = null;
	  
	  @Override
	  public void reduce(IndexSortComparable key, Iterator<LongWritable> values, OutputCollector<MatrixIndexes, MatrixBlock> out, Reporter report) 
		 throws IOException 
	  {
		  //cache output collector
		  _out = out;
		  
		  //output binary block
		  int count = 0;
		  while( values.hasNext() )
		  {
			  //flush full matrix block
			  if( _pos >= _brlen ) {
				  _indexes.setIndexes(_indexes.getRowIndex()+1, _indexes.getColumnIndex());
				  out.collect(_indexes, _data);
				  _pos = 0;
				  _data.reset(_brlen,1,false);
			  }
				  
			  _data.quickSetValue(_pos, 0, values.next().get());
			  _pos++;
			  count++;  
		  }
		  
		  report.incrCounter(SortMR.NUM_VALUES_PREFIX, _taskID, count);	
	  }  
		
	  @Override
	  public void configure(JobConf job) 
	  {
		  _taskID = MapReduceTool.getUniqueKeyPerTask(job, false);
		  _brlen = MRJobConfiguration.getNumRowsPerBlock(job, (byte) 0);
		  _pos = 0;
		  _data = new MatrixBlock(_brlen, 1, false);
		  //note the output indexes are a sequence for rows and the taskID for columns
		  //this is useful because the counts are collected over taskIDs as well, which
		  //later on makes the task of reshifting self contained 
		  _indexes = new MatrixIndexes(0, Long.parseLong(_taskID));
	  }
	  
	  @Override
	  public void close() 
		  throws IOException
	  {  
		  //flush final matrix block
		  if( _pos > 0 ){
			  _indexes.setIndexes(_indexes.getRowIndex()+1, _indexes.getColumnIndex());
			  _data.setNumRows(_pos);
			  _out.collect(_indexes, _data);
		  }
	  }
}

