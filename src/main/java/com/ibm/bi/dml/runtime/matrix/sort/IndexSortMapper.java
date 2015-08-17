/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.matrix.sort;

import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.matrix.SortMR;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;

public class IndexSortMapper extends MapReduceBase 
   implements Mapper<MatrixIndexes, MatrixBlock, IndexSortComparable, LongWritable>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	
	  private int _brlen = -1;
	  
	  //reuse writables
	  private LongWritable   _tmpLong = new LongWritable();
	  private IndexSortComparable _tmpSortKey = null;
		
	  @Override
	  public void map(MatrixIndexes key, MatrixBlock value, OutputCollector<IndexSortComparable, LongWritable> out, Reporter reporter) 
        throws IOException 
	  {
		  if( value.getNumColumns()>1 )
			  throw new IOException("IndexSort only supports column vectors, but found matrix block with clen="+value.getNumColumns());
		  
		  long row_offset = (key.getRowIndex()-1)*_brlen+1;
		  for( int i=0; i<value.getNumRows(); i++ )
		  {
			  double dval = value.quickGetValue(i, 0);
			  long lix = row_offset+i;
			  _tmpSortKey.set( dval, lix );
			  _tmpLong.set(lix);
			  out.collect(_tmpSortKey, _tmpLong);  
		  }
	  }
	
	  @Override
	  public void configure(JobConf job)
	  {
		 super.configure(job);
		 _brlen = MRJobConfiguration.getNumRowsPerBlock(job, (byte) 0);
		 boolean desc = job.getBoolean(SortMR.SORT_DECREASING, false);
		 if( !desc )
			 _tmpSortKey = new IndexSortComparable();
		 else
			 _tmpSortKey = new IndexSortComparableDesc();
	  }
}
