/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package org.apache.sysml.runtime.matrix.sort;

import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import org.apache.sysml.runtime.matrix.SortMR;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.mapred.MRJobConfiguration;

public class IndexSortMapper extends MapReduceBase 
   implements Mapper<MatrixIndexes, MatrixBlock, IndexSortComparable, LongWritable>
{
		
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
