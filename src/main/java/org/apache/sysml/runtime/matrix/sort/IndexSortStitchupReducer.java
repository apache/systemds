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
import java.util.Iterator;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.mapred.MRJobConfiguration;

public class IndexSortStitchupReducer extends MapReduceBase 
		implements Reducer<MatrixIndexes, MatrixBlock, MatrixIndexes, MatrixBlock>
{
	
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
