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
import java.util.StringTokenizer;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import org.apache.sysml.runtime.matrix.SortMR;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.mapred.MRJobConfiguration;

public class IndexSortStitchupMapper extends MapReduceBase 
 	  implements Mapper<MatrixIndexes, MatrixBlock, MatrixIndexes, MatrixBlock>
{
	
  	private long[] _offsets = null;
  	private long _rlen = -1;
  	private long _brlen = -1;
  	
  	private MatrixBlock _tmpBlk = null;
  	private MatrixIndexes _tmpIx = null;
  	
	@Override
	public void map(MatrixIndexes key, MatrixBlock value, OutputCollector<MatrixIndexes, MatrixBlock> out, Reporter reporter) 
		throws IOException 
	{
		//compute starting cell offset
		int id = (int)key.getColumnIndex();
		long offset = _offsets[id];
		offset += (key.getRowIndex()-1)*_brlen;
		
		//SPECIAL CASE: block aligned
		int blksize = computeOutputBlocksize(_rlen, _brlen, offset);
		if( offset%_brlen==0 && value.getNumRows()==blksize ) 
		{ 
			_tmpIx.setIndexes(offset/_brlen+1, 1);
			out.collect(_tmpIx, value);
		}
		//GENERAL CASE: not block aligned
		else 
		{
			int loffset = (int) (offset%_brlen);
			//multiple output blocks
			if( value.getNumRows()+loffset>_brlen ) 
			{
				long tmpnnz = 0;
				//output first part
				_tmpBlk.reset( (int)_brlen, 1 );
				for( int i=0; i<_brlen-loffset; i++ )
					_tmpBlk.quickSetValue(loffset+i, 0, value.quickGetValue(i, 0));
				tmpnnz += _tmpBlk.getNonZeros();
				_tmpIx.setIndexes(offset/_brlen+1, 1);
				out.collect(_tmpIx, _tmpBlk);		
			
				//output second block
				blksize = computeOutputBlocksize(_rlen, _brlen, offset+(_brlen-loffset));
				_tmpBlk.reset( blksize, 1 );
				for( int i=(int)_brlen-loffset; i<value.getNumRows(); i++ )
					_tmpBlk.quickSetValue(i-((int)_brlen-loffset), 0, value.quickGetValue(i, 0));
				tmpnnz += _tmpBlk.getNonZeros();
				_tmpIx.setIndexes(offset/_brlen+2, 1);
				out.collect(_tmpIx, _tmpBlk);	
				
				//sanity check for correctly redistributed non-zeros
				if( tmpnnz != value.getNonZeros() )
					throw new IOException("Number of split non-zeros does not match non-zeros of original block ("+tmpnnz+" vs "+value.getNonZeros()+")");
			}
			//single output block
			else 
			{	
				_tmpBlk.reset( blksize, 1 );
				for( int i=0; i<value.getNumRows(); i++ )
					_tmpBlk.quickSetValue(loffset+i, 0, value.quickGetValue(i, 0));
				_tmpIx.setIndexes(offset/_brlen+1, 1);
				out.collect(_tmpIx, _tmpBlk);		
			}
		}
	}
	
	@Override
	public void configure(JobConf job)
	{
		super.configure(job);
		_offsets = parseOffsets(job.get(SortMR.SORT_INDEXES_OFFSETS));
		_rlen = MRJobConfiguration.getNumRows(job, (byte) 0);
		_brlen = MRJobConfiguration.getNumRowsPerBlock(job, (byte) 0);
		
		_tmpIx = new MatrixIndexes();
		_tmpBlk = new MatrixBlock((int)_brlen, 1, false);
	}
	
	
	/**
	 * 
	 * @param str
	 * @return
	 */
	private static long[] parseOffsets(String str)
	{
		String counts = str.substring(1, str.length() - 1);
		StringTokenizer st = new StringTokenizer(counts, ",");
		int len = st.countTokens();
		long[] ret = new long[len];
		for( int i=0; i<len; i++ )
			ret[i] = Long.parseLong(st.nextToken().trim());
		
		return ret;
	}
	
	private static int computeOutputBlocksize( long rlen, long brlen, long offset )
	{
		long rix = offset/brlen+1;
		int blksize = (int) Math.min(brlen, rlen-(rix-1)*brlen);

		return blksize;
	}
}
