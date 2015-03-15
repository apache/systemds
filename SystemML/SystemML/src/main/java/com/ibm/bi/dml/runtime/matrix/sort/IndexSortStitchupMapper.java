/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.matrix.sort;

import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.matrix.SortMR;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;

public class IndexSortStitchupMapper extends MapReduceBase 
 	  implements Mapper<MatrixIndexes, MatrixBlock, MatrixIndexes, MatrixBlock>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
