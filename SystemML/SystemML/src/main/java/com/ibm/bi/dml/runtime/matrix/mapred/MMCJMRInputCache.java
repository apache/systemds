/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.matrix.mapred;

import java.io.IOException;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.mapred.JobConf;

import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.data.Pair;

public class MMCJMRInputCache extends MMCJMRCache
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//in-memory buffer
 	private int cacheSize = 0;
 	private boolean flushedAll = false;
	private boolean memOnly;
	
	public MMCJMRInputCache( JobConf conf, long memSize, long rlen, long clen, int brlen, int bclen, boolean leftCached, Class<? extends MatrixValue> valueClass ) 
		throws IOException, IllegalAccessException, InstantiationException
	{
		long elementSize = 77+8*Math.min(rlen,brlen)*Math.min(clen,bclen)+20+12+12+4;//matrix block, matrix index, pair, integer in the linked list
		long numRowBlocks = leftCached ? (long)Math.ceil((double)rlen/(double)brlen) : 1;
		long numColBlocks = leftCached ? 1 : (long)Math.ceil((double)clen/(double)bclen);
		
		int buffCapacity = (int)Math.max(Math.min((memSize/elementSize), (numRowBlocks*numColBlocks)), 1);
		super.allocateBuffer(buffCapacity, valueClass, false);
		
		//local file management (if necessary)
		int n = (int)Math.ceil((double)(numRowBlocks*numColBlocks)/(double)_bufferCapacity);
		memOnly = (n==1);
		if( !memOnly )
		{
			_job = conf;
			_fs = FileSystem.getLocal(_job);
			_fileN = n;
			super.constructLocalFilePrefix("_input_cache_");
			super.deleteAllWorkingFiles();
		}		
	}
	
	public int getCacheSize()
	{
		return cacheSize;
	}
	
	/**
	 * 
	 * @param inIndex
	 * @param inValue
	 * @throws Exception
	 */
	public void put( long inIndex, MatrixValue inValue ) 
		throws Exception
	{
		if( !memOnly )
		{
			int newFileCursor = (int) (cacheSize / _bufferCapacity);
			if( _fileCursor!=-1 && _fileCursor!=newFileCursor ){
				super.writeBuffer();
				_bufferSize = 0;
			}
			_fileCursor = newFileCursor;
		}

		int lpos = cacheSize % _bufferCapacity;
		
		Pair<MatrixIndexes,MatrixValue> tmp = _buffer[lpos];
		tmp.getKey().setIndexes(inIndex,inIndex);
		tmp.getValue().copy(inValue);
		
		_bufferSize++;
		cacheSize++;
	}
	
	/**
	 * 
	 * @param pos
	 * @return
	 * @throws IOException
	 */
	public Pair<MatrixIndexes,MatrixValue> get( int pos ) 
		throws IOException
	{
		if( !memOnly )
		{
			int newFileCursor = (int) (pos / _bufferCapacity);
			if( _fileCursor!=newFileCursor )
			{
				//flush last cursor (on first get after put sequence)
				if( !flushedAll ) {
					super.writeBuffer();
					_bufferSize = 0;
					flushedAll = true;
				}
				
				//load new filecursor partition
				_fileCursor = newFileCursor;
				super.loadBuffer();
			}
		}
		
		//get cached in-memory block
		int lpos = pos % _bufferCapacity;
		return _buffer[lpos];
	}
	
	/**
	 * 
	 * @throws IOException
	 */
	public void resetCache() 
		throws IOException
	{
		//by default don't reset buffersize (e.g., for aggregator)
		resetCache(false);
	}
	
	/**
	 * @throws IOException 
	 * 
	 */
	public void resetCache(boolean fullreset) 
		throws IOException
	{
		cacheSize = 0;
		flushedAll = false;
		
		if(fullreset)
			_bufferSize = 0;
		
		if( !memOnly )
			super.deleteAllWorkingFiles();
	}
	
	public void close() 
		throws IOException
	{
		if( !memOnly )
			super.deleteAllWorkingFiles();	
	}
}
