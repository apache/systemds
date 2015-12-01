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

package org.apache.sysml.runtime.matrix.mapred;

import java.io.IOException;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.mapred.JobConf;

import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.data.Pair;

public class MMCJMRInputCache extends MMCJMRCache
{
	
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
