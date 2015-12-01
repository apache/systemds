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
import java.util.HashMap;
import java.util.Random;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;

import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.data.Pair;
import org.apache.sysml.runtime.util.LocalFileUtils;
import org.apache.sysml.runtime.util.MapReduceTool;

/**
 * Base class for MMCJ partial aggregator and MMCJ input cache to factor out
 * common file management.
 * 
 */
public abstract class MMCJMRCache 
{
	
	protected Pair<MatrixIndexes,MatrixValue>[] _buffer = null;
 	protected int _bufferCapacity = 0;
	protected int _bufferSize = 0;
	protected HashMap<MatrixIndexes,Integer> _bufferMap = null; //optional

	protected JobConf _job = null;
	protected FileSystem _fs = null;
	protected int _fileCursor = -1;
	protected String _filePrefix = null;
	protected int _fileN = -1;

	/**
	 * 
	 * @return
	 */
	public HashMap<MatrixIndexes,Integer> getBufferMap()
	{
		return _bufferMap;
	}
	
	/**
	 * 
	 * @param buffCapacity
	 * @param valueClass
	 * @param buffMap
	 * @throws IllegalAccessException
	 * @throws InstantiationException
	 */
	@SuppressWarnings("unchecked")
	protected void allocateBuffer( int buffCapacity, Class<? extends MatrixValue> valueClass, boolean buffMap ) 
		throws IllegalAccessException, InstantiationException
	{
		_bufferCapacity = buffCapacity;
		_buffer = new Pair[_bufferCapacity];
		for(int i=0; i<_bufferCapacity; i++)
			_buffer[i] = new Pair<MatrixIndexes, MatrixValue>(new MatrixIndexes(), valueClass.newInstance());
		if( buffMap )
			_bufferMap = new HashMap<MatrixIndexes, Integer>();
	
		//System.out.println("allocated buffer: "+_bufferCapacity);
	}
	
	/**
	 * 
	 * @param fname
	 */
	protected void constructLocalFilePrefix(String fname)
	{
		//get random localdir (to spread load across available disks)
		String[] localDirs = _job.get("mapred.local.dir").split(",");
		Random rand = new Random();
		int randPos = rand.nextInt(localDirs.length);
				
		//construct file prefix
		String hadoopLocalDir = localDirs[ randPos ];
		String uniqueSubDir = MapReduceTool.getGloballyUniqueName(_job);
		_filePrefix = new Path(hadoopLocalDir, uniqueSubDir + fname).toString();
	}
	
	/**
	 * 
	 * @param fileCursor
	 * @return
	 */
	protected Path getFilePath( int fileCursor )
	{
		Path path = new Path( _filePrefix + fileCursor );
		return path;
	}
	
	/**
	 * 
	 * @throws IOException
	 */
	protected void loadBuffer() 
		throws IOException
	{
		_bufferSize=0;
		if( _bufferMap!=null )
			_bufferMap.clear();
		
		Path path = getFilePath(_fileCursor);
		
		if( _fs.exists(path) ) {
			//System.out.println( "load buffer: "+path.toString() );
			_bufferSize = LocalFileUtils.readBlockSequenceFromLocal(path.toString(), _buffer, _bufferMap);
		}
	}
	
	/**
	 * 
	 * @throws IOException
	 */
	protected void writeBuffer() 
		throws IOException 
	{
		if(_fileCursor<0 || _bufferSize<=0)
			return;
		
		//the old file will be overwritten
		Path path = getFilePath(_fileCursor);
		//System.out.println( "write buffer: "+path.toString() );
		LocalFileUtils.writeBlockSequenceToLocal(path.toString(), _buffer, _bufferSize);
	}


	/**
	 * 
	 * @throws IOException
	 */
	protected void deleteAllWorkingFiles() 
		throws IOException
	{
		//delete existing files
		for(int i=0; i<_fileN; i++) {
			Path ifile = new Path(_filePrefix+i);
			MapReduceTool.deleteFileIfExistOnLFS(ifile, _job);
		}
	}
}
