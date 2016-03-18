/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.runtime.io;

import java.io.EOFException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.util.MapReduceTool;

/**
 * Base class for all format-specific frame readers. Every reader is required to implement the basic 
 * read functionality but might provide additional custom functionality. Any non-default parameters
 * (e.g., CSV read properties) should be passed into custom constructors. There is also a factory
 * for creating format-specific readers. 
 * 
 */
public abstract class FrameReader 
{
	/**
	 * 
	 * @param fname
	 * @param schema
	 * @param names
	 * @param rlen
	 * @param clen
	 * @return
	 */
	public abstract FrameBlock readFrameFromHDFS( String fname, List<ValueType> schema, List<String> names,
			long rlen, long clen)
		throws IOException, DMLRuntimeException;
	
	/**
	 * 
	 * @param fname
	 * @param schema
	 * @param rlen
	 * @param clen
	 * @return
	 */
	public FrameBlock readFrameFromHDFS( String fname, List<ValueType> schema, long rlen, long clen )
		throws IOException, DMLRuntimeException
	{
		return readFrameFromHDFS(fname, schema, getDefColNames(schema.size()), rlen, clen);
	}
	
	/**
	 * 
	 * @param fname
	 * @param rlen
	 * @param clen
	 * @return
	 */
	public FrameBlock readFrameFromHDFS( String fname, long rlen, long clen )
		throws IOException, DMLRuntimeException
	{
		return readFrameFromHDFS(fname, getDefSchema(clen), getDefColNames(clen), rlen, clen);
	}
	
	/**
	 * 
	 * @param iNumColumns
	 * @return
	 */
	public List<ValueType> getDefSchema( long lNumColumns )
		throws IOException, DMLRuntimeException
	{
		List<ValueType> schema = new ArrayList<ValueType>();
		for (int i=0; i < lNumColumns; ++i)
			schema.add(ValueType.STRING);
		return schema;
	}

	/**
	 * 
	 * @param iNumColumns
	 * @return
	 */
	public List<String> getDefColNames( long lNumColumns )
		throws IOException, DMLRuntimeException
	{
		List<String> colNames = new ArrayList<String>();
		for (int i=0; i < lNumColumns; ++i)
			colNames.add("C"+i);
		return colNames;
	}

	/**
	 * 
	 * @param fs
	 * @param file
	 * @return
	 * @throws IOException
	 */
	public static Path[] getSequenceFilePaths( FileSystem fs, Path file ) 
		throws IOException
	{
		Path[] ret = null;
		
		if( fs.isDirectory(file) )
		{
			LinkedList<Path> tmp = new LinkedList<Path>();
			FileStatus[] dStatus = fs.listStatus(file);
			for( FileStatus fdStatus : dStatus )
				if( !fdStatus.getPath().getName().startsWith("_") ) //skip internal files
					tmp.add(fdStatus.getPath());
			ret = tmp.toArray(new Path[0]);
		}
		else
		{
			ret = new Path[]{ file };
		}
		
		return ret;
	}
	
	/**
	 * NOTE: mallocDense controls if the output matrix blocks is fully allocated, this can be redundant
	 * if binary block read and single block. 
	 * 
	 * @param schema
	 * @param names
	 * @return
	 * @throws DMLRuntimeException 
	 * @throws IOException 
	 */
	protected static FrameBlock createOutputFrameBlock(List<ValueType> schema, List<String> names)
		throws IOException, DMLRuntimeException
	{
		//check schema and column names
		if( !OptimizerUtils.isValidCPDimensions(schema, names) )
			throw new DMLRuntimeException("Schema and names to be define with equal size.");
		
		//prepare result frame block
		FrameBlock ret = new FrameBlock(schema, names);
		
		return ret;
	}
	
	/**
	 * 
	 * @param fs
	 * @param path
	 * @throws IOException 
	 */
	protected static void checkValidInputFile(FileSystem fs, Path path) 
		throws IOException
	{
		//check non-existing file
		if( !fs.exists(path) )	
			throw new IOException("File "+path.toString()+" does not exist on HDFS/LFS.");
	
		//check for empty file
		if( MapReduceTool.isFileEmpty( fs, path.toString() ) )
			throw new EOFException("Empty input file "+ path.toString() +".");
		
	}
}
