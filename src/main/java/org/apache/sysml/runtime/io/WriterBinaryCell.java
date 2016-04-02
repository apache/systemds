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

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.matrix.data.IJV;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixCell;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.util.MapReduceTool;

public class WriterBinaryCell extends MatrixWriter
{

	@Override
	public void writeMatrixToHDFS(MatrixBlock src, String fname, long rlen, long clen, int brlen, int bclen, long nnz) 
		throws IOException, DMLRuntimeException, DMLUnsupportedOperationException 
	{
		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path( fname );

		//if the file already exists on HDFS, remove it.
		MapReduceTool.deleteFileIfExistOnHDFS( fname );
			
		//core write
		writeBinaryCellMatrixToHDFS(path, job, src, rlen, clen, brlen, bclen);
	}

	@Override
	@SuppressWarnings("deprecation")
	public void writeEmptyMatrixToHDFS(String fname, long rlen, long clen, int brlen, int bclen) 
		throws IOException, DMLRuntimeException 
	{
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path( fname );
		FileSystem fs = FileSystem.get(job);

		SequenceFile.Writer writer = new SequenceFile.Writer(fs, job, path,
                MatrixIndexes.class, MatrixCell.class);
		
		MatrixIndexes index = new MatrixIndexes(1, 1);
		MatrixCell cell = new MatrixCell(0);
		writer.append(index, cell);
		writer.close();
	}

	/**
	 * 
	 * @param path
	 * @param job
	 * @param src
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @throws IOException
	 */
	@SuppressWarnings("deprecation")
	protected void writeBinaryCellMatrixToHDFS( Path path, JobConf job, MatrixBlock src, long rlen, long clen, int brlen, int bclen )
		throws IOException
	{
		boolean sparse = src.isInSparseFormat();
		boolean entriesWritten = false;
		FileSystem fs = FileSystem.get(job);
		SequenceFile.Writer writer = new SequenceFile.Writer(fs, job, path, MatrixIndexes.class, MatrixCell.class);
		
		MatrixIndexes indexes = new MatrixIndexes();
		MatrixCell cell = new MatrixCell();

		int rows = src.getNumRows(); 
		int cols = src.getNumColumns();
        
		try
		{
			//bound check per block
			if( rows > rlen || cols > clen )
			{
				throw new IOException("Matrix block [1:"+rows+",1:"+cols+"] " +
						              "out of overall matrix range [1:"+rlen+",1:"+clen+"].");
			}
		
			if( sparse ) //SPARSE
			{
				
				Iterator<IJV> iter = src.getSparseBlockIterator();
				while( iter.hasNext() )
				{
					IJV lcell = iter.next();
					indexes.setIndexes(lcell.getI()+1, lcell.getJ()+1);
					cell.setValue(lcell.getV());
					writer.append(indexes, cell);
					entriesWritten = true;
				}
			}
			else //DENSE
			{
				for( int i=0; i<rows; i++ )
					for( int j=0; j<cols; j++ )
					{
						double lvalue  = src.getValueDenseUnsafe(i, j);
						if( lvalue != 0 ) //for nnz
						{
							indexes.setIndexes(i+1, j+1);
							cell.setValue(lvalue);
							writer.append(indexes, cell);
							entriesWritten = true;
						}
					}
			}
	
			//handle empty result
			if ( !entriesWritten ) {
				writer.append(new MatrixIndexes(1, 1), new MatrixCell(0));
			}
		}
		finally
		{
			IOUtilFunctions.closeSilently(writer);
		}
	}
}
