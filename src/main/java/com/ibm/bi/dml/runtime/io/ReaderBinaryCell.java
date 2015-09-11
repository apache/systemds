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

package com.ibm.bi.dml.runtime.io;

import java.io.IOException;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.JobConf;

import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;

/**
 * 
 * 
 */
public class ReaderBinaryCell extends MatrixReader
{

	@Override
	public MatrixBlock readMatrixFromHDFS(String fname, long rlen, long clen, int brlen, int bclen, long estnnz) 
		throws IOException, DMLRuntimeException 
	{
		//allocate output matrix block
		MatrixBlock ret = createOutputMatrixBlock(rlen, clen, estnnz, true, false);
		
		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());	
		FileSystem fs = FileSystem.get(job);
		Path path = new Path( fname );
		
		//check existence and non-empty file
		checkValidInputFile(fs, path); 
	
		//core read 
		readBinaryCellMatrixFromHDFS(path, job, fs, ret, rlen, clen, brlen, bclen);
		
		//finally check if change of sparse/dense block representation required
		if( !ret.isInSparseFormat() )
			ret.recomputeNonZeros();
		ret.examSparsity();
		
		return ret;
	}

	/**
	 * 
	 * @param path
	 * @param job
	 * @param fs
	 * @param dest
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @throws IOException
	 */
	@SuppressWarnings("deprecation")
	private void readBinaryCellMatrixFromHDFS( Path path, JobConf job, FileSystem fs, MatrixBlock dest, long rlen, long clen, int brlen, int bclen )
		throws IOException
	{
		boolean sparse = dest.isInSparseFormat();		
		MatrixIndexes key = new MatrixIndexes();
		MatrixCell value = new MatrixCell();
		int row = -1;
		int col = -1;
		
		try
		{
			for( Path lpath : getSequenceFilePaths(fs,path) ) //1..N files 
			{
				//directly read from sequence files (individual partfiles)
				SequenceFile.Reader reader = new SequenceFile.Reader(fs,lpath,job);
				
				try
				{
					if( sparse )
					{
						while(reader.next(key, value))
						{
							row = (int)key.getRowIndex()-1;
							col = (int)key.getColumnIndex()-1;
							double lvalue = value.getValue();
							//dest.quickSetValue( row, col, lvalue );
							dest.appendValue(row, col, lvalue);
						}
					}
					else
					{
						while(reader.next(key, value))
						{
							row = (int)key.getRowIndex()-1;
							col = (int)key.getColumnIndex()-1;
							double lvalue = value.getValue();
							dest.setValueDenseUnsafe( row, col, lvalue );
						}
					}
				}
				finally
				{
					IOUtilFunctions.closeSilently(reader);
				}
			}
			
			if( sparse )
				dest.sortSparseRows();
		}
		catch(Exception ex)
		{
			//post-mortem error handling and bounds checking
			if( row < 0 || row + 1 > rlen || col < 0 || col + 1 > clen )
			{
				throw new IOException("Matrix cell ["+(row+1)+","+(col+1)+"] " +
									  "out of overall matrix range [1:"+rlen+",1:"+clen+"].");
			}
			else
			{
				throw new IOException( "Unable to read matrix in binary cell format.", ex );
			}
		}
	}
	
}
