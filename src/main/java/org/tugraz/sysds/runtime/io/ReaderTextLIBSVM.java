/*
 * Copyright 2019 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tugraz.sysds.runtime.io;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.commons.lang.mutable.MutableInt;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.tugraz.sysds.conf.ConfigurationManager;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.data.SparseRowVector;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;

public class ReaderTextLIBSVM extends MatrixReader 
{
	public ReaderTextLIBSVM() {

	}

	@Override
	public MatrixBlock readMatrixFromHDFS(String fname, long rlen, long clen, int blen, long estnnz) 
		throws IOException, DMLRuntimeException 
	{
		//allocate output matrix block
		MatrixBlock ret = null;
		if( rlen>=0 && clen>=0 ) //otherwise allocated on read
			ret = createOutputMatrixBlock(rlen, clen, (int)rlen, estnnz, true, false);
		
		//prepare file access
		JobConf    job  =  new JobConf(ConfigurationManager.getCachedJobConf());
		Path       path =  new Path( fname );
		FileSystem fs   =  IOUtilFunctions.getFileSystem(path, job);
		
		//check existence and non-empty file
		checkValidInputFile(fs, path); 
	
		//core read 
		ret = readLIBSVMMatrixFromHDFS(path, job, fs, ret, rlen, clen, blen);
		
		//finally check if change of sparse/dense block representation required
		//(nnz explicitly maintained during read)
		ret.examSparsity();
		
		return ret;
	}
	
	@Override
	public MatrixBlock readMatrixFromInputStream(InputStream is, long rlen, long clen, int blen, long estnnz) 
		throws IOException, DMLRuntimeException 
	{
		//allocate output matrix block
		MatrixBlock ret = createOutputMatrixBlock(rlen, clen, (int)rlen, estnnz, true, false);
		
		//core read 
		long lnnz = readLIBSVMMatrixFromInputStream(is, "external inputstream", ret,
			new MutableInt(0), rlen, clen, blen);
		
		//finally check if change of sparse/dense block representation required
		ret.setNonZeros( lnnz );
		ret.examSparsity();
		
		return ret;
	}
	
	@SuppressWarnings("unchecked")
	private static MatrixBlock readLIBSVMMatrixFromHDFS( Path path, JobConf job, FileSystem fs, MatrixBlock dest, 
			long rlen, long clen, int blen)
		throws IOException, DMLRuntimeException
	{
		//prepare file paths in alphanumeric order
		ArrayList<Path> files=new ArrayList<>();
		if(fs.isDirectory(path)) {
			for(FileStatus stat: fs.listStatus(path, IOUtilFunctions.hiddenFileFilter))
				files.add(stat.getPath());
			Collections.sort(files);
		}
		else
			files.add(path);
		
		//determine matrix size via additional pass if required
		if ( dest == null ) {
			dest = computeLIBSVMSize(files, clen, job, fs);
			clen = dest.getNumColumns();
		}
		
		//actual read of individual files
		long lnnz = 0;
		MutableInt row = new MutableInt(0);
		for(int fileNo=0; fileNo<files.size(); fileNo++) {
			lnnz += readLIBSVMMatrixFromInputStream(fs.open(files.get(fileNo)),
				path.toString(), dest, row, rlen, clen, blen);
		}
		
		//post processing
		dest.setNonZeros( lnnz );
		
		return dest;
	}
	
	private static long readLIBSVMMatrixFromInputStream( InputStream is, String srcInfo, MatrixBlock dest, MutableInt rowPos, 
			long rlen, long clen, int blen )
		throws IOException
	{
		SparseRowVector vect = new SparseRowVector(1024);
		String value = null;
		int row = rowPos.intValue();
		long lnnz = 0;
		
		// Read the data
		try( BufferedReader br = new BufferedReader(new InputStreamReader(is)) ) {
			while( (value=br.readLine())!=null ) { //for each line
				String rowStr = value.toString().trim();
				lnnz += ReaderTextLIBSVM.parseLibsvmRow(rowStr, vect, (int)clen);
				dest.appendRow(row, vect);
				row++;
			}
		}
		
		rowPos.setValue(row);
		return lnnz;
	}

	private static MatrixBlock computeLIBSVMSize(List<Path> files, long ncol, JobConf job, FileSystem fs) 
		throws IOException, DMLRuntimeException 
	{
		int nrow = -1;
		for(int fileNo=0; fileNo<files.size(); fileNo++) {
			try( BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(files.get(fileNo)))) ) {
				while( br.readLine() != null )
					nrow++;
			}
		}
		
		// allocate target matrix block based on given size; 
		return createOutputMatrixBlock(nrow, ncol, 
			nrow, (long)nrow*ncol, true, false);
	}
	
	protected static int parseLibsvmRow(String rowStr, SparseRowVector vect, int clen) {
		// reset row buffer (but keep allocated arrays)
		vect.setSize(0);
		
		//parse row w/ first entry being the label
		String[] parts = rowStr.split(IOUtilFunctions.LIBSVM_DELIM);
		double label = Double.parseDouble(parts[0]);
		
		//parse entire row
		for( int i=1; i<parts.length; i++ ) {
			//parse non-zero: <index#>:<value#>
			String[] pair = parts[i].split(IOUtilFunctions.LIBSVM_INDEX_DELIM);
			vect.append(Integer.parseInt(pair[0])-1, Double.parseDouble(pair[1]));
		}
		vect.append(clen-1, label);
		return vect.size();
	}
}