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

package org.tugraz.sysds.runtime.io;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;
import org.tugraz.sysds.conf.ConfigurationManager;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.data.DenseBlock;
import org.tugraz.sysds.runtime.matrix.data.IJV;
import org.tugraz.sysds.runtime.matrix.data.InputInfo;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.util.FastStringTokenizer;
import org.tugraz.sysds.runtime.util.HDFSTool;

public class ReaderTextCell extends MatrixReader
{
	protected final boolean _allowRawRead; 
	protected final boolean _isMMFile;
	protected FileFormatPropertiesMM _mmProps = null;
	
	public ReaderTextCell(InputInfo info) {
		this(info, true);
	}
	
	public ReaderTextCell(InputInfo info, boolean allowRaw) {
		_allowRawRead = allowRaw;
		_isMMFile = (info == InputInfo.MatrixMarketInputInfo);
	}
	
	@Override
	public MatrixBlock readMatrixFromHDFS(String fname, long rlen, long clen, int blen, long estnnz) 
		throws IOException, DMLRuntimeException 
	{
		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path( fname );
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
		
		//check existence and non-empty file
		checkValidInputFile(fs, path); 
		
		//read matrix market header
		if( _isMMFile )
			_mmProps = IOUtilFunctions.readAndParseMatrixMarketHeader(fname);
		
		//allocate output matrix block
		if( estnnz < 0 )
			estnnz = HDFSTool.estimateNnzBasedOnFileSize(path, rlen, clen, blen, 3);
		MatrixBlock ret = createOutputMatrixBlock(rlen, clen, (int)rlen, estnnz, true, false);
		
		//core read 
		if( fs.isDirectory(path) || !_allowRawRead )
			readTextCellMatrixFromHDFS(path, job, ret, rlen, clen, blen);
		else
			readRawTextCellMatrixFromHDFS(path, job, fs, ret, rlen, clen, blen, _isMMFile);
		
		//finally check if change of sparse/dense block representation required
		if( !AGGREGATE_BLOCK_NNZ )
			ret.recomputeNonZeros();
		ret.examSparsity();
		
		return ret;
	}

	@Override
	public MatrixBlock readMatrixFromInputStream(InputStream is, long rlen, long clen, int blen, long estnnz) 
		throws IOException, DMLRuntimeException 
	{
		//allocate output matrix block
		MatrixBlock ret = createOutputMatrixBlock(rlen, clen, blen, estnnz, true, false);
	
		//core read 
		readRawTextCellMatrixFromInputStream(is, ret, rlen, clen, blen, _isMMFile);
		
		//finally check if change of sparse/dense block representation required
		if( !AGGREGATE_BLOCK_NNZ )
			ret.recomputeNonZeros();
		ret.examSparsity();
		
		return ret;
	}

	protected void readTextCellMatrixFromHDFS( Path path, JobConf job, MatrixBlock dest, long rlen, long clen, int blen )
		throws IOException
	{
		boolean sparse = dest.isInSparseFormat();
		FileInputFormat.addInputPath(job, path);
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);
		InputSplit[] splits = informat.getSplits(job, 1);
		
		LongWritable key = new LongWritable();
		Text value = new Text();
		IJV cell = new IJV();
		long nnz = 0;
		
		try
		{
			FastStringTokenizer st = new FastStringTokenizer(' ');
			
			for(InputSplit split: splits) {
				RecordReader<LongWritable,Text> reader = informat.getRecordReader(split, job, Reporter.NULL);
				try {
					if( sparse ) { //SPARSE<-value
						while( reader.next(key, value) ) {
							cell = parseCell(value.toString(), st, cell, _mmProps);
							appendCell(cell, dest, _mmProps);
						}
						dest.sortSparseRows();
					} 
					else { //DENSE<-value
						DenseBlock a = dest.getDenseBlock();
						while( reader.next(key, value) ) {
							cell = parseCell(value.toString(), st, cell, _mmProps);
							nnz += appendCell(cell, a, _mmProps);
						}
					}
				}
				finally {
					IOUtilFunctions.closeSilently(reader);
				}
			}
			
			if( !dest.isInSparseFormat() )
				dest.setNonZeros(nnz);
		}
		catch(Exception ex) {
			//post-mortem error handling and bounds checking
			if( cell.getI() < 0 || cell.getI() + 1 > rlen || cell.getJ() < 0 || cell.getJ() + 1 > clen )
				throw new IOException("Matrix cell ["+(cell.getI()+1)+","+(cell.getJ()+1)+"] "
					+ "out of overall matrix range [1:"+rlen+",1:"+clen+"].");
			else
				throw new IOException( "Unable to read matrix in text cell format.", ex );
		}
	}
	
	protected static IJV parseCell(String line, FastStringTokenizer st, IJV cell, FileFormatPropertiesMM mmProps) {
		st.reset( line ); //reinit tokenizer
		int row = st.nextInt() - 1;
		int col = st.nextInt() - 1;
		double value = (mmProps == null) ? st.nextDouble() : 
			mmProps.isPatternField() ? 1 : mmProps.isIntField() ? st.nextLong() : st.nextDouble();
		return cell.set(row, col, value);
	}
	
	protected static int appendCell(IJV cell, MatrixBlock dest, FileFormatPropertiesMM mmProps) {
		if( cell.getV() == 0 ) return 0;
		dest.appendValue(cell.getI(), cell.getJ(), cell.getV());
		if( mmProps != null && mmProps.isSymmetric() && !cell.onDiag() ) {
			dest.appendValue(cell.getJ(), cell.getI(), cell.getV());
			return 2;
		}
		return 1;
	}
	
	protected static int appendCell(IJV cell, DenseBlock dest, FileFormatPropertiesMM mmProps) {
		if( cell.getV() == 0 ) return 0;
		dest.set(cell.getI(), cell.getJ(), cell.getV());
		if( mmProps != null && mmProps.isSymmetric() && ! cell.onDiag() ) {
			dest.set(cell.getJ(), cell.getI(), cell.getV());
			return 2;
		}
		return 1;
	}

	private static void readRawTextCellMatrixFromHDFS( Path path, JobConf job, FileSystem fs, MatrixBlock dest, long rlen, long clen, int blen, boolean matrixMarket )
		throws IOException
	{
		//create input stream for path
		InputStream inputStream = fs.open(path);
		
		//actual read
		readRawTextCellMatrixFromInputStream(inputStream, dest, rlen, clen, blen, matrixMarket);
	}

	private static void readRawTextCellMatrixFromInputStream( InputStream is, MatrixBlock dest, long rlen, long clen, int blen, boolean matrixMarket )
			throws IOException
	{
		BufferedReader br = new BufferedReader(new InputStreamReader( is ));
		FileFormatPropertiesMM mmProps = null;
		
		boolean sparse = dest.isInSparseFormat();
		String value = null;
		IJV cell = new IJV();
		long nnz = 0;
		
		// Read the header lines, if reading from a matrixMarket file
		if ( matrixMarket ) {
			value = br.readLine(); // header line
			if ( value==null || !value.startsWith("%%") ) {
				throw new IOException("Error while reading file in MatrixMarket format. Expecting a header line, but encountered, \"" + value +"\".");
			}
			mmProps = FileFormatPropertiesMM.parse(value);
			
			// skip until end-of-comments
			while( (value = br.readLine())!=null && value.charAt(0) == '%' ) {
				//do nothing just skip comments
			}
			
			// the first line after comments is the one w/ matrix dimensions
			// validate (rlen clen nnz)
			String[] fields = value.trim().split("\\s+"); 
			long mm_rlen = Long.parseLong(fields[0]);
			long mm_clen = Long.parseLong(fields[1]);
			if ( rlen != mm_rlen || clen != mm_clen ) {
				throw new IOException("Unexpected matrix dimensions while reading file in MatrixMarket format. Expecting dimensions [" + rlen + " rows, " + clen + " cols] but encountered [" + mm_rlen + " rows, " + mm_clen + "cols].");
			}
		}
		
		try
		{
			FastStringTokenizer st = new FastStringTokenizer(' ');
			
			if( sparse ) { //SPARSE<-value
				while( (value=br.readLine())!=null ) {
					cell = parseCell(value.toString(), st, cell, mmProps);
					appendCell(cell, dest, mmProps);
				}
				dest.sortSparseRows();
			} 
			else { //DENSE<-value
				DenseBlock a = dest.getDenseBlock();
				while( (value=br.readLine())!=null ) {
					cell = parseCell(value.toString(), st, cell, mmProps);
					nnz += appendCell(cell, a, mmProps);
				}
				dest.setNonZeros(nnz);
			}
		}
		catch(Exception ex) {
			//post-mortem error handling and bounds checking
			if( cell.getI() < 0 || cell.getI() + 1 > rlen || cell.getJ() < 0 || cell.getJ() + 1 > clen ) 
				throw new IOException("Matrix cell ["+(cell.getI()+1)+","+(cell.getJ()+1)+"] "
					+ "out of overall matrix range [1:"+rlen+",1:"+clen+"].", ex);
			else
				throw new IOException( "Unable to read matrix in raw text cell format.", ex );
		}
		finally {
			IOUtilFunctions.closeSilently(br);
		}
	}
}
