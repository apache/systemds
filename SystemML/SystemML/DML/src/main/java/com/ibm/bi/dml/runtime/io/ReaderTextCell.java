/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.io;

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

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.util.FastStringTokenizer;

public class ReaderTextCell extends MatrixReader
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private boolean _isMMFile = false;
	
	public ReaderTextCell(InputInfo info)
	{
		_isMMFile = (info == InputInfo.MatrixMarketInputInfo);
	}
	
	@Override
	public MatrixBlock readMatrixFromHDFS(String fname, long rlen, long clen, int brlen, int bclen, long estnnz) 
		throws IOException, DMLRuntimeException 
	{
		//allocate output matrix block
		MatrixBlock ret = createOutputMatrixBlock(rlen, clen, estnnz, true);
		
		//prepare file access
		JobConf job = new JobConf();	
		FileSystem fs = FileSystem.get(job);
		Path path = new Path( fname );
		
		//check existence and non-empty file
		checkValidInputFile(fs, path); 
	
		//core read 
		if( fs.isDirectory(path) )
			readTextCellMatrixFromHDFS(path, job, ret, rlen, clen, brlen, bclen);
		else
			readRawTextCellMatrixFromHDFS(path, job, fs, ret, rlen, clen, brlen, bclen, _isMMFile);
		
		//finally check if change of sparse/dense block representation required
		if( !ret.isInSparseFormat() )
			ret.recomputeNonZeros();
		ret.examSparsity();
		
		return ret;
	}

	public MatrixBlock readMatrixFromInputStream(InputStream is, long rlen, long clen, int brlen, int bclen, long estnnz) 
		throws IOException, DMLRuntimeException 
	{
		//allocate output matrix block
		MatrixBlock ret = createOutputMatrixBlock(rlen, clen, estnnz, true);
	
		//core read 
		readRawTextCellMatrixFromInputStream(is, ret, rlen, clen, brlen, bclen, _isMMFile);
		
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
	 * @param dest
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @throws IOException
	 * @throws IllegalAccessException
	 * @throws InstantiationException
	 */
	private void readTextCellMatrixFromHDFS( Path path, JobConf job, MatrixBlock dest, long rlen, long clen, int brlen, int bclen )
		throws IOException
	{
		boolean sparse = dest.isInSparseFormat();
		FileInputFormat.addInputPath(job, path);
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);
		InputSplit[] splits = informat.getSplits(job, 1);
		
		LongWritable key = new LongWritable();
		Text value = new Text();
		int row = -1;
		int col = -1;
		
		try
		{
			FastStringTokenizer st = new FastStringTokenizer(' ');
			
			for(InputSplit split: splits)
			{
				RecordReader<LongWritable,Text> reader = informat.getRecordReader(split, job, Reporter.NULL);
			
				try
				{
					if( sparse ) //SPARSE<-value
					{
						while( reader.next(key, value) )
						{
							st.reset( value.toString() ); //reinit tokenizer
							row = st.nextInt() - 1;
							col = st.nextInt() - 1;
							double lvalue = st.nextDouble();
							dest.appendValue(row, col, lvalue);
						}
						
						dest.sortSparseRows();
					} 
					else //DENSE<-value
					{
						while( reader.next(key, value) )
						{
							st.reset( value.toString() ); //reinit tokenizer
							row = st.nextInt()-1;
							col = st.nextInt()-1;
							double lvalue = st.nextDouble();
							dest.setValueDenseUnsafe( row, col, lvalue );
						}
					}
				}
				finally
				{
					if( reader != null )
						reader.close();
				}
			}
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
				throw new IOException( "Unable to read matrix in text cell format.", ex );
			}
		}
	}

	
	/**
	 * 
	 * @param path
	 * @param job
	 * @param dest
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @throws IOException
	 * @throws IllegalAccessException
	 * @throws InstantiationException
	 */
	private void readRawTextCellMatrixFromHDFS( Path path, JobConf job, FileSystem fs, MatrixBlock dest, long rlen, long clen, int brlen, int bclen, boolean matrixMarket )
		throws IOException
	{
		//create input stream for path
		InputStream inputStream = fs.open(path);
		
		//actual read
		readRawTextCellMatrixFromInputStream(inputStream, dest, rlen, clen, brlen, bclen, matrixMarket);
	}
	
	/**
	 * 
	 * @param is
	 * @param dest
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @param matrixMarket
	 * @throws IOException
	 * @throws IllegalAccessException
	 * @throws InstantiationException
	 */
	private void readRawTextCellMatrixFromInputStream( InputStream is, MatrixBlock dest, long rlen, long clen, int brlen, int bclen, boolean matrixMarket )
			throws IOException
	{
		BufferedReader br = new BufferedReader(new InputStreamReader( is ));	
		
		boolean sparse = dest.isInSparseFormat();
		String value = null;
		int row = -1;
		int col = -1;
		
		// Read the header lines, if reading from a matrixMarket file
		if ( matrixMarket ) {
			value = br.readLine(); // header line
			if ( value==null || !value.startsWith("%%") ) {
				throw new IOException("Error while reading file in MatrixMarket format. Expecting a header line, but encountered, \"" + value +"\".");
			}
			
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
			
			if( sparse ) //SPARSE<-value
			{
				while( (value=br.readLine())!=null )
				{
					st.reset( value ); //reinit tokenizer
					row = st.nextInt()-1;
					col = st.nextInt()-1;
					double lvalue = st.nextDouble();
					dest.appendValue(row, col, lvalue);
				}
				
				dest.sortSparseRows();
			} 
			else //DENSE<-value
			{
				while( (value=br.readLine())!=null )
				{
					st.reset( value ); //reinit tokenizer
					row = st.nextInt()-1;
					col = st.nextInt()-1;	
					double lvalue = st.nextDouble();
					dest.setValueDenseUnsafe( row, col, lvalue );
				}
			}
		}
		catch(Exception ex)
		{
			//post-mortem error handling and bounds checking
			if( row < 0 || row + 1 > rlen || col < 0 || col + 1 > clen ) 
			{
				throw new IOException("Matrix cell ["+(row+1)+","+(col+1)+"] " +
									  "out of overall matrix range [1:"+rlen+",1:"+clen+"].", ex);
			}
			else
			{
				throw new IOException( "Unable to read matrix in raw text cell format.", ex );
			}
		}
		finally
		{
			IOUtilFunctions.closeSilently(br);
		}
	}
}
