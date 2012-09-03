package com.ibm.bi.dml.runtime.util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.InputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.TextInputFormat;

import com.ibm.bi.dml.runtime.matrix.io.BinaryBlockToBinaryCellConverter;
import com.ibm.bi.dml.runtime.matrix.io.BinaryBlockToTextCellConverter;
import com.ibm.bi.dml.runtime.matrix.io.Converter;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlockDSM;
import com.ibm.bi.dml.runtime.matrix.io.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.io.Pair;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;


/**
 * This class provides methods to read and write matrix blocks from to HDFS using different data formats.
 * Those functionalities are used especially for CP read/write and exporting in-memory matrices to HDFS
 * (before executing MR jobs).
 * 
 */
public class DataConverter 
{
	//////////////
	// READING and WRITING of matrix blocks to/from HDFS
	// (textcell, binarycell, binaryblock)
	///////
	
	/**
	 * 
	 * @param mat
	 * @param dir
	 * @param outputinfo
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @throws IOException
	 */
	public static void writeMatrixToHDFS(MatrixBlock mat, String dir, OutputInfo outputinfo, 
										 long rlen, long clen, int brlen, int bclen )
		throws IOException
	{
		JobConf job = new JobConf();
		Path path = new Path(dir);
		FileOutputFormat.setOutputPath(job, path);

		//NOTE: MB creating the sparse map was slower than iterating over the array, however, this might change with an iterator interface
		// for sparsity=0.1 iterating over the whole block was 2x faster
		try
		{
			// If the file already exists on HDFS, remove it.
			MapReduceTool.deleteFileIfExistOnHDFS(dir);

			// core matrix writing
			if ( outputinfo == OutputInfo.TextCellOutputInfo ) 
			{	
				writeTextCellMatrixToHDFS(path, job, mat, rlen, clen, brlen, bclen);
			}
			else if ( outputinfo == OutputInfo.BinaryCellOutputInfo ) 
			{
				writeBinaryCellMatrixToHDFS(path, job, mat, rlen, clen, brlen, bclen);
			}
			else if( outputinfo == OutputInfo.BinaryBlockOutputInfo )
			{
				writeBinaryBlockMatrixToHDFS(path, job, mat, rlen, clen, brlen, bclen);
			}
		}
		catch(Exception e)
		{
			throw new IOException(e);
		}
	}	

	/**
	 * 
	 * @param dir
	 * @param inputinfo
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @return
	 * @throws IOException
	 */
	public static MatrixBlock readMatrixFromHDFS(String dir, InputInfo inputinfo, long rlen, long clen, 
			int brlen, int bclen) 
	throws IOException
	{	
		//expected matrix is sparse (default SystemML usecase)
		return readMatrixFromHDFS(dir, inputinfo, rlen, clen, brlen, bclen, 0.1d);
	}
	
	/**
	 * NOTE: providing an exact estimate of 'expected sparsity' can prevent a full copy of the result
	 * matrix block (required for changing sparse->dense, or vice versa)
	 * 
	 * @param dir
	 * @param inputinfo
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @param expectedSparsity
	 * @return
	 * @throws IOException
	 */
	public static MatrixBlock readMatrixFromHDFS(String dir, InputInfo inputinfo, long rlen, long clen, 
			int brlen, int bclen, double expectedSparsity) 
	throws IOException
	{	
		boolean sparse = expectedSparsity < MatrixBlockDSM.SPARCITY_TURN_POINT;

		//System.out.println("DataConverter: read matrix from HDFS ("+dir+").");
		
		// TODO: fix memory problem, and remove forced dense afterwards
		// force dense representation for 1D matrices (vectors)
		if ( rlen == 1 || clen == 1 )
			sparse = false;
		
		//prepare result matrix block
		MatrixBlock ret = new MatrixBlock((int)rlen, (int)clen, sparse);
		if( !sparse )
			ret.spaceAllocForDenseUnsafe((int)rlen, (int)clen);
		
		//prepare file access
		JobConf job = new JobConf();	
		Path path = new Path(dir);
		FileSystem fs = FileSystem.get(job);
		if( !fs.exists(path) )	
			throw new IOException("File "+dir+" does not exist on HDFS.");
		FileInputFormat.addInputPath(job, path); 
		
		try 
		{
			//core matrix reading 
			if( inputinfo == InputInfo.TextCellInputInfo )
			{			
				if( fs.getFileStatus(path).isDir() )
					readTextCellMatrixFromHDFS(path, job, ret, rlen, clen, brlen, bclen);
				else
					readRawTextCellMatrixFromHDFS(path, job, ret, rlen, clen, brlen, bclen);
			}
			else if( inputinfo == InputInfo.BinaryCellInputInfo )
			{
				readBinaryCellMatrixFromHDFS( path, job, ret, rlen, clen, brlen, bclen );
			}
			else if( inputinfo == InputInfo.BinaryBlockInputInfo )
			{
				readBinaryBlockMatrixFromHDFS( path, job, ret, rlen, clen, brlen, bclen );
			}
			
			//finally check if change of sparse/dense block representation required
			if( !sparse )
				ret.recomputeNonZeros();
			ret.examSparsity();	
		} 
		catch (Exception e) 
		{
			throw new IOException(e);
		}
		
		return ret;
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
	private static void writeTextCellMatrixToHDFS( Path path, JobConf job, MatrixBlock src, long rlen, long clen, int brlen, int bclen )
		throws IOException
	{
		boolean sparse = src.isInSparseFormat();
		boolean entriesWritten = false;
		FileSystem fs = FileSystem.get(job);
        BufferedWriter br=new BufferedWriter(new OutputStreamWriter(fs.create(path,true)));		
        
    	int rows = src.getNumRows();
		int cols = src.getNumColumns();

		//bound check per block
		if( rows > rlen || cols > clen )
		{
			throw new IOException("Matrix block [1:"+rows+",1:"+cols+"] " +
					              "out of overall matrix range [1:"+rlen+",1:"+clen+"].");
		}
		
		try
		{
			if( sparse ) //SPARSE
			{			   
				for( int i=0; i<rows; i++ )
					for( int j=0; j<cols; j++ )
					{
						double lvalue = src.getValueSparseUnsafe(i, j);
						if( lvalue != 0 ) //for nnz
						{
							StringBuilder sb = new StringBuilder();
							sb.append(i+1);
							sb.append(" ");
							sb.append(j+1);
							sb.append(" ");
							sb.append(lvalue);
							sb.append("\n");
							br.write( sb.toString() ); //same as append
							entriesWritten = true;
						}
					}
			}
			else //DENSE
			{
				for( int i=0; i<rows; i++ )
					for( int j=0; j<cols; j++ )
					{
						double lvalue = src.getValueDenseUnsafe(i, j);
						if( lvalue != 0 ) //for nnz
						{
							StringBuilder sb = new StringBuilder();
							sb.append(i+1);
							sb.append(" ");
							sb.append(j+1);
							sb.append(" ");
							sb.append(lvalue);
							sb.append("\n");
							br.write( sb.toString() ); //same as append
							entriesWritten = true;
						}
					}
			}
	
			//handle empty result
			if ( !entriesWritten ) {
				br.write("1 1 0\n");
			}
		}
		finally
		{
			if( br != null )
				br.close();
		}
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
	private static void writeBinaryCellMatrixToHDFS( Path path, JobConf job, MatrixBlock src, long rlen, long clen, int brlen, int bclen )
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
        
		//bound check per block
		if( rows > rlen || cols > clen )
		{
			throw new IOException("Matrix block [1:"+rows+",1:"+cols+"] " +
					              "out of overall matrix range [1:"+rlen+",1:"+clen+"].");
		}
		
		try
		{
			if( sparse ) //SPARSE
			{
				for( int i=0; i<rows; i++ )
					for( int j=0; j<cols; j++ )
					{
						double lvalue  = src.getValueSparseUnsafe(i, j); 
						if( lvalue != 0 ) //for nnz
						{
							indexes.setIndexes(i+1, j+1);
							cell.setValue(lvalue);
							writer.append(indexes, cell);
							entriesWritten = true;
						}
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
			if( writer != null )
				writer.close();
		}
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
	private static void writeBinaryBlockMatrixToHDFS( Path path, JobConf job, MatrixBlock src, long rlen, long clen, int brlen, int bclen )
		throws IOException
	{
		boolean sparse = src.isInSparseFormat();
		FileSystem fs = FileSystem.get(job);
		SequenceFile.Writer writer = new SequenceFile.Writer(fs, job, path, MatrixIndexes.class, MatrixBlock.class);
		
		//reblock and write		
		MatrixBlock fullBlock = new MatrixBlock(brlen, bclen, false);	
		fullBlock.spaceAllocForDenseUnsafe(brlen, bclen);
		MatrixBlock block = null;
		
		//bound check per block
		if( src.getNumRows() > rlen || src.getNumColumns() > clen )
		{
			throw new IOException("Matrix block [1:"+src.getNumRows()+",1:"+src.getNumColumns()+"] " +
					              "out of overall matrix range [1:"+rlen+",1:"+clen+"].");
		}
		
		try
		{
			for(int blockRow = 0; blockRow < (int)Math.ceil(src.getNumRows()/(double)brlen); blockRow++)
				for(int blockCol = 0; blockCol < (int)Math.ceil(src.getNumColumns()/(double)bclen); blockCol++)
				{
					int maxRow = (blockRow*brlen + brlen < src.getNumRows()) ? brlen : src.getNumRows() - blockRow*brlen;
					int maxCol = (blockCol*bclen + bclen < src.getNumColumns()) ? bclen : src.getNumColumns() - blockCol*bclen;
			
					int row_offset = blockRow*brlen;
					int col_offset = blockCol*bclen;
					
					//memory allocation
					if(maxRow < brlen || maxCol < bclen)
					{
						block = new MatrixBlock(maxRow, maxCol, false);
						block.spaceAllocForDenseUnsafe(maxRow, maxCol);
					}
					else 
					{
						block = fullBlock;
					}
					
					//TODO: monitor written entries and write only blocks with nnz>0
					//      (this requires changes of the runtime level before)
					
					//NOTE: set all values (incl nnz) due to block reuse
					if(sparse) //DENSE<-SPARSE
					{
						for(int i = 0; i < maxRow; i++)
							for(int j = 0; j < maxCol; j++)
							{
								double value = src.getValueSparseUnsafe( row_offset + i, col_offset + j);
								block.setValueDenseUnsafe(i, j, value); 
							}
					}
					else //DENSE<-DENSE
					{
						for(int i = 0; i < maxRow; i++) 
							for(int j = 0; j < maxCol; j++)
							{
								double value = src.getValueDenseUnsafe( row_offset + i, col_offset + j);
								block.setValueDenseUnsafe(i, j, value);
							}
					}	
					
					block.recomputeNonZeros();
					
					//handle empty result, append and reset
					if ( blockRow == 0 && blockCol == 0 & block.getNonZeros() == 0 )
						block.addDummyZeroValue();
					writer.append(new MatrixIndexes(blockRow+1, blockCol+1), block);
					block.reset();
				}
		}
		finally
		{
			if( writer != null )
				writer.close();
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
	private static void readTextCellMatrixFromHDFS( Path path, JobConf job, MatrixBlock dest, long rlen, long clen, int brlen, int bclen )
		throws IOException, IllegalAccessException, InstantiationException
	{
		boolean sparse = dest.isInSparseFormat();
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);
		InputSplit[] splits = informat.getSplits(job, 1);
		
		LongWritable key = new LongWritable();
		Text value = new Text();
		int row = -1;
		int col = -1;
		
		try
		{
			for(InputSplit split: splits)
			{
				RecordReader<LongWritable,Text> reader = informat.getRecordReader(split, job, Reporter.NULL);
			
				try
				{
					if( sparse ) //SPARSE<-value
					{
						while( reader.next(key, value) )
						{
							String cellStr = value.toString().trim();							
							StringTokenizer st = new StringTokenizer(cellStr, " ");
							row = Integer.parseInt( st.nextToken() )-1;
							col = Integer.parseInt( st.nextToken() )-1;
							double lvalue = Double.parseDouble( st.nextToken() );
							dest.setValue( row, col, lvalue );
						}
					} 
					else //DENSE<-value
					{
						while( reader.next(key, value) )
						{
							String cellStr = value.toString().trim();
							StringTokenizer st = new StringTokenizer(cellStr, " ");
							row = Integer.parseInt( st.nextToken() )-1;
							col = Integer.parseInt( st.nextToken() )-1;
							double lvalue = Double.parseDouble( st.nextToken() );
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
	private static void readRawTextCellMatrixFromHDFS( Path path, JobConf job, MatrixBlock dest, long rlen, long clen, int brlen, int bclen )
		throws IOException, IllegalAccessException, InstantiationException
	{
		boolean sparse = dest.isInSparseFormat();
		FileSystem fs = FileSystem.get(job);
		BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(path)));	
		
		String value = null;
		int row = -1;
		int col = -1;

		try
		{
			if( sparse ) //SPARSE<-value
			{
				while( (value=br.readLine())!=null )
				{
					String cellStr = value.toString().trim();							
					StringTokenizer st = new StringTokenizer(cellStr, " ");
					row = Integer.parseInt( st.nextToken() )-1;
					col = Integer.parseInt( st.nextToken() )-1;
					double lvalue = Double.parseDouble( st.nextToken() );
					dest.setValue( row, col, lvalue );
				}
			} 
			else //DENSE<-value
			{
				while( (value=br.readLine())!=null )
				{
					String cellStr = value.toString().trim();
					StringTokenizer st = new StringTokenizer(cellStr, " ");
					row = Integer.parseInt( st.nextToken() )-1;
					col = Integer.parseInt( st.nextToken() )-1;
					double lvalue = Double.parseDouble( st.nextToken() );
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
									  "out of overall matrix range [1:"+rlen+",1:"+clen+"].");
			}
			else
			{
				throw new IOException( "Unable to read matrix in raw text cell format.", ex );
			}
		}
		finally
		{
			if( br != null )
				br.close();
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
	private static void readBinaryCellMatrixFromHDFS( Path path, JobConf job, MatrixBlock dest, long rlen, long clen, int brlen, int bclen )
		throws IOException, IllegalAccessException, InstantiationException
	{
		boolean sparse = dest.isInSparseFormat();
		SequenceFileInputFormat<MatrixIndexes,MatrixCell> informat = new SequenceFileInputFormat<MatrixIndexes,MatrixCell>();
		InputSplit[] splits = informat.getSplits(job, 1);
		
		MatrixIndexes key = new MatrixIndexes();
		MatrixCell value = new MatrixCell();
		int row = -1;
		int col = -1;
		
		try
		{
			for(InputSplit split: splits)
			{
				RecordReader<MatrixIndexes,MatrixCell> reader = informat.getRecordReader(split, job, Reporter.NULL);
				try
				{
					if( sparse )
					{
						while(reader.next(key, value))
						{
							row = (int)key.getRowIndex()-1;
							col = (int)key.getColumnIndex()-1;
							double lvalue = value.getValue();
							dest.setValue( row, col, lvalue );
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
				throw new IOException( "Unable to read matrix in binary cell format.", ex );
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
	private static void readBinaryBlockMatrixFromHDFS( Path path, JobConf job, MatrixBlock dest, long rlen, long clen, int brlen, int bclen )
		throws IOException, IllegalAccessException, InstantiationException
	{
		boolean sparse = dest.isInSparseFormat();
		SequenceFileInputFormat<MatrixIndexes,MatrixBlock> informat = new SequenceFileInputFormat<MatrixIndexes,MatrixBlock>();
		InputSplit[] splits = informat.getSplits(job, 1);				
		MatrixIndexes key = new MatrixIndexes(); 
		MatrixBlock value = new MatrixBlock();
		
		for(InputSplit split: splits)
		{
			RecordReader<MatrixIndexes,MatrixBlock> reader = informat.getRecordReader(split, job, Reporter.NULL);
			
			try
			{
				while( reader.next(key, value) )
				{
					int row_offset = (int)(key.getRowIndex()-1)*brlen;
					int col_offset = (int)(key.getColumnIndex()-1)*bclen;
					
					int rows = value.getNumRows();
					int cols = value.getNumColumns();
					
					//bound check per block
					if( row_offset + rows < 0 || row_offset + rows > rlen || col_offset + cols<0 || col_offset + cols > clen )
					{
						throw new IOException("Matrix block ["+(row_offset+1)+":"+(row_offset+rows)+","+(col_offset+1)+":"+(col_offset+cols)+"] " +
								              "out of overall matrix range [1:"+rlen+",1:"+clen+"].");
					}
						
					//NOTE: at this point value is typically in dense format (see write)
					
					//copy block to result
					if( value.isInSparseFormat() ) //sparse input format
					{					
						if( sparse ) //SPARSE<-SPARSE
						{
							for( int i=0; i<rows; i++ )
								for( int j=0; j<cols; j++ )
								{
								    double lvalue = value.getValueSparseUnsafe(i, j);  //input value
									if( lvalue != 0  ) 					//for all nnz
										dest.setValue(row_offset+i, col_offset+j, lvalue );	
								}
						}
						else //DENSE<-SPARSE
						{
							for( int i=0; i<rows; i++ )
								for( int j=0; j<cols; j++ )
								{
								    double lvalue = value.getValueSparseUnsafe(i, j);  //input value
									if( lvalue != 0  ) 					//for all nnz
										dest.setValueDenseUnsafe(row_offset+i, col_offset+j, lvalue );	
								}
						}
						
					}
					else //dense input format
					{
						if( sparse ) //SPARSE<-DENSE
						{
							for( int i=0; i<rows; i++ )
								for( int j=0; j<cols; j++ )
								{
								    double lvalue = value.getValueDenseUnsafe(i, j);  //input value
									if( lvalue != 0  ) 					//for all nnz
										dest.setValue(row_offset+i, col_offset+j, lvalue );	
								}
						}
						else //DENSE<-DENSE
						{
							for( int i=0; i<rows; i++ )
								for( int j=0; j<cols; j++ )
								{
									double lvalue = value.getValueDenseUnsafe(i, j);  //input value
									if( lvalue != 0  ) 					//for all nnz
										dest.setValueDenseUnsafe(row_offset+i, col_offset+j, lvalue );	
								}
						}
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
	
	
	
	//////////////
	// Utils for CREATING and COPYING matrix blocks 
	///////
	
	/**
	 * Creates a two-dimensional double matrix of the input matrix block. 
	 * 
	 * @param mb
	 * @return
	 */
	public static double[][] convertToDoubleMatrix( MatrixBlock mb )
	{
		int rows = mb.getNumRows();
		int cols = mb.getNumColumns();
		double[][] ret = new double[rows][cols];
		
		if( mb.isInSparseFormat() )
		{
			for( int i=0; i<rows; i++ )
				for( int j=0; j<cols; j++ )
					ret[i][j] = mb.getValueSparseUnsafe(i, j);
		}
		else
		{
			for( int i=0; i<rows; i++ )
				for( int j=0; j<cols; j++ )
					ret[i][j] = mb.getValueDenseUnsafe(i, j);
		}
				
		return ret;
	}
	
	/**
	 * Creates a dense Matrix Block and copies the given double matrix into it.
	 * 
	 * @param data
	 * @return
	 */
	public static MatrixBlock convertToMatrixBlock( double[][] data )
	{
		int rows = data.length;
		int cols = (rows > 0)? data[0].length : 0;
		MatrixBlock mb = new MatrixBlock(rows, cols, false);
		try
		{ 
			//copy data to mb (can be used because we create a dense matrix)
			mb.init( data, rows, cols );
		} 
		catch (Exception e){} //can never happen
		
		return mb;
	}

	
	
	//////////////
	// OLD/UNUSED functionality
	///////
	
	@SuppressWarnings("unchecked")
	public static void writeMatrixToHDFSOld(MatrixBlock mat, 
										 String dir, 
										 OutputInfo outputinfo, 
										 long rlen, 
										 long clen, 
										 int brlen, 
										 int bclen)
		throws IOException
	{
		JobConf job = new JobConf();
		FileOutputFormat.setOutputPath(job, new Path(dir));
		
		try{
			long numEntriesWritten = 0;
			// If the file already exists on HDFS, remove it.
			MapReduceTool.deleteFileIfExistOnHDFS(dir);
			
			if ( outputinfo == OutputInfo.TextCellOutputInfo ) {
		        Path pt=new Path(dir);
		        FileSystem fs = FileSystem.get(new Configuration());
		        BufferedWriter br=new BufferedWriter(new OutputStreamWriter(fs.create(pt,true)));		
		        Converter outputConverter = new BinaryBlockToTextCellConverter();

				outputConverter.setBlockSize((int)rlen, (int)clen);
				
				outputConverter.convert(new MatrixIndexes(1, 1), mat);
				while(outputConverter.hasNext()){
					br.write(outputConverter.next().getValue().toString() + "\n");
					numEntriesWritten++;
				}
				
				if ( numEntriesWritten == 0 ) {
					br.write("1 1 0\n");
				}
				
				br.close();
			}
			else if ( outputinfo == OutputInfo.BinaryCellOutputInfo ) {
				FileSystem fs = FileSystem.get(job);
				SequenceFile.Writer writer = new SequenceFile.Writer(fs, job, new Path(dir), outputinfo.outputKeyClass, outputinfo.outputValueClass);
				Converter outputConverter = new BinaryBlockToBinaryCellConverter();
				
				outputConverter.setBlockSize((int)rlen, (int)clen);
				
				outputConverter.convert(new MatrixIndexes(1, 1), mat);
				Pair pair;
				Writable index, cell;
				while(outputConverter.hasNext()){
					pair = outputConverter.next();
					index = (Writable) pair.getKey();
					cell = (Writable) pair.getValue();
					
					writer.append(index, cell);
					numEntriesWritten++;
				}
				
				if ( numEntriesWritten == 0 ) {
					writer.append(new MatrixIndexes(1, 1), new MatrixCell(0));
				}
				writer.close();
			}
			else{
				FileSystem fs = FileSystem.get(job);
				SequenceFile.Writer writer = new SequenceFile.Writer(fs, job, new Path(dir), outputinfo.outputKeyClass, outputinfo.outputValueClass);
				//reblock
				MatrixBlock fullBlock = new MatrixBlock(brlen, bclen, false);
				
				MatrixBlock block;
				for(int blockRow = 0; blockRow < (int)Math.ceil(mat.getNumRows()/(double)brlen); blockRow++){
					for(int blockCol = 0; blockCol < (int)Math.ceil(mat.getNumColumns()/(double)bclen); blockCol++){
						int maxRow = (blockRow*brlen + brlen < mat.getNumRows()) ? brlen : mat.getNumRows() - blockRow*brlen;
						int maxCol = (blockCol*bclen + bclen < mat.getNumColumns()) ? bclen : mat.getNumColumns() - blockCol*bclen;
						
						if(maxRow < brlen || maxCol < bclen)
							block = new MatrixBlock(maxRow, maxCol, false);
						else block = fullBlock;
						
						for(int row = 0; row < maxRow; row++) {
							for(int col = 0; col < maxCol; col++){
								double value = mat.getValue(row + blockRow*brlen, col + blockCol*bclen);
								block.setValue(row, col, value);
							}
						}
						if ( blockRow == 0 && blockCol == 0 & block.getNonZeros() == 0 )
							block.addDummyZeroValue();
						writer.append(new MatrixIndexes(blockRow+1, blockCol+1), block);
						block.reset();
					}
				}
				
				writer.close();
			}
		}catch(Exception e){
			throw new IOException(e);
		}
	}
	
	
	@SuppressWarnings("unchecked")
	public static MatrixBlock readMatrixFromHDFSOld(String dir, InputInfo inputinfo, long rlen, long clen, 
			int brlen, int bclen) 
		throws IOException
	{	
		// force dense representation for 1D matrices (vectors)
		boolean sp = true;
		if ( rlen == 1 || clen == 1 )
			sp = false;
		MatrixBlock ret = new MatrixBlock((int)rlen, (int)clen, sp);
		
	//	String filename = getSubDirsIgnoreLogs(dir);
		JobConf job = new JobConf();
		
		if(!FileSystem.get(job).exists(new Path(dir)))	
			return null;
		
		FileInputFormat.addInputPath(job, new Path(dir));
		
		try {

			InputFormat informat=inputinfo.inputFormatClass.newInstance();
			if(informat instanceof TextInputFormat)
				((TextInputFormat)informat).configure(job);
			InputSplit[] splits= informat.getSplits(job, 1);
			
			Converter inputConverter=MRJobConfiguration.getConverterClass(inputinfo, false, brlen, bclen).newInstance();
			inputConverter.setBlockSize(brlen, bclen);
    		
			Writable key=inputinfo.inputKeyClass.newInstance();
			Writable value=inputinfo.inputValueClass.newInstance();
			
			for(InputSplit split: splits)
			{
				RecordReader reader=informat.getRecordReader(split, job, Reporter.NULL);
				while(reader.next(key, value))
				{
					inputConverter.convert(key, value);
					while(inputConverter.hasNext())
					{
						Pair pair=inputConverter.next();
						MatrixIndexes index=(MatrixIndexes) pair.getKey();
						MatrixCell cell=(MatrixCell) pair.getValue();
						ret.setValue((int)index.getRowIndex()-1, (int)index.getColumnIndex()-1, cell.getValue());
					}
				}
				reader.close();
			}
			
			ret.examSparsity();
		} catch (Exception e) {
			throw new IOException(e);
		}
		
		return ret;
	}
	
	
	
	
	
	/**
	 * Reads a matrix from HDFS (in block format) and returns its values in a 1D
	 * array containing double values.
	 * 
	 * @param dir
	 * @param numRows
	 *            Number of rows which the matrix should have
	 * @param numCols
	 *            Number of columns which the matrix should have
	 * @param blockSizeRows
	 *            Number of rows in normal blocks
	 * @param blockSizeCols
	 *            Number of cols in normal blocks
	 * @return
	 * @throws IOException
	 * @throws DMLRuntimeException
	 */
	/*private static void readDouble1DArrayMatrixFromHDFSBlock(String dir, 
															MatrixBlock1D mat,
															int blockSizeRows, 
															int blockSizeCols) 
			throws DMLRuntimeException {		
		try {
			Path[] subpaths = getSubDirs(dir);
			FileSystem fs = FileSystem.get(conf);
			MatrixIndexes indexes = new MatrixIndexes();
			MatrixBlock value = new MatrixBlock();

			for (Path path : subpaths) {
				SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
				BinaryBlockToBinaryCellConverter conv = new BinaryBlockToBinaryCellConverter();
				conv.setBlockSize(blockSizeRows, blockSizeCols);

				while (reader.next(indexes, value)) {
					conv.convert(indexes, value);
					while (conv.hasNext()) {
						Pair<MatrixIndexes, MatrixCell> pair = conv.next();
						int row = (int)(pair.getKey().getRowIndex()) - 1;
						int col = (int)(pair.getKey().getColumnIndex()) - 1;
						
						if(row >= mat.getNumRows() || col >= mat.getNumColumns())
							throw new DMLRuntimeException("matrix on disk "
														  + dir
														  + " does not match size of matrix object");
						
						mat.setValue(row,
									 col,
									 pair.getValue().getValue());
					}
				}
			}
		} catch (IOException e) {
			throw new DMLRuntimeException(e);
		}
	}*/

	/**
	 * Reads a matrix from HDFS (in block format) and returns its values in a 1D
	 * array containing double values.
	 * 
	 * @param dir
	 * @param numRows
	 *            Number of rows which the matrix should have
	 * @param numCols
	 *            Number of columns which the matrix should have
	 * @return
	 * @throws IOException
	 * @throws DMLRuntimeException
	 */
	/*private static void readDouble1DArrayMatrixFromHDFSText(String dir, MatrixBlock1D mat)
			throws DMLRuntimeException {
		try {
			Path[] subpaths = getSubDirs(dir);
			FileSystem fs = FileSystem.get(conf);
			if (!fs.isDirectory(new Path(dir))) {
				subpaths = new Path[] { new Path(dir) };
			}

			LongWritable indexes = new LongWritable();
			Text value = new Text();

			for (Path path : subpaths) {
				// SequenceFile.Reader reader = new SequenceFile.Reader(fs,
				// path, conf);
				TextToBinaryCellConverter conv = new TextToBinaryCellConverter();
				FSDataInputStream fi = fs.open(path);
				BufferedReader br = new BufferedReader(new InputStreamReader(fi));
				String line = null;
				while ((line = br.readLine()) != null) {
					value = new Text(line);

					conv.convert(indexes, value);
					while (conv.hasNext()) {
						Pair<MatrixIndexes, MatrixCell> pair = conv.next();
						
						int row = (int)(pair.getKey().getRowIndex() - 1);
						int col = (int)(pair.getKey().getColumnIndex() - 1);
						
						if(row >= mat.getNumRows() || col >= mat.getNumColumns())
							throw new DMLRuntimeException("matrix on disk "
														  + dir
														  + " does not match size of matrix object");
						
						mat.setValue(row, 
									  col, 
									  pair.getValue().getValue());
					}
				}
			}
		} catch (IOException e) {
			throw new DMLRuntimeException(e);
		}
	}*/

	/**
	 * Reads a matrix from HDFS (in cell format) and returns its values in a 1D
	 * array containing double values.
	 * 
	 * @param dir
	 * @param numRows
	 *            Number of rows which the matrix is expected to have
	 * @param numCols
	 *            Number of columns which the matrix is expected to have
	 * @return
	 * @throws IOException
	 * @throws DMLRuntimeException
	 */
	/*private static void readDouble1DArrayMatrixFromHDFSCell(String dir, MatrixBlock1D mat)
			throws DMLRuntimeException {
		try {
			Path[] subpaths = getSubDirs(dir);
			FileSystem fs = FileSystem.get(conf);
			MatrixIndexes indexes = new MatrixIndexes();
			MatrixCell value = new MatrixCell();

			for (Path path : subpaths) {
				SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);

				while (reader.next(indexes, value)) {
					long i = indexes.getRowIndex() - 1;
					long j = indexes.getColumnIndex() - 1;
					long p = i * mat.getNumColumns() + j;
					if (p > (int) p)
						throw new DMLRuntimeException("Matrix is too large");

					int row = (int)i;
					int col = (int)j;
					
					if(row >= mat.getNumRows() || col >= mat.getNumColumns())
						throw new DMLRuntimeException("matrix on disk "
													  + dir
													  + " does not match size of matrix object");
					
					mat.setValue(row, col, value.getValue());
				}
			}
		} catch (IOException e) {
			throw new DMLRuntimeException(e);
		}
	}*/

	/*private static void writeDoubleMatrixToHDFSBlock(String dir, 
													int blockSizeRows, 
													int blockSizeCols,
													MatrixBlock1D mat) 
			throws DMLRuntimeException {
		int numRows = mat.getNumRows();
		int numCols = mat.getNumColumns();
		try {
			SequenceFile.Writer writer = new SequenceFile.Writer(FileSystem.get(conf), conf, new Path(dir),
					MatrixIndexes.class, MatrixBlock.class);

			MatrixIndexes index = new MatrixIndexes();
			MatrixBlock value = new MatrixBlock();
			for (int i = 0; i < mat.getNumRows(); i += numRows) {
				int rows = Math.min(numRows, (mat.getNumRows() - i));
				for (int j = 0; j < mat.getNumColumns(); j += numCols) {
					int cols = Math.min(numCols, (mat.getNumColumns() - j));
					index.setIndexes(((i / numRows) + 1), ((j / numCols) + 1));
					value = new MatrixBlock(rows, cols, true);
					for (int k = 0; k < rows; k++) {
						for (int l = 0; l < cols; l++) 
							value.setValue(k, l, mat.getValue(i+k, j+l));
					}
					writer.append(index, value);
				}
			}

			writer.close();
		} catch (IOException e) {
			throw new DMLRuntimeException(e);
		}
	}*/

	//writes to one file only
	/*private static void writeDoubleMatrixToHDFSText(String dir, MatrixBlock1D mat)
		throws DMLRuntimeException{
		try{
			Path path = new Path(dir);
			FileSystem fs = FileSystem.get(conf);
			PrintWriter writer;
			if(fs.exists(path))
				System.err.println(path.toString()+" already exists");
			writer = new PrintWriter(fs.create(path, true));
			
			if(mat.isInSparseFormat()){
				HashMap<CellIndex, Double> map = mat.getSparseMap();
				Iterator<Map.Entry<CellIndex, Double>> it = map.entrySet().iterator();
				while(it.hasNext()){
					Map.Entry<CellIndex, Double> elt = it.next();
					int row = elt.getKey().row;
					int col = elt.getKey().column;
					double v = elt.getValue().doubleValue();
					writer.println((row+1)+" "+(col+1)+" "+v);
				}
			}else
				for(int i=0; i<mat.getNumRows(); i++)
					for(int j=0; j<mat.getNumColumns(); j++)
						writer.println((i+1)+" "+(j+1)+" "+mat.getValue(i, j));
			
			writer.close();
		}catch(IOException e){
			throw new DMLRuntimeException(e);
		}
	}*/
	
	/*private static Path[] getSubDirs(String dir) throws IOException {
		FileSystem fs = FileSystem.get(new Configuration());
		ArrayList<Path> paths = new ArrayList<Path>();
		for (FileStatus cur : fs.listStatus(new Path(dir))) {
			paths.add(cur.getPath());
		}
		return paths.toArray(new Path[paths.size()]);
	}*/

}
