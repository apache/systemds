package com.ibm.bi.dml.runtime.util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.LinkedList;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
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
import org.apache.hadoop.mapred.TextInputFormat;

import com.ibm.bi.dml.parser.DMLTranslator;
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
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlockDSM.IJV;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlockDSM.SparseCellIterator;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


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
		//FileOutputFormat.setOutputPath(job, path); FIXME

		//System.out.println("write matrix (sparse="+mat.isInSparseFormat()+") to HDFS: "+dir);
		
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
	public static MatrixBlock readMatrixFromHDFS(String dir, InputInfo inputinfo, long rlen, long clen, int brlen, int bclen, boolean localFS) 
		throws IOException
	{	
		//expected matrix is sparse (default SystemML usecase)
		return readMatrixFromHDFS(dir, inputinfo, rlen, clen, brlen, bclen, 0.1d, localFS);
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
	public static MatrixBlock readMatrixFromHDFS(String dir, InputInfo inputinfo, long rlen, long clen, int brlen, int bclen) 
		throws IOException
	{	
		//expected matrix is sparse (default SystemML usecase)
		return readMatrixFromHDFS(dir, inputinfo, rlen, clen, brlen, bclen, 0.1d, false);
	}

	/**
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
	public static MatrixBlock readMatrixFromHDFS(String dir, InputInfo inputinfo, long rlen, long clen, int brlen, int bclen, double expectedSparsity) 
		throws IOException
	{	
		return readMatrixFromHDFS(dir, inputinfo, rlen, clen, brlen, bclen, expectedSparsity, false);
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
			int brlen, int bclen, double expectedSparsity, boolean localFS) 
	throws IOException
	{	
		boolean sparse = expectedSparsity < MatrixBlockDSM.SPARCITY_TURN_POINT;

		//System.out.println("read matrix (sparse="+sparse+") from HDFS: "+dir);
		
		//handle vectors specially
		//if result is a column vector, use dense format, otherwise use the normal process to decide
		if ( clen <=MatrixBlock.SKINNY_MATRIX_TURN_POINT )
			sparse = false;
		
		//prepare result matrix block
		MatrixBlock ret = new MatrixBlock((int)rlen, (int)clen, sparse, (int)(expectedSparsity*rlen*clen));
		if( !sparse && inputinfo!=InputInfo.BinaryBlockInputInfo )
			ret.spaceAllocForDenseUnsafe((int)rlen, (int)clen);
		
		//prepare file access
		JobConf job = new JobConf();	
		Path path = null; 
		FileSystem fs = null;
		if ( localFS ) {
			path = new Path("file:///" + dir);
			//System.out.println("local file system path ..." + path.toUri() + ", " + path.toString());
			fs = FileSystem.getLocal(job);
		}
		else {
			path = new Path(dir);
			fs = FileSystem.get(job);
		}
		if( !fs.exists(path) )	
			throw new IOException("File "+dir+" does not exist on HDFS/LFS.");
		//System.out.println("dataconverter: reading file " + path + " [" + rlen + "," + clen + "] from localFS=" + localFS);
		//FileInputFormat.addInputPath(job, path); //FIXME 
		
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
				readBinaryCellMatrixFromHDFS( path, job, fs, ret, rlen, clen, brlen, bclen );
			}
			else if( inputinfo == InputInfo.BinaryBlockInputInfo )
			{
				readBinaryBlockMatrixFromHDFS( path, job, fs, ret, rlen, clen, brlen, bclen );
			}
			
			//finally check if change of sparse/dense block representation required
			if( !sparse || inputinfo==InputInfo.BinaryBlockInputInfo )
				ret.recomputeNonZeros();
			ret.examSparsity();	
		} 
		catch (Exception e) 
		{
			throw new IOException(e);
		}

		//System.out.println("read matrix (after exec sparse="+ret.isInSparseFormat()+") from HDFS: "+dir);
		
		return ret;
	}

	/**
	 * Reads a partition that contains the given block index (rowBlockIndex, colBlockIndex)
	 * from a file on HDFS/LocalFS as specified by <code>dir</code>, which is partitioned using 
	 * <code>pformat</code> and <code>psize</code>.
	 * 
	 * @param dir
	 * @param localFS
	 * @param rows
	 * @param cols
	 * @param pformat
	 * @param psize
	 * @param rowBlockIndex
	 * @param colBlockIndex
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static MatrixBlock readPartitionFromDistCache(String dir, boolean localFS, 
			long rows, long cols,
			int partID, int partSize) throws DMLRuntimeException {
		
		dir = dir + "/" + partID;
		
		try {
			MatrixBlock partition = readMatrixFromHDFS(dir, InputInfo.BinaryBlockInputInfo, partSize, 1, DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize, localFS);
			System.out.println("Reading complete...");
			return partition;
		} catch (IOException e) {
			throw new DMLRuntimeException(e);
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
			//for obj reuse and preventing repeated buffer re-allocations
			StringBuilder sb = new StringBuilder();
			
			if( sparse ) //SPARSE
			{			   
				SparseCellIterator iter = src.getSparseCellIterator();
				while( iter.hasNext() )
				{
					IJV cell = iter.next();

					sb.append(cell.i+1);
					sb.append(' ');
					sb.append(cell.j+1);
					sb.append(' ');
					sb.append(cell.v);
					sb.append('\n');
					br.write( sb.toString() ); //same as append
					sb.setLength(0); 
					entriesWritten = true;					
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
							sb.append(i+1);
							sb.append(' ');
							sb.append(j+1);
							sb.append(' ');
							sb.append(lvalue);
							sb.append('\n');
							br.write( sb.toString() ); //same as append
							sb.setLength(0); 
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
				SparseCellIterator iter = src.getSparseCellIterator();
				while( iter.hasNext() )
				{
					IJV lcell = iter.next();
					indexes.setIndexes(lcell.i+1, lcell.j+1);
					cell.setValue(lcell.v);
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
	 * @throws DMLUnsupportedOperationException 
	 * @throws DMLRuntimeException 
	 */
	private static void writeBinaryBlockMatrixToHDFS( Path path, JobConf job, MatrixBlock src, long rlen, long clen, int brlen, int bclen )
		throws IOException, DMLRuntimeException, DMLUnsupportedOperationException
	{
		boolean sparse = src.isInSparseFormat();
		FileSystem fs = FileSystem.get(job);
		SequenceFile.Writer writer = new SequenceFile.Writer(fs, job, path, MatrixIndexes.class, MatrixBlock.class);
		
		//bound check for src block
		if( src.getNumRows() > rlen || src.getNumColumns() > clen )
		{
			throw new IOException("Matrix block [1:"+src.getNumRows()+",1:"+src.getNumColumns()+"] " +
					              "out of overall matrix range [1:"+rlen+",1:"+clen+"].");
		}
		//reblock and write
		try
		{
			MatrixIndexes indexes = new MatrixIndexes();

			if( rlen <= brlen && clen <= bclen ) //opt for single block
			{
				//directly write single block
				indexes.setIndexes(1, 1);
				writer.append(indexes, src);
			}
			else //general case
			{
				//initialize blocks for reuse (at most 4 different blocks required)
				MatrixBlock[] blocks = createMatrixBlocksForReuse(rlen, clen, brlen, bclen, sparse, src.getNonZeros());  
				
				//create and write subblocks of matrix
				for(int blockRow = 0; blockRow < (int)Math.ceil(src.getNumRows()/(double)brlen); blockRow++)
					for(int blockCol = 0; blockCol < (int)Math.ceil(src.getNumColumns()/(double)bclen); blockCol++)
					{
						int maxRow = (blockRow*brlen + brlen < src.getNumRows()) ? brlen : src.getNumRows() - blockRow*brlen;
						int maxCol = (blockCol*bclen + bclen < src.getNumColumns()) ? bclen : src.getNumColumns() - blockCol*bclen;
				
						int row_offset = blockRow*brlen;
						int col_offset = blockCol*bclen;
						
						//get reuse matrix block
						MatrixBlock block = getMatrixBlockForReuse(blocks, maxRow, maxCol, brlen, bclen);
	
						//copy submatrix to block TODO rename to slice!
						src.slideOperations( row_offset+1, row_offset+maxRow, 
								             col_offset+1, col_offset+maxCol, 
								             block );
						
						//append block to sequence file
						indexes.setIndexes(blockRow+1, blockCol+1);
						writer.append(indexes, block);
						
						//reset block for later reuse
						block.reset();
					}
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
							dest.quickSetValue( row, col, lvalue );
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
					dest.quickSetValue( row, col, lvalue );
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
	 * Note: see readBinaryBlockMatrixFromHDFS for why we directly use SequenceFile.Reader.
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
	 * @throws IllegalAccessException
	 * @throws InstantiationException
	 */
	private static void readBinaryCellMatrixFromHDFS( Path path, JobConf job, FileSystem fs, MatrixBlock dest, long rlen, long clen, int brlen, int bclen )
		throws IOException, IllegalAccessException, InstantiationException
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
							dest.quickSetValue( row, col, lvalue );
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
	 * Note: For efficiency, we directly use SequenceFile.Reader instead of SequenceFileInputFormat-
	 * InputSplits-RecordReader (SequenceFileRecordReader). First, this has no drawbacks since the
	 * SequenceFileRecordReader internally uses SequenceFile.Reader as well. Second, it is 
	 * advantageous if the actual sequence files are larger than the file splits created by   
	 * informat.getSplits (which is usually aligned to the HDFS block size) because then there is 
	 * overhead for finding the actual split between our 1k-1k blocks. This case happens
	 * if the read matrix was create by CP or when jobs directly write to large output files 
	 * (e.g., parfor matrix partitioning).
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
	 * @throws IllegalAccessException
	 * @throws InstantiationException
	 */
	private static void readBinaryBlockMatrixFromHDFS( Path path, JobConf job, FileSystem fs, MatrixBlock dest, long rlen, long clen, int brlen, int bclen )
		throws IOException, IllegalAccessException, InstantiationException
	{
		MatrixIndexes key = new MatrixIndexes(); 
		MatrixBlock value = new MatrixBlock();
		
		for( Path lpath : getSequenceFilePaths(fs, path) ) //1..N files 
		{
			//directly read from sequence files (individual partfiles)
			SequenceFile.Reader reader = new SequenceFile.Reader(fs,lpath,job);
			
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
					
					//copy block to result
					dest.copy( row_offset, row_offset+rows-1, 
							   col_offset, col_offset+cols-1,
							   value, false );
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
			SparseCellIterator iter = mb.getSparseCellIterator();
			while( iter.hasNext() )
			{
				IJV cell = iter.next();
				ret[cell.i][cell.j] = cell.v;
			}
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
	 * @throws DMLRuntimeException 
	 */
	public static MatrixBlock convertToMatrixBlock( double[][] data ) 
		throws DMLRuntimeException
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
		
		//check and convert internal representation
		mb.examSparsity();
		
		return mb;
	}

	
	
	/////////////////////////////////////////////
	// Helper methods for the specific formats //
	/////////////////////////////////////////////

	/**
	 * 
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @param sparse
	 * @return
	 */
	public static MatrixBlock[] createMatrixBlocksForReuse( long rlen, long clen, int brlen, int bclen, boolean sparse, long nonZeros )
	{
		MatrixBlock[] blocks = new MatrixBlock[4];
		double sparsity = ((double)nonZeros)/(rlen*clen);
		long estNNZ = -1;
		
		//full block 
		if( rlen >= brlen && clen >= bclen )
		{
			estNNZ = (long) (brlen*bclen*sparsity);
			blocks[0] = new MatrixBlock( brlen, bclen, sparse, (int)estNNZ );
		}
		//partial col block
		if( rlen >= brlen && clen%bclen!=0 )
		{
			estNNZ = (long) (brlen*(clen%bclen)*sparsity);
			blocks[1] = new MatrixBlock( brlen, (int)(clen%bclen), sparse, (int)estNNZ );
		}
		//partial row block
		if( rlen%brlen!=0 && clen>=bclen )
		{
			estNNZ = (long) ((rlen%brlen)*bclen*sparsity);
			blocks[2] = new MatrixBlock( (int)(rlen%brlen), bclen, sparse, (int)estNNZ );
		}
		//partial row/col block
		if( rlen%brlen!=0 && clen%bclen!=0 )
		{
			estNNZ = (long) ((rlen%brlen)*(clen%bclen)*sparsity);
			blocks[3] = new MatrixBlock( (int)(rlen%brlen), (int)(clen%bclen), sparse, (int)estNNZ );
		}
		
		//space allocation
		for( MatrixBlock b : blocks )
			if( b != null )
				if( !sparse )
					b.spaceAllocForDenseUnsafe(b.getNumRows(), b.getNumColumns());		
		//NOTE: no preallocation for sparse (preallocate sparserows with estnnz) in order to reduce memory footprint
		
		return blocks;
	}
	
	/**
	 * 
	 * @param blocks
	 * @param rows
	 * @param cols
	 * @param brlen
	 * @param bclen
	 * @return
	 */
	public static MatrixBlock getMatrixBlockForReuse( MatrixBlock[] blocks, int rows, int cols, int brlen, int bclen )
	{
		int index = -1;
		
		if( rows==brlen && cols==bclen )
			index = 0;
		else if( rows==brlen && cols<bclen )
			index = 1;
		else if( rows<brlen && cols==bclen )
			index = 2;
		else //if( rows<brlen && cols<bclen )
			index = 3;

		return blocks[ index ];
	}

	/**
	 * 
	 * @param file
	 * @return
	 * @throws IOException
	 */
	public static Path[] getSequenceFilePaths( FileSystem fs, Path file ) 
		throws IOException
	{
		FileStatus fStatus = fs.getFileStatus(file);
		Path[] ret = null;
		
		if( fStatus.isDir() )
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
	
}
