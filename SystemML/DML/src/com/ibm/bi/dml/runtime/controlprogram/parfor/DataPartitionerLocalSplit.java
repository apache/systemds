package com.ibm.bi.dml.runtime.controlprogram.parfor;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.StringTokenizer;
import java.util.Map.Entry;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.TextInputFormat;

import com.ibm.bi.dml.hops.Hops;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDSequence;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.MatrixObjectNew;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.utils.DMLRuntimeException;

/**
 * Partitions a given matrix into row or column partitions with a two pass-approach.
 * In the first phase the input matrix is read from HDFS and sorted into block partitions
 * in a staging area in the local file system according to the partition format. 
 * In order to allow for scalable partitioning, we process one block at a time. 
 * Furthermore, in the second phase, all blocks of a partition are append to a sequence file
 * on HDFS. Block-wise partitioning and write-once semantics of sequence files require the
 * indirection over the local staging area. For scalable computation, we process one 
 * sequence file at a time.
 *
 * NOTE: For the resulting partitioned matrix, we store block and cell indexes wrt partition boundaries.
 *       This means that the partitioned matrix CANNOT be read as a traditional matrix because there are
 *       for example multiple blocks with same index (while the actual index is encoded in the path).
 *       In order to enable full read of partition matrices, data converter would need to handle negative
 *       row/col offsets for partitioned read. Currently not done in order to avoid overhead from normal read
 *       and since partitioning only applied if exclusively indexed access.
 *
 *
 * TODO metadata file for partitioned matrices ?
 * TODO parallel writing of sequence files (most expensive part and nicely parallelizable, but larger memory requirements)
 *
 */
public class DataPartitionerLocalSplit extends DataPartitioner
{
	private IDSequence _seq = null;
	
	public DataPartitionerLocalSplit(PDataPartitionFormat dpf) 
	{
		super(dpf);
		
		_seq = new IDSequence();
	}

	@Override
	public MatrixObjectNew createPartitionedMatrix(MatrixObjectNew in, boolean force)
			throws DMLRuntimeException 
	{
		//check for naive partitioning
		if( _format == PDataPartitionFormat.NONE )
			return in;
		
		//analyze input matrix object
		ValueType vt = in.getValueType();
		String varname = in.getVarName();
		MatrixFormatMetaData meta = (MatrixFormatMetaData)in.getMetaData();
		MatrixCharacteristics mc = meta.getMatrixCharacteristics();
		String fname = in.getFileName();
		InputInfo ii = meta.getInputInfo();
		OutputInfo oi = meta.getOutputInfo();
		int rows = (int)mc.get_rows(); 
		int cols = (int)mc.get_cols();
		int brlen = mc.get_rows_per_block();
		int bclen = mc.get_cols_per_block();
		
		if( !force ) //try to optimize, if format not forced
		{
			//check lower bound of useful data partitioning
			if( ( rows == 1 || cols == 1 ) ||                            //is vector
				( rows < Hops.CPThreshold && cols < Hops.CPThreshold) )  //or matrix already fits in mem
			{
				return in;
			}
			
			//check for changing to blockwise representations
			if( _format == PDataPartitionFormat.ROW_WISE && cols < Hops.CPThreshold )
			{
				System.out.println("INFO: DataPartitioner: Changing format from "+PDataPartitionFormat.ROW_WISE+" to "+PDataPartitionFormat.ROW_BLOCK_WISE+".");
				_format = PDataPartitionFormat.ROW_BLOCK_WISE;
			}
			if( _format == PDataPartitionFormat.COLUMN_WISE && rows < Hops.CPThreshold )
			{
				System.out.println("INFO: DataPartitioner: Changing format from "+PDataPartitionFormat.COLUMN_WISE+" to "+PDataPartitionFormat.ROW_BLOCK_WISE+".");
				_format = PDataPartitionFormat.COLUMN_BLOCK_WISE;
			}
		}
		
		//force writing to disk (typically not required since partitioning only applied if dataset exceeds CP size)
		in.exportData(); //written to disk iff dirty
		
		//prepare filenames and cleanup if required
		String fnameNew = fname + NAME_SUFFIX;
		String fnameStaging = STAGING_DIR+"/"+fname;
		try{
			MapReduceTool.deleteFileIfExistOnHDFS(fnameNew);
			cleanupStagingDir(fnameStaging);
		}
		catch(Exception ex){
			throw new DMLRuntimeException( ex );
		}
		
		//reblock input matrix
		if( ii == InputInfo.TextCellInputInfo )
			partitionTextcellToTextcell( fname, fnameStaging, fnameNew, rows, cols, brlen, bclen );
		else if( ii == InputInfo.BinaryCellInputInfo )
			partitionBinarycellToBinarycell( fname, fnameStaging, fnameNew, rows, cols, brlen, bclen );
		else if( ii == InputInfo.BinaryBlockInputInfo )
			partitionBinaryblockToBinaryblock( fname, fnameStaging, fnameNew, rows, cols, brlen, bclen );
		else	
		{
			System.out.println("Warning: Cannot create data partitions of format: "+ii.toString());
			return in; //return unmodified input matrix, similar to no partitioning
		}
		
		//create output matrix object
		MatrixObjectNew mobj = new MatrixObjectNew(vt, fnameNew );
		mobj.setDataType(DataType.MATRIX);
		mobj.setVarName( varname+NAME_SUFFIX );
		mobj.setPartitioned( _format ); 
		MatrixCharacteristics mcNew = new MatrixCharacteristics( rows, cols,
				                           (_format==PDataPartitionFormat.ROW_WISE)? 1 : (int)brlen, //for blockwise brlen anyway
				                           (_format==PDataPartitionFormat.COLUMN_WISE)? 1 : (int)bclen ); //for blockwise bclen anyway
		MatrixFormatMetaData metaNew = new MatrixFormatMetaData(mcNew,oi,ii);
		mobj.setMetaData(metaNew);	 
		
		return mobj;
	}
	
	/**
	 * 
	 * @param fname
	 * @param fnameStaging
	 * @param fnameNew
	 * @param brlen
	 * @param bclen
	 * @throws DMLRuntimeException
	 */
	private void partitionTextcellToTextcell( String fname, String fnameStaging, String fnameNew, int rlen, int clen, int brlen, int bclen ) 
		throws DMLRuntimeException
	{
		int row = -1;
		int col = -1;
		
		try 
		{
			//STEP 1: read matrix from HDFS and write blocks to local staging area			
			//check and add input path
			JobConf job = new JobConf();
			Path path = new Path(fname);
			FileInputFormat.addInputPath(job, path);
			TextInputFormat informat = new TextInputFormat();
			informat.configure(job);
			InputSplit[] splits = informat.getSplits(job, 1);
			
			LinkedList<PartitionCell> buffer = new LinkedList<PartitionCell>();
			LongWritable key = new LongWritable();
			Text value = new Text();

			for(InputSplit split: splits)
			{
				RecordReader<LongWritable,Text> reader = informat.getRecordReader(split, job, Reporter.NULL);
				try
				{
					while(reader.next(key, value))
					{
						String cellStr = value.toString().trim();							
						StringTokenizer st = new StringTokenizer(cellStr, " ");
						row = Integer.parseInt( st.nextToken() );
						col = Integer.parseInt( st.nextToken() );
						double lvalue = Double.parseDouble( st.nextToken() );
						PartitionCell tmp = new PartitionCell( row, col, lvalue ); 
		
						buffer.addLast( tmp );
						if( buffer.size() > CELL_BUFFER_SIZE ) //periodic flush
						{
							appendCellBufferToStagingArea(fnameStaging, buffer, brlen, bclen);
							buffer.clear();
						}
					}
					
					//final flush
					appendCellBufferToStagingArea(fnameStaging, buffer, brlen, bclen);
					buffer.clear();
				}
				finally
				{
					if( reader != null )
						reader.close();
				}
			}

			//STEP 2: read matrix blocks from staging area and write matrix to HDFS
			String[] fnamesPartitions = new File(fnameStaging).list();			
			for( String pdir : fnamesPartitions  )
				writeTextCellSequenceFileToHDFS( job, fnameNew, fnameStaging+"/"+pdir );			
		} 
		catch (Exception e) 
		{
			//post-mortem error handling and bounds checking
			if( row < 1 || row > rlen || col < 1 || col > clen )
			{
				throw new DMLRuntimeException("Matrix cell ["+(row)+","+(col)+"] " +
									          "out of overall matrix range [1:"+rlen+",1:"+clen+"].");
			}
			else
				throw new DMLRuntimeException("Unable to partition text cell matrix.", e);
		}
	}	

	/**
	 * 
	 * @param fname
	 * @param fnameStaging
	 * @param fnameNew
	 * @param brlen
	 * @param bclen
	 * @throws DMLRuntimeException
	 */
	private void partitionBinarycellToBinarycell( String fname, String fnameStaging, String fnameNew, int rlen, int clen, int brlen, int bclen ) 
		throws DMLRuntimeException
	{
		int row = -1;
		int col = -1;
		
		try 
		{
			//STEP 1: read matrix from HDFS and write blocks to local staging area
			//check and add input path
			JobConf job = new JobConf();
			Path path = new Path(fname);
			FileInputFormat.addInputPath(job, path);
			
			//prepare sequence file reader, and write to local staging area
			SequenceFileInputFormat<MatrixIndexes,MatrixCell> informat = new SequenceFileInputFormat<MatrixIndexes,MatrixCell>();
			InputSplit[] splits = informat.getSplits(job, 1);
	
			LinkedList<PartitionCell> buffer = new LinkedList<PartitionCell>();
			MatrixIndexes key = new MatrixIndexes();
			MatrixCell value = new MatrixCell();
	
			for(InputSplit split: splits)
			{
				RecordReader<MatrixIndexes,MatrixCell> reader = informat.getRecordReader(split, job, Reporter.NULL);
				try
				{
					while(reader.next(key, value))
					{
						row = (int)key.getRowIndex();
						col = (int)key.getColumnIndex();
						PartitionCell tmp = new PartitionCell( row, col, value.getValue() ); 
		
						buffer.addLast( tmp );
						if( buffer.size() > CELL_BUFFER_SIZE ) //periodic flush
						{
							appendCellBufferToStagingArea(fnameStaging, buffer, brlen, bclen);
							buffer.clear();
						}
					}
					
					//final flush
					appendCellBufferToStagingArea(fnameStaging, buffer, brlen, bclen);
					buffer.clear();
				}
				finally
				{
					if( reader != null )
						reader.close();
				}
			}
			
			//STEP 2: read matrix blocks from staging area and write matrix to HDFS
			String[] fnamesPartitions = new File(fnameStaging).list();			
			for( String pdir : fnamesPartitions  )
				writeBinaryCellSequenceFileToHDFS( job, fnameNew, fnameStaging+"/"+pdir );			
		} 
		catch (Exception e) 
		{
			//post-mortem error handling and bounds checking
			if( row < 1 || row > rlen || col < 1 || col > clen )
			{
				throw new DMLRuntimeException("Matrix cell ["+(row)+","+(col)+"] " +
									  		  "out of overall matrix range [1:"+rlen+",1:"+clen+"].");
			}
			else
				throw new DMLRuntimeException("Unable to partition binary cell matrix.", e);
		}
	}	
	
	/**
	 * 
	 * @param fname
	 * @param fnameStaging
	 * @param fnameNew
	 * @param brlen
	 * @param bclen
	 * @throws DMLRuntimeException
	 */
	private void partitionBinaryblockToBinaryblock( String fname, String fnameStaging, String fnameNew, int rlen, int clen, int brlen, int bclen ) 
		throws DMLRuntimeException
	{
		try 
		{		
			//STEP 1: read matrix from HDFS and write blocks to local staging area	
			//check and add input path
			JobConf job = new JobConf();
			Path path = new Path(fname);
			FileInputFormat.addInputPath(job, path);
			
			//prepare sequence file reader, and write to local staging area
			SequenceFileInputFormat<MatrixIndexes,MatrixBlock> informat = new SequenceFileInputFormat<MatrixIndexes,MatrixBlock>();
			InputSplit[] splits = informat.getSplits(job, 1);				
			MatrixIndexes key = new MatrixIndexes(); 
			MatrixBlock value = new MatrixBlock();
			
			for(InputSplit split : splits)
			{
				RecordReader<MatrixIndexes,MatrixBlock> reader=informat.getRecordReader(split, job, Reporter.NULL);
				try
				{
					while(reader.next(key, value)) //for each block
					{
						long row_offset = (key.getRowIndex()-1)*brlen;
						long col_offset = (key.getColumnIndex()-1)*bclen;
						int rows = value.getNumRows();
						int cols = value.getNumColumns();
						
						//bound check per block
						if( row_offset + rows < 1 || row_offset + rows > rlen || col_offset + cols<1 || col_offset + cols > clen )
						{
							throw new IOException("Matrix block ["+(row_offset+1)+":"+(row_offset+rows)+","+(col_offset+1)+":"+(col_offset+cols)+"] " +
									              "out of overall matrix range [1:"+rlen+",1:"+clen+"].");
						}
						
					    appendBlockToStagingArea(fnameStaging, value, row_offset, col_offset, brlen, bclen);
					}
				}
				finally
				{
					if( reader != null )
						reader.close();
				}
			}

			//STEP 2: read matrix blocks from staging area and write matrix to HDFS
			String[] fnamesPartitions = new File(fnameStaging).list();			
			for( String pdir : fnamesPartitions  )
				writeBinaryBlockSequenceFileToHDFS( job, fnameNew, fnameStaging+"/"+pdir );			
		} 
		catch (Exception e) 
		{
			throw new DMLRuntimeException("Unable to partition binary block matrix.", e);
		}
	}

	/**
	 * 
	 * @param dir
	 * @param mb
	 * @param row_offset
	 * @param col_offset
	 * @param brlen
	 * @param bclen
	 * @throws DMLRuntimeException
	 * @throws IOException
	 */
	private void appendBlockToStagingArea( String dir, MatrixBlock mb, long row_offset, long col_offset, long brlen, long bclen ) 
		throws DMLRuntimeException, IOException
	{
		//NOTE: for temporary block we always create dense representations
		boolean sparse = mb.isInSparseFormat();
		int rows = mb.getNumRows();
		int cols = mb.getNumColumns();

		if( _format == PDataPartitionFormat.ROW_WISE )
		{	
			MatrixBlock tmp = new MatrixBlock( 1, cols, false ); 
			tmp.spaceAllocForDenseUnsafe(1, cols);
			
			for( int i=0; i<rows; i++ )
			{
				String pdir = checkAndCreateStagingDir(dir+"/"+(row_offset+1+i));
				String pfname = pdir+"/"+"block_"+(col_offset/bclen+1);
				if( sparse )
				{
					for( int j=0; j<cols; j++ )
					{
						double value = mb.getValueSparseUnsafe(i, j);
						tmp.setValueDenseUnsafe(0, j, value);
					}
				}
				else
				{
					for( int j=0; j<cols; j++ )
					{
						double value = mb.getValueDenseUnsafe(i, j);
						tmp.setValueDenseUnsafe(0, j, value);
					}
				}
				tmp.recomputeNonZeros();
				writeBlockToLocal(pfname, tmp);
			}
		}
		else if( _format == PDataPartitionFormat.ROW_BLOCK_WISE )
		{
			String pdir = checkAndCreateStagingDir(dir+"/"+(row_offset/brlen+1));
			String pfname = pdir+"/"+"block_"+(col_offset/bclen+1);
			writeBlockToLocal(pfname, mb);
		}
		else if( _format == PDataPartitionFormat.COLUMN_WISE )
		{
			//create object for reuse
			MatrixBlock tmp = new MatrixBlock( rows, 1, false ); 
			tmp.spaceAllocForDenseUnsafe(rows, 1);
						
			for( int i=0; i<cols; i++ )
			{
				String pdir = checkAndCreateStagingDir(dir+"/"+(col_offset+1+i));
				String pfname = pdir+"/"+"block_"+(row_offset/brlen+1); 			
				if( sparse )
				{
					for( int j=0; j<rows; j++ )
					{
						double value = mb.getValueSparseUnsafe(j, i);
						tmp.setValueDenseUnsafe(j, 0, value);
					}
				}
				else
				{
					for( int j=0; j<rows; j++ )
					{
						double value = mb.getValueDenseUnsafe(j, i);
						tmp.setValueDenseUnsafe(j, 0, value);
					}					
				}
				tmp.recomputeNonZeros();
				writeBlockToLocal(pfname, tmp);
			}				
		}
		else if( _format == PDataPartitionFormat.COLUMN_BLOCK_WISE )
		{
			String pdir = checkAndCreateStagingDir(dir+"/"+(col_offset/bclen+1));
			String pfname = pdir+"/"+"block_"+(row_offset/brlen+1);
			writeBlockToLocal(pfname, mb);
		}
	}
	
	/**
	 * 
	 * @param dir
	 * @param buffer
	 * @param brlen
	 * @param bclen
	 * @throws DMLRuntimeException
	 * @throws IOException
	 */
	private void appendCellBufferToStagingArea( String dir, LinkedList<PartitionCell> buffer, int brlen, int bclen ) 
		throws DMLRuntimeException, IOException
	{
		HashMap<Integer,LinkedList<PartitionCell>> sortedBuffer = new HashMap<Integer,LinkedList<PartitionCell>>();
		
		//sort cells in buffer wrt key
		int key = -1;
		for( PartitionCell c : buffer )
		{
			switch(_format)
			{
				case ROW_WISE:
					key = c.row;
					c.row = 1;
					break;
				case ROW_BLOCK_WISE:
					key = (c.row-1)/brlen+1;
					c.row = (c.row-1)%brlen+1;
					break;
				case COLUMN_WISE:
					key = c.col;
					c.col = 1;
					break;
				case COLUMN_BLOCK_WISE:
					key = (c.col-1)/bclen+1;
					c.col = (c.col-1)%bclen+1;
					break;
			}
			
			if( !sortedBuffer.containsKey(key) )
				sortedBuffer.put(key, new LinkedList<PartitionCell>());
			sortedBuffer.get(key).addLast(c);
		}	
		
		//write lists of cells to local files
		for( Entry<Integer,LinkedList<PartitionCell>> e : sortedBuffer.entrySet() )
		{
			String pdir = checkAndCreateStagingDir(dir+"/"+e.getKey());
			String pfname = pdir+"/"+"block_"+_seq.getNextID();
			writeCellListToLocal(pfname, e.getValue());
		}
	}	

	
	/////////////////////////////////////
	//     Helper methods for HDFS     //
	// read/write in different formats //
	/////////////////////////////////////
	
	public void writeBinaryBlockSequenceFileToHDFS( JobConf job, String dir, String lpdir ) 
		throws IOException
	{
		long key = getKeyFromFilePath(lpdir);
		FileSystem fs = FileSystem.get(job);
		Path path =  new Path(dir+"/"+key);
		SequenceFile.Writer writer = new SequenceFile.Writer(fs, job, path, MatrixIndexes.class, MatrixBlock.class); //beware ca 50ms

		try
		{
			String[] fnameBlocks = new File( lpdir ).list();
			for( String fnameBlock : fnameBlocks  )
			{
				MatrixBlock tmp = readBlockFromLocal(lpdir+"/"+fnameBlock);
				long key2 = getKey2FromFileName(fnameBlock);
				
				if( _format == PDataPartitionFormat.ROW_WISE || _format == PDataPartitionFormat.ROW_BLOCK_WISE )
				{
					writer.append(new MatrixIndexes(1,key2), tmp);
				}
				else if( _format == PDataPartitionFormat.COLUMN_WISE || _format == PDataPartitionFormat.COLUMN_BLOCK_WISE )
				{
					writer.append(new MatrixIndexes(key2,1), tmp);
				}
			}
		}
		finally
		{
			if( writer != null )
				writer.close();
		}
	}
	
	public void writeBinaryCellSequenceFileToHDFS( JobConf job, String dir, String lpdir ) 
		throws IOException
	{
		long key = getKeyFromFilePath(lpdir);
		FileSystem fs = FileSystem.get(job);
		Path path =  new Path(dir+"/"+key);
		SequenceFile.Writer writer = new SequenceFile.Writer(fs, job, path, MatrixIndexes.class, MatrixCell.class); //beware ca 50ms
	
		try
		{
			MatrixIndexes indexes = new MatrixIndexes();
			MatrixCell cell = new MatrixCell();
			
			String[] fnameBlocks = new File( lpdir ).list();
			for( String fnameBlock : fnameBlocks  )
			{
				LinkedList<PartitionCell> tmp = readCellListFromLocal(lpdir+"/"+fnameBlock);
				for( PartitionCell c : tmp )
				{
					indexes.setIndexes(c.row, c.col);
					cell.setValue(c.value);
					writer.append(indexes, cell);
				}
			}
		}
		finally
		{
			if( writer != null )
				writer.close();
		}
	}
	
	public void writeTextCellSequenceFileToHDFS( JobConf job, String dir, String lpdir ) 
		throws IOException
	{
		long key = getKeyFromFilePath(lpdir);
		FileSystem fs = FileSystem.get(job);
		Path path = new Path(dir+"/"+key);
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(fs.create(path,true)));		
		try
		{
			String[] fnameBlocks = new File( lpdir ).list();
			for( String fnameBlock : fnameBlocks  )
			{
				LinkedList<PartitionCell> tmp = readCellListFromLocal(lpdir+"/"+fnameBlock);
				for( PartitionCell c : tmp )
				{
					StringBuilder sb = new StringBuilder();
					sb.append(c.row);
					sb.append(" ");
					sb.append(c.col);
					sb.append(" ");
					sb.append(c.value);
					sb.append("\n");
					out.write( sb.toString() );
				}
			}
		}
		finally
		{
			if( out != null )
				out.close();
		}
	}
	
	
	/////////////////////////////////
	// Helper methods for local fs //
	//         read/write          //
	/////////////////////////////////
	
	/**
	 * 
	 * @param fname
	 * @param mb
	 * @throws IOException
	 */
	private void writeBlockToLocal(String fname, MatrixBlock mb) 
		throws IOException
	{
		FileOutputStream fos = new FileOutputStream( fname );
		DataOutputStream out = new DataOutputStream( fos );
		try 
		{
			mb.write(out);
		}
		finally
		{
			if( out != null )
				out.close();	
		}	
	}
	
	/**
	 * 
	 * @param fname
	 * @param buffer
	 * @throws IOException
	 */
	private void writeCellListToLocal( String fname, LinkedList<PartitionCell> buffer ) 
		throws IOException
	{
		FileOutputStream fos = new FileOutputStream( fname );
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(fos));	
		try 
		{
			for( PartitionCell c : buffer )
			{
				StringBuilder sb = new StringBuilder();
				sb.append(c.row);
				sb.append(" ");
				sb.append(c.col);
				sb.append(" ");
				sb.append(c.value);
				sb.append("\n");
				out.write( sb.toString() );
			}
		}
		finally
		{
			if( out != null )
				out.close();	
		}	
	}
	
	/**
	 * 
	 * @param fname
	 * @return
	 * @throws IOException
	 */
	private MatrixBlock readBlockFromLocal(String fname) 
		throws IOException
	{
		FileInputStream fis = new FileInputStream( fname );
		DataInputStream in = new DataInputStream( fis );
		MatrixBlock mb = new MatrixBlock();
		try 
		{
			mb.readFields(in);
		}
		finally
		{
			if( in != null )
				in.close ();
		}
   		
		return mb;
	}
	
	/**
	 * 
	 * @param fname
	 * @return
	 * @throws IOException
	 */
	private LinkedList<PartitionCell> readCellListFromLocal( String fname ) 
		throws IOException
	{
		FileInputStream fis = new FileInputStream( fname );
		BufferedReader in = new BufferedReader(new InputStreamReader(fis));	
		LinkedList<PartitionCell> buffer = new LinkedList<PartitionCell>();
		try 
		{
			String value = null;
			while( (value=in.readLine())!=null )
			{
				String cellStr = value.toString().trim();							
				StringTokenizer st = new StringTokenizer(cellStr, " ");
				int row = Integer.parseInt( st.nextToken() );
				int col = Integer.parseInt( st.nextToken() );
				double lvalue = Double.parseDouble( st.nextToken() );
				PartitionCell c =  new PartitionCell( row, col, lvalue );
				buffer.addLast( c );
			}
		}
		finally
		{
			if( in != null )
				in.close();
		}
   		
		return buffer;
	}
	
	/**
	 * 
	 * @param dir
	 * @return
	 */
	private String checkAndCreateStagingDir(String dir) 
	{
		File f =  new File(dir);		
		if( !f.exists() )
			f.mkdirs();
		
		return dir;
	}
	
	/**
	 * 
	 * @param dir
	 * @return
	 */
	private String cleanupStagingDir(String dir) 
	{
		File f =  new File(dir);
		if( f.exists() )
			rDelete(f);
		
		return dir;
	}
	
	/**
	 * 
	 * @param dir
	 * @return
	 */
	private long getKeyFromFilePath( String dir )
	{
		String[] dirparts = dir.split("/");
		long key = Long.parseLong( dirparts[dirparts.length-1] );
		return key;
	}
	
	/**
	 * 
	 * @param fname
	 * @return
	 */
	private long getKey2FromFileName( String fname )
	{
		return Long.parseLong( fname.split("_")[1] );
	}
	
	/**
	 * Helper class for representing text cell and binary cell records in order to
	 * allow for buffering and buffered read/write.
	 */
	private class PartitionCell
	{
		private int row;
		private int col;
		private double value;
		
		private PartitionCell( int r, int c, double v )
		{
			row = r;
			col = c;
			value = v;
		}
	}
}
