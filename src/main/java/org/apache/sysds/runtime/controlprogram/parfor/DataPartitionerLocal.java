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

package org.apache.sysds.runtime.controlprogram.parfor;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.HashMap;
import java.util.LinkedList;
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
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import org.apache.sysds.runtime.controlprogram.ParForProgramBlock.PartitionFormat;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.parfor.util.Cell;
import org.apache.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysds.runtime.controlprogram.parfor.util.StagingFileUtils;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixCell;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.util.FastStringTokenizer;
import org.apache.sysds.runtime.util.LocalFileUtils;

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
 */
public class DataPartitionerLocal extends DataPartitioner
{
	private static final boolean PARALLEL = true; 
	
	private IDSequence _seq = null;
	private MatrixBlock _reuseBlk = null;
	
	private int _par = -1;
	
	/**
	 * DataPartitionerLocal constructor.
	 * 
	 * @param dpf data partitionformat
	 * @param par -1 for serial otherwise number of threads, can be ignored by implementation
	 */
	public DataPartitionerLocal(PartitionFormat dpf, int par) {
		super(dpf._dpf, dpf._N);
		if( dpf.isBlockwise() )
			throw new DMLRuntimeException("Data partitioning formt '"+dpf+"' not supported by DataPartitionerLocal" );
		_seq = new IDSequence();
		_par = (par > 0) ? par : 1;
	}
	
	@Override
	protected void partitionMatrix(MatrixObject in, String fnameNew, FileFormat fmt, long rlen, long clen, int blen)
	{
		//force writing to disk (typically not required since partitioning only applied if dataset exceeds CP size)
		in.exportData(); //written to disk iff dirty
		
		String fname = in.getFileName();
		String fnameStaging = LocalFileUtils.getUniqueWorkingDir( LocalFileUtils.CATEGORY_PARTITIONING );
		
		//reblock input matrix
		if( fmt == FileFormat.TEXT )
			partitionTextCell( fname, fnameStaging, fnameNew, rlen, clen, blen );
		else if( fmt == FileFormat.BINARY )
			partitionBinaryBlock( fname, fnameStaging, fnameNew, rlen, clen, blen );
		else
			throw new DMLRuntimeException("Cannot create data partitions of format: "+fmt.toString());
	
		LocalFileUtils.cleanupWorkingDirectory(fnameStaging);
	}

	private void partitionTextCell( String fname, String fnameStaging, String fnameNew, long rlen, long clen, int blen ) 
	{
		long row = -1;
		long col = -1;
		
		try 
		{
			//STEP 1: read matrix from HDFS and write blocks to local staging area
			//check and add input path
			JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
			Path path = new Path(fname);
			FileInputFormat.addInputPath(job, path);
			TextInputFormat informat = new TextInputFormat();
			informat.configure(job);
			InputSplit[] splits = informat.getSplits(job, 1);
			
			LinkedList<Cell> buffer = new LinkedList<>();
			LongWritable key = new LongWritable();
			Text value = new Text();
			FastStringTokenizer st = new FastStringTokenizer(' ');
			
			for(InputSplit split: splits)
			{
				RecordReader<LongWritable,Text> reader = informat.getRecordReader(split, job, Reporter.NULL);
				try
				{
					while(reader.next(key, value))
					{
						st.reset( value.toString() ); //reset tokenizer
						row = st.nextLong();
						col = st.nextLong();
						double lvalue = st.nextDouble();
						Cell tmp = new Cell( row, col, lvalue ); 
		
						buffer.addLast( tmp );
						if( buffer.size() > StagingFileUtils.CELL_BUFFER_SIZE ) //periodic flush
						{
							appendCellBufferToStagingArea(fnameStaging, buffer, blen);
							buffer.clear();
						}
					}
					
					//final flush
					if( !buffer.isEmpty() )
					{
						appendCellBufferToStagingArea(fnameStaging, buffer, blen);
						buffer.clear();
					}
				}
				finally {
					IOUtilFunctions.closeSilently(reader);
				}
			}

			//STEP 2: read matrix blocks from staging area and write matrix to HDFS
			String[] fnamesPartitions = new File(fnameStaging).list();	
			if(PARALLEL) 
			{
				int len = Math.min(fnamesPartitions.length, _par);
				Thread[] threads = new Thread[len];
				for( int i=0;i<len;i++ )
				{
					int start = i*(int)Math.ceil(((double)fnamesPartitions.length)/len);
					int end = (i+1)*(int)Math.ceil(((double)fnamesPartitions.length)/len)-1;
					end = Math.min(end, fnamesPartitions.length-1);
					threads[i] = new Thread(new DataPartitionerWorkerTextCell(job, fnameNew, fnameStaging, fnamesPartitions, start, end));
					threads[i].start();
				}
				
				for( Thread t : threads )
					t.join();
			}
			else
			{
				for( String pdir : fnamesPartitions  )
					writeTextCellFileToHDFS( job, fnameNew, fnameStaging+"/"+pdir );	
			}
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

	@SuppressWarnings("deprecation")
	private void partitionBinaryBlock( String fname, String fnameStaging, String fnameNew, long rlen, long clen, int blen ) 
	{
		//STEP 1: read matrix from HDFS and write blocks to local staging area	
		//check and add input path
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path(fname);
		
		try(FileSystem fs = IOUtilFunctions.getFileSystem(path, job)) {
			//create reuse object
			_reuseBlk = DataPartitioner.createReuseMatrixBlock(_format, blen, blen);
			
			//prepare sequence file reader, and write to local staging area
			MatrixIndexes key = new MatrixIndexes(); 
			MatrixBlock value = new MatrixBlock();
			
			for(Path lpath : IOUtilFunctions.getSequenceFilePaths(fs, path) )
			{
				try(SequenceFile.Reader reader = new SequenceFile.Reader(fs,lpath,job)) {
					while(reader.next(key, value)) //for each block
					{
						long row_offset = (key.getRowIndex()-1)*blen;
						long col_offset = (key.getColumnIndex()-1)*blen;
						long rows = value.getNumRows();
						long cols = value.getNumColumns();
						
						//bound check per block
						if( row_offset + rows < 1 || row_offset + rows > rlen || col_offset + cols<1 || col_offset + cols > clen )
						{
							throw new IOException("Matrix block ["+(row_offset+1)+":"+(row_offset+rows)+","+(col_offset+1)+":"+(col_offset+cols)+"] " +
									              "out of overall matrix range [1:"+rlen+",1:"+clen+"].");
						}
						
					    appendBlockToStagingArea(fnameStaging, value, row_offset, col_offset, blen);
					}
				}
			}

			//STEP 2: read matrix blocks from staging area and write matrix to HDFS
			String[] fnamesPartitions = new File(fnameStaging).list();
			if(PARALLEL) 
			{
				int len = Math.min(fnamesPartitions.length, _par);
				Thread[] threads = new Thread[len];
				for( int i=0;i<len;i++ )
				{
					int start = i*(int)Math.ceil(((double)fnamesPartitions.length)/len);
					int end = (i+1)*(int)Math.ceil(((double)fnamesPartitions.length)/len)-1;
					end = Math.min(end, fnamesPartitions.length-1);
					threads[i] = new Thread(new DataPartitionerWorkerBinaryBlock(job, fnameNew, fnameStaging, fnamesPartitions, start, end));
					threads[i].start();
				}
				
				for( Thread t : threads )
					t.join();
			}
			else {
				for( String pdir : fnamesPartitions  )
					writeBinaryBlockSequenceFileToHDFS( job, fnameNew, fnameStaging+"/"+pdir, false );
			}
		} 
		catch (Exception e)  {
			throw new DMLRuntimeException("Unable to partition binary block matrix.", e);
		}
	}

	private void appendBlockToStagingArea( String dir, MatrixBlock mb, long row_offset, long col_offset, long blen ) 
		throws IOException
	{
		//NOTE: for temporary block we always create dense representations
		boolean sparse = mb.isInSparseFormat();
		long nnz = mb.getNonZeros();
		long rows = mb.getNumRows();
		long cols = mb.getNumColumns();
		double sparsity = ((double)nnz)/(rows*cols);

		if( _format == PDataPartitionFormat.ROW_WISE ) 
		{	
			_reuseBlk.reset( 1, (int)cols, sparse, (int) (cols*sparsity) );
			for( int i=0; i<rows; i++ )
			{
				String pdir = LocalFileUtils.checkAndCreateStagingDir(dir+"/"+(row_offset+1+i));
				String pfname = pdir+"/"+"block_"+(col_offset/blen+1);
				mb.slice(i, i, 0, (int)(cols-1), _reuseBlk);
				LocalFileUtils.writeMatrixBlockToLocal(pfname, _reuseBlk);
				_reuseBlk.reset();
			}
		}
		else if( _format == PDataPartitionFormat.ROW_BLOCK_WISE )
		{
			String pdir = LocalFileUtils.checkAndCreateStagingDir(dir+"/"+(row_offset/blen+1));
			String pfname = pdir+"/"+"block_"+(col_offset/blen+1);
			LocalFileUtils.writeMatrixBlockToLocal(pfname, mb);
		}
		else if( _format == PDataPartitionFormat.COLUMN_WISE )
		{
			//create object for reuse
			_reuseBlk.reset( (int)rows, 1, false );
			
			for( int i=0; i<cols; i++ )
			{
				String pdir = LocalFileUtils.checkAndCreateStagingDir(dir+"/"+(col_offset+1+i));
				String pfname = pdir+"/"+"block_"+(row_offset/blen+1);
				mb.slice(0, (int)(rows-1), i, i, _reuseBlk);
				LocalFileUtils.writeMatrixBlockToLocal(pfname, _reuseBlk);
				_reuseBlk.reset();
			}				
		}
		else if( _format == PDataPartitionFormat.COLUMN_BLOCK_WISE )
		{
			String pdir = LocalFileUtils.checkAndCreateStagingDir(dir+"/"+(col_offset/blen+1));
			String pfname = pdir+"/"+"block_"+(row_offset/blen+1);
			LocalFileUtils.writeMatrixBlockToLocal(pfname, mb);
		}
	}

	private void appendCellBufferToStagingArea( String dir, LinkedList<Cell> buffer, int blen ) 
		throws IOException
	{
		HashMap<Long,LinkedList<Cell>> sortedBuffer = new HashMap<>();
		
		//sort cells in buffer wrt key
		long key = -1;
		for( Cell c : buffer )
		{
			switch(_format)
			{
				case ROW_WISE:
					key = c.getRow();
					c.setRow(1);
					break;
				case ROW_BLOCK_WISE:
					key = (c.getRow()-1)/blen+1;
					c.setRow((c.getRow()-1)%blen+1);
					break;
				case COLUMN_WISE:
					key = c.getCol();
					c.setCol(1);
					break;
				case COLUMN_BLOCK_WISE:
					key = (c.getCol()-1)/blen+1;
					c.setCol((c.getCol()-1)%blen+1);
					break;
				default:
					//do nothing
			}
			
			if( !sortedBuffer.containsKey(key) )
				sortedBuffer.put(key, new LinkedList<Cell>());
			sortedBuffer.get(key).addLast(c);
		}	
		
		//write lists of cells to local files
		for( Entry<Long,LinkedList<Cell>> e : sortedBuffer.entrySet() )
		{
			String pdir = LocalFileUtils.checkAndCreateStagingDir(dir+"/"+e.getKey());
			String pfname = pdir+"/"+"block_"+_seq.getNextID();
			StagingFileUtils.writeCellListToLocal(pfname, e.getValue());
		}
	}	

	
	/////////////////////////////////////
	//     Helper methods for HDFS     //
	// read/write in different formats //
	/////////////////////////////////////
	
	@SuppressWarnings({ "deprecation"})
	public void writeBinaryBlockSequenceFileToHDFS( JobConf job, String dir, String lpdir, boolean threadsafe ) 
		throws IOException
	{
		long key = getKeyFromFilePath(lpdir);
		Path path =  new Path(dir+"/"+key);
		SequenceFile.Writer writer = null;
		
		try {
			FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
			writer = new SequenceFile.Writer(fs, job, path, MatrixIndexes.class, MatrixBlock.class); //beware ca 50ms
			String[] fnameBlocks = new File( lpdir ).list();
			for( String fnameBlock : fnameBlocks  )
			{
				long key2 = getKey2FromFileName(fnameBlock);
				MatrixBlock tmp = null;
				if( threadsafe )
					tmp = LocalFileUtils.readMatrixBlockFromLocal(lpdir+"/"+fnameBlock);
				else
					tmp = LocalFileUtils.readMatrixBlockFromLocal(lpdir+"/"+fnameBlock, _reuseBlk);
				
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
		finally {
			IOUtilFunctions.closeSilently(writer);
		}
	}
	
	@SuppressWarnings({ "deprecation" })
	public void writeBinaryCellSequenceFileToHDFS( JobConf job, String dir, String lpdir ) 
		throws IOException
	{
		long key = getKeyFromFilePath(lpdir);
		Path path =  new Path(dir+"/"+key);
		SequenceFile.Writer writer = null;
		
		try {
			FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
			writer = new SequenceFile.Writer(fs, job, path, MatrixIndexes.class, MatrixCell.class); //beware ca 50ms
			
			MatrixIndexes indexes = new MatrixIndexes();
			MatrixCell cell = new MatrixCell();
			
			String[] fnameBlocks = new File( lpdir ).list();
			for( String fnameBlock : fnameBlocks  )
			{
				LinkedList<Cell> tmp = StagingFileUtils.readCellListFromLocal(lpdir+"/"+fnameBlock);
				for( Cell c : tmp )
				{
					indexes.setIndexes(c.getRow(), c.getCol());
					cell.setValue(c.getValue());
					writer.append(indexes, cell);
				}
			}
		}
		finally {
			IOUtilFunctions.closeSilently(writer);
		}
	}
	
	public void writeTextCellFileToHDFS( JobConf job, String dir, String lpdir ) 
		throws IOException
	{
		long key = getKeyFromFilePath(lpdir);
		Path path = new Path(dir+"/"+key);
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
		try(BufferedWriter out = new BufferedWriter(new OutputStreamWriter(fs.create(path,true))))
		{
			//for obj reuse and preventing repeated buffer re-allocations
			StringBuilder sb = new StringBuilder();
			
			String[] fnameBlocks = new File( lpdir ).list();
			for( String fnameBlock : fnameBlocks  )
			{
				LinkedList<Cell> tmp = StagingFileUtils.readCellListFromLocal(lpdir+"/"+fnameBlock);
				for( Cell c : tmp )
				{
					sb.append(c.getRow());
					sb.append(' ');
					sb.append(c.getCol());
					sb.append(' ');
					sb.append(c.getValue());
					sb.append('\n');
					out.write( sb.toString() );
					sb.setLength(0);
				}
			}
		}
	}
	
	
	/////////////////////////////////
	// Helper methods for local fs //
	//         read/write          //
	/////////////////////////////////

	private static long getKeyFromFilePath( String dir ) {
		String[] dirparts = dir.split("/");
		long key = Long.parseLong( dirparts[dirparts.length-1] );
		return key;
	}

	private static long getKey2FromFileName( String fname ) {
		return Long.parseLong( fname.split("_")[1] );
	}

	private abstract class DataPartitionerWorker implements Runnable
	{
		private JobConf _job = null;
		private String _fnameNew = null;
		private String _fnameStaging = null;
		private String[] _fnamesPartitions = null;
		private int _start = -1;
		private int _end = -1;
		
		public DataPartitionerWorker(JobConf job, String fnameNew, String fnameStaging, String[] fnamesPartitions, int start, int end)
		{
			_job = job;
			_fnameNew = fnameNew;
			_fnameStaging = fnameStaging;
			_fnamesPartitions = fnamesPartitions;
			_start = start;
			_end = end;
		}

		@Override
		public void run() 
		{
			//read each input if required
			try
			{
				for( int i=_start; i<=_end; i++ )
				{
					String pdir = _fnamesPartitions[i];
					writeFileToHDFS( _job, _fnameNew, _fnameStaging+"/"+pdir );	
				}
			}
			catch(Exception ex)
			{
				throw new RuntimeException("Failed on parallel data partitioning.", ex);
			}
		}
		
		public abstract void writeFileToHDFS( JobConf job, String fnameNew, String stagingDir ) 
			throws IOException;
	}
	
	private class DataPartitionerWorkerTextCell extends DataPartitionerWorker
	{
		public DataPartitionerWorkerTextCell(JobConf job, String fnameNew, String fnameStaging, String[] fnamesPartitions, int start, int end) {
			super(job, fnameNew, fnameStaging, fnamesPartitions, start, end);
		}

		@Override
		public void writeFileToHDFS(JobConf job, String fnameNew, String stagingDir) 
			throws IOException 
		{
			writeTextCellFileToHDFS( job, fnameNew, stagingDir );
		}
	}
	
	private class DataPartitionerWorkerBinaryBlock extends DataPartitionerWorker
	{
		public DataPartitionerWorkerBinaryBlock(JobConf job, String fnameNew, String fnameStaging, String[] fnamesPartitions, int start, int end) {
			super(job, fnameNew, fnameStaging, fnamesPartitions, start, end);
		}

		@Override
		public void writeFileToHDFS(JobConf job, String fnameNew, String stagingDir) 
			throws IOException
		{
			writeBinaryBlockSequenceFileToHDFS( job, fnameNew, stagingDir, true );
		}
	}
}
