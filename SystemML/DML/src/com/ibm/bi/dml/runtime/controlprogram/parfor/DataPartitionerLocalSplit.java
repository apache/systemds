package com.ibm.bi.dml.runtime.controlprogram.parfor;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.hops.Hops;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.MatrixObjectNew;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.utils.DMLRuntimeException;

/**
 * Partitions a given matrix into row or column partitions with a two pass-approach.
 * In the first phase the input matrix is read from HDFS and sorted into block partitions
 * in a staging area in the local filesystem ccording to the partition format. 
 * In order to allow for scalable partitioning, we process one block at a time. 
 * Furthermore, in the second phase, all blocks of a partition are append to a sequence file
 * on HDFS. Block-wise partitioning and write-once semantics of sequence files require the
 * indirection over the local staging area. For scalable computation, we process one 
 * sequence file at a time.
 *
 * TODO metadata file for partitioned matrices ?
 * TODO parallel writing of sequence files (most expensive part and nicely parallelizable, but larger memory requirements)
 *
 */
public class DataPartitionerLocalSplit extends DataPartitioner
{
	private static Configuration conf = new Configuration();
		
	public DataPartitionerLocalSplit(PDataPartitionFormat dpf) 
	{
		super(dpf);
	}

	@Override
	public MatrixObjectNew createPartitionedMatrix(MatrixObjectNew in)
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
		long rows = mc.get_rows(); 
		long cols = mc.get_cols();
		int brlen = mc.get_rows_per_block();
		int bclen = mc.get_cols_per_block();
		
		//check lower bound of useful data partitioning
		if( ( rows == 1 || cols == 1 ) ||                            //is vector
			( rows < Hops.CPThreshold && cols < Hops.CPThreshold) )  //or matrix already fits in mem
		{
			return in;
		}
		
		//reblock input matrix
		String fnameNew = null;
		if( ii == InputInfo.BinaryBlockInputInfo )
			fnameNew = reblockBinaryToBinary( fname, brlen, bclen );
		else if( ii == InputInfo.TextCellInputInfo )
			fnameNew = reblockTextcellToTextcell( fname, brlen, bclen );
		else
		{
			System.out.println("Warning: Cannot create data partitions of format: "+ii.toString());
			return in; //return unmodified input matrix, similar to no partitioning
		}
		
		
		//create output matrix object
		MatrixObjectNew mobj = new MatrixObjectNew(vt, fnameNew );
		mobj.setDataType(DataType.MATRIX);
		mobj.setVarName( varname+NAME_SUFFIX );
		mobj.setPartitioned(); 
		MatrixCharacteristics mcNew = new MatrixCharacteristics( rows, cols,
				                           (_format==PDataPartitionFormat.ROW_WISE)? 1 : (int)brlen,
				                           (_format==PDataPartitionFormat.COLUMN_WISE)? 1 : (int)bclen );
		MatrixFormatMetaData metaNew = new MatrixFormatMetaData(mcNew,oi,ii);
		mobj.setMetaData(metaNew);	 
		
		return mobj;
	}

	
	private String reblockBinaryToBinary( String fname, int brlen, int bclen ) 
		throws DMLRuntimeException
	{
		String fnameNew = fname + NAME_SUFFIX;
		String fnameStaging = STAGING_DIR+"/"+fname;
		InputInfo ii = InputInfo.BinaryBlockInputInfo;
		
		try 
		{
			//STEP 1: read matrix from HDFS and write blocks to local staging area
			
			//check and add input path
			JobConf job = new JobConf();
			Path path = new Path(fname);
			FileInputFormat.addInputPath(job, path);
			
			//prepare sequence file reader, and write to local staging area
			InputFormat informat = ii.inputFormatClass.newInstance();
			InputSplit[] splits= informat.getSplits(job, 1);
			Writable key = ii.inputKeyClass.newInstance();
			Writable value = ii.inputValueClass.newInstance();
			
			for(InputSplit split : splits)
			{
				RecordReader reader=informat.getRecordReader(split, job, Reporter.NULL);
				while(reader.next(key, value)) //for each block
				{
					MatrixIndexes index0 = (MatrixIndexes) key;
					MatrixBlock block0 = (MatrixBlock) value;
					
					long row_offset = (index0.getRowIndex()-1)*brlen;
					long col_offset = (index0.getColumnIndex()-1)*bclen;
					
				    appendBlockToStagingArea(fnameStaging, block0, row_offset, col_offset, brlen, bclen);
				}
				reader.close();
			}
			
			//STEP 2: read matrix blocks from staging area and write matrix to HDFS
			String[] fnamesPartitions = new File(fnameStaging).list();
			for( String pdir : fnamesPartitions  )
				writeSequenceFileToHDFS( fnameNew, fnameStaging+"/"+pdir );			
		} 
		catch (Exception e) 
		{
			throw new DMLRuntimeException(e);
		}
		
		return fnameNew;
	}
	
	
	private String reblockTextcellToTextcell( String fname, int brlen, int bclen )
		throws DMLRuntimeException
	{
		throw new DMLRuntimeException("not implemented yet.");
	}
	

	private void appendBlockToStagingArea( String dir, MatrixBlock mb, long row_offset, long col_offset, long brlen, long bclen ) 
		throws DMLRuntimeException, IOException
	{
		if( _format == PDataPartitionFormat.ROW_WISE )
		{	
			for( int i=0; i<mb.getNumRows(); i++ )
			{
				String pdir = checkAndCreateStagingDir(dir+"/"+(row_offset+1+i));
				String pfname = pdir+"/"+"block_"+(col_offset/bclen+1);

				MatrixBlock tmp = new MatrixBlock( 1,mb.getNumColumns(), mb.isInSparseFormat() ); 
				for( int j=0; j<mb.getNumColumns(); j++ )
				{
					double value = mb.getValue(i, j);
					if( value != 0 )//for sparse matrices
						tmp.setValue(0, j, value);
				}
				writeBlockToLocal(pfname, tmp);
			}
		}
		else if( _format == PDataPartitionFormat.COLUMN_WISE )
		{
			for( int i=0; i<mb.getNumColumns(); i++ )
			{
				String pdir = checkAndCreateStagingDir(dir+"/"+(col_offset+1+i));
				String pfname = pdir+"/"+"block_"+(row_offset/brlen+1); 
			
				MatrixBlock tmp = new MatrixBlock( mb.getNumRows(),1, mb.isInSparseFormat() ); 
				for( int j=0; j<mb.getNumRows(); j++ )
				{
					double value = mb.getValue(j, i);
					if( value != 0 )//for sparse matrices
						tmp.setValue(j, 0, value);
				}
				
				writeBlockToLocal(pfname, tmp);
			}				
		}
	}
	
	public void writeSequenceFileToHDFS( String dir, String lpdir ) 
		throws IOException
	{
		long key = getKeyFromFilePath(lpdir);
		SequenceFile.Writer writer = new SequenceFile.Writer(FileSystem.get(conf), conf, 
											                new Path(dir+"/"+key),
											                MatrixIndexes.class, MatrixBlock.class);

		String[] fnameBlocks = new File( lpdir ).list();
		for( String fnameBlock : fnameBlocks  )
		{
			MatrixBlock tmp = readBlockFromLocal(lpdir+"/"+fnameBlock);
			long key2 = getKey2FromFileName(fnameBlock);
			
			if( _format == PDataPartitionFormat.ROW_WISE )
			{
				writer.append(new MatrixIndexes(key,key2), tmp);
			}
			else if( _format == PDataPartitionFormat.COLUMN_WISE )
			{
				writer.append(new MatrixIndexes(key2,key), tmp);
			}
		}
		
		writer.close();
	}
	
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
			out.close();	
		}	
	}
	
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
			in.close ();
		}
   		
		return mb;
	}
	
	
	
	private String checkAndCreateStagingDir(String dir) 
	{
		File f =  new File(dir);
		if( !f.exists() )
			f.mkdirs();
		
		return dir;
	}
	
	private long getKeyFromFilePath( String dir )
	{
		String[] dirparts = dir.split("/");
		long key = Long.parseLong( dirparts[dirparts.length-1] );
		return key;
	}
	
	private long getKey2FromFileName( String fname )
	{
		return Long.parseLong( fname.split("_")[1] );
	}
}
