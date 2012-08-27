package com.ibm.bi.dml.runtime.controlprogram.parfor;

import java.io.File;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.parser.ParseException;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.ConfigurationManager;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.MatrixObjectNew;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.configuration.DMLConfig;


/**
 * This is the base class for all data partitioner. 
 * 
 */
public abstract class DataPartitioner 
{	
	protected static final String NAME_SUFFIX = "_dp";
	protected static final String STAGING_DIR = "/tmp/partitioning/";
	
	protected PDataPartitionFormat _format = null;
	
	protected DataPartitioner( PDataPartitionFormat dpf )
	{
		_format = dpf;
	}
	
	/**
	 * Creates a partitioned matrix object based on the given input matrix object, 
	 * according to the specified split format. The input matrix can be in-memory
	 * or still on HDFS and the partitioned output matrix is written to HDFS. The
	 * created matrix object can be used transparently for obtaining the full matrix
	 * or reading 1 or multiple partitions based on given index ranges. 
	 * 
	 * @param in
	 * @return
	 * @throws DMLRuntimeException
	 */
	public abstract MatrixObjectNew createPartitionedMatrix( MatrixObjectNew in )
		throws DMLRuntimeException;

	/**
	 * @throws ParseException 
	 * 
	 */
	public static void cleanupWorkingDirectory() 
		throws ParseException
	{
		//build dir name to be cleaned up
		StringBuilder sb = new StringBuilder();
		sb.append(STAGING_DIR);
		sb.append(Lops.FILE_SEPARATOR);
		sb.append(ConfigurationManager.getConfig().getTextValue(DMLConfig.SCRATCH_SPACE));
		sb.append(Lops.FILE_SEPARATOR);
		sb.append(Lops.PROCESS_PREFIX);
		sb.append(DMLScript.getUUID());
		String dir =  sb.toString();
		
		//cleanup
		File fdir = new File(dir);		
		rDelete( fdir );
	}
	
	private static void rDelete(File dir)
	{
		//recursively delete files if required
		if( dir.isDirectory() )
		{
			File[] files = dir.listFiles();
			for( File f : files )
				rDelete( f );	
		}
		
		//delete file itself
		dir.delete();
	}
}
