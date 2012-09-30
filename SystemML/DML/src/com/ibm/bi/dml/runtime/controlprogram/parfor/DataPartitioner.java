package com.ibm.bi.dml.runtime.controlprogram.parfor;

import com.ibm.bi.dml.hops.Hops;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.ConfigurationManager;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.MatrixObjectNew;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.runtime.util.UtilFunctions;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.configuration.DMLConfig;


/**
 * This is the base class for all data partitioner. 
 * 
 */
public abstract class DataPartitioner 
{	
	protected static final String NAME_SUFFIX = "_dp";
	protected static final int CELL_BUFFER_SIZE = 100000;
	
	protected static String STAGING_DIR = null;	
	
	protected PDataPartitionFormat _format = null;
	
	protected DataPartitioner( PDataPartitionFormat dpf )
	{
		_format = dpf;
		
		//configure staging dir root
		DMLConfig conf = ConfigurationManager.getConfig();
		if( conf != null )
			STAGING_DIR = conf.getTextValue(DMLConfig.LOCAL_TMP_DIR) + "/partitioning/";
		else
			STAGING_DIR = "tmp/systemml/partitioning/";
		
		//create shared staging dir if not existing
		UtilFunctions.createLocalFileIfNotExist(STAGING_DIR, DMLConfig.DEFAULT_SHARED_DIR_PERMISSION);
	}
	
	/**
	 * see createPartitionedMatrix( MatrixObjectNew in, boolean force )
	 * (with default not to force)
	 * 
	 * @param in
	 * @return
	 * @throws DMLRuntimeException
	 */
	public MatrixObjectNew createPartitionedMatrixObject( MatrixObjectNew in )
		throws DMLRuntimeException
	{
		return createPartitionedMatrixObject(in, false);
	}

	/**
	 * Creates a partitioned matrix object based on the given input matrix object, 
	 * according to the specified split format. The input matrix can be in-memory
	 * or still on HDFS and the partitioned output matrix is written to HDFS. The
	 * created matrix object can be used transparently for obtaining the full matrix
	 * or reading 1 or multiple partitions based on given index ranges. 
	 * 
	 * @param in
	 * @param force
	 * @return
	 * @throws DMLRuntimeException
	 */
	public MatrixObjectNew createPartitionedMatrixObject( MatrixObjectNew in, boolean force )
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
		long nonZeros = mc.getNonZeros();
		
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
		
		try{
			MapReduceTool.deleteFileIfExistOnHDFS(fnameNew);
		}
		catch(Exception ex){
			throw new DMLRuntimeException( ex );
		}
		
		//core partitioning (depending on subclass)
		partitionMatrix( fname, fnameNew, ii, oi, rows, cols, brlen, bclen );
		
		//create output matrix object
		MatrixObjectNew mobj = new MatrixObjectNew(vt, fnameNew );
		mobj.setDataType(DataType.MATRIX);
		mobj.setVarName( varname+NAME_SUFFIX );
		mobj.setPartitioned( _format ); 
		MatrixCharacteristics mcNew = new MatrixCharacteristics( rows, cols,
				                           (_format==PDataPartitionFormat.ROW_WISE)? 1 : (int)brlen, //for blockwise brlen anyway
				                           (_format==PDataPartitionFormat.COLUMN_WISE)? 1 : (int)bclen ); //for blockwise bclen anyway
		mcNew.setNonZeros( nonZeros );
		MatrixFormatMetaData metaNew = new MatrixFormatMetaData(mcNew,oi,ii);
		mobj.setMetaData(metaNew);	 
		
		return mobj;
	}

	protected abstract void partitionMatrix( String fname, String fnameNew, InputInfo ii, OutputInfo oi, long rlen, long clen, int brlen, int bclen )
		throws DMLRuntimeException;

}
