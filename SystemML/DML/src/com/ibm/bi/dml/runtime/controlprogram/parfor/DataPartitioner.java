package com.ibm.bi.dml.runtime.controlprogram.parfor;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.ibm.bi.dml.hops.Hops;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.utils.DMLRuntimeException;


/**
 * This is the base class for all data partitioner. 
 * 
 */
public abstract class DataPartitioner 
{	
	protected static final Log LOG = LogFactory.getLog(DataPartitioner.class.getName());
	
	//note: the following value has been empirically determined but might change in the future,
	//MatrixBlockDSM.SPARCITY_TURN_POINT (with 0.4) was too high because we create 3-4 values per nnz and 
	//have some computation overhead for binary cell.
	protected static final double SPARSITY_CELL_THRESHOLD = 0.1d; 
	
	protected static final String NAME_SUFFIX = "_dp";
	
	//instance variables
	protected PDataPartitionFormat _format = null;
	protected int _n = -1; //blocksize if applicable
	protected boolean _allowBinarycell = true;
	
	protected DataPartitioner( PDataPartitionFormat dpf, int n )
	{
		_format = dpf;
		_n = n;
	}
	
	/**
	 * see createPartitionedMatrix( MatrixObjectNew in, boolean force )
	 * (with default not to force)
	 * 
	 * @param in
	 * @return
	 * @throws DMLRuntimeException
	 */
	public MatrixObject createPartitionedMatrixObject( MatrixObject in )
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
	public MatrixObject createPartitionedMatrixObject( MatrixObject in, boolean force )
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
		double sparsity = ((double)nonZeros)/(rows*cols);
		
		if( !force ) //try to optimize, if format not forced
		{
			//check lower bound of useful data partitioning
			if( rows < Hops.CPThreshold && cols < Hops.CPThreshold )  //or matrix already fits in mem
			{
				return in;
			}
			
			//check for changing to blockwise representations
			if( _format == PDataPartitionFormat.ROW_WISE && cols < Hops.CPThreshold )
			{
				LOG.debug("Changing format from "+PDataPartitionFormat.ROW_WISE+" to "+PDataPartitionFormat.ROW_BLOCK_WISE+".");
				_format = PDataPartitionFormat.ROW_BLOCK_WISE;
			}
			if( _format == PDataPartitionFormat.COLUMN_WISE && rows < Hops.CPThreshold )
			{
				LOG.debug("Changing format from "+PDataPartitionFormat.COLUMN_WISE+" to "+PDataPartitionFormat.ROW_BLOCK_WISE+".");
				_format = PDataPartitionFormat.COLUMN_BLOCK_WISE;
			}
			
			//FIXME: just for test
			//_n = 16000000; //16000
			//_format = PDataPartitionFormat.ROW_BLOCK_WISE_N;
		}
		
		//check changing to binarycell in case of sparse cols (robustness)
		boolean convertBlock2Cell = false;
		if(    ii == InputInfo.BinaryBlockInputInfo 
			&& _allowBinarycell
			&& _format == PDataPartitionFormat.COLUMN_WISE	
			&& sparsity < SPARSITY_CELL_THRESHOLD )
		{
			LOG.debug("Changing partition outputinfo from binaryblock to binarycell due to sparsity="+sparsity);
			oi = OutputInfo.BinaryCellOutputInfo;
			convertBlock2Cell = true;
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
		MatrixObject mobj = new MatrixObject(vt, fnameNew );
		mobj.setDataType(DataType.MATRIX);
		mobj.setVarName( varname+NAME_SUFFIX );
		mobj.setPartitioned( _format, _n ); 
		MatrixCharacteristics mcNew = new MatrixCharacteristics( rows, cols,
				                           (_format==PDataPartitionFormat.ROW_WISE)? 1 : (int)brlen, //for blockwise brlen anyway
				                           (_format==PDataPartitionFormat.COLUMN_WISE)? 1 : (int)bclen ); //for blockwise bclen anyway
		mcNew.setNonZeros( nonZeros );
		if( convertBlock2Cell )
			ii = InputInfo.BinaryCellInputInfo;
		MatrixFormatMetaData metaNew = new MatrixFormatMetaData(mcNew,oi,ii);
		mobj.setMetaData(metaNew);	 
		
		return mobj;
	}
	
	/**
	 * 
	 */
	public void disableBinaryCell()
	{
		_allowBinarycell = false;
	}
	
	/**
	 * 
	 * @param fname
	 * @param fnameNew
	 * @param ii
	 * @param oi
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @throws DMLRuntimeException
	 */
	protected abstract void partitionMatrix( String fname, String fnameNew, InputInfo ii, OutputInfo oi, long rlen, long clen, int brlen, int bclen )
		throws DMLRuntimeException;

	
	public static MatrixBlock createReuseMatrixBlock( PDataPartitionFormat dpf, int rows, int cols ) 
	{
		MatrixBlock tmp = null;
		
		switch( dpf )
		{
			case ROW_WISE:
				//default assumption sparse, but reset per input block anyway
				tmp = new MatrixBlock( 1, (int)cols, true, (int)(cols*0.1) );
				break;
			case COLUMN_WISE:
				//default dense because single column alwyas below SKINNY_MATRIX_TURN_POINT
				tmp = new MatrixBlock( (int)rows, 1, false );
				break;
		}
		
		return tmp;
	}
}
