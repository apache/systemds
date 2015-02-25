/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor;

import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.controlprogram.caching.CacheStatistics;
import com.ibm.bi.dml.runtime.controlprogram.caching.CacheableData;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.parfor.Task.TaskType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.StatisticMonitor;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDHandler;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.PairWritableBlock;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.PairWritableCell;
import com.ibm.bi.dml.runtime.instructions.cp.Data;
import com.ibm.bi.dml.runtime.instructions.cp.IntObject;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.util.LocalFileUtils;
import com.ibm.bi.dml.utils.Statistics;

/**
 *
 */
public class RemoteDPParWorkerReducer extends ParWorker
	implements Reducer<LongWritable, Writable, Writable, Writable>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	//MR data partitioning attributes
	private String _inputVar = null;
	private String _iterVar = null;
	private PDataPartitionFormat _dpf = null;
	private OutputInfo _info = null;
	private int _rlen = -1;
	private int _clen = -1;
	private int _brlen = -1;
	private int _bclen = -1;
	
	//reuse matrix partition
	private MatrixBlock _partition = null; 
	private boolean _tSparseCol = false;
		
	//MR ParWorker attributes  
	protected String  _stringID       = null; 
	protected HashMap<String, String> _rvarFnames = null; 

	//cached collector/reporter
	protected OutputCollector<Writable, Writable> _out = null;
	protected Reporter _report = null;
	
	/**
	 * 
	 */
	public RemoteDPParWorkerReducer() 
	{
		
	}
	
	@Override
	public void reduce(LongWritable key, Iterator<Writable> valueList, OutputCollector<Writable, Writable> out, Reporter reporter)
		throws IOException 
	{
		//cache collector/reporter (for write in close)
		_out = out;
		_report = reporter;
		
		//collect input partition
		if( _info == OutputInfo.BinaryBlockOutputInfo )
			_partition = collectBinaryBlock( valueList );
		else
			_partition = collectBinaryCellInput( valueList );
			
		//update in-memory matrix partition
		MatrixObject mo = (MatrixObject)_ec.getVariable( _inputVar );
		mo.setInMemoryPartition( _partition );
		
		//execute program
		LOG.trace("execute RemoteDPParWorkerReducer "+_stringID+" ("+_workerID+")");
		try {
			//create tasks for input data
			Task lTask = new Task(TaskType.SET);
			lTask.addIteration( new IntObject(_iterVar,key.get()) );
			
			//execute program
			executeTask( lTask );
		}
		catch(Exception ex)
		{
			throw new IOException("ParFOR: Failed to execute task.",ex);
		}
		
		//statistic maintenance (after final export)
		RemoteParForUtils.incrementParForMRCounters(_report, 1, 1);
	}

	/**
	 * 
	 */
	@Override
	public void configure(JobConf job)
	{
		//Step 1: configure data partitioning information
		_rlen = (int)MRJobConfiguration.getPartitioningNumRows( job );
		_clen = (int)MRJobConfiguration.getPartitioningNumCols( job );
		_brlen = MRJobConfiguration.getPartitioningBlockNumRows( job );
		_bclen = MRJobConfiguration.getPartitioningBlockNumCols( job );
		_iterVar = MRJobConfiguration.getPartitioningItervar( job );
		_dpf = MRJobConfiguration.getPartitioningFormat( job );		
		switch( _dpf ) { //create matrix partition for reuse
			case ROW_WISE:    _rlen = 1; break;
			case COLUMN_WISE: _clen = 1; break;
			default:  throw new RuntimeException("Partition format not yet supported in fused partition-execute: "+_dpf);
		}
		_info = MRJobConfiguration.getPartitioningOutputInfo( job );
		_tSparseCol = MRJobConfiguration.getPartitioningTransposedCol( job ); 
		if( _tSparseCol )
			_partition = new MatrixBlock((int)_clen, _rlen, true);
		else
			_partition = new MatrixBlock((int)_rlen, _clen, false);

		//Step 1: configure parworker
		String taskID = job.get("mapred.tip.id");		
		LOG.trace("configure RemoteDPParWorkerReducer "+taskID);
			
		try
		{
			_stringID = taskID;
			_workerID = IDHandler.extractIntID(_stringID); //int task ID
			
			//create local runtime program
			String in = MRJobConfiguration.getProgramBlocks(job);
			ParForBody body = ProgramConverter.parseParForBody(in, (int)_workerID);
			_childBlocks = body.getChildBlocks();
			_ec          = body.getEc();				
			_resultVars  = body.getResultVarNames();
	
			//init local cache manager 
			if( !CacheableData.isCachingActive() ) 
			{
				String uuid = IDHandler.createDistributedUniqueID();
				LocalFileUtils.createWorkingDirectoryWithUUID( uuid );
				CacheableData.initCaching( uuid ); //incl activation, cache dir creation (each map task gets its own dir for simplified cleanup)
			}
			
			if( !CacheableData.cacheEvictionLocalFilePrefix.contains("_") ) //account for local mode
			{
				CacheableData.cacheEvictionLocalFilePrefix = CacheableData.cacheEvictionLocalFilePrefix +"_" + _workerID; 
				CacheableData.cacheEvictionHDFSFilePrefix = CacheableData.cacheEvictionHDFSFilePrefix +"_" + _workerID;
			}
			
			//ensure that resultvar files are not removed
			super.pinResultVariables();
		
			//enable/disable caching (if required)
			boolean cpCaching = MRJobConfiguration.getParforCachingConfig( job );
			if( !cpCaching )
				CacheableData.disableCaching();
			
			_numTasks    = 0;
			_numIters    = 0;			
		}
		catch(Exception ex)
		{
			throw new RuntimeException(ex);
		}
		
		//disable parfor stat monitoring, reporting execution times via counters not useful 
		StatisticMonitor.disableStatMonitoring();
		
		//always reset stats because counters per map task (for case of JVM reuse)
		if( DMLScript.STATISTICS && !InfrastructureAnalyzer.isLocalMode() )
		{
			CacheStatistics.reset();
			Statistics.reset();
		}
		
		//get partitioned matrix varname (this needs to happen after program parsing) 
		_inputVar = getPartionedMatrixName();	
	}
	
	/**
	 * 
	 */
	@Override
	public void close() 
	    throws IOException 
	{
		try
		{
			//write output if required (matrix indexed write)
			RemoteParForUtils.exportResultVariables( _workerID, _ec.getVariables(), _resultVars, _out );
		
			//statistic maintenance (after final export)
			RemoteParForUtils.incrementParForMRCounters(_report, 0, 0);
			
			//print heaver hitter per task
			if( DMLScript.STATISTICS && !InfrastructureAnalyzer.isLocalMode() )
				LOG.info("\nSystemML Statistics:\nHeavy hitter instructions (name, time, count):\n" + Statistics.getHeavyHitters(10));		
		}
		catch(Exception ex)
		{
			throw new IOException( ex );
		}
		
		//cleanup cache and local tmp dir
		RemoteParForUtils.cleanupWorkingDirectories();
		
		//ensure caching is not disabled for CP in local mode
		CacheableData.enableCaching();
	}

	/**
	 * 
	 * @return
	 * @throws DMLRuntimeException
	 * @throws IOException
	 */
	private String getPartionedMatrixName() 
		throws RuntimeException
	{
		for( String var : _ec.getVariables().keySet() )
		{
			Data dat = _ec.getVariable( var );
			if ( dat instanceof MatrixObject && ((MatrixObject)dat).isPartitioned() ) 
				return var;
		}
		
		throw new RuntimeException("Unable to find partioned matrix object in symboltable.");
	}
	
	/**
	 * Collects a matrixblock partition from a given input iterator over 
	 * binary blocks.
	 * 
	 * Note it reuses the instance attribute _partition - multiple calls
	 * will overwrite the result.
	 * 
	 * @param valueList
	 * @return
	 * @throws IOException 
	 */
	private MatrixBlock collectBinaryBlock( Iterator<Writable> valueList ) 
		throws IOException 
	{
		try
		{
			//reset reuse block, keep configured representation
			_partition.reset(_rlen, _clen);	

			while( valueList.hasNext() )
			{
				PairWritableBlock pairValue = (PairWritableBlock)valueList.next();
				int row_offset = (int)(pairValue.indexes.getRowIndex()-1)*_brlen;
				int col_offset = (int)(pairValue.indexes.getColumnIndex()-1)*_bclen;
				MatrixBlock block = pairValue.block;
				if( !_partition.isInSparseFormat() ) //DENSE
				{
					_partition.copy( row_offset, row_offset+block.getNumRows()-1, 
							   col_offset, col_offset+block.getNumColumns()-1,
							   pairValue.block, false ); 
				}
				else //SPARSE 
				{
					_partition.appendToSparse(pairValue.block, row_offset, col_offset);
				}
			}

			//final partition cleanup
			cleanupCollectedMatrixPartition( _partition.isInSparseFormat() );
		}
		catch(DMLRuntimeException ex)
		{
			throw new IOException(ex);
		}
		
		return _partition;
	}
	
	
	/**
	 * Collects a matrixblock partition from a given input iterator over 
	 * binary cells.
	 * 
	 * Note it reuses the instance attribute _partition - multiple calls
	 * will overwrite the result.
	 * 
	 * @param valueList
	 * @return
	 * @throws IOException 
	 */
	private MatrixBlock collectBinaryCellInput( Iterator<Writable> valueList ) 
		throws IOException 
	{
		//reset reuse block, keep configured representation
		if( _tSparseCol )
			_partition.reset(_clen, _rlen);	
		else
			_partition.reset(_rlen, _clen);
		
		switch( _dpf )
		{
			case ROW_WISE:
				while( valueList.hasNext() )
				{
					PairWritableCell pairValue = (PairWritableCell)valueList.next();
					if( pairValue.indexes.getColumnIndex()<0 )
						continue; //cells used to ensure empty partitions
					_partition.quickSetValue(0, (int)pairValue.indexes.getColumnIndex()-1, pairValue.cell.getValue());
				}
				break;
			case COLUMN_WISE:
				while( valueList.hasNext() )
				{
					PairWritableCell pairValue = (PairWritableCell)valueList.next();
					if( pairValue.indexes.getRowIndex()<0 )
						continue; //cells used to ensure empty partitions
					if( _tSparseCol )
						_partition.appendValue(0,(int)pairValue.indexes.getRowIndex()-1, pairValue.cell.getValue());
					else
						_partition.quickSetValue((int)pairValue.indexes.getRowIndex()-1, 0, pairValue.cell.getValue());
				}
				break;
			default: 
				throw new IOException("Partition format not yet supported in fused partition-execute: "+_dpf);
		}
		
		//final partition cleanup
		cleanupCollectedMatrixPartition(_tSparseCol);
		
		return _partition;
	}
	
	/**
	 * 
	 * @param sort
	 * @throws IOException
	 */
	private void cleanupCollectedMatrixPartition(boolean sort) 
		throws IOException
	{
		//sort sparse row contents if required
		if( _partition.isInSparseFormat() && sort )
			_partition.sortSparseRows();

		//ensure right number of nnz
		if( !_partition.isInSparseFormat() )
			_partition.recomputeNonZeros();
			
		//exam and switch dense/sparse representation
		try {
			_partition.examSparsity();
		}
		catch(Exception ex){
			throw new IOException(ex);
		}
	}
}
