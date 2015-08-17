/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.context;

import java.util.LinkedList;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.storage.StorageLevel;

import scala.Tuple2;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.MLContext;
import com.ibm.bi.dml.api.MLContextProxy;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.instructions.spark.SPInstruction;
import com.ibm.bi.dml.runtime.instructions.spark.data.BroadcastObject;
import com.ibm.bi.dml.runtime.instructions.spark.data.LineageObject;
import com.ibm.bi.dml.runtime.instructions.spark.data.PartitionedMatrixBlock;
import com.ibm.bi.dml.runtime.instructions.spark.data.RDDObject;
import com.ibm.bi.dml.runtime.instructions.spark.functions.CopyBinaryCellFunction;
import com.ibm.bi.dml.runtime.instructions.spark.functions.CopyBlockFunction;
import com.ibm.bi.dml.runtime.instructions.spark.functions.CopyTextInputFunction;
import com.ibm.bi.dml.runtime.instructions.spark.utils.SparkUtils;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.util.MapReduceTool;


public class SparkExecutionContext extends ExecutionContext
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private static final Log LOG = LogFactory.getLog(SparkExecutionContext.class.getName());

	private static boolean ASYNCHRONOUS_VAR_DESTROY = true;
	
	//executor memory and relative fractions as obtained from the spark configuration
	private static long _memExecutors = -1; //mem per executors
	private static double _memRatioData = -1; 
	private static double _memRatioShuffle = -1;
	private static int _numExecutors = -1; //total executors
	private static int _defaultPar = -1; //total vcores  
	
	// TODO: This needs to be debugged further. For now getting around the problem with Singleton
	// Only one SparkContext may be active per JVM. You must stop() the active SparkContext before creating a new one. 
	// This limitation may eventually be removed; see SPARK-2243 for more details.
	private static JavaSparkContext _singletonSpctx = null;
	
	private JavaSparkContext _spctx = null; 
	
	protected SparkExecutionContext(Program prog) 
	{
		//protected constructor to force use of ExecutionContextFactory
		this( true, prog );
	}

	protected SparkExecutionContext(boolean allocateVars, Program prog) 
	{
		//protected constructor to force use of ExecutionContextFactory
		super( allocateVars, prog );
		
		synchronized(SparkExecutionContext.class) 
		{
			if(_singletonSpctx != null) {
				// Reuse the context
				_spctx = _singletonSpctx;
			}
			else {
				//create a default spark context (master, appname, etc refer to system properties
				//as given in the spark configuration or during spark-submit)
				MLContext mlCtx = MLContextProxy.getActiveMLContext();
				if(mlCtx != null) {
					// This is when DML is called through spark shell
					// Will clean the passing of static variables later as this involves minimal change to DMLScript
					_spctx = new JavaSparkContext(mlCtx.getSparkContext());
				}
				else {
					if(DMLScript.USE_LOCAL_SPARK_CONFIG) {
						// For now set 4 cores for integration testing :)
						SparkConf conf = new SparkConf().setMaster("local[*]").setAppName("My local integration test app");
						// This is discouraged in spark but have added only for those testcase that cannot stop the context properly
						// conf.set("spark.driver.allowMultipleContexts", "true");
						conf.set("spark.ui.enabled", "false");
						// conf.set("spark.ui.port", "69389"); // some random port
						_spctx = new JavaSparkContext(conf);
					}
					else {
						_spctx = new JavaSparkContext();
					}
				}
				
				//globally add binaryblock serialization framework for all hdfs read/write operations
				//TODO if spark context passed in from outside (mlcontext), we need to clean this up at the end 
				if( MRJobConfiguration.USE_BINARYBLOCK_SERIALIZATION )
					MRJobConfiguration.addBinaryBlockSerializationFramework( _spctx.hadoopConfiguration() );
				
				_singletonSpctx = _spctx;
			}
		}
	}

	public void close() {
		synchronized(SparkExecutionContext.class) {
			_spctx.stop();
			_singletonSpctx = null;
		}
	}
	
	public JavaSparkContext getSparkContext()
	{
		return _spctx;
	}
	
	
	/**
	 * Spark instructions should call this for all matrix inputs except broadcast
	 * variables.
	 * 
	 * @param varname
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	@SuppressWarnings("unchecked")
	public JavaPairRDD<MatrixIndexes,MatrixBlock> getBinaryBlockRDDHandleForVariable( String varname ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		return (JavaPairRDD<MatrixIndexes,MatrixBlock>) getRDDHandleForVariable( varname, InputInfo.BinaryBlockInputInfo);
	}
	
	/**
	 * 
	 * @param varname
	 * @param inputInfo
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	public JavaPairRDD<?,?> getRDDHandleForVariable( String varname, InputInfo inputInfo ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		MatrixObject mo = getMatrixObject(varname);
		return getRDDHandleForMatrixObject(mo, inputInfo);
	}
	
	/**
	 * This call returns an RDD handle for a given matrix object. This includes 
	 * the creation of RDDs for in-memory or binary-block HDFS data. 
	 * 
	 * 
	 * @param varname
	 * @return
	 * @throws DMLRuntimeException 
	 * @throws DMLUnsupportedOperationException 
	 */
	@SuppressWarnings("unchecked")
	public JavaPairRDD<?,?> getRDDHandleForMatrixObject( MatrixObject mo, InputInfo inputInfo ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		
		//NOTE: MB this logic should be integrated into MatrixObject
		//However, for now we cannot assume that spark libraries are 
		//always available and hence only store generic references in 
		//matrix object while all the logic is in the SparkExecContext
		
		JavaPairRDD<?,?> rdd = null;
		//CASE 1: rdd already existing 
		if( mo.getRDDHandle()!=null )
		{
			// TODO: Currently unchecked handling as it ignores inputInfo
			// This is ok since this method is only supposed to be called only by Reblock which performs appropriate casting
			rdd = mo.getRDDHandle().getRDD();
		}
		//CASE 2: dirty in memory data
		else if( mo.isDirty() )
		{
			//get in-memory matrix block and parallelize it
			MatrixBlock mb = mo.acquireRead(); //pin matrix in memory
			rdd = toJavaPairRDD(_spctx, mb, (int)mo.getNumRowsPerBlock(), (int)mo.getNumColumnsPerBlock());
			mo.release(); //unpin matrix
			
			//keep rdd handle for future operations on it
			RDDObject rddhandle = new RDDObject(rdd, mo.getVarName());
			mo.setRDDHandle(rddhandle);
		}
		//CASE 3: non-dirty (file exists on HDFS)
		else
		{
			// parallelize hdfs-resident file
			// For binary block, these are: SequenceFileInputFormat.class, MatrixIndexes.class, MatrixBlock.class
			if(inputInfo == InputInfo.BinaryBlockInputInfo) {
				rdd = _spctx.hadoopFile( mo.getFileName(), inputInfo.inputFormatClass, inputInfo.inputKeyClass, inputInfo.inputValueClass);
				//note: this copy is still required in Spark 1.4 because spark hands out whatever the inputformat
				//recordreader returns; the javadoc explicitly recommend to copy all key/value pairs
				rdd = ((JavaPairRDD<MatrixIndexes, MatrixBlock>)rdd).mapToPair( new CopyBlockFunction() ); //cp is workaround for read bug
			}
			else if(inputInfo == InputInfo.TextCellInputInfo || inputInfo == InputInfo.CSVInputInfo || inputInfo == InputInfo.MatrixMarketInputInfo) {
				rdd = _spctx.hadoopFile( mo.getFileName(), inputInfo.inputFormatClass, inputInfo.inputKeyClass, inputInfo.inputValueClass);
				rdd = ((JavaPairRDD<LongWritable, Text>)rdd).mapToPair( new CopyTextInputFunction() ); //cp is workaround for read bug
			}
			else if(inputInfo == InputInfo.BinaryCellInputInfo) {
				rdd = _spctx.hadoopFile( mo.getFileName(), inputInfo.inputFormatClass, inputInfo.inputKeyClass, inputInfo.inputValueClass);
				rdd = ((JavaPairRDD<MatrixIndexes, MatrixCell>)rdd).mapToPair( new CopyBinaryCellFunction() ); //cp is workaround for read bug
			}
			else {
				throw new DMLRuntimeException("Incorrect input format in getRDDHandleForVariable");
			}
			
			//keep rdd handle for future operations on it
			RDDObject rddhandle = new RDDObject(rdd, mo.getVarName());
			rddhandle.setHDFSFile(true);
			mo.setRDDHandle(rddhandle);
		}
		
		return rdd;
	}
	
	/**
	 * TODO So far we only create broadcast variables but never destroy
	 * them. This is a memory leak which might lead to executor out-of-memory.
	 * However, in order to handle this, we need to keep track when broadcast 
	 * variables are no longer required.
	 * 
	 * @param varname
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	public Broadcast<PartitionedMatrixBlock> getBroadcastForVariable( String varname ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		MatrixObject mo = getMatrixObject(varname);
		
		Broadcast<PartitionedMatrixBlock> bret = null;
		if(    mo.getBroadcastHandle()!=null 
			&& mo.getBroadcastHandle().getBroadcast().isValid() ) 
		{
			//reuse existing broadcast handle
			bret = mo.getBroadcastHandle().getBroadcast();
		}
		else 
		{
			int brlen = (int) mo.getNumRowsPerBlock();
			int bclen = (int) mo.getNumColumnsPerBlock();
			
			//read data into memory (no matter where it comes from)
			MatrixBlock mb = mo.acquireRead();
			PartitionedMatrixBlock pmb = new PartitionedMatrixBlock(mb, brlen, bclen);
			bret = _spctx.broadcast(pmb);
			BroadcastObject bchandle = new BroadcastObject(bret, varname);
			mo.setBroadcastHandle(bchandle);
			mo.release();
		}
		
		return bret;
	}
	
	/**
	 * Keep the output rdd of spark rdd operations as meta data of matrix objects in the 
	 * symbol table.
	 * 
	 * Spark instructions should call this for all matrix outputs.
	 * 
	 * 
	 * @param varname
	 * @param rdd
	 * @throws DMLRuntimeException 
	 */
	public void setRDDHandleForVariable(String varname, JavaPairRDD<MatrixIndexes,?> rdd) 
		throws DMLRuntimeException
	{
		MatrixObject mo = getMatrixObject(varname);
		RDDObject rddhandle = new RDDObject(rdd, varname);
		mo.setRDDHandle( rddhandle );
	}
	
	/**
	 * Utility method for creating an RDD out of an in-memory matrix block.
	 * 
	 * @param sc
	 * @param block
	 * @return
	 * @throws DMLUnsupportedOperationException 
	 * @throws DMLRuntimeException 
	 */
	public static JavaPairRDD<MatrixIndexes,MatrixBlock> toJavaPairRDD(JavaSparkContext sc, MatrixBlock src, int brlen, int bclen) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		LinkedList<Tuple2<MatrixIndexes,MatrixBlock>> list = new LinkedList<Tuple2<MatrixIndexes,MatrixBlock>>();
		
		if(    src.getNumRows() <= brlen 
		    && src.getNumColumns() <= bclen )
		{
			list.addLast(new Tuple2<MatrixIndexes,MatrixBlock>(new MatrixIndexes(1,1), src));
		}
		else
		{
			boolean sparse = src.isInSparseFormat();
			
			//create and write subblocks of matrix
			for(int blockRow = 0; blockRow < (int)Math.ceil(src.getNumRows()/(double)brlen); blockRow++)
				for(int blockCol = 0; blockCol < (int)Math.ceil(src.getNumColumns()/(double)bclen); blockCol++)
				{
					int maxRow = (blockRow*brlen + brlen < src.getNumRows()) ? brlen : src.getNumRows() - blockRow*brlen;
					int maxCol = (blockCol*bclen + bclen < src.getNumColumns()) ? bclen : src.getNumColumns() - blockCol*bclen;
					
					MatrixBlock block = new MatrixBlock(maxRow, maxCol, sparse);
						
					int row_offset = blockRow*brlen;
					int col_offset = blockCol*bclen;
	
					//copy submatrix to block
					src.sliceOperations( row_offset+1, row_offset+maxRow, 
							             col_offset+1, col_offset+maxCol, 
							             block );							
					
					//append block to sequence file
					MatrixIndexes indexes = new MatrixIndexes(blockRow+1, blockCol+1);
					list.addLast(new Tuple2<MatrixIndexes,MatrixBlock>(indexes, block));
				}
		}
		
		return sc.parallelizePairs(list);
	}
	
	/**
	 * This method is a generic abstraction for calls from the buffer pool.
	 * See toMatrixBlock(JavaPairRDD<MatrixIndexes,MatrixBlock> rdd, int numRows, int numCols);
	 * 
	 * @param rdd
	 * @param numRows
	 * @param numCols
	 * @return
	 * @throws DMLRuntimeException 
	 */
	@SuppressWarnings("unchecked")
	public static MatrixBlock toMatrixBlock(Object rdd, int rlen, int clen, int brlen, int bclen) 
		throws DMLRuntimeException
	{
		RDDObject rddo = (RDDObject)rdd;		
		JavaPairRDD<MatrixIndexes,MatrixBlock> lrdd = (JavaPairRDD<MatrixIndexes, MatrixBlock>) rddo.getRDD();
		
		return toMatrixBlock(lrdd, rlen, clen, brlen, bclen);
	}
	
	/**
	 * Utility method for creating a single matrix block out of an RDD. Note that this collect call
	 * might trigger execution of any pending transformation. 
	 * 
	 * TODO add a more efficient path for sparse matrices, see BinaryBlockReader.
	 * 
	 * @param rdd
	 * @param numRows
	 * @param numCols
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static MatrixBlock toMatrixBlock(JavaPairRDD<MatrixIndexes,MatrixBlock> rdd, int rlen, int clen, int brlen, int bclen) 
		throws DMLRuntimeException
	{
		MatrixBlock out = null;
		
		if( rlen <= brlen && clen <= bclen ) //SINGLE BLOCK
		{
			//special case without copy and nnz maintenance
			List<Tuple2<MatrixIndexes,MatrixBlock>> list = rdd.collect();
			if( list.size()>1 )
				throw new DMLRuntimeException("Expecting no more than one result block.");
			else if( list.size()==1 )
				out = list.get(0)._2();
		}
		else //MULTIPLE BLOCKS
		{
			//current assumption always dense
			out = new MatrixBlock(rlen, clen, false);
			List<Tuple2<MatrixIndexes,MatrixBlock>> list = rdd.collect();
			
			for( Tuple2<MatrixIndexes,MatrixBlock> keyval : list )
			{
				MatrixIndexes ix = keyval._1();
				MatrixBlock block = keyval._2();
				
				int row_offset = (int)(ix.getRowIndex()-1)*brlen;
				int col_offset = (int)(ix.getColumnIndex()-1)*bclen;
				int rows = block.getNumRows();
				int cols = block.getNumColumns();
				
				out.copy( row_offset, row_offset+rows-1, 
						  col_offset, col_offset+cols-1,
						  block, false );			
			}
			
			out.recomputeNonZeros();
			out.examSparsity();
		}
		
		return out;
	}
	
	/**
	 * 
	 * @param rdd
	 * @param oinfo
	 */
	@SuppressWarnings("unchecked")
	public static void writeRDDtoHDFS( Object rdd, String path, OutputInfo oinfo )
	{
		RDDObject rddo = (RDDObject) rdd;
		JavaPairRDD<MatrixIndexes,MatrixBlock> lrdd = (JavaPairRDD<MatrixIndexes, MatrixBlock>) rddo.getRDD();
		
		lrdd.saveAsHadoopFile(path, 
				oinfo.outputKeyClass, 
				oinfo.outputValueClass, 
				oinfo.outputFormatClass);
	}
	
	/**
	 * Returns the available memory budget for broadcast variables in bytes.
	 * In detail, this takes into account the total executor memory as well
	 * as relative ratios for data and shuffle. Note, that this is a conservative
	 * estimate since both data memory and shuffle memory might not be fully
	 * utilized. 
	 * 
	 * @return
	 */
	public static double getBroadcastMemoryBudget()
	{
		if( _memExecutors < 0 || _memRatioData < 0 || _memRatioShuffle < 0 )
			analyzeSparkConfiguation();
		
		//70% of remaining free memory
		double membudget = OptimizerUtils.MEM_UTIL_FACTOR *
			              (  _memExecutors 
			               - _memExecutors*(_memRatioData+_memRatioShuffle) );
		
		return membudget;
	}
	
	/**
	 * 
	 * @return
	 */
	public static double getConfiguredTotalDataMemory()
	{
		if( _memExecutors < 0 || _memRatioData < 0 )
			analyzeSparkConfiguation();
		
		return ( _memExecutors * _memRatioData * _numExecutors );
	}
	
	public static int getNumExecutors()
	{
		if( _numExecutors < 0 )
			analyzeSparkConfiguation();
		
		return _numExecutors;
	}
	
	public static int getDefaultParallelism()
	{
		if( _defaultPar < 0 )
			analyzeSparkConfiguation();
		
		return _defaultPar;
	}
	
	/**
	 * 
	 */
	public static void analyzeSparkConfiguation() 
	{
		SparkConf sconf = new SparkConf();
		
		//parse absolute executor memory
		String tmp = sconf.get("spark.executor.memory", "512m");
		if ( tmp.endsWith("g") || tmp.endsWith("G") )
			_memExecutors = Long.parseLong(tmp.substring(0,tmp.length()-1)) * 1024 * 1024 * 1024;
		else if ( tmp.endsWith("m") || tmp.endsWith("M") )
			_memExecutors = Long.parseLong(tmp.substring(0,tmp.length()-1)) * 1024 * 1024;
		else if( tmp.endsWith("k") || tmp.endsWith("K") )
			_memExecutors = Long.parseLong(tmp.substring(0,tmp.length()-1)) * 1024;
		else 
			_memExecutors = Long.parseLong(tmp.substring(0,tmp.length()-2));
		
		//get data and shuffle memory ratios (defaults not specified in job conf)
		_memRatioData = sconf.getDouble("spark.storage.memoryFraction", 0.6); //default 60%
		_memRatioShuffle = sconf.getDouble("spark.shuffle.memoryFraction", 0.2); //default 20%
		
		//get default parallelism (total number of executors and cores)
		//note: spark context provides this information while conf does not
		//(for num executors we need to correct for driver and local mode)
		SparkExecutionContext sec = new SparkExecutionContext(false, null);
		_numExecutors = Math.max(sec._spctx.sc().getExecutorMemoryStatus().size() - 1, 1);  
		_defaultPar = sec._spctx.defaultParallelism(); 

		//note: required time for infrastructure analysis on 5 node cluster: ~5-20ms. 
	}

	/**
	 * 
	 */
	public void checkAndRaiseValidationWarningJDKVersion()
	{
		//get the jre version 
		String version = System.getProperty("java.version");
		
		//parse jre version
		int ix1 = version.indexOf('.');
		int ix2 = version.indexOf('.', ix1+1);
		int versionp1 = Integer.parseInt(version.substring(0, ix1));
		int versionp2 = Integer.parseInt(version.substring(ix1+1, ix2));
		
		//check multi-threaded executors
		int numExecutors = getNumExecutors();
		int numCores = getDefaultParallelism();
		boolean multiThreaded = (numCores > numExecutors);
		
		//check for jdk version less than 8 (and raise warning if multi-threaded)
		if( versionp1 == 1 && versionp2 < 8 && multiThreaded) 
		{
			LOG.warn("########################################################################################");
			LOG.warn("### WARNING: Multi-threaded text reblock may lead to thread contention on JRE < 1.8 ####");
			LOG.warn("### java.version = " + version);
			LOG.warn("### total number of executors = " + numExecutors);
			LOG.warn("### total number of cores = " + numCores);
			LOG.warn("### JDK-7032154: Performance tuning of sun.misc.FloatingDecimal/FormattedFloatingDecimal");
			LOG.warn("### Workaround: Convert text to binary w/ changed configuration of one executor per core");
			LOG.warn("########################################################################################");
		}
	}
	
	///////////////////////////////////////////
	// Cleanup of RDDs and Broadcast variables
	///////
	
	/**
	 * Adds a child rdd object to the lineage of a parent rdd.
	 * 
	 * @param varParent
	 * @param varChild
	 * @throws DMLRuntimeException
	 */
	public void addLineageRDD(String varParent, String varChild) 
		throws DMLRuntimeException 
	{
		RDDObject parent = getMatrixObject(varParent).getRDDHandle();
		RDDObject child = getMatrixObject(varChild).getRDDHandle();
		
		parent.addLineageChild( child );
	}
	
	/**
	 * Adds a child broadcast object to the lineage of a parent rdd.
	 * 
	 * @param varParent
	 * @param varChild
	 * @throws DMLRuntimeException
	 */
	public void addLineageBroadcast(String varParent, String varChild) 
		throws DMLRuntimeException 
	{
		RDDObject parent = getMatrixObject(varParent).getRDDHandle();
		BroadcastObject child = getMatrixObject(varChild).getBroadcastHandle();
		
		parent.addLineageChild( child );
	}
	
	@Override
	public void cleanupMatrixObject( MatrixObject mo ) 
		throws DMLRuntimeException
	{
		//NOTE: this method overwrites the default behavior of cleanupMatrixObject
		//and hence is transparently used by rmvar instructions and other users. The
		//core difference is the lineage-based cleanup of RDD and broadcast variables.
		
		try
		{
			if ( mo.isCleanupEnabled() ) 
			{
				//compute ref count only if matrix cleanup actually necessary
				if ( !getVariables().hasReferences(mo) ) 
				{
					//clean cached data	
					mo.clearData(); 
					
					//clean hdfs data
					if( mo.isFileExists() ) {
						String fpath = mo.getFileName();
						if (fpath != null) {
							MapReduceTool.deleteFileIfExistOnHDFS(fpath);
							MapReduceTool.deleteFileIfExistOnHDFS(fpath + ".mtd");
						}
					}
					
					//cleanup RDD and broadcast variables (recursive)
					//note: requires that mo.clearData already removed back references
					if( mo.getRDDHandle()!=null ) { 
 						rCleanupLineageObject(mo.getRDDHandle());
					}	
					if( mo.getBroadcastHandle()!=null ) {
						rCleanupLineageObject(mo.getBroadcastHandle());
					}
				}
			}
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException(ex);
		}
	}
	
	private void rCleanupLineageObject(LineageObject lob)
	{		
		//abort recursive cleanup if still consumers
		if( lob.getNumReferences() > 0 )
			return;
			
		//abort if still reachable through matrix object (via back references for 
		//robustness in function calls and to prevent repeated scans of the symbol table)
		if( lob.hasBackReference() )
			return;
		
		//cleanup current lineage object (from driver/executors)
		if( lob instanceof RDDObject )
			cleanupRDDVariable(((RDDObject)lob).getRDD());
		else if( lob instanceof BroadcastObject )
			cleanupBroadcastVariable(((BroadcastObject)lob).getBroadcast());
	
		//recursively process lineage children
		for( LineageObject c : lob.getLineageChilds() ){
			c.decrementNumReferences();
			rCleanupLineageObject(c);
		}
	}
	
	/**
	 * This call destroys a broadcast variable at all executors and the driver.
	 * Hence, it is intended to be used on rmvar only. Depending on the
	 * ASYNCHRONOUS_VAR_DESTROY configuration, this is asynchronous or not.
	 * 
	 * 
	 * @param inV
	 */
	public void cleanupBroadcastVariable(Broadcast<?> bvar) 
	{
		//in comparison to 'unpersist' (which would only delete the broadcast from the executors),
		//this call also deletes related data from the driver.
		if( bvar.isValid() ) {
			bvar.destroy( ASYNCHRONOUS_VAR_DESTROY );
		}
	}
	
	/**
	 * This call removes an rdd variable from executor memory and disk if required.
	 * Hence, it is intended to be used on rmvar only. Depending on the
	 * ASYNCHRONOUS_VAR_DESTROY configuration, this is asynchronous or not.
	 * 
	 * @param rvar
	 */
	public void cleanupRDDVariable(JavaPairRDD<?,?> rvar) 
	{
		if( rvar.getStorageLevel()!=StorageLevel.NONE() ) {
			rvar.unpersist( ASYNCHRONOUS_VAR_DESTROY );
		}
	}
	
	///////////////////////////////////////////
	// Debug String Handling (see explain); TODO to be removed
	///////

	/**
	 * 
	 * @param inst
	 * @param outputVarName
	 * @throws DMLRuntimeException
	 */
	public void setDebugString(SPInstruction inst, String outputVarName) 
		throws DMLRuntimeException 
	{
		RDDObject parentLineage = getMatrixObject(outputVarName).getRDDHandle();
		
		if( parentLineage == null || parentLineage.getRDD() == null )
			return;
		
		MLContextProxy.addRDDForInstructionForMonitoring(inst, parentLineage.getRDD().id());
		
		JavaPairRDD<?, ?> out = parentLineage.getRDD();
		JavaPairRDD<?, ?> in1 = null; 
		JavaPairRDD<?, ?> in2 = null;
		String input1VarName = null; 
		String input2VarName = null;
		if(parentLineage.getLineageChilds() != null) {
			for(LineageObject child : parentLineage.getLineageChilds()) {
				if(child instanceof RDDObject) {
					if(in1 == null) {
						in1 = ((RDDObject) child).getRDD();
						input1VarName = child.getVarName();
					}
					else if(in2 == null) {
						in2 = ((RDDObject) child).getRDD();
						input2VarName = child.getVarName();
					}
					else {
						throw new DMLRuntimeException("PRINT_EXPLAIN_WITH_LINEAGE not yet supported for three outputs");
					}
				}
			}
		}
		setLineageInfoForExplain(inst, out, in1, input1VarName, in2, input2VarName);
	}
	
	// The most expensive operation here is rdd.toDebugString() which can be a major hit because
	// of unrolling lazy evaluation of Spark. Hence, it is guarded against it along with flag 'PRINT_EXPLAIN_WITH_LINEAGE' which is 
	// enabled only through MLContext. This way, it doesnot affect our performance evaluation through non-MLContext path
	private void setLineageInfoForExplain(SPInstruction inst, 
			JavaPairRDD<?, ?> out, 
			JavaPairRDD<?, ?> in1, String in1Name, 
			JavaPairRDD<?, ?> in2, String in2Name) throws DMLRuntimeException {
		
			
		// RDDInfo outInfo = org.apache.spark.storage.RDDInfo.fromRdd(out.rdd());
		
		// First fetch start lines from input RDDs
		String startLine1 = null; 
		String startLine2 = null;
		int i1length = 0, i2length = 0;
		if(in1 != null) {
			String [] lines = in1.toDebugString().split("\\r?\\n");
			startLine1 = SparkUtils.getStartLineFromSparkDebugInfo(lines[0]); // lines[0].substring(4, lines[0].length());
			i1length = lines.length;
		}
		if(in2 != null) {
			String [] lines = in2.toDebugString().split("\\r?\\n");
			startLine2 =  SparkUtils.getStartLineFromSparkDebugInfo(lines[0]); // lines[0].substring(4, lines[0].length());
			i2length = lines.length;
		}
		
		String outDebugString = "";
		int skip = 0;
		
		// Now process output RDD and replace inputRDD debug string by the matrix variable name
		String [] outLines = out.toDebugString().split("\\r?\\n");
		for(int i = 0; i < outLines.length; i++) {
			if(skip > 0) {
				skip--;
				// outDebugString += "\nSKIP:" + outLines[i];
			}
			else if(startLine1 != null && outLines[i].contains(startLine1)) {
				String prefix = SparkUtils.getPrefixFromSparkDebugInfo(outLines[i]); // outLines[i].substring(0, outLines[i].length() - startLine1.length());
				outDebugString += "\n" + prefix + "[[" + in1Name + "]]";
				//outDebugString += "\n{" + prefix + "}[[" + in1Name + "]] => " + outLines[i];
				skip = i1length - 1;  
			}
			else if(startLine2 != null && outLines[i].contains(startLine2)) {
				String prefix = SparkUtils.getPrefixFromSparkDebugInfo(outLines[i]); // outLines[i].substring(0, outLines[i].length() - startLine2.length());
				outDebugString += "\n" + prefix + "[[" + in2Name + "]]";
				skip = i2length - 1;
			}
			else {
				outDebugString += "\n" + outLines[i];
			}
		}
		
		MLContext mlContext = MLContextProxy.getActiveMLContext();
		if(mlContext != null && mlContext.getMonitoringUtil() != null) {
			mlContext.getMonitoringUtil().setLineageInfo(inst, outDebugString);
		}
		else {
			throw new DMLRuntimeException("The method setLineageInfoForExplain should be called only through MLContext");
		}
		
	}
	

}
