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

package org.apache.sysds.runtime.controlprogram.context;

import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.lang.annotation.Target;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.Collectors;
import java.util.stream.LongStream;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.storage.RDDInfo;
import org.apache.spark.storage.StorageLevel;
import org.apache.spark.util.LongAccumulator;
import org.apache.sysds.api.DMLException;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.api.mlcontext.MLContext;
import org.apache.sysds.api.mlcontext.MLContextUtil;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.Checkpoint;
import org.apache.sysds.resource.CloudUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.io.ReaderSparkCompressed;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.caching.TensorObject;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.TensorBlock;
import org.apache.sysds.runtime.data.TensorIndexes;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.spark.DeCompressionSPInstruction;
import org.apache.sysds.runtime.instructions.spark.data.BroadcastObject;
import org.apache.sysds.runtime.instructions.spark.data.LineageObject;
import org.apache.sysds.runtime.instructions.spark.data.PartitionedBlock;
import org.apache.sysds.runtime.instructions.spark.data.PartitionedBroadcast;
import org.apache.sysds.runtime.instructions.spark.data.RDDObject;
import org.apache.sysds.runtime.instructions.spark.functions.ComputeBinaryBlockNnzFunction;
import org.apache.sysds.runtime.instructions.spark.functions.CopyFrameBlockPairFunction;
import org.apache.sysds.runtime.instructions.spark.functions.CopyTextInputFunction;
import org.apache.sysds.runtime.instructions.spark.functions.CreateSparseBlockFunction;
import org.apache.sysds.runtime.instructions.spark.utils.FrameRDDConverterUtils.LongFrameToLongWritableFrameFunction;
import org.apache.sysds.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.apache.sysds.runtime.instructions.spark.utils.SparkUtils;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.io.InputOutputInfo;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixCell;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.meta.TensorCharacteristics;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.utils.MLContextProxy;
import org.apache.sysds.utils.Statistics;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;
import org.apache.sysds.utils.stats.SparkStatistics;

import scala.Tuple2;


public class SparkExecutionContext extends ExecutionContext
{

	//internal configurations
	private static final boolean LAZY_SPARKCTX_CREATION = true;
	private static final boolean ASYNCHRONOUS_VAR_DESTROY = true;
	public static final boolean FAIR_SCHEDULER_MODE = true;

	//executor memory and relative fractions as obtained from the spark configuration
	private static SparkClusterConfig _sconf = null;

	//singleton spark context (as there can be only one spark context per JVM)
	private static JavaSparkContext _spctx = null;

	//registry of parallelized RDDs to enforce that at any time, we spent at most
	//10% of JVM max heap size for parallelized RDDs; if this is not sufficient,
	//matrices or frames are exported to HDFS and the RDDs are created from files.
	//TODO unify memory management for CP, par RDDs, and potentially broadcasts
	private static final MemoryManagerParRDDs _parRDDs = new MemoryManagerParRDDs(0.1);
	
	//pool of reused fair scheduler pool names (unset bits indicate availability)
	private static boolean[] _poolBuff = FAIR_SCHEDULER_MODE ?
		new boolean[InfrastructureAnalyzer.getLocalParallelism()] : null;
	
	private static boolean localSparkWarning = false;

	protected SparkExecutionContext(boolean allocateVars, boolean allocateLineage, Program prog) {
		//protected constructor to force use of ExecutionContextFactory
		super( allocateVars, allocateLineage, prog );
		
		//spark context creation via internal initializer
		if( !LAZY_SPARKCTX_CREATION || DMLScript.getGlobalExecMode()==ExecMode.SPARK ) {
			initSparkContext();
		}
	}

	/**
	 * Returns the used singleton spark context. In case of lazy spark context
	 * creation, this methods blocks until the spark context is created.
	 *
	 * @return java spark context
	 */
	public JavaSparkContext getSparkContext() {
		//lazy spark context creation on demand (lazy instead of asynchronous
		//to avoid wait for uninitialized spark context on close)
		if( LAZY_SPARKCTX_CREATION ) {
			initSparkContext();
		}

		//return the created spark context
		return _spctx;
	}

	public static void initLocalSparkContext(SparkConf sparkConf) {
		// allows re-initialization
		_sconf = new SparkClusterConfig(sparkConf);
	}

	public synchronized static JavaSparkContext getSparkContextStatic() {
		initSparkContext();
		if(_spctx.sc().isStopped()){
			_spctx = null;
			initSparkContext();
		}
		return _spctx;
	}

	/**
	 * Indicates if the spark context has been created or has
	 * been passed in from outside.
	 *
	 * @return true if spark context created
	 */
	public synchronized static boolean isSparkContextCreated() {
		return (_spctx != null);
	}

	public static void resetSparkContextStatic() {
		_spctx = null;
	}

	public void close() {
		synchronized( SparkExecutionContext.class) {
			if(_spctx != null) {
				Logger spL = Logger.getLogger("org.apache.spark.network.client.TransportResponseHandler");
				spL.setLevel(Level.FATAL);
				OutputStream buff = new OutputStream() {
					@Override
					public void write(int b) {
						// do Nothing
					}
				};
				PrintStream old = System.err;
				System.setErr(new PrintStream(buff));
				// stop the spark context if existing
				_spctx.stop();
				// make sure stopped context is never used again
				_spctx = null;
				System.setErr(old);
				spL.setLevel(Level.ERROR);
			}
		}
	}

	public static boolean isLazySparkContextCreation(){
		return LAZY_SPARKCTX_CREATION;
	}

	public static void handleIllegalReflectiveAccessSpark(){
		Module pf = org.apache.spark.unsafe.Platform.class.getModule();
		Target.class.getModule().addOpens("java.nio", pf);
		Target.class.getModule().addOpens("java.io", pf);

		Module se = org.apache.spark.util.SizeEstimator.class.getModule();
		Target.class.getModule().addOpens("java.util", se);
		Target.class.getModule().addOpens("java.lang", se);
		Target.class.getModule().addOpens("java.lang.ref", se);
		Target.class.getModule().addOpens("java.util.concurrent", se);
	}

	public synchronized static void initSparkContext(){
		//check for redundant spark context init
		if( _spctx != null )
			return;
		handleIllegalReflectiveAccessSpark();

		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;

		//create a default spark context (master, appname, etc refer to system properties
		//as given in the spark configuration or during spark-submit)

		MLContext mlCtxObj = MLContextProxy.getActiveMLContext();
		if(mlCtxObj != null)
		{
			// This is when DML is called through spark shell
			// Will clean the passing of static variables later as this involves minimal change to DMLScript
			_spctx = MLContextUtil.getJavaSparkContext(mlCtxObj);
		}
		else
		{
			final SparkConf conf = createSystemDSSparkConf();
			if(DMLScript.USE_LOCAL_SPARK_CONFIG)
				setLocalConfSettings(conf);
			
			_spctx = createContext(conf);
			if(DMLScript.USE_LOCAL_SPARK_CONFIG)
				_spctx.setCheckpointDir("/tmp/systemds_spark_cache_" + DMLScript.getUUID());

			_parRDDs.clear();
		}

		// Set warning if spark.driver.maxResultSize is not set. It needs to be set before starting Spark Context for CP collect
		String strDriverMaxResSize = _spctx.getConf().get("spark.driver.maxResultSize", "1g");
		long driverMaxResSize = UtilFunctions.parseMemorySize(strDriverMaxResSize);
		if (driverMaxResSize != 0 && driverMaxResSize<OptimizerUtils.getLocalMemBudget() && !DMLScript.USE_LOCAL_SPARK_CONFIG)
			LOG.warn("Configuration parameter spark.driver.maxResultSize set to " + UtilFunctions.formatMemorySize(driverMaxResSize) + "."
					+ " You can set it through Spark default configuration setting either to 0 (unlimited) or to available memory budget of size "
					+ UtilFunctions.formatMemorySize((long)OptimizerUtils.getLocalMemBudget()) + ".");

		//globally add binaryblock serialization framework for all hdfs read/write operations
		//TODO if spark context passed in from outside (mlcontext), we need to clean this up at the end
		if( HDFSTool.USE_BINARYBLOCK_SERIALIZATION )
			HDFSTool.addBinaryBlockSerializationFramework( _spctx.hadoopConfiguration() );

		//statistics maintenance
		if( DMLScript.STATISTICS ){
			SparkStatistics.setCtxCreateTime(System.nanoTime()-t0);
		}
	}


	private static JavaSparkContext createContext(SparkConf conf){
		try{
			return new JavaSparkContext(conf);
		} 
		catch(Exception e){
			if(e.getMessage().contains("A master URL must be set in your configuration")){
				if(!localSparkWarning){
					LOG.warn("Error constructing Spark Context, falling back to local Spark context creation");
					localSparkWarning = true;
				}
				setLocalConfSettings(conf);
				return createContext(conf);
			}
			else
				throw new DMLException("Error while creating Spark context", e);
		}
	}

	private static void setLocalConfSettings(SparkConf conf){
		final String threads = ConfigurationManager.getDMLConfig().getTextValue(DMLConfig.LOCAL_SPARK_NUM_THREADS);
		conf.setMaster("local[" + threads + "]");
		conf.setAppName("LocalSparkContextApp");
		conf.set("spark.ui.showConsoleProgress", "false");
		conf.set("spark.ui.enabled", "false");
	}

	/**
	 * Sets up a SystemDS-preferred Spark configuration based on the implicit
	 * default configuration (as passed via configurations from outside).
	 *
	 * @return spark configuration
	 */
	public static SparkConf createSystemDSSparkConf() {
		SparkConf conf = new SparkConf();

		//always set unlimited result size (required for cp collect)
		conf.set("spark.driver.maxResultSize", "0");

		//always use the fair scheduler (for single jobs, it's equivalent to fifo
		//but for concurrent jobs in parfor it ensures better data locality because
		//round robin assignment mitigates the problem of 'sticky slots')
		if( FAIR_SCHEDULER_MODE ) {
			conf.set("spark.scheduler.mode", "FAIR");
		}

		//increase scheduler delay (usually more robust due to better data locality)
		if( !conf.contains("spark.locality.wait") ) { //default 3s
			conf.set("spark.locality.wait", "5s");
		}
		
		//increase max message size for robustness
		String sparkVersion = org.apache.spark.package$.MODULE$.SPARK_VERSION();
		String msgSizeConf = (UtilFunctions.compareVersion(sparkVersion, "2.0.0") < 0) ?
			"spark.akka.frameSize" : "spark.rpc.message.maxSize";
		if( !conf.contains(msgSizeConf) ) { //default 128MB
			conf.set(msgSizeConf, "512");
		}
		
		return conf;
	}
	
	@SuppressWarnings("resource")
	public static boolean isLocalMaster() {
		return isSparkContextCreated() ? 
			getSparkContextStatic().isLocal() : 
			DMLScript.USE_LOCAL_SPARK_CONFIG;
	}

	/**
	 * Spark instructions should call this for all matrix inputs except broadcast
	 * variables.
	 *
	 * @param varname variable name
	 * @return JavaPairRDD of MatrixIndexes-MatrixBlocks
	 */
	@SuppressWarnings("unchecked")
	public JavaPairRDD<MatrixIndexes,MatrixBlock> getBinaryMatrixBlockRDDHandleForVariable(String varname ) {
		MatrixObject mo = getMatrixObject(varname);
		return (JavaPairRDD<MatrixIndexes,MatrixBlock>)
			getRDDHandleForMatrixObject(mo, FileFormat.BINARY, -1, true);
	}
	
	@SuppressWarnings("unchecked")
	public JavaPairRDD<MatrixIndexes,MatrixBlock> getBinaryMatrixBlockRDDHandleForVariable(String varname, int numParts, boolean inclEmpty ) {
		MatrixObject mo = getMatrixObject(varname);
		return (JavaPairRDD<MatrixIndexes,MatrixBlock>)
			getRDDHandleForMatrixObject(mo, FileFormat.BINARY, numParts, inclEmpty);
	}

	/**
	 * Spark instructions should call this for all tensor inputs except broadcast
	 * variables.
	 *
	 * @param varname variable name
	 * @return JavaPairRDD of TensorIndexes-HomogTensors
	 */
	@SuppressWarnings("unchecked")
	public JavaPairRDD<TensorIndexes, TensorBlock> getBinaryTensorBlockRDDHandleForVariable(String varname ) {
		TensorObject to = getTensorObject(varname);
		return (JavaPairRDD<TensorIndexes, TensorBlock>)
				getRDDHandleForTensorObject(to, FileFormat.BINARY, -1, true);
	}

	@SuppressWarnings("unchecked")
	public JavaPairRDD<TensorIndexes, TensorBlock> getBinaryTensorBlockRDDHandleForVariable(String varname, int numParts, boolean inclEmpty ) {
		TensorObject to = getTensorObject(varname);
		return (JavaPairRDD<TensorIndexes, TensorBlock>)
				getRDDHandleForTensorObject(to, FileFormat.BINARY, numParts, inclEmpty);
	}

	/**
	 * Spark instructions should call this for all frame inputs except broadcast
	 * variables.
	 *
	 * @param varname variable name
	 * @return JavaPairRDD of Longs-FrameBlocks
	 */
	@SuppressWarnings("unchecked")
	public JavaPairRDD<Long,FrameBlock> getFrameBinaryBlockRDDHandleForVariable( String varname ) {
		FrameObject fo = getFrameObject(varname);
		JavaPairRDD<Long,FrameBlock> out = (JavaPairRDD<Long,FrameBlock>)
			getRDDHandleForFrameObject(fo, FileFormat.BINARY);
		return out;
	}

	public JavaPairRDD<?,?> getRDDHandleForVariable( String varname, FileFormat fmt, int numParts, boolean inclEmpty ) {
		Data dat = getVariable(varname);
		if( dat instanceof MatrixObject ) {
			MatrixObject mo = getMatrixObject(varname);
			return getRDDHandleForMatrixObject(mo, fmt, numParts, inclEmpty);
		}
		else if( dat instanceof FrameObject ) {
			FrameObject fo = getFrameObject(varname);
			return getRDDHandleForFrameObject(fo, fmt);
		}
		else {
			throw new DMLRuntimeException("Failed to obtain RDD for data type other than matrix or frame.");
		}
	}

	public JavaPairRDD<?,?> getRDDHandleForMatrixObject( MatrixObject mo, FileFormat fmt ) {
		return getRDDHandleForMatrixObject(mo, fmt, -1, true);
	}
	
	@SuppressWarnings({ "unchecked" })
	public JavaPairRDD<?,?> getRDDHandleForMatrixObject( MatrixObject mo, FileFormat fmt, int numParts, boolean inclEmpty ) {
		//NOTE: MB this logic should be integrated into MatrixObject
		//However, for now we cannot assume that spark libraries are
		//always available and hence only store generic references in
		//matrix object while all the logic is in the SparkExecContext

		JavaSparkContext sc = getSparkContext();
		JavaPairRDD<?,?> rdd = null;
		InputOutputInfo inputInfo = InputOutputInfo.get(DataType.MATRIX, fmt);
		//CASE 1: rdd already existing (reuse if checkpoint or trigger
		//pending rdd operations if not yet cached but prevent to re-evaluate
		//rdd operations if already executed and cached
		if(    mo.getRDDHandle()!=null
			&& (mo.getRDDHandle().isCheckpointRDD() || !mo.isCached(false)) )
		{
			//return existing rdd handling (w/o input format change)
			rdd = mo.getRDDHandle().getRDD();
		}
		//CASE 2: dirty in memory data or cached result of rdd operations
		else if( mo.isDirty() || mo.isCached(false) || mo.isFederated() || mo instanceof MatrixObjectFuture)
		{
			//get in-memory matrix block and parallelize it
			//w/ guarded parallelize (fallback to export, rdd from file if too large)
			DataCharacteristics dc = mo.getDataCharacteristics();
			boolean fromFile = false;
			if( !mo.isFederated() && !(mo instanceof MatrixObjectFuture) && (!OptimizerUtils.checkSparkCollectMemoryBudget(dc, 0)
				|| !_parRDDs.reserve(OptimizerUtils.estimatePartitionedSizeExactSparsity(dc)))) {
				if( mo.isDirty() || !mo.isHDFSFileExists() ) //write if necessary
					mo.exportData();
				rdd = sc.hadoopFile( mo.getFileName(), inputInfo.inputFormatClass, inputInfo.keyClass, inputInfo.valueClass);
				rdd = SparkUtils.copyBinaryBlockMatrix((JavaPairRDD<MatrixIndexes, MatrixBlock>)rdd); //cp is workaround for read bug
				fromFile = true;
			}
			else { //default case
				MatrixBlock mb = mo.acquireRead(); //pin matrix in memory
				rdd = toMatrixJavaPairRDD(sc, mb, mo.getBlocksize(), numParts, inclEmpty);
				mo.release(); //unpin matrix
				_parRDDs.registerRDD(rdd.id(), OptimizerUtils.estimatePartitionedSizeExactSparsity(dc), true);
			}

			//keep rdd handle for future operations on it
			RDDObject rddhandle = new RDDObject(rdd);
			rddhandle.setHDFSFile(fromFile);
			rddhandle.setParallelizedRDD(!fromFile);
			mo.setRDDHandle(rddhandle);
		}
		//CASE 3: non-dirty (file exists on HDFS)
		else
		{
			// parallelize hdfs-resident file
			// For binary block, these are: SequenceFileInputFormat.class, MatrixIndexes.class, MatrixBlock.class
			// TODO: Is that a bug because it triggers extra file reading? rdd is needed only for the first case
			rdd = sc.hadoopFile( mo.getFileName(), inputInfo.inputFormatClass, inputInfo.keyClass, inputInfo.valueClass);
			if(fmt == FileFormat.BINARY) 
				//note: this copy is still required in Spark 1.4 because spark hands out whatever the inputformat
				//recordreader returns; the javadoc explicitly recommend to copy all key/value pairs
				// cp is workaround for read bug
				rdd = SparkUtils.copyBinaryBlockMatrix((JavaPairRDD<MatrixIndexes, MatrixBlock>)rdd); 
			else if(fmt == FileFormat.COMPRESSED){
				// initial RDDS.
				rdd = ReaderSparkCompressed.getRDD(sc, mo.getFileName()); 
			}
			else if(fmt.isTextFormat()){
				JavaPairRDD<LongWritable, Text> textRDD = sc.hadoopFile(mo.getFileName(), //
					inputInfo.inputFormatClass, LongWritable.class, Text.class);
				// cp is workaround for read bug
				rdd = textRDD.mapToPair(new CopyTextInputFunction());
			}
			else
				throw new DMLRuntimeException("Incorrect input format in getRDDHandleForVariable");


			//keep rdd handle for future operations on it
			RDDObject rddhandle = new RDDObject(rdd);
			rddhandle.setHDFSFile(true);
			mo.setRDDHandle(rddhandle);
		}

		return rdd;
	}

	public JavaPairRDD<?, ?> getRDDHandleForTensorObject(TensorObject to, FileFormat fmt, int numParts, boolean inclEmpty) {
		//NOTE: MB this logic should be integrated into MatrixObject
		//However, for now we cannot assume that spark libraries are
		//always available and hence only store generic references in
		//matrix object while all the logic is in the SparkExecContext

		JavaSparkContext sc = getSparkContext();
		JavaPairRDD<?, ?> rdd;
		//CASE 1: rdd already existing (reuse if checkpoint or trigger
		//pending rdd operations if not yet cached but prevent to re-evaluate
		//rdd operations if already executed and cached
		if (to.getRDDHandle() != null && (to.getRDDHandle().isCheckpointRDD() || !to.isCached(false))) {
			//return existing rdd handling (w/o input format change)
			rdd = to.getRDDHandle().getRDD();
		}
		//CASE 2: dirty in memory data or cached result of rdd operations
		else if (to.isDirty() || to.isCached(false)) {
			//get in-memory matrix block and parallelize it
			//w/ guarded parallelize (fallback to export, rdd from file if too large)
			DataCharacteristics dc = to.getDataCharacteristics();
			//boolean fromFile = false;
			if (!OptimizerUtils.checkSparkCollectMemoryBudget(dc, 0) || !_parRDDs.reserve(
					OptimizerUtils.estimatePartitionedSizeExactSparsity(dc))) {
				if (to.isDirty() || !to.isHDFSFileExists()) //write if necessary
					to.exportData();
				// TODO implement hadoop read write for tensor
				throw new DMLRuntimeException("Tensor can not yet be written or read to hadoopFile");
				/*rdd = sc.hadoopFile(to.getFileName(), inputInfo.inputFormatClass, inputInfo.inputKeyClass, inputInfo.inputValueClass);
				rdd = SparkUtils.copyBinaryBlockTensor((JavaPairRDD<TensorIndexes, HomogTensor>) rdd); //cp is workaround for read bug
				fromFile = true;*/
			} else { //default case
				TensorBlock tb = to.acquireRead(); //pin matrix in memory
				int blen = dc.getBlocksize();
				rdd = toTensorJavaPairRDD(sc, tb, blen, numParts, inclEmpty);
				to.release(); //unpin matrix
				_parRDDs.registerRDD(rdd.id(), OptimizerUtils.estimatePartitionedSizeExactSparsity(dc), true);
			}

			//keep rdd handle for future operations on it
			RDDObject rddhandle = new RDDObject(rdd);
			// TODO use fromFile instead of false and true
			rddhandle.setHDFSFile(false);
			rddhandle.setParallelizedRDD(true);
			to.setRDDHandle(rddhandle);
		}
		//CASE 3: non-dirty (file exists on HDFS)
		else {
			// parallelize hdfs-resident file
			// For binary block, these are: SequenceFileInputFormat.class, MatrixIndexes.class, MatrixBlock.class
			if (fmt == FileFormat.BINARY) {
				// TODO implement hadoop read write for tensor
				throw new DMLRuntimeException("Tensor can not yet be written or read to hadoopFile");
				//rdd = sc.hadoopFile(to.getFileName(), inputInfo.inputFormatClass, inputInfo.inputKeyClass, inputInfo.inputValueClass);
				//note: this copy is still required in Spark 1.4 because spark hands out whatever the inputformat
				//recordreader returns; the javadoc explicitly recommend to copy all key/value pairs
				//rdd = SparkUtils.copyBinaryBlockTensor((JavaPairRDD<TensorIndexes, HomogTensor>) rdd); //cp is workaround for read bug
				// TODO: TensorMarket?
			} else {
				// TODO support other Input formats
				throw new DMLRuntimeException("Incorrect input format in getRDDHandleForVariable");
			}

			//keep rdd handle for future operations on it
			//RDDObject rddhandle = new RDDObject(rdd);
			//rddhandle.setHDFSFile(true);
			//to.setRDDHandle(rddhandle);
		}

		return rdd;
	}

	/**
	 * FIXME: currently this implementation assumes matrix representations but frame signature
	 * in order to support the old transform implementation.
	 *
	 * @param fo frame object
	 * @param fmt file format type
	 * @return JavaPairRDD handle for a frame object
	 */
	@SuppressWarnings({ "unchecked" })
	public JavaPairRDD<?,?> getRDDHandleForFrameObject( FrameObject fo, FileFormat fmt )
	{
		//NOTE: MB this logic should be integrated into FrameObject
		//However, for now we cannot assume that spark libraries are
		//always available and hence only store generic references in
		//matrix object while all the logic is in the SparkExecContext

		InputOutputInfo inputInfo2 = InputOutputInfo.get(DataType.FRAME, fmt); 
		JavaSparkContext sc = getSparkContext();
		JavaPairRDD<?,?> rdd = null;
		//CASE 1: rdd already existing (reuse if checkpoint or trigger
		//pending rdd operations if not yet cached but prevent to re-evaluate
		//rdd operations if already executed and cached
		if(    fo.getRDDHandle()!=null
			&& (fo.getRDDHandle().isCheckpointRDD() || !fo.isCached(false)) )
		{
			//return existing rdd handling (w/o input format change)
			rdd = fo.getRDDHandle().getRDD();
		}
		//CASE 2: dirty in memory data or cached result of rdd operations
		else if( fo.isDirty() || fo.isCached(false) )
		{
			//get in-memory matrix block and parallelize it
			//w/ guarded parallelize (fallback to export, rdd from file if too large)
			DataCharacteristics dc = fo.getDataCharacteristics();
			boolean fromFile = false;
			if( !OptimizerUtils.checkSparkCollectMemoryBudget(dc, 0) || !_parRDDs.reserve(
					OptimizerUtils.estimatePartitionedSizeExactSparsity(dc)) ) {
				if( fo.isDirty() ) { //write only if necessary
					fo.exportData();
				}
				rdd = sc.hadoopFile( fo.getFileName(), inputInfo2.inputFormatClass, inputInfo2.keyClass, inputInfo2.valueClass);
				rdd = ((JavaPairRDD<LongWritable, FrameBlock>)rdd).mapToPair( new CopyFrameBlockPairFunction() ); //cp is workaround for read bug
				fromFile = true;
			}
			else { //default case
				FrameBlock fb = fo.acquireRead(); //pin frame in memory
				rdd = toFrameJavaPairRDD(sc, fb);
				fo.release(); //unpin frame
				_parRDDs.registerRDD(rdd.id(), OptimizerUtils.estimatePartitionedSizeExactSparsity(dc), true);
			}

			//keep rdd handle for future operations on it
			RDDObject rddhandle = new RDDObject(rdd);
			rddhandle.setHDFSFile(fromFile);
			fo.setRDDHandle(rddhandle);
		}
		//CASE 3: non-dirty (file exists on HDFS)
		else
		{
			// parallelize hdfs-resident file
			// For binary block, these are: SequenceFileInputFormat.class, MatrixIndexes.class, MatrixBlock.class
			if(fmt == FileFormat.BINARY) {
				rdd = sc.hadoopFile( fo.getFileName(), inputInfo2.inputFormatClass, inputInfo2.keyClass, inputInfo2.valueClass);
				//note: this copy is still required in Spark 1.4 because spark hands out whatever the inputformat
				//recordreader returns; the javadoc explicitly recommend to copy all key/value pairs
				rdd = ((JavaPairRDD<LongWritable, FrameBlock>)rdd).mapToPair( new CopyFrameBlockPairFunction() ); //cp is workaround for read bug
			}
			else if(fmt.isTextFormat()) {
				rdd = sc.hadoopFile( fo.getFileName(), inputInfo2.inputFormatClass, inputInfo2.keyClass, inputInfo2.valueClass);
				rdd = ((JavaPairRDD<LongWritable, Text>)rdd).mapToPair( new CopyTextInputFunction() ); //cp is workaround for read bug
			}
			else {
				throw new DMLRuntimeException("Incorrect input format in getRDDHandleForVariable");
			}

			//keep rdd handle for future operations on it
			RDDObject rddhandle = new RDDObject(rdd);
			rddhandle.setHDFSFile(true);
			fo.setRDDHandle(rddhandle);
		}

		return rdd;
	}

	public Broadcast<CacheBlock<?>> broadcastVariable(CacheableData<CacheBlock<?>> cd) {
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		Broadcast<CacheBlock<?>> brBlock = null;

		// reuse existing non partitioned broadcast handle
		if (cd.getBroadcastHandle() != null && cd.getBroadcastHandle().isNonPartitionedBroadcastValid()) {
			brBlock = cd.getBroadcastHandle().getNonPartitionedBroadcast();
		}

		//TODO: synchronize
		if (brBlock == null) {
			//create new broadcast handle (never created, evicted)
			// account for overwritten invalid broadcast (e.g., evicted)
			if (cd.getBroadcastHandle() != null)
				CacheableData.addBroadcastSize(-cd.getBroadcastHandle().getSize());

			// read the matrix block
			CacheBlock<?> cb = cd.acquireRead();
			cd.release();

			// broadcast a non-empty frame whose size is smaller than 2G
			if (cb.getExactSerializedSize() > 0 && cb.getExactSerializedSize() <= Integer.MAX_VALUE) {
				brBlock = getSparkContext().broadcast(cb);
				// create the broadcast handle if the matrix or frame has never been broadcasted
				if (cd.getBroadcastHandle() == null) {
					cd.setBroadcastHandle(new BroadcastObject<>());
				}
				cd.getBroadcastHandle().setNonPartitionedBroadcast(brBlock,
					OptimizerUtils.estimateSize(cd.getDataCharacteristics()));
				CacheableData.addBroadcastSize(cd.getBroadcastHandle().getSize());

				if (DMLScript.STATISTICS) {
					SparkStatistics.accBroadCastTime(System.nanoTime() - t0);
					SparkStatistics.incBroadcastCount(1);
				}
			}
		}
		return brBlock;
	}

	@SuppressWarnings("unchecked")
	public PartitionedBroadcast<MatrixBlock> getBroadcastForMatrixObject(MatrixObject mo) {
		//NOTE: The memory consumption of this method is the in-memory size of the 
		//matrix object plus the partitioned size in 1k-1k blocks. Since the call
		//to broadcast happens after the matrix object has been released, the memory
		//requirements of blockified chunks in Spark's block manager are covered under
		//this maximum. Also note that we explicitly clear the in-memory blocks once
		//the broadcasts are created (other than in local mode) in order to avoid 
		//unnecessary memory requirements during the lifetime of this broadcast handle.
		

		PartitionedBroadcast<MatrixBlock> bret = null;

		synchronized (mo) {  //synchronize with the async. broadcast thread
			long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
			//reuse existing broadcast handle
			if (mo.getBroadcastHandle() != null && mo.getBroadcastHandle().isPartitionedBroadcastValid()) {
				bret = mo.getBroadcastHandle().getPartitionedBroadcast();
			}

			//create new broadcast handle (never created, evicted)
			if (bret == null) {
				//account for overwritten invalid broadcast (e.g., evicted)
				if (mo.getBroadcastHandle() != null)
					CacheableData.addBroadcastSize(-mo.getBroadcastHandle().getSize());

				//obtain meta data for matrix
				int blen = mo.getBlocksize();

				//create partitioned matrix block and release memory consumed by input
				MatrixBlock mb = mo.acquireRead();
				PartitionedBlock<MatrixBlock> pmb = new PartitionedBlock<>(mb, blen);
				mo.release();

				//determine coarse-grained partitioning
				int numPerPart = PartitionedBroadcast.computeBlocksPerPartition(mo.getNumRows(), mo.getNumColumns(), blen);
				int numParts = (int) Math.ceil((double) pmb.getNumRowBlocks() * pmb.getNumColumnBlocks() / numPerPart);
				Broadcast<PartitionedBlock<MatrixBlock>>[] ret = new Broadcast[numParts];

				//create coarse-grained partitioned broadcasts
				if (numParts > 1) {
					Arrays.parallelSetAll(ret, i -> createPartitionedBroadcast(pmb, numPerPart, i));
				} else { //single partition
					ret[0] = getSparkContext().broadcast(pmb);
					if (!isLocalMaster())
						pmb.clearBlocks();
				}
				
				bret = new PartitionedBroadcast<>(ret, mo.getDataCharacteristics());
				// create the broadcast handle if the matrix or frame has never been broadcasted
				if (mo.getBroadcastHandle() == null) {
					mo.setBroadcastHandle(new BroadcastObject<MatrixBlock>());
				}
				mo.getBroadcastHandle().setPartitionedBroadcast(bret,
					OptimizerUtils.estimatePartitionedSizeExactSparsity(mo.getDataCharacteristics()));
				CacheableData.addBroadcastSize(mo.getBroadcastHandle().getSize());

			}
			if (DMLScript.STATISTICS) {
				SparkStatistics.accBroadCastTime(System.nanoTime() - t0);
				SparkStatistics.incBroadcastCount(1);
			}
		}
		return bret;
	}
	
	public void setBroadcastHandle(MatrixObject mo) {
		getBroadcastForMatrixObject(mo);
	}

	@SuppressWarnings("unchecked")
	public PartitionedBroadcast<TensorBlock> getBroadcastForTensorObject(TensorObject to) {
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;

		PartitionedBroadcast<TensorBlock> bret = null;

		//reuse existing broadcast handle
		if (to.getBroadcastHandle() != null && to.getBroadcastHandle().isPartitionedBroadcastValid()) {
			bret = to.getBroadcastHandle().getPartitionedBroadcast();
		}

		//create new broadcast handle (never created, evicted)
		if (bret == null) {
			//account for overwritten invalid broadcast (e.g., evicted)
			if (to.getBroadcastHandle() != null)
				CacheableData.addBroadcastSize(-to.getBroadcastHandle().getSize());

			//obtain meta data for matrix
			DataCharacteristics dc = to.getDataCharacteristics();
			long[] dims = dc.getDims();
			int blen = dc.getBlocksize();

			//create partitioned matrix block and release memory consumed by input
			PartitionedBlock<TensorBlock> pmb = new PartitionedBlock<>(to.acquireReadAndRelease(), dims, blen);

			//determine coarse-grained partitioning
			int numPerPart = PartitionedBroadcast.computeBlocksPerPartition(dims, blen);
			int numParts = (int) Math.ceil((double) pmb.getNumRowBlocks() * pmb.getNumColumnBlocks() / numPerPart);
			Broadcast<PartitionedBlock<TensorBlock>>[] ret = new Broadcast[numParts];

			//create coarse-grained partitioned broadcasts
			if (numParts > 1) {
				Arrays.parallelSetAll(ret, i -> createPartitionedBroadcast(pmb, numPerPart, i));
			} else { //single partition
				ret[0] = getSparkContext().broadcast(pmb);
				if (!isLocalMaster())
					pmb.clearBlocks();
			}

			bret = new PartitionedBroadcast<>(ret, to.getDataCharacteristics());
			// create the broadcast handle if the matrix or frame has never been broadcasted
			if (to.getBroadcastHandle() == null) {
				to.setBroadcastHandle(new BroadcastObject<MatrixBlock>());
			}
			to.getBroadcastHandle().setPartitionedBroadcast(bret,
					OptimizerUtils.estimatePartitionedSizeExactSparsity(to.getDataCharacteristics()));
			CacheableData.addBroadcastSize(to.getBroadcastHandle().getSize());

			if (DMLScript.STATISTICS) {
				SparkStatistics.accBroadCastTime(System.nanoTime() - t0);
				SparkStatistics.incBroadcastCount(1);
			}
		}
		return bret;
	}

	public PartitionedBroadcast<MatrixBlock> getBroadcastForVariable(String varname) {
		return getBroadcastForMatrixObject(getMatrixObject(varname));
	}

	public PartitionedBroadcast<TensorBlock> getBroadcastForTensorVariable(String varname) {
		return getBroadcastForTensorObject(getTensorObject(varname));
	}

	@SuppressWarnings("unchecked")
	public PartitionedBroadcast<FrameBlock> getBroadcastForFrameVariable(String varname) {
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;

		FrameObject fo = getFrameObject(varname);

		PartitionedBroadcast<FrameBlock> bret = null;

		//reuse existing broadcast handle
		if (fo.getBroadcastHandle() != null && fo.getBroadcastHandle().isPartitionedBroadcastValid()) {
			bret = fo.getBroadcastHandle().getPartitionedBroadcast();
		}

		//create new broadcast handle (never created, evicted)
		if (bret == null) {
			//account for overwritten invalid broadcast (e.g., evicted)
			if (fo.getBroadcastHandle() != null)
				CacheableData.addBroadcastSize(-fo.getBroadcastHandle().getSize());

			//obtain meta data for frame
			int blen = OptimizerUtils.getDefaultFrameSize();

			//create partitioned frame block and release memory consumed by input
			FrameBlock mb = fo.acquireRead();
			PartitionedBlock<FrameBlock> pmb = new PartitionedBlock<>(mb, blen);
			fo.release();

			//determine coarse-grained partitioning
			int numPerPart = PartitionedBroadcast.computeBlocksPerPartition(fo.getNumRows(), fo.getNumColumns(), blen);
			int numParts = (int) Math.ceil((double) pmb.getNumRowBlocks() * pmb.getNumColumnBlocks() / numPerPart);
			Broadcast<PartitionedBlock<FrameBlock>>[] ret = new Broadcast[numParts];

			//create coarse-grained partitioned broadcasts
			if (numParts > 1) {
				Arrays.parallelSetAll(ret, i -> createPartitionedBroadcast(pmb, numPerPart, i));
			} else { //single partition
				ret[0] = getSparkContext().broadcast(pmb);
				if (!isLocalMaster())
					pmb.clearBlocks();
			}
			
			bret = new PartitionedBroadcast<>(ret, new MatrixCharacteristics(
				fo.getDataCharacteristics()).setBlocksize(blen));
			if (fo.getBroadcastHandle() == null)
				fo.setBroadcastHandle(new BroadcastObject<FrameBlock>());
			
			fo.getBroadcastHandle().setPartitionedBroadcast(bret,
				OptimizerUtils.estimatePartitionedSizeExactSparsity(fo.getDataCharacteristics()));
			CacheableData.addBroadcastSize(fo.getBroadcastHandle().getSize());

			if (DMLScript.STATISTICS) {
				SparkStatistics.accBroadCastTime(System.nanoTime() - t0);
				SparkStatistics.incBroadcastCount(1);
			}
		}
		return bret;
	}
	
	private Broadcast<PartitionedBlock<? extends CacheBlock<?>>> createPartitionedBroadcast(
			PartitionedBlock<? extends CacheBlock<?>> pmb, int numPerPart, int pos) {
		int offset = pos * numPerPart;
		int numBlks = Math.min(numPerPart, pmb.getNumRowBlocks() * pmb.getNumColumnBlocks() - offset);
		PartitionedBlock<? extends CacheBlock<?>> tmp = pmb.createPartition(offset, numBlks);
		Broadcast<PartitionedBlock<? extends CacheBlock<?>>> ret = getSparkContext().broadcast(tmp);
		if (!isLocalMaster())
			tmp.clearBlocks();
		return ret;
	}

	/**
	 * Keep the output rdd of spark rdd operations as meta data of matrix/frame
	 * objects in the symbol table.
	 *
	 * @param varname variable name
	 * @param rdd JavaPairRDD handle for variable
	 */
	public void setRDDHandleForVariable(String varname, JavaPairRDD<?,?> rdd) {
		CacheableData<?> obj = getCacheableData(varname);
		RDDObject rddhandle = new RDDObject(rdd);
		obj.setRDDHandle( rddhandle );
	}

	public void setRDDHandleForVariable(String varname, RDDObject rddhandle) {
		CacheableData<?> obj = getCacheableData(varname);
		obj.setRDDHandle(rddhandle);
	}

	public static JavaPairRDD<MatrixIndexes,MatrixBlock> toMatrixJavaPairRDD(JavaSparkContext sc, MatrixBlock src, int blen) {
		return toMatrixJavaPairRDD(sc, src, blen, -1, true);
	}
	
	public static JavaPairRDD<MatrixIndexes,MatrixBlock> toMatrixJavaPairRDD(JavaSparkContext sc, MatrixBlock src,
			int blen, int numParts, boolean inclEmpty) {
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		List<Tuple2<MatrixIndexes,MatrixBlock>> list = null;

		if( src.getNumRows() <= blen && src.getNumColumns() <= blen ) {
			list = Arrays.asList(new Tuple2<>(new MatrixIndexes(1,1), src));
		}
		else {
			MatrixCharacteristics mc = new MatrixCharacteristics(
				src.getNumRows(), src.getNumColumns(), blen, src.getNonZeros());
			list = LongStream.range(0, mc.getNumBlocks()).parallel()
				.mapToObj(i -> createIndexedMatrixBlock(src, mc, i))
				.filter(kv -> inclEmpty || !kv._2.isEmptyBlock(false))
				.collect(Collectors.toList());
		}

		JavaPairRDD<MatrixIndexes,MatrixBlock> result = (numParts > 1) ?
			sc.parallelizePairs(list, numParts) : sc.parallelizePairs(list);
		
		if (DMLScript.STATISTICS) {
			SparkStatistics.accParallelizeTime(System.nanoTime() - t0);
			SparkStatistics.incParallelizeCount(1);
		}

		return result;
	}

	public static JavaPairRDD<TensorIndexes, TensorBlock> toTensorJavaPairRDD(JavaSparkContext sc, TensorBlock src, int blen) {
		return toTensorJavaPairRDD(sc, src, blen, -1, true);
	}

	public static JavaPairRDD<TensorIndexes, TensorBlock> toTensorJavaPairRDD(JavaSparkContext sc, TensorBlock src,
			int blen, int numParts, boolean inclEmpty) {
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		List<Tuple2<TensorIndexes, TensorBlock>> list;

		int numDims = src.getNumDims();
		boolean singleBlock = true;
		for (int i = 0; i < numDims; i++) {
			if (blen > src.getDim(i)) {
				singleBlock = false;
				break;
			}
		}
		if (singleBlock) {
			long[] ix = new long[numDims];
			Arrays.fill(ix, 1);
			list = Arrays.asList(new Tuple2<>(new TensorIndexes(ix), src));
		} else {
			// TODO rows and columns for matrix characteristics
			long[] dims = src.getLongDims();
			TensorCharacteristics mc = new TensorCharacteristics(dims, src.getNonZeros());
			list = LongStream.range(0, mc.getNumBlocks()).parallel()
					.mapToObj(i -> createIndexedTensorBlock(src, mc, i))
					.filter(kv -> inclEmpty || !kv._2.isEmpty(false))
					.collect(Collectors.toList());
		}

		JavaPairRDD<TensorIndexes, TensorBlock> result = (numParts > 1) ?
				sc.parallelizePairs(list, numParts) : sc.parallelizePairs(list);

		if (DMLScript.STATISTICS) {
			SparkStatistics.accParallelizeTime(System.nanoTime() - t0);
			SparkStatistics.incParallelizeCount(1);
		}

		return result;
	}

	private static Tuple2<MatrixIndexes,MatrixBlock> createIndexedMatrixBlock(MatrixBlock mb, MatrixCharacteristics mc, long ix) {
		try {
			//compute block indexes
			long blockRow = ix / mc.getNumColBlocks();
			long blockCol = ix % mc.getNumColBlocks();
			//compute block sizes
			int maxRow = UtilFunctions.computeBlockSize(mc.getRows(), blockRow+1, mc.getBlocksize());
			int maxCol = UtilFunctions.computeBlockSize(mc.getCols(), blockCol+1, mc.getBlocksize());
			//copy sub-matrix to block
			MatrixBlock block = new MatrixBlock(maxRow, maxCol, mb.isInSparseFormat());
			int row_offset = (int)blockRow*mc.getBlocksize();
			int col_offset = (int)blockCol*mc.getBlocksize();
			block = mb.slice( row_offset, row_offset+maxRow-1,
				col_offset, col_offset+maxCol-1, false, block );
			//create key-value pair
			return new Tuple2<>(new MatrixIndexes(blockRow+1, blockCol+1), block);
		}
		catch(DMLRuntimeException ex) {
			throw new RuntimeException(ex);
		}
	}

	private static Tuple2<TensorIndexes, TensorBlock> createIndexedTensorBlock(TensorBlock mb, TensorCharacteristics tc, long ix) {
		try {
			//compute block indexes
			long[] blockIx = UtilFunctions.computeTensorIndexes(tc, ix);
			int[] outDims = new int[tc.getNumDims()];
			int[] offset = new int[tc.getNumDims()];
			UtilFunctions.computeSliceInfo(tc, blockIx, outDims, offset);
			TensorBlock outBlock = new TensorBlock(mb.getValueType(), outDims);
			outBlock = mb.slice(offset, outBlock);
			return new Tuple2<>(new TensorIndexes(blockIx), outBlock);
		}
		catch(DMLRuntimeException ex) {
			throw new RuntimeException(ex);
		}
	}

	public static JavaPairRDD<Long,FrameBlock> toFrameJavaPairRDD(JavaSparkContext sc, FrameBlock src) {
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		LinkedList<Tuple2<Long,FrameBlock>> list = new LinkedList<>();

		//create and write subblocks of matrix
		int blksize = ConfigurationManager.getBlocksize();
		for(int blockRow = 0; blockRow < (int)Math.ceil(src.getNumRows()/(double)blksize); blockRow++)
		{
			int maxRow = (blockRow*blksize + blksize < src.getNumRows()) ? blksize : src.getNumRows() - blockRow*blksize;
			int roffset = blockRow*blksize;

			FrameBlock block = new FrameBlock(src.getSchema());

			//copy sub frame to block, incl meta data on first
			src.slice( roffset, roffset+maxRow-1, 0, src.getNumColumns()-1, block );
			if( roffset == 0 )
				block.setColumnMetadata(src.getColumnMetadata());

			//append block to sequence file
			list.addLast(new Tuple2<>((long)roffset+1, block));
		}

		JavaPairRDD<Long,FrameBlock> result = sc.parallelizePairs(list);
		if (DMLScript.STATISTICS) {
			SparkStatistics.accParallelizeTime(System.nanoTime() - t0);
			SparkStatistics.incParallelizeCount(1);
		}

		return result;
	}

	/**
	 * This method is a generic abstraction for calls from the buffer pool.
	 *
	 * @param rdd rdd object
	 * @param rlen number of rows
	 * @param clen number of columns
	 * @param blen block length
	 * @param nnz number of non-zeros
	 * @return matrix block
	 */
	@SuppressWarnings("unchecked")
	public static MatrixBlock toMatrixBlock(RDDObject rdd, int rlen, int clen, int blen, long nnz) {
		return toMatrixBlock(
			(JavaPairRDD<MatrixIndexes, MatrixBlock>) rdd.getRDD(),
			rlen, clen, blen, nnz);
	}

	/**
	 * Utility method for creating a single matrix block out of a binary block RDD. Note that this collect call might
	 * trigger execution of any pending transformations.
	 *
	 * NOTE: This is an unguarded utility function, which requires memory for both the output matrix and its collected,
	 * blocked representation.
	 *
	 * @param rdd  JavaPairRDD for matrix block
	 * @param rlen number of rows
	 * @param clen number of columns
	 * @param blen block length
	 * @param nnz  number of non-zeros
	 * @return Local matrix block
	 */
	public static MatrixBlock toMatrixBlock(JavaPairRDD<MatrixIndexes,MatrixBlock> rdd, int rlen, int clen, int blen, long nnz) {
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		final Boolean singleBlock = rlen <= blen && clen <= blen ;
		final MatrixBlock out = singleBlock ? toMatrixBlockSingleBlock(rdd, rlen, clen) : toMatrixBlockMultiBlock(rdd, rlen, clen, blen, nnz);

		if (DMLScript.STATISTICS)
			SparkStatistics.accCollectTime(System.nanoTime() - t0);

		return out;
	}

	private static MatrixBlock toMatrixBlockSingleBlock(JavaPairRDD<MatrixIndexes, MatrixBlock> rdd, int rlen,
		int clen) {
		// special case without copy and nnz maintenance
		List<Tuple2<MatrixIndexes, MatrixBlock>> list = rdd.collect();
		MatrixBlock out;

		if(list.size() > 1)
			throw new DMLRuntimeException("Expecting no more than one result block but got: " + list.size());
		else if(list.size() == 1)
			out = list.get(0)._2();
		else // empty (e.g., after ops w/ outputEmpty=false)
			out = new MatrixBlock(rlen, clen, true);
		out.examSparsity();
		return out;
	}

	private static MatrixBlock toMatrixBlockMultiBlock(JavaPairRDD<MatrixIndexes, MatrixBlock> rdd, int rlen, int clen,
		int blen, long nnz) {
		// determine target sparse/dense representation
		final long lnnz = (nnz >= 0) ? nnz : (long) rlen * clen;
		final boolean sparse = MatrixBlock.evalSparseFormatInMemory(rlen, clen, lnnz);

		// create output matrix block (w/ lazy allocation)
		MatrixBlock out = new MatrixBlock(rlen, clen, sparse, lnnz);

		// kickoff asynchronous allocation
		Future<MatrixBlock> fout = out.allocateBlockAsync();

		// trigger pending RDD operations and collect blocks
		List<Tuple2<MatrixIndexes, MatrixBlock>> list = rdd.collect();
		out = IOUtilFunctions.get(fout); // wait for allocation
		
		LongAdder aNnz = new LongAdder();
		// copy blocks one-at-a-time into output matrix block
		blockPartitionsToMatrixBlock(list, out, aNnz,  blen);
		
		// post-processing output matrix
		if(sparse)
			out.sortSparseRows();
		if(containsCompressedMatrixBlock(list))
			// Recompute zeros since compressed does not maintain it perfectly
			out.recomputeNonZeros();
		else 
			out.setNonZeros(aNnz.longValue());
		out.examSparsity();
		return out;
	}

	private static boolean containsCompressedMatrixBlock(List<Tuple2<MatrixIndexes, MatrixBlock>> list) {
		for(Tuple2<MatrixIndexes, MatrixBlock> t : list)
			if(t._2() instanceof CompressedMatrixBlock)
				return true;
		return false;
	}

	private static void blockPartitionsToMatrixBlock(List<Tuple2<MatrixIndexes, MatrixBlock>> tuples, MatrixBlock out,
		LongAdder aNnz, int blen) {
		if(out.isInSparseFormat())
			blockPartitionsToMatrixBlockSingleThread(tuples, out, aNnz, blen);
		else
			blockPartitionsToMatrixBlockMultiThreaded(tuples, out, aNnz, blen);
	}

	private static void blockPartitionsToMatrixBlockSingleThread(List<Tuple2<MatrixIndexes, MatrixBlock>> tuples,
		MatrixBlock out, LongAdder aNnz, int blen) {
		final boolean sparse = out.isInSparseFormat();
		final boolean sparseCopyShallow = sparse && out.getNumColumns() <= blen;
		for(Tuple2<MatrixIndexes, MatrixBlock> tuple : tuples)
			blockPartitionToMatrixBlock(tuple, out, aNnz, blen, sparseCopyShallow);
	}

	private static void blockPartitionsToMatrixBlockMultiThreaded(List<Tuple2<MatrixIndexes, MatrixBlock>> tuples,
		MatrixBlock out, LongAdder aNnz, int blen) {
		final int k = InfrastructureAnalyzer.getLocalParallelism();
		final ExecutorService pool = CommonThreadPool.get(k);
		try {
			final ArrayList<BlockPartitionToMatrixBlockTask> tasks = new ArrayList<>();
			final int tSize = tuples.size();
			final int blockSize = Math.max(tSize / k / 2, 1);
			for(int i = 0; i < tSize; i += blockSize)
				tasks.add(new BlockPartitionToMatrixBlockTask(tuples, i, Math.min(i + blockSize, tSize), out, aNnz, blen, false));
			// for(Tuple2<MatrixIndexes, MatrixBlock> tuple : tuples)

			for(Future<Object> f : pool.invokeAll(tasks))
				f.get();

		}
		catch(InterruptedException | ExecutionException ex) {
			throw new DMLRuntimeException("Parallel block partitions to matrix block failed", ex);
		}
		finally{
			pool.shutdown();
		}
	}

	private static void blockPartitionToMatrixBlock(Tuple2<MatrixIndexes, MatrixBlock> tuple, MatrixBlock out,
		LongAdder aNnz, int blen, boolean sparseCopyShallow) {
		// unpack index-block pair
		final MatrixIndexes ix = tuple._1();
		final MatrixBlock block = tuple._2();

		// compute row/column block offsets
		final int row_offset = (int) (ix.getRowIndex() - 1) * blen;
		final int col_offset = (int) (ix.getColumnIndex() - 1) * blen;

		// Add the values of this block into the output block.
		block.putInto(out, row_offset, col_offset, sparseCopyShallow);

		// incremental maintenance nnz
		aNnz.add(block.getNonZeros());
	}

	private static class BlockPartitionToMatrixBlockTask implements Callable<Object> {
		private final List<Tuple2<MatrixIndexes, MatrixBlock>> _tuples;
		private final int _start;
		private final int _end;
		private final MatrixBlock _out;
		private final LongAdder _aNnz;
		private final int _blen;
		private final boolean _sparseCopyShallow;

		protected BlockPartitionToMatrixBlockTask(List<Tuple2<MatrixIndexes, MatrixBlock>> tuples, int start, int end, MatrixBlock out,
		LongAdder aNnz, int blen, boolean sparseCopyShallow){
			_tuples = tuples;
			_start = start;
			_end = end;
			_out = out;
			_aNnz = aNnz;
			_blen = blen;
			_sparseCopyShallow = sparseCopyShallow;
		}

		@Override
		public Object call(){
			for(int i = _start; i < _end; i ++)
				blockPartitionToMatrixBlock(_tuples.get(i), _out, _aNnz, _blen, _sparseCopyShallow);
			
			return null;
		}
	}

	@SuppressWarnings("unchecked")
	public static MatrixBlock toMatrixBlock(RDDObject rdd, int rlen, int clen, long nnz) {
		return toMatrixBlock(
				(JavaPairRDD<MatrixIndexes, MatrixCell>) rdd.getRDD(),
				rlen, clen, nnz);
	}

	/**
	 * Utility method for creating a single matrix block out of a binary cell RDD.
	 * Note that this collect call might trigger execution of any pending transformations.
	 *
	 * @param rdd JavaPairRDD for matrix block
	 * @param rlen number of rows
	 * @param clen number of columns
	 * @param nnz number of non-zeros
	 * @return matrix block
	 */
	public static MatrixBlock toMatrixBlock(JavaPairRDD<MatrixIndexes, MatrixCell> rdd, int rlen, int clen, long nnz)
	{
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;

		MatrixBlock out = null;

		//determine target sparse/dense representation
		long lnnz = (nnz >= 0) ? nnz : (long)rlen * clen;
		boolean sparse = MatrixBlock.evalSparseFormatInMemory(rlen, clen, lnnz);

		//create output matrix block (w/ lazy allocation)
		out = new MatrixBlock(rlen, clen, sparse);

		List<Tuple2<MatrixIndexes,MatrixCell>> list = rdd.collect();

		//copy blocks one-at-a-time into output matrix block
		for( Tuple2<MatrixIndexes,MatrixCell> keyval : list )
		{
			//unpack index-block pair
			MatrixIndexes ix = keyval._1();
			MatrixCell cell = keyval._2();

			//append cell to dense/sparse target in order to avoid shifting for sparse
			//note: this append requires a final sort of sparse rows
			out.appendValue((int)ix.getRowIndex()-1, (int)ix.getColumnIndex()-1, cell.getValue());
		}

		//post-processing output matrix
		if( sparse )
			out.sortSparseRows();
		out.recomputeNonZeros();
		out.examSparsity();

		if (DMLScript.STATISTICS)
			SparkStatistics.accCollectTime(System.nanoTime() - t0);

		return out;
	}

	public static TensorBlock toTensorBlock(JavaPairRDD<TensorIndexes, TensorBlock> rdd, DataCharacteristics dc) {
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;

		// TODO special case single block
		int[] idims = dc.getIntDims();
		// TODO asynchronous allocation
		List<Tuple2<TensorIndexes, TensorBlock>> list = rdd.collect();
		ValueType vt = (list.get(0)._2).getValueType();
		TensorBlock out = new TensorBlock(vt, idims).allocateBlock();

		//copy blocks one-at-a-time into output matrix block
		for( Tuple2<TensorIndexes, TensorBlock> keyval : list )
		{
			//unpack index-block pair
			TensorIndexes ix = keyval._1();
			TensorBlock block = keyval._2();

			//compute row/column block offsets
			int[] lower = new int[ix.getNumDims()];
			int[] upper = new int[ix.getNumDims()];
			for (int i = 0; i < lower.length; i++) {
				lower[i] = (int) ((ix.getIndex(i) - 1) * dc.getBlocksize());
				upper[i] = lower[i] + block.getDim(i) - 1;
			}
			upper[upper.length - 1]++;
			for (int i = upper.length - 1; i > 0; i--) {
				if (upper[i] == block.getDim(i)) {
					upper[i] = 0;
					upper[i - 1]++;
				}
			}

			// TODO sparse copy
			out.copy(lower, upper, block);
			// TODO keep track of nnz
		}

		// TODO post-processing output tensor (nnz, sparsity)

		if (DMLScript.STATISTICS)
			SparkStatistics.accCollectTime(System.nanoTime() - t0);

		return out;
	}

	public static PartitionedBlock<MatrixBlock> toPartitionedMatrixBlock(JavaPairRDD<MatrixIndexes,MatrixBlock> rdd, int rlen, int clen, int blen, long nnz)
	{
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;

		PartitionedBlock<MatrixBlock> out = new PartitionedBlock<>(rlen, clen, blen);
		List<Tuple2<MatrixIndexes,MatrixBlock>> list = rdd.collect();

		//copy blocks one-at-a-time into output matrix block
		for( Tuple2<MatrixIndexes,MatrixBlock> keyval : list ) {
			//unpack index-block pair
			MatrixIndexes ix = keyval._1();
			MatrixBlock block = keyval._2();
			out.setBlock((int)ix.getRowIndex(), (int)ix.getColumnIndex(), block);
		}

		if (DMLScript.STATISTICS)
			SparkStatistics.accCollectTime(System.nanoTime() - t0);

		return out;
	}

	@SuppressWarnings("unchecked")
	public static FrameBlock toFrameBlock(RDDObject rdd, ValueType[] schema, int rlen, int clen) {
		JavaPairRDD<Long,FrameBlock> lrdd = (JavaPairRDD<Long,FrameBlock>) rdd.getRDD();
		return toFrameBlock(lrdd, schema, rlen, clen);
	}

	public static FrameBlock toFrameBlock(JavaPairRDD<Long,FrameBlock> rdd, ValueType[] schema, int rlen, int clen) {
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;

		if(schema == null)
			schema = UtilFunctions.nCopies(clen, ValueType.STRING);

		//create output frame block (w/ lazy allocation)
		FrameBlock out = new FrameBlock(schema, rlen);

		List<Tuple2<Long,FrameBlock>> list = rdd.collect();

		//copy blocks one-at-a-time into output matrix block
		for( Tuple2<Long,FrameBlock> keyval : list )
		{
			//unpack index-block pair
			int ix = (int)(keyval._1() - 1);
			FrameBlock block = keyval._2();

			//copy into output frame
			out.copy( ix, ix+block.getNumRows()-1, 0, block.getNumColumns()-1, block );
			if( ix == 0 ) {
				out.setColumnNames(block.getColumnNames());
				out.setColumnMetadata(block.getColumnMetadata());
			}
		}

		if (DMLScript.STATISTICS)
			SparkStatistics.accCollectTime(System.nanoTime() - t0);

		return out;
	}

	@SuppressWarnings("unchecked")
	public static long writeMatrixRDDtoHDFS( RDDObject rdd, String path, FileFormat fmt )
	{
		JavaPairRDD<MatrixIndexes,MatrixBlock> lrdd = (JavaPairRDD<MatrixIndexes, MatrixBlock>) rdd.getRDD();
		InputOutputInfo oinfo = InputOutputInfo.get(DataType.MATRIX, fmt);
		
		// if compression is enabled decompress all blocks before writing to disk to ensure type of MatrixBlock.
		if(ConfigurationManager.isCompressionEnabled())
			lrdd = lrdd.mapValues(new DeCompressionSPInstruction.DeCompressionFunction());

		//piggyback nnz maintenance on write
		LongAccumulator aNnz = getSparkContextStatic().sc().longAccumulator("nnz");
		lrdd = lrdd.mapValues(new ComputeBinaryBlockNnzFunction(aNnz));


		//save file is an action which also triggers nnz maintenance
		lrdd.saveAsHadoopFile(path,
			oinfo.keyClass,
			oinfo.valueClass,
			oinfo.outputFormatClass);

		//return nnz aggregate of all blocks
		return aNnz.value();
	}

	@SuppressWarnings("unchecked")
	public static void writeFrameRDDtoHDFS( RDDObject rdd, String path, FileFormat fmt)
	{
		JavaPairRDD<?, FrameBlock> lrdd = (JavaPairRDD<Long, FrameBlock>) rdd.getRDD();
		InputOutputInfo oinfo = InputOutputInfo.get(DataType.FRAME, fmt);
		
		//convert keys to writables if necessary
		if( fmt == FileFormat.BINARY ) {
			lrdd = ((JavaPairRDD<Long, FrameBlock>)lrdd).mapToPair(
					new LongFrameToLongWritableFrameFunction());
		}

		//save file is an action which also triggers nnz maintenance
		lrdd.saveAsHadoopFile(path,
			oinfo.keyClass,
			oinfo.valueClass,
			oinfo.outputFormatClass);
	}

	///////////////////////////////////////////
	// Cleanup of RDDs and Broadcast variables
	///////

	/**
	 * Adds a child rdd object to the lineage of a parent rdd.
	 *
	 * @param varParent parent variable
	 * @param varChild child variable
	 */
	public void addLineageRDD(String varParent, String varChild) {
		RDDObject parent = getCacheableData(varParent).getRDDHandle();
		RDDObject child = getCacheableData(varChild).getRDDHandle();
		parent.addLineageChild( child );
	}

	/**
	 * Adds a child broadcast object to the lineage of a parent rdd.
	 *
	 * @param varParent parent variable
	 * @param varChild child variable
	 */
	public void addLineageBroadcast(String varParent, String varChild) {
		RDDObject parent = getCacheableData(varParent).getRDDHandle();
		BroadcastObject<?> child = getCacheableData(varChild).getBroadcastHandle();
		parent.addLineageChild( child );
	}

	public void addLineage(String varParent, String varChild, boolean broadcast) {
		if( broadcast )
			addLineageBroadcast(varParent, varChild);
		else
			addLineageRDD(varParent, varChild);
	}

	@Override
	public void cleanupCacheableData(CacheableData<?> mo)
	{
		//NOTE: this method overwrites the default behavior of cleanupMatrixObject
		//and hence is transparently used by rmvar instructions and other users. The
		//core difference is the lineage-based cleanup of RDD and broadcast variables.

		if (DMLScript.JMLC_MEM_STATISTICS)
			Statistics.removeCPMemObject(System.identityHashCode(mo));

		if( !mo.isCleanupEnabled() )
			return;
		
		try
		{
			//compute ref count only if matrix cleanup actually necessary
			if( !getVariables().hasReferences(mo) ) {
				//clean cached data
				mo.clearData(getTID());

				//clean hdfs data if no pending rdd operations on it
				if( mo.isHDFSFileExists() && mo.getFileName()!=null ) {
					if( mo.getRDDHandle()==null ) {
						HDFSTool.deleteFileWithMTDIfExistOnHDFS(mo.getFileName());
					}
					else { //deferred file removal
						RDDObject rdd = mo.getRDDHandle();
						rdd.setHDFSFilename(mo.getFileName());
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
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
	}

	private void rCleanupLineageObject(LineageObject lob)
		throws IOException
	{
		//abort recursive cleanup if still consumers
		if( lob.getNumReferences() > 0 )
			return;

		//abort if still reachable through matrix object (via back references for
		//robustness in function calls and to prevent repeated scans of the symbol table)
		if( lob.hasBackReference() )
			return;

		//abort if a RDD is cached locally
		if (lob.isInLineageCache())
			return;

		//cleanup current lineage object (from driver/executors)
		//incl deferred hdfs file removal (only if metadata set by cleanup call)
		cleanupSingleLineageObject(lob);

		//recursively process lineage children
		for( LineageObject c : lob.getLineageChilds() ){
			c.decrementNumReferences();
			rCleanupLineageObject(c);
		}
	}

	@SuppressWarnings({ "rawtypes", "unchecked" })
	public static void cleanupSingleLineageObject(LineageObject lob) {
		//cleanup current lineage object (from driver/executors)
		//incl deferred hdfs file removal (only if metadata set by cleanup call)
		if( lob instanceof RDDObject ) {
			RDDObject rdd = (RDDObject)lob;
			int rddID = rdd.getRDD().id();
			cleanupRDDVariable(rdd.getRDD());
			if( rdd.getHDFSFilename()!=null ) { //deferred file removal
				try {
					HDFSTool.deleteFileWithMTDIfExistOnHDFS(rdd.getHDFSFilename());
				}
				catch(IOException e) {
					throw new DMLRuntimeException(e);
				}
			}
			if( rdd.isParallelizedRDD() )
				_parRDDs.deregisterRDD(rddID);
		}
		else if( lob instanceof BroadcastObject ) {
			BroadcastObject bob = (BroadcastObject) lob;
			// clean the partitioned broadcast
			if (bob.isPartitionedBroadcastValid()) {
				PartitionedBroadcast pbm = bob.getPartitionedBroadcast();
				if( pbm != null ) //robustness evictions
					pbm.destroy();
			}
			// clean the non-partitioned broadcast
			if (((BroadcastObject) lob).isNonPartitionedBroadcastValid()) {
				Broadcast<CacheableData> bc = bob.getNonPartitionedBroadcast();
				if( bc != null ) //robustness evictions
					cleanupBroadcastVariable(bc);
			}
			CacheableData.addBroadcastSize(-bob.getSize());
		}
	}

	/**
	 * This call destroys a broadcast variable at all executors and the driver.
	 * Hence, it is intended to be used on rmvar only. Depending on the
	 * ASYNCHRONOUS_VAR_DESTROY configuration, this is asynchronous or not.
	 *
	 * @param bvar broadcast variable
	 */
	public static void cleanupBroadcastVariable(Broadcast<?> bvar)
	{
		//In comparison to 'unpersist' (which would only delete the broadcast
		//from the executors), this call also deletes related data from the driver.
		if( bvar.isValid() ) {
			bvar.destroy( !ASYNCHRONOUS_VAR_DESTROY );
		}
	}

	/**
	 * This call removes an rdd variable from executor memory and disk if required.
	 * Hence, it is intended to be used on rmvar only. Depending on the
	 * ASYNCHRONOUS_VAR_DESTROY configuration, this is asynchronous or not.
	 *
	 * @param rvar rdd variable to remove
	 */
	public static void cleanupRDDVariable(JavaPairRDD<?,?> rvar)
	{
		if( rvar.getStorageLevel()!=StorageLevel.NONE() ) {
			rvar.unpersist( !ASYNCHRONOUS_VAR_DESTROY );
		}
	}

	@SuppressWarnings("unchecked")
	public void repartitionAndCacheMatrixObject( String var ) {
		MatrixObject mo = getMatrixObject(var);
		DataCharacteristics dcIn = mo.getDataCharacteristics();

		//double check size to avoid unnecessary spark context creation
		if( !OptimizerUtils.exceedsCachingThreshold(mo.getNumColumns(),
				OptimizerUtils.estimateSizeExactSparsity(dcIn)) )
			return;

		//get input rdd and default storage level
		JavaPairRDD<MatrixIndexes,MatrixBlock> in = (JavaPairRDD<MatrixIndexes, MatrixBlock>)
			getRDDHandleForMatrixObject(mo, FileFormat.BINARY);

		//avoid unnecessary repartitioning/caching if data already partitioned
		if( SparkUtils.isHashPartitioned(in) )
			return;
		
		//avoid unnecessary caching of input in order to reduce memory pressure
		if( mo.getRDDHandle().allowsShortCircuitRead()
			&& isRDDMarkedForCaching(in.id()) && !isRDDCached(in.id()) ) {
			in = (JavaPairRDD<MatrixIndexes,MatrixBlock>)
				((RDDObject)mo.getRDDHandle().getLineageChilds().get(0)).getRDD();

			//investigate issue of unnecessarily large number of partitions
			int numPartitions = SparkUtils.getNumPreferredPartitions(dcIn, in);
			if( numPartitions < in.getNumPartitions() )
				in = in.coalesce( numPartitions );
		}

		//repartition rdd (force creation of shuffled rdd via merge), note: without deep copy albeit
		//executed on the original data, because there will be no merge, i.e., no key duplicates
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = RDDAggregateUtils.mergeByKey(in, false);

		//convert mcsr into memory-efficient csr if potentially sparse
		if( OptimizerUtils.checkSparseBlockCSRConversion(dcIn) ) {
			out = out.mapValues(new CreateSparseBlockFunction(SparseBlock.Type.CSR));
		}

		//persist rdd in default storage level
		out.persist( Checkpoint.DEFAULT_STORAGE_LEVEL )
			.count(); //trigger caching to prevent contention

		//create new rdd handle, in-place of current matrix object
		RDDObject inro =  mo.getRDDHandle();  //guaranteed to exist (see above)
		RDDObject outro = new RDDObject(out); //create new rdd object
		outro.setCheckpointRDD(true);         //mark as checkpointed
		outro.addLineageChild(inro);          //keep lineage to prevent cycles on cleanup
		mo.setRDDHandle(outro);
	}

	@SuppressWarnings("unchecked")
	public void cacheMatrixObject( String var ) {
		//get input rdd and default storage level
		MatrixObject mo = getMatrixObject(var);

		//double check size to avoid unnecessary spark context creation
		if( !OptimizerUtils.exceedsCachingThreshold(mo.getNumColumns(),
				OptimizerUtils.estimateSizeExactSparsity(mo.getDataCharacteristics())) )
			return;

		JavaPairRDD<MatrixIndexes,MatrixBlock> in = (JavaPairRDD<MatrixIndexes, MatrixBlock>)
			getRDDHandleForMatrixObject(mo, FileFormat.BINARY);

		//persist rdd (force rdd caching, if not already cached)
		if( !isRDDCached(in.id()) )
			in.count(); //trigger caching to prevent contention
	}

	public int setThreadLocalSchedulerPool() {
		int pool = -1;
		if( FAIR_SCHEDULER_MODE ) {
			pool = allocSchedulerPoolName();
			getSparkContext().sc().setLocalProperty(
				"spark.scheduler.pool", "parforPool"+pool);
		}
		return pool;
	}

	public void cleanupThreadLocalSchedulerPool(int pool) {
		if( FAIR_SCHEDULER_MODE ) {
			freeSchedulerPoolName(pool);
			getSparkContext().sc().setLocalProperty(
				"spark.scheduler.pool", null);
		}
	}
	
	private static synchronized int allocSchedulerPoolName() {
		int pool = ArrayUtils.indexOf(_poolBuff, false);
		//grow pool on demand
		if( pool < 0 ) {
			pool = _poolBuff.length;
			_poolBuff = Arrays.copyOf(_poolBuff,
				(int)Math.min(2L*pool, Integer.MAX_VALUE));
		}
		//mark pool name for in use
		_poolBuff[pool] = true;
		return pool;
	}
	
	private static synchronized void freeSchedulerPoolName(int pool) {
		_poolBuff[pool] = false;
	}

	private boolean isRDDMarkedForCaching( int rddID ) {
		JavaSparkContext jsc = getSparkContext();
		return jsc.sc().getPersistentRDDs().contains(rddID);
	}

	public static boolean isRDDCached( int rddID ) {
		if (!isSparkContextCreated())
			return false;

		JavaSparkContext jsc = _spctx;
		//check that rdd is marked for caching
		if( !jsc.sc().getPersistentRDDs().contains(rddID) ) {
			return false;
		}

		//check that rdd is actually already cached
		for( RDDInfo info : jsc.sc().getRDDStorageInfo() ) {
			if( info.id() == rddID )
				return info.isCached();
		}
		return false;
	}

	public static long getMemCachedRDDSize(int rddID) {
		if (!isSparkContextCreated())
			return 0;

		JavaSparkContext jsc = _spctx;
		//check that rdd is marked for caching
		if( !jsc.sc().getPersistentRDDs().contains(rddID) )
			return 0;

		for (RDDInfo info : jsc.sc().getRDDStorageInfo()) {
			if (info.id() == rddID && info.isCached())
				return info.memSize(); //total size summing all executors
		}
		return 0;
	}

	public static long getStorageSpaceUsed() {
		//return the sum of the sizes of the cached RDDs in all executors
		if (!isSparkContextCreated())
			return 0;

		JavaSparkContext jsc = _spctx;
		return Arrays.stream(jsc.sc().getRDDStorageInfo()).mapToLong(RDDInfo::memSize).sum();
	}

	///////////////////////////////////////////
	// Spark configuration handling
	///////

	/**
	 * Obtains the lazily analyzed spark cluster configuration.
	 *
	 * @return spark cluster configuration
	 */
	public synchronized static SparkClusterConfig getSparkClusterConfig() {
		//lazy creation of spark cluster config
		if( _sconf == null )
			_sconf = new SparkClusterConfig();
		return _sconf;
	}

	/**
	 * Obtains the available memory budget for broadcast variables in bytes.
	 *
	 * @return broadcast memory budget
	 */
	public static double getBroadcastMemoryBudget() {
		return getSparkClusterConfig().getBroadcastMemoryBudget();
	}

	/**
	 * Obtain the available memory budget for data storage in bytes.
	 *
	 * @param min      flag for minimum data budget
	 * @param refresh  flag for refresh with spark context
	 * @return data memory budget
	 */
	public static double getDataMemoryBudget(boolean min, boolean refresh) {
		return getSparkClusterConfig()
			.getDataMemoryBudget(min, refresh);
	}

	/**
	 * Obtain the number of executors in the cluster (excluding the driver).
	 *
	 * @return number of executors
	 */
	public static int getNumExecutors() {
		return getSparkClusterConfig()
			.getNumExecutors();
	}

	/**
	 * Obtain the default degree of parallelism (cores in the cluster).
	 *
	 * @param refresh  flag for refresh with spark context
	 * @return default degree of parallelism
	 */
	public static int getDefaultParallelism(boolean refresh) {
		return getSparkClusterConfig()
			.getDefaultParallelism(refresh);
	}

	/**
	 * Captures relevant spark cluster configuration properties, e.g., memory budgets and
	 * degree of parallelism. This configuration abstracts legacy ({@literal <} Spark 1.6) and current
	 * configurations and provides a unified view.
	 */
	public static class SparkClusterConfig
	{
		//broadcasts are stored in mem-and-disk in storage space, this config
		//defines the fraction of min storage space to be used as broadcast budget
		private static final double BROADCAST_DATA_FRACTION = 0.70;
		private static final double BROADCAST_DATA_FRACTION_LEGACY = 0.35;

		//forward private config from Spark's UnifiedMemoryManager.scala (>1.6)
		public static final long RESERVED_SYSTEM_MEMORY_BYTES = 300 * 1024 * 1024;

		//meta configurations
		private boolean _legacyVersion = false; //spark version <1.6
		private boolean _confOnly = false; //infrastructure info based on config

		//memory management configurations
		private long _memExecutor = -1; //mem per executor
		private double _memDataMinFrac = -1; //minimum data fraction
		private double _memDataMaxFrac = -1; //maximum data fraction
		private double _memBroadcastFrac = -1; //broadcast fraction

		//degree of parallelism configurations
		private int _numExecutors = -1; //total executors
		private int _defaultPar = -1; //total vcores

		public SparkClusterConfig()
		{
			handleIllegalReflectiveAccessSpark();
			SparkConf sconf = createSystemDSSparkConf();
			_confOnly = true;

			//parse version and config
			String sparkVersion = getSparkVersionString();
			_legacyVersion = (UtilFunctions.compareVersion(sparkVersion, "1.6.0") < 0
					|| sconf.getBoolean("spark.memory.useLegacyMode", false) );

			//obtain basic spark configurations
			if( _legacyVersion )
				analyzeSparkConfiguationLegacy(sconf);
			else
				analyzeSparkConfiguation(sconf);

			//log debug of created spark cluster config
			if( LOG.isDebugEnabled() )
				LOG.debug( this.toString() );
		}

		// Meant to be used only resource optimization
		public SparkClusterConfig(SparkConf sconf)
		{
			_confOnly = true;

			//parse version and config
			String sparkVersion = CloudUtils.SPARK_VERSION;
			_legacyVersion = (UtilFunctions.compareVersion(sparkVersion, "1.6.0") < 0
					|| sconf.getBoolean("spark.memory.useLegacyMode", false) );

			//obtain basic spark configurations
			if( _legacyVersion )
				analyzeSparkConfiguationLegacy(sconf);
			else
				analyzeSparkConfiguation(sconf);
		}

		public long getBroadcastMemoryBudget() {
			return (long) (_memExecutor * _memBroadcastFrac);
		}

		public long getDataMemoryBudget(boolean min, boolean refresh) {
			//always get the current num executors on refresh because this might
			//change if not all executors are initially allocated and it is plan-relevant
			int numExec = _numExecutors;
			if( (refresh && !_confOnly) || isSparkContextCreated() ) {
				numExec = Math.max(getSparkContextStatic().sc()
					.getExecutorMemoryStatus().size() - 1, 1);
			}

			//compute data memory budget
			return (long) ( numExec * _memExecutor *
				(min ? _memDataMinFrac : _memDataMaxFrac) );
		}

		public int getNumExecutors() {
			if( _numExecutors < 0 )
				analyzeSparkParallelismConfiguation(null);
			return _numExecutors;
		}

		public int getDefaultParallelism(boolean refresh) {
			if( _defaultPar < 0 && !refresh )
				analyzeSparkParallelismConfiguation(null);

			//always get the current default parallelism on refresh because this might
			//change if not all executors are initially allocated and it is plan-relevant
			int par = ( (refresh && !_confOnly) || isSparkContextCreated() ) ?
				getSparkContextStatic().defaultParallelism() : _defaultPar;
			return Math.max(par, 1); //robustness min parallelism
		}

		public void analyzeSparkConfiguationLegacy(SparkConf conf)  {
			//ensure allocated spark conf
			SparkConf sconf = (conf == null) ? createSystemDSSparkConf() : conf;
			
			//parse absolute executor memory
			_memExecutor = UtilFunctions.parseMemorySize(
					sconf.get("spark.executor.memory", "1g"));

			//get data and shuffle memory ratios (defaults not specified in job conf)
			double dataFrac = sconf.getDouble("spark.storage.memoryFraction", 0.6); //default 60%
			_memDataMinFrac = dataFrac;
			_memDataMaxFrac = dataFrac;
			_memBroadcastFrac = dataFrac * BROADCAST_DATA_FRACTION_LEGACY; //default 18%

			//analyze spark degree of parallelism
			analyzeSparkParallelismConfiguation(sconf);
		}

		public void analyzeSparkConfiguation(SparkConf conf) {
			//ensure allocated spark conf
			SparkConf sconf = (conf == null) ? createSystemDSSparkConf() : conf;
			
			//parse absolute executor memory, incl fixed cut off
			_memExecutor = UtilFunctions.parseMemorySize(
					sconf.get("spark.executor.memory", "1g"))
					- RESERVED_SYSTEM_MEMORY_BYTES;

			//get data and shuffle memory ratios (defaults not specified in job conf)
			//first get the unified memory fraction (60%) comprising execution and storage
			double unifiedMem = sconf.getDouble("spark.memory.fraction", 0.6);
			//minimum default storage expressed as 50% of unified memory (= 30% of heap)
			_memDataMinFrac = unifiedMem * sconf.getDouble("spark.memory.storageFraction", 0.5);
			//storage memory can expand and take up the full unified memory
			_memDataMaxFrac = unifiedMem;
			//Heuristic-based broadcast fraction (70% of min storage = 21% of heap)
			_memBroadcastFrac = _memDataMinFrac * BROADCAST_DATA_FRACTION;
			//analyze spark degree of parallelism
			analyzeSparkParallelismConfiguation(sconf);
		}

		private void analyzeSparkParallelismConfiguation(SparkConf conf) {
			//ensure allocated spark conf
			SparkConf sconf = (conf == null) ? createSystemDSSparkConf() : conf;
			
			int numExecutors = sconf.getInt("spark.executor.instances", -1);
			int numCoresPerExec = sconf.getInt("spark.executor.cores", -1);
			int defaultPar = sconf.getInt("spark.default.parallelism", -1);

			if( numExecutors > 1 && (defaultPar > 1 || numCoresPerExec > 1) ) {
				_numExecutors = numExecutors;
				_defaultPar = (defaultPar>1) ? defaultPar : numExecutors * numCoresPerExec;
				_confOnly &= true;
			}
			else if( DMLScript.USE_LOCAL_SPARK_CONFIG ) {
				//avoid unnecessary spark context creation in local mode (e.g., tests, resource opt.)
				_numExecutors = 1;
				_defaultPar = 2;
				_confOnly &= true;
			}
			else if (ConfigurationManager.getCompilerConfigFlag(CompilerConfig.ConfigType.RESOURCE_OPTIMIZATION)) {
				_numExecutors = numExecutors;
				_defaultPar = numExecutors * numCoresPerExec;
				_confOnly = true;
			}
			else {
				//get default parallelism (total number of executors and cores)
				//note: spark context provides this information while conf does not
				//(for num executors we need to correct for driver and local mode)
				JavaSparkContext jsc = getSparkContextStatic();
				_numExecutors = Math.max(jsc.sc().getExecutorMemoryStatus().size() - 1, 1);
				_defaultPar = jsc.defaultParallelism();
				_confOnly &= false; //implies env info refresh w/ spark context
			}
		}

		/**
		 * Obtains the spark version string. If the spark context has been created,
		 * we simply get it from the context; otherwise, we use Spark internal
		 * constants to avoid creating the spark context just for the version.
		 *
		 * @return spark version string
		 */
		private static String getSparkVersionString() {
			//check for existing spark context
			if( isSparkContextCreated() )
				return getSparkContextStatic().version();

			//use spark internal constant to avoid context creation
			return org.apache.spark.package$.MODULE$.SPARK_VERSION();
		}

		@Override
		public String toString() {
			StringBuilder sb = new StringBuilder("SparkClusterConfig: \n");
			sb.append("-- legacyVersion    = " + _legacyVersion + " ("+getSparkVersionString()+")\n" );
			sb.append("-- confOnly         = " + _confOnly + "\n");
			sb.append("-- numExecutors     = " + _numExecutors + "\n");
			sb.append("-- defaultPar       = " + _defaultPar + "\n");
			sb.append("-- memExecutor      = " + _memExecutor + "\n");
			sb.append("-- memDataMinFrac   = " + _memDataMinFrac + "\n");
			sb.append("-- memDataMaxFrac   = " + _memDataMaxFrac + "\n");
			sb.append("-- memBroadcastFrac = " + _memBroadcastFrac + "\n");
			return sb.toString();
		}
	}

	public static class MemoryManagerParRDDs
	{
		private final long _limit;
		private long _size;
		private HashMap<Integer, Long> _rdds;

		public MemoryManagerParRDDs(double fractionMem) {
			_limit = (long)(fractionMem * InfrastructureAnalyzer.getLocalMaxMemory());
			_size = 0;
			_rdds = new HashMap<>();
		}

		public synchronized boolean reserve(long rddSize) {
			boolean ret = (rddSize + _size < _limit);
			_size += ret ? rddSize : 0;
			return ret;
		}

		public synchronized void registerRDD(int rddID, long rddSize, boolean reserved) {
			if( !reserved ) {
				throw new RuntimeException("Unsupported rdd registration "
						+ "without size reservation for "+rddSize+" bytes.");
			}
			_rdds.put(rddID, rddSize);
		}

		public synchronized void deregisterRDD(int rddID) {
			Long rddSize = _rdds.remove(rddID);
			_size -= (rddSize!=null) ? rddSize : 0;
		}

		public synchronized void clear() {
			_size = 0;
			_rdds.clear();
		}
	}
}
