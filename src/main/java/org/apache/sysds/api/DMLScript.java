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

package org.apache.sysds.api;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.InetAddress;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Map;
import java.util.Scanner;

import org.apache.commons.cli.AlreadySelectedException;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.codegen.SpoofCompiler;
import org.apache.sysds.hops.codegen.SpoofCompiler.GeneratorAPI;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.DMLTranslator;
import org.apache.sysds.parser.LanguageException;
import org.apache.sysds.parser.ParseException;
import org.apache.sysds.parser.ParserFactory;
import org.apache.sysds.parser.ParserWrapper;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.DMLScriptException;
import org.apache.sysds.runtime.codegen.CodegenUtils;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedData;
import org.apache.sysds.runtime.controlprogram.federated.FederatedWorker;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.FederatedMonitoringServer;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.CoordinatorModel;
import org.apache.sysds.runtime.controlprogram.parfor.util.IDHandler;
import org.apache.sysds.runtime.instructions.gpu.context.GPUContextPool;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.lineage.LineageCacheConfig;
import org.apache.sysds.runtime.lineage.LineageCacheConfig.LineageCachePolicy;
import org.apache.sysds.runtime.lineage.LineageCacheConfig.ReuseCacheType;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.runtime.util.LocalFileUtils;
import org.apache.sysds.utils.Explain;
import org.apache.sysds.utils.Explain.ExplainCounts;
import org.apache.sysds.utils.Explain.ExplainType;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;
import org.apache.sysds.utils.NativeHelper;
import org.apache.sysds.utils.SettingsChecker;
import org.apache.sysds.utils.Statistics;

import com.fasterxml.jackson.databind.ObjectMapper;

public class DMLScript 
{
	// Set the execution mode
	private static ExecMode   EXEC_MODE                  = DMLOptions.defaultOptions.execMode;
	// Enable/disable to print statistics
	public static boolean     STATISTICS                 = DMLOptions.defaultOptions.stats;
	// Enable/disable to print statistics n-grams
	public static boolean     STATISTICS_NGRAMS          = DMLOptions.defaultOptions.statsNGrams;
	// Enable/disable to gather memory use stats in JMLC
	public static boolean     JMLC_MEM_STATISTICS        = false;
	// Set maximum heavy hitter count
	public static int         STATISTICS_COUNT           = DMLOptions.defaultOptions.statsCount;
	// The sizes of recorded n-gram tuples
	public static int[]         STATISTICS_NGRAM_SIZES   = DMLOptions.defaultOptions.statsNGramSizes;
	// Set top k displayed n-grams limit
	public static int         STATISTICS_TOP_K_NGRAMS    = DMLOptions.defaultOptions.statsTopKNGrams;
	// Set if N-Grams use lineage for data-dependent tracking
	public static boolean     STATISTICS_NGRAMS_USE_LINEAGE = DMLOptions.defaultOptions.statsNGramsUseLineage;
	// Set statistics maximum wrap length
	public static int         STATISTICS_MAX_WRAP_LEN    = 30;
	// Enable/disable to print federated statistics
	public static boolean     FED_STATISTICS             = DMLOptions.defaultOptions.fedStats;
	// Set federated statistics maximum heavy hitter count
	public static int         FED_STATISTICS_COUNT       = DMLOptions.defaultOptions.fedStatsCount;
	// Enable/disable this instance is a federated worker
	public static boolean     FED_WORKER                 = DMLOptions.defaultOptions.fedWorker;
	// Set explain type
	public static ExplainType EXPLAIN                    = DMLOptions.defaultOptions.explainType;
	// Set filename of dml script
	public static String      DML_FILE_PATH_ANTLR_PARSER = DMLOptions.defaultOptions.filePath;
	// Set data type to use internally
	public static String      FLOATING_POINT_PRECISION   = "double";
	// Enable/disable to print GPU memory-related information
	public static boolean     PRINT_GPU_MEMORY_INFO      = false;
	// Set maximum number of bytes to use for shadow buffer
	public static long        EVICTION_SHADOW_BUFFER_MAX_BYTES = 0;
	// Set number of bytes to use for shadow buffer
	public static long        EVICTION_SHADOW_BUFFER_CURR_BYTES = 0;
	// Set fraction of available GPU memory to use
	public static double      GPU_MEMORY_UTILIZATION_FACTOR = 0.9;
	// Set GPU memory allocator to use
	public static String      GPU_MEMORY_ALLOCATOR       = "cuda";
	// Enable/disable lineage tracing
	public static boolean     LINEAGE                    = DMLOptions.defaultOptions.lineage;
	// Enable/disable deduplicate lineage items
	public static boolean     LINEAGE_DEDUP              = DMLOptions.defaultOptions.lineage_dedup;
	// Enable/disable lineage-based reuse
	public static ReuseCacheType LINEAGE_REUSE           = DMLOptions.defaultOptions.linReuseType;
	// Set lineage cache eviction policy
	public static LineageCachePolicy LINEAGE_POLICY      = DMLOptions.defaultOptions.linCachePolicy;
	// Enable/disable estimate reuse benefits
	public static boolean     LINEAGE_ESTIMATE           = DMLOptions.defaultOptions.lineage_estimate;
	// Enable/disable lineage debugger
	public static boolean     LINEAGE_DEBUGGER           = DMLOptions.defaultOptions.lineage_debugger;
	// Set accelerator
	public static boolean           USE_ACCELERATOR      = DMLOptions.defaultOptions.gpu;
	public static boolean           FORCE_ACCELERATOR    = DMLOptions.defaultOptions.forceGPU;
	// Enable synchronizing GPU after every instruction
	public static boolean           SYNCHRONIZE_GPU      = true;
	// Set OOC backend
	public static boolean           USE_OOC              = DMLOptions.defaultOptions.ooc;
	// Enable eager CUDA free on rmvar
	public static boolean           EAGER_CUDA_FREE      = false;

	// Global seed 
	public static int               SEED                 = -1;

	public static String MONITORING_ADDRESS = null;

	// flag that indicates whether or not to suppress any prints to stdout
	public static boolean _suppressPrint2Stdout = false;
	//set default local spark configuration - used for local testing
	public static boolean USE_LOCAL_SPARK_CONFIG = false;
	public static boolean _activeAM = false;
	/**
	 * If true, allow DMLProgram to be generated while not halting due to validation errors/warnings
	 */
	public static boolean VALIDATOR_IGNORE_ISSUES = false;

	public static String _uuid = IDHandler.createDistributedUniqueID();
	private static final Log LOG = LogFactory.getLog(DMLScript.class.getName());

	///////////////////////////////
	// public external interface
	////////
	
	public static String getUUID() {
		return _uuid;
	}

	/**
	 * Used to set master UUID on all nodes (in parfor remote, where DMLScript passed) 
	 * in order to simplify cleanup of scratch_space and local working dirs.
	 * 
	 * @param uuid master UUID to set on all nodes
	 */
	public static void setUUID(String uuid) 
	{
		_uuid = uuid;
	}
	
	public static boolean suppressPrint2Stdout() {
		return _suppressPrint2Stdout;
	}
	
	public static void setActiveAM(){
		_activeAM = true;
	}
	
	public static boolean isActiveAM(){
		return _activeAM;
	}

	/**
	 * Main entry point for systemDS dml script execution
	 *
	 * @param args command-line arguments
	 */
	public static void main(String[] args)
	{
		try{
			DMLScript.executeScript(args);
		} catch(Exception e){
			for(String s: args){
				if(s.trim().contains("-debug")){
					e.printStackTrace();
					return;
				}
			}
			errorPrint(e);
		}
	}

	/**
	 * Single entry point for all public invocation alternatives (e.g.,
	 * main, executeScript, JaqlUdf etc)
	 * 
	 * @param args arguments
	 * @return true if success, false otherwise
	 * @throws IOException If an internal IOException happens.
	 */
	public static boolean executeScript( String[] args )
		throws IOException, ParseException, DMLScriptException
	{
		//parse arguments and set execution properties
		ExecMode oldrtplatform  = EXEC_MODE;  //keep old rtplatform
		ExplainType oldexplain  = EXPLAIN;     //keep old explain

		DMLOptions dmlOptions = null;
		
		try{
			dmlOptions = DMLOptions.parseCLArguments(args);
		}
		catch(AlreadySelectedException e) {
			LOG.error("Mutually exclusive options were selected. " + e.getMessage());
			//TODO fix null default options
			//HelpFormatter formatter = new HelpFormatter();
			//formatter.printHelp( "systemds", DMLOptions.defaultOptions.options );
			return false;
		}
		catch(org.apache.commons.cli.ParseException e) {
			LOG.error("Parsing Exception " + e.getMessage());
			//TODO fix null default options
			//HelpFormatter formatter = new HelpFormatter();
			//formatter.printHelp( "systemds", DMLOptions.defaultOptions.options );
			return false;
		}

		try
		{
			STATISTICS            = dmlOptions.stats;
			STATISTICS_COUNT      = dmlOptions.statsCount;
			STATISTICS_NGRAMS     = dmlOptions.statsNGrams;
			STATISTICS_NGRAM_SIZES = dmlOptions.statsNGramSizes;
			STATISTICS_TOP_K_NGRAMS = dmlOptions.statsTopKNGrams;
			FED_STATISTICS        = dmlOptions.fedStats;
			FED_STATISTICS_COUNT  = dmlOptions.fedStatsCount;
			JMLC_MEM_STATISTICS   = dmlOptions.memStats;
			USE_ACCELERATOR       = dmlOptions.gpu;
			FORCE_ACCELERATOR     = dmlOptions.forceGPU;
			USE_OOC	              = dmlOptions.ooc;
			EXPLAIN               = dmlOptions.explainType;
			EXEC_MODE             = dmlOptions.execMode;
			LINEAGE               = dmlOptions.lineage;
			LINEAGE_DEDUP         = dmlOptions.lineage_dedup;
			LINEAGE_REUSE         = dmlOptions.linReuseType;
			LINEAGE_POLICY        = dmlOptions.linCachePolicy;
			LINEAGE_ESTIMATE      = dmlOptions.lineage_estimate;
			LINEAGE_DEBUGGER      = dmlOptions.lineage_debugger;
			SEED                  = dmlOptions.seed;


			String fnameOptConfig = dmlOptions.configFile;
			boolean isFile = dmlOptions.filePath != null;
			String fileOrScript = isFile ? dmlOptions.filePath : dmlOptions.script;

			boolean help = dmlOptions.help;

			SettingsChecker.check();

			if (help) {
				HelpFormatter formatter = new HelpFormatter();
				formatter.printHelp( "systemds", dmlOptions.options );
				return true;
			}

			if (dmlOptions.clean) {
				cleanSystemDSWorkspace();
				return true;
			}
			
			if(dmlOptions.fedWorker) {
				loadConfiguration(fnameOptConfig);
				new FederatedWorker(dmlOptions.fedWorkerPort, dmlOptions.debug);
				return true;
			}

			if(dmlOptions.fedMonitoring) {
				loadConfiguration(fnameOptConfig);
				new FederatedMonitoringServer(dmlOptions.fedMonitoringPort, dmlOptions.debug);
				return true;
			}

			if(dmlOptions.fedMonitoringAddress != null) {
				MONITORING_ADDRESS = dmlOptions.fedMonitoringAddress;
			}

			LineageCacheConfig.setConfig(LINEAGE_REUSE);
			LineageCacheConfig.setCachePolicy(LINEAGE_POLICY);
			LineageCacheConfig.setEstimator(LINEAGE_ESTIMATE);

			String dmlScriptStr = readDMLScript(isFile, fileOrScript);
			Map<String, String> argVals = dmlOptions.argVals;

			DML_FILE_PATH_ANTLR_PARSER = dmlOptions.filePath;
			
			//Step 3: invoke dml script
			printInvocationInfo(fileOrScript, fnameOptConfig, argVals);
			execute(dmlScriptStr, fnameOptConfig, argVals, args);
		}
		finally {
			//reset runtime platform and visualize flag
			setGlobalExecMode(oldrtplatform);
			EXPLAIN = oldexplain;
			CommonThreadPool.shutdownAsyncPools();
		}
		
		return true;
	}

	/**
	 * Reads the DML/PyDML script into a String
	 * @param isFile	Whether the string argument is a path to a file or the script itself
	 * @param scriptOrFilename script or filename
	 * @return a string representation of the script
	 * @throws IOException	if error
	 */
	public static String readDMLScript( boolean isFile, String scriptOrFilename )
		throws IOException
	{
		String dmlScriptStr;
		
		if( isFile )
		{
			String fileName = scriptOrFilename;
			//read DML script from file
			if(fileName == null)
				throw new LanguageException("DML script path was not specified!");
			
			StringBuilder sb = new StringBuilder();
			BufferedReader in = null;
			try 
			{
				//read from hdfs or gpfs file system
				if(    fileName.startsWith("hdfs:") || fileName.startsWith("gpfs:")
					|| IOUtilFunctions.isObjectStoreFileScheme(new Path(fileName)) )
				{ 
					Path scriptPath = new Path(fileName);
					FileSystem fs = IOUtilFunctions.getFileSystem(scriptPath);
					in = new BufferedReader(new InputStreamReader(fs.open(scriptPath)));
				}
				// from local file system
				else { 
					in = new BufferedReader(new FileReader(fileName));
				}
				
				//core script reading
				String tmp = null;
				while ((tmp = in.readLine()) != null) {
					sb.append( tmp );
					sb.append( "\n" );
				}
			}
			catch (IOException ex) {
				LOG.error("Failed to read the script from the file system", ex);
				throw ex;
			}
			finally {
				IOUtilFunctions.closeSilently(in);
			}
			
			dmlScriptStr = sb.toString();
		}
		else
		{
			String scriptString = scriptOrFilename;
			//parse given script string 
			if(scriptString == null)
				throw new LanguageException("DML script was not specified!");
			
			InputStream is = new ByteArrayInputStream(scriptString.getBytes());
			try( Scanner scan = new Scanner(is) ) {
				dmlScriptStr = scan.useDelimiter("\\A").next();	
			}
		}
		
		return dmlScriptStr;
	}
	
	///////////////////////////////
	// private internal interface 
	// (core compilation and execute)
	////////

	public static void loadConfiguration(String fnameOptConfig) throws IOException {
		DMLConfig dmlconf = DMLConfig.readConfigurationFile(fnameOptConfig);
		ConfigurationManager.setGlobalConfig(dmlconf);
		CompilerConfig cconf = OptimizerUtils.constructCompilerConfig(dmlconf);
		ConfigurationManager.setGlobalConfig(cconf);
		LOG.debug("\nDML config: \n" + dmlconf.getConfigInfo());
		setGlobalFlags(dmlconf);
	}

	/**
	 * The running body of DMLScript execution. This method should be called after execution properties have been correctly set,
	 * and customized parameters have been put into _argVals
	 * 
	 * @param dmlScriptStr DML script string
	 * @param fnameOptConfig configuration file
	 * @param argVals map of argument values
	 * @param allArgs arguments
	 * @throws IOException if IOException occurs
	 */
	private static void execute(String dmlScriptStr, String fnameOptConfig, Map<String,String> argVals, String[] allArgs)
		throws IOException
	{
		// print basic time, environment info, and process id
		printStartExecInfo(dmlScriptStr);

		// optionally register for monitoring
		registerForMonitoring();
		
		//Step 1: parse configuration files & write any configuration specific global variables
		loadConfiguration(fnameOptConfig);

		//Step 2: configure codegen
		configureCodeGen();

		//Step 3: parse dml script
		Statistics.startCompileTimer();
		ParserWrapper parser = ParserFactory.createParser();
		DMLProgram prog = parser.parse(DML_FILE_PATH_ANTLR_PARSER, dmlScriptStr, argVals);
		
		//Step 4: construct HOP DAGs (incl LVA, validate, and setup)
		DMLTranslator dmlt = new DMLTranslator(prog);
		dmlt.liveVariableAnalysis(prog);
		dmlt.validateParseTree(prog);
		dmlt.constructHops(prog);
		
		//init working directories (before usage by following compilation steps)
		initHadoopExecution( ConfigurationManager.getDMLConfig() );
	
		//Step 5: rewrite HOP DAGs (incl IPA and memory estimates)
		dmlt.rewriteHopsDAG(prog);
		
		//Step 6: construct lops (incl exec type and op selection)
		dmlt.constructLops(prog);

		//Step 7: rewrite LOP DAGs (incl adding new LOPs s.a. prefetch, broadcast)
		dmlt.rewriteLopDAG(prog);
		
		//Step 8: generate runtime program, incl codegen
		Program rtprog = dmlt.getRuntimeProgram(prog, ConfigurationManager.getDMLConfig());
		
		//Step 9: prepare statistics [and optional explain output]
		//count number compiled MR jobs / SP instructions	
		ExplainCounts counts = Explain.countDistributedOperations(rtprog);
		Statistics.resetNoOfCompiledJobs( counts.numJobs );
		
		//explain plan of program (hops or runtime)
		if( EXPLAIN != ExplainType.NONE )
			System.out.println(Explain.display(prog, rtprog, EXPLAIN, counts));
		
		Statistics.stopCompileTimer();
		
		//double costs = CostEstimationWrapper.getTimeEstimate(rtprog, ExecutionContextFactory.createContext());
		//System.out.println("Estimated costs: "+costs);
		
		//Step 10: execute runtime program
		ExecutionContext ec = null;
		try {
			ec = ExecutionContextFactory.createContext(rtprog);
			ScriptExecutorUtils.executeRuntimeProgram(rtprog, ec, ConfigurationManager.getDMLConfig(), STATISTICS ? STATISTICS_COUNT : 0, null);
		}
		finally {
			//cleanup scratch_space and all working dirs
			cleanupHadoopExecution(ConfigurationManager.getDMLConfig());
			FederatedData.clearWorkGroup();
			//stop spark context (after cleanup of federated workers and other pools,
			//otherwise federated spark cleanups in local tests throw errors in same JVM)
			if(ec != null && ec instanceof SparkExecutionContext)
				((SparkExecutionContext) ec).close();
			LOG.info("END DML run " + getDateTime() );
		}
	}

	/**
	 * Sets the global flags in DMLScript based on user provided configuration
	 * 
	 * @param dmlconf user provided configuration
	 */
	public static void setGlobalFlags(DMLConfig dmlconf) {
		// Sets the GPUs to use for this process (a range, all GPUs, comma separated list or a specific GPU)
		GPUContextPool.AVAILABLE_GPUS = dmlconf.getTextValue(DMLConfig.AVAILABLE_GPUS);
		DMLScript.STATISTICS_MAX_WRAP_LEN = dmlconf.getIntValue(DMLConfig.STATS_MAX_WRAP_LEN);
		NativeHelper.initialize(dmlconf.getTextValue(DMLConfig.NATIVE_BLAS_DIR), dmlconf.getTextValue(DMLConfig.NATIVE_BLAS).trim());
		DMLScript.SYNCHRONIZE_GPU = dmlconf.getBooleanValue(DMLConfig.SYNCHRONIZE_GPU);
		DMLScript.EAGER_CUDA_FREE = dmlconf.getBooleanValue(DMLConfig.EAGER_CUDA_FREE);
		DMLScript.PRINT_GPU_MEMORY_INFO = dmlconf.getBooleanValue(DMLConfig.PRINT_GPU_MEMORY_INFO);
		DMLScript.GPU_MEMORY_UTILIZATION_FACTOR = dmlconf.getDoubleValue(DMLConfig.GPU_MEMORY_UTILIZATION_FACTOR);
		DMLScript.GPU_MEMORY_ALLOCATOR = dmlconf.getTextValue(DMLConfig.GPU_MEMORY_ALLOCATOR);
		if(DMLScript.GPU_MEMORY_UTILIZATION_FACTOR < 0) {
			throw new RuntimeException("Incorrect value (" + DMLScript.GPU_MEMORY_UTILIZATION_FACTOR + ") for the configuration:" + DMLConfig.GPU_MEMORY_UTILIZATION_FACTOR);
		}
		DMLScript.USE_LOCAL_SPARK_CONFIG |= dmlconf.getBooleanValue(DMLConfig.USE_LOCAL_SPARK_CONFIG);
		DMLScript.FLOATING_POINT_PRECISION = dmlconf.getTextValue(DMLConfig.FLOATING_POINT_PRECISION);
		org.apache.sysds.runtime.matrix.data.LibMatrixCUDA.resetFloatingPointPrecision();
		if(DMLScript.FLOATING_POINT_PRECISION.equals("double")) {
			DMLScript.EVICTION_SHADOW_BUFFER_MAX_BYTES = 0;
		}
		else {
			double shadowBufferSize = dmlconf.getDoubleValue(DMLConfig.EVICTION_SHADOW_BUFFERSIZE);
			if(shadowBufferSize < 0 || shadowBufferSize > 1) 
				throw new RuntimeException("Incorrect value (" + shadowBufferSize + ") for the configuration:" + DMLConfig.EVICTION_SHADOW_BUFFERSIZE);
			DMLScript.EVICTION_SHADOW_BUFFER_MAX_BYTES = (long) (InfrastructureAnalyzer.getLocalMaxMemory()*shadowBufferSize);
			if(DMLScript.EVICTION_SHADOW_BUFFER_MAX_BYTES > 0 && 
					DMLScript.EVICTION_SHADOW_BUFFER_CURR_BYTES > DMLScript.EVICTION_SHADOW_BUFFER_MAX_BYTES) {
				// This will be printed in a very rare situation when:
				// 1. There is a memory leak which leads to non-cleared shadow buffer OR
				// 2. MLContext is registering to bunch of outputs that are all part of shadow buffer
				System.out.println("WARN: Cannot use the shadow buffer due to potentially cached GPU objects. Current shadow buffer size (in bytes):" 
					+ DMLScript.EVICTION_SHADOW_BUFFER_CURR_BYTES + " > Max shadow buffer size (in bytes):" + DMLScript.EVICTION_SHADOW_BUFFER_MAX_BYTES);
			}
		}
	}
	
	public static void initHadoopExecution( DMLConfig config ) 
		throws IOException, ParseException, DMLRuntimeException
	{
		//create scratch space with appropriate permissions
		String scratch = config.getTextValue(DMLConfig.SCRATCH_SPACE);
		HDFSTool.createDirIfNotExistOnHDFS(scratch, DMLConfig.DEFAULT_SHARED_DIR_PERMISSION);
		
		//cleanup working dirs from previous aborted runs with same pid in order to prevent conflicts
		cleanupHadoopExecution(config); 
		
		//init caching (incl set active)
		LocalFileUtils.createWorkingDirectory();
		CacheableData.initCaching();
		
		//reset statistics (required if multiple scripts executed in one JVM)
		Statistics.resetNoOfExecutedJobs();
		if( STATISTICS )
			Statistics.reset();
	}
	
	public static void cleanupHadoopExecution( DMLConfig config ) 
		throws IOException, ParseException
	{
		//create dml-script-specific suffix
		StringBuilder sb = new StringBuilder();
		sb.append(Lop.FILE_SEPARATOR);
		sb.append(Lop.PROCESS_PREFIX);
		sb.append(DMLScript.getUUID());
		String dirSuffix = sb.toString();
		
		//0) cleanup federated workers if necessary
		FederatedData.clearFederatedWorkers();
		
		//1) cleanup scratch space (everything for current uuid)
		//(required otherwise export to hdfs would skip assumed unnecessary writes if same name)
		HDFSTool.deleteFileIfExistOnHDFS( config.getTextValue(DMLConfig.SCRATCH_SPACE) + dirSuffix );
		
		//2) cleanup systemds-internal working dirs
		CacheableData.cleanupCacheDir(); //might be local/hdfs
		LocalFileUtils.cleanupWorkingDirectory();
	}

	
	///////////////////////////////
	// private internal helper functionalities
	////////

	private static void printInvocationInfo(String fnameScript, String fnameOptConfig, Map<String,String> argVals) {
		LOG.debug("****** args to DML Script ******\n" + "UUID: " + getUUID() + "\n" + "SCRIPT PATH: " + fnameScript + "\n" 
			+ "RUNTIME: " + getGlobalExecMode() + "\n" + "BUILTIN CONFIG: " + DMLConfig.DEFAULT_SYSTEMDS_CONFIG_FILEPATH + "\n"
			+ "OPTIONAL CONFIG: " + fnameOptConfig + "\n");
		if( !argVals.isEmpty() ) {
			LOG.debug("Script arguments are: \n");
			for (int i=1; i<= argVals.size(); i++)
				LOG.debug("Script argument $" + i + " = " + argVals.get("$" + i) );
		}
	}
	
	private static void printStartExecInfo(String dmlScriptString) {
		boolean info = LOG.isInfoEnabled();
		boolean debug = LOG.isDebugEnabled();
		if(info)
			LOG.info("BEGIN DML run " + getDateTime());
		if(debug)
			LOG.debug("DML script: \n" + dmlScriptString);
		if(info)
			LOG.info("Process id:  " + IDHandler.getProcessID());
	}

	private static void registerForMonitoring() {

		if (MONITORING_ADDRESS != null && !MONITORING_ADDRESS.isBlank() && !MONITORING_ADDRESS.isEmpty()) {
			try {

				String uriString = MONITORING_ADDRESS + "/coordinators";

				ObjectMapper objectMapper = new ObjectMapper();

				var model = new CoordinatorModel();
				model.name = InetAddress.getLocalHost().getHostName();
				// TODO fix and replace localhost identifyer with hostname in federated instructions SYSTEMDS-3440
				// https://issues.apache.org/jira/browse/SYSTEMDS-3440
				model.host = "localhost"; 
				model.processId = IDHandler.getProcessID();

				String requestBody = objectMapper
					.writerWithDefaultPrettyPrinter()
					.writeValueAsString(model);

				var client = HttpClient.newHttpClient();
				var request = HttpRequest.newBuilder(URI.create(uriString))
						.header("accept", "application/json")
						.POST(HttpRequest.BodyPublishers.ofString(requestBody))
						.build();

				client.send(request, HttpResponse.BodyHandlers.ofString());
			} catch (IOException | InterruptedException e) {
				throw new RuntimeException(e);
			}
		}
	}
	
	private static String getDateTime() {
		DateFormat dateFormat = new SimpleDateFormat("MM/dd/yyyy HH:mm:ss");
		Date date = new Date();
		return dateFormat.format(date);
	}

	private static void cleanSystemDSWorkspace() {
		try {
			//read the default config
			DMLConfig conf = DMLConfig.readConfigurationFile(null);
			
			//cleanup scratch space (on HDFS)
			String scratch = conf.getTextValue(DMLConfig.SCRATCH_SPACE);
			if( scratch != null )
				HDFSTool.deleteFileIfExistOnHDFS(scratch);
			
			//cleanup local working dir
			String localtmp = conf.getTextValue(DMLConfig.LOCAL_TMP_DIR);
			if( localtmp != null )
				LocalFileUtils.cleanupRcWorkingDirectory(localtmp);
		}
		catch(Exception ex) {
			throw new DMLException("Failed to run SystemDS workspace cleanup.", ex);
		}
	}

	public static ExecMode getGlobalExecMode() {
		return EXEC_MODE;
	}
	
	public static void setGlobalExecMode(ExecMode mode) {
		EXEC_MODE = mode;
	}

	/**
	 * Print the error in a user friendly manner.
	 *
	 * @param e The exception thrown.
	 */
	public static void errorPrint(Exception e){
		final String ANSI_RED = "\u001B[31m";
		final String ANSI_RESET = "\u001B[0m";
		StringBuilder sb = new StringBuilder();
		sb.append(ANSI_RED + "\n");
		sb.append("An Error Occurred : ");
		sb.append("\n" );
		sb.append(StringUtils.leftPad(e.getClass().getSimpleName(),25));
		sb.append(" -- ");
		sb.append(e.getMessage());
		Throwable s =  e.getCause();
		while(s != null){
			sb.append("\n" );
			sb.append(StringUtils.leftPad(s.getClass().getSimpleName(),25));
			sb.append(" -- ");
			sb.append(s.getMessage());
			s = s.getCause();
		}
		sb.append("\n" + ANSI_RESET);
		System.out.println(sb.toString());
	}

	private static void configureCodeGen() {
		// load native codegen if configured
		if(ConfigurationManager.isCodegenEnabled()) {
			GeneratorAPI configured_generator = GeneratorAPI.valueOf(
				ConfigurationManager.getDMLConfig().getTextValue(DMLConfig.CODEGEN_API).toUpperCase());
			try {
				SpoofCompiler.loadNativeCodeGenerator(configured_generator);
			}
			catch(Exception e) {
				LOG.error("Failed to load native cuda codegen library\n" + e);
			}
		}
		// set the global class loader to make Spark's MutableURLClassLoader
		// available for all parfor worker threads without pass-through
		CodegenUtils.setClassLoader(Thread.currentThread().getContextClassLoader());
	}
}
