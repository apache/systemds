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

package org.apache.sysml.api;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URI;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.HashSet;
import java.util.Map;
import java.util.Scanner;

import org.apache.commons.cli.AlreadySelectedException;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.lang.StringUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.security.UserGroupInformation;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.sysml.api.mlcontext.ScriptType;
import org.apache.sysml.conf.CompilerConfig;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.conf.DMLOptions;
import org.apache.sysml.debug.DMLDebugger;
import org.apache.sysml.debug.DMLDebuggerProgramInfo;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.DMLTranslator;
import org.apache.sysml.parser.LanguageException;
import org.apache.sysml.parser.ParseException;
import org.apache.sysml.parser.ParserFactory;
import org.apache.sysml.parser.ParserWrapper;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLScriptException;
import org.apache.sysml.runtime.controlprogram.Program;
import org.apache.sysml.runtime.controlprogram.caching.CacheableData;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.controlprogram.parfor.util.IDHandler;
import org.apache.sysml.runtime.instructions.gpu.context.GPUContextPool;
import org.apache.sysml.runtime.io.IOUtilFunctions;
import org.apache.sysml.runtime.matrix.CleanupMR;
import org.apache.sysml.runtime.matrix.mapred.MRConfigurationNames;
import org.apache.sysml.runtime.matrix.mapred.MRJobConfiguration;
import org.apache.sysml.runtime.util.LocalFileUtils;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.utils.Explain;
import org.apache.sysml.utils.NativeHelper;
import org.apache.sysml.utils.Explain.ExplainCounts;
import org.apache.sysml.utils.Explain.ExplainType;
import org.apache.sysml.utils.Statistics;
import org.apache.sysml.yarn.DMLAppMasterUtils;
import org.apache.sysml.yarn.DMLYarnClientProxy;


public class DMLScript 
{	
	public enum RUNTIME_PLATFORM { 
		HADOOP,         // execute all matrix operations in MR
		SINGLE_NODE,    // execute all matrix operations in CP
		HYBRID,         // execute matrix operations in CP or MR
		HYBRID_SPARK,   // execute matrix operations in CP or Spark
		SPARK           // execute matrix operations in Spark
	}
	
	/**
	 * Eviction policies for eviction of GPU objects.
	 */
	public enum EvictionPolicy {
		LRU, 				// Evict the least recently used GPUObject. 
		LFU, 				// Evict the least frequently used GPUObject. 
		MIN_EVICT,
		MRU, 				// http://www.vldb.org/conf/1985/P127.PDF
		ALIGN_MEMORY
		// TODO:
		// ARC, // https://dbs.uni-leipzig.de/file/ARC.pdf
		// LOOP_AWARE 		// different policies for operations in for/while/parfor loop vs out-side the loop
	}
	
	// TODO: Anthony
	public static boolean           JMLC_MEM_STATISTICS = false;                                 // whether to gather memory use stats in JMLC
	
	// debug mode is deprecated and will be removed soon.
	public static boolean           ENABLE_DEBUG_MODE   = DMLOptions.defaultOptions.debug;       // debug mode
	
	// These flags are not used by JMLC and hence are not cleaned up. We can revisit this later.
	public static ExplainType       EXPLAIN             = DMLOptions.defaultOptions.explainType; // explain type
	public static String            DML_FILE_PATH_ANTLR_PARSER = DMLOptions.defaultOptions.filePath; // filename of dml/pydml script
	
	// TODO: For now, assume that multiple threads won't attempt to use different floating point precision.
	public static String            FLOATING_POINT_PRECISION = "double";                         // data type to use internally. 
	
	/**
	 * Global variable indicating the script type (DML or PYDML). Can be used
	 * for DML/PYDML-specific tasks, such as outputting booleans in the correct
	 * case (TRUE/FALSE for DML and True/False for PYDML).
	 */
	public static ScriptType        SCRIPT_TYPE         = DMLOptions.defaultOptions.scriptType;
	
	public static boolean _suppressPrint2Stdout = false;  // flag that indicates whether or not to suppress any prints to stdout
	public static boolean USE_LOCAL_SPARK_CONFIG = false; //set default local spark configuration - used for local testing
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
	 * Used to set master UUID on all nodes (in parfor remote_mr, where DMLScript passed) 
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
	 *
	 * @param args command-line arguments
	 * @throws IOException if an IOException occurs
	 */
	public static void main(String[] args)
		throws IOException
	{
		Configuration conf = new Configuration(ConfigurationManager.getCachedJobConf());
		String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();

		try {
			DMLScript.executeScript(conf, otherArgs);
		} catch (ParseException pe) {
			System.err.println(pe.getMessage());
		} catch (DMLScriptException e){
			// In case of DMLScriptException, simply print the error message.
			System.err.println(e.getMessage());
		}
	}

	/**
	 * Single entry point for all public invocation alternatives (e.g.,
	 * main, executeScript, JaqlUdf etc)
	 * 
	 * @param conf Hadoop configuration
	 * @param args arguments
	 * @return true if success, false otherwise
	 */
	@SuppressWarnings("null")
	public static boolean executeScript( Configuration conf, String[] args ) {
		//parse arguments and set execution properties
		RUNTIME_PLATFORM oldExecMode  = ConfigurationManager.getExecutionMode();  //keep old ConfigurationManager.getExecutionMode()
		ExplainType oldexplain          = EXPLAIN;     //keep old explain

		DMLOptions dmlOptions = null;
		
		try
		{
			dmlOptions = DMLOptions.parseCLArguments(args);
			ConfigurationManager.setGlobalOptions(dmlOptions);
			
			JMLC_MEM_STATISTICS = dmlOptions.memStats;
			EXPLAIN             = dmlOptions.explainType;
			ENABLE_DEBUG_MODE   = dmlOptions.debug;
			SCRIPT_TYPE         = dmlOptions.scriptType;

			String fnameOptConfig = dmlOptions.configFile;
			boolean isFile = dmlOptions.filePath != null;
			String fileOrScript = isFile ? dmlOptions.filePath : dmlOptions.script;

			boolean help = dmlOptions.help;

			if (help) {
				HelpFormatter formatter = new HelpFormatter();
				formatter.printHelp( "systemml", dmlOptions.options );
				return true;
			}

			if (dmlOptions.clean) {
				cleanSystemMLWorkspace();
				return true;
			}

			//set log level
			if (!ENABLE_DEBUG_MODE)
				setLoggingProperties( conf );
		
			//Step 2: prepare script invocation
			if (isFile && StringUtils.endsWithIgnoreCase(fileOrScript, ".pydml")) {
				SCRIPT_TYPE = ScriptType.PYDML;
			}

			String dmlScriptStr = readDMLScript(isFile, fileOrScript);
			Map<String, String> argVals = dmlOptions.argVals;

			DML_FILE_PATH_ANTLR_PARSER = dmlOptions.filePath;
			
			//Step 3: invoke dml script
			printInvocationInfo(fileOrScript, fnameOptConfig, argVals);
			if (ENABLE_DEBUG_MODE) {
				// inner try loop is just to isolate the debug exception, which will allow to manage the bugs from debugger v/s runtime
				launchDebugger(dmlScriptStr, fnameOptConfig, argVals, SCRIPT_TYPE);
			}
			else {
				execute(dmlScriptStr, fnameOptConfig, argVals, args, SCRIPT_TYPE);
			}
		}
		catch(AlreadySelectedException e) {
			System.err.println("Mutually exclusive options were selected. " + e.getMessage());
			HelpFormatter formatter = new HelpFormatter();
			formatter.printHelp( "systemml", dmlOptions.options );
			return false;
		}
		catch(org.apache.commons.cli.ParseException e) {
			System.err.println(e.getMessage());
			HelpFormatter formatter = new HelpFormatter();
			formatter.printHelp( "systemml", dmlOptions.options );
		}
		catch (ParseException | DMLScriptException e) {
			throw e;
		}
		catch(Exception ex) {
			LOG.error("Failed to execute DML script.", ex);
			throw new DMLException(ex);
		}
		finally {
			//reset runtime platform and visualize flag
			ConfigurationManager.getDMLOptions().setExecutionMode(oldExecMode);
			EXPLAIN = oldexplain;
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
	protected static String readDMLScript( boolean isFile, String scriptOrFilename )
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
				else 
				{ 
					in = new BufferedReader(new FileReader(fileName));
				}
				
				//core script reading
				String tmp = null;
				while ((tmp = in.readLine()) != null)
				{
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
			Scanner scan = new Scanner(is);
			dmlScriptStr = scan.useDelimiter("\\A").next();	
			scan.close();
		}
		
		return dmlScriptStr;
	}

	
	private static void setLoggingProperties( Configuration conf ) {
		String debug = conf.get("systemml.logging");
		if (debug == null)
			debug = System.getProperty("systemml.logging");
		if (debug != null){
			if (debug.equalsIgnoreCase("debug"))
				Logger.getLogger("org.apache.sysml").setLevel((Level) Level.DEBUG);
			else if (debug.equalsIgnoreCase("trace"))
				Logger.getLogger("org.apache.sysml").setLevel((Level) Level.TRACE);
		}
	}
	
	///////////////////////////////
	// private internal interface 
	// (core compilation and execute)
	////////

	/**
	 * The running body of DMLScript execution. This method should be called after execution properties have been correctly set,
	 * and customized parameters have been put into _argVals
	 * 
	 * @param dmlScriptStr DML script string
	 * @param fnameOptConfig configuration file
	 * @param argVals map of argument values
	 * @param allArgs arguments
	 * @param scriptType type of script (DML or PyDML)
	 * @throws IOException if IOException occurs
	 */
	private static void execute(String dmlScriptStr, String fnameOptConfig, Map<String,String> argVals, String[] allArgs, ScriptType scriptType)
		throws IOException
	{	
		SCRIPT_TYPE = scriptType;

		//print basic time and environment info
		printStartExecInfo( dmlScriptStr );
		
		//Step 1: parse configuration files & write any configuration specific global variables
		DMLConfig dmlconf = DMLConfig.readConfigurationFile(fnameOptConfig);
		ConfigurationManager.setGlobalConfig(dmlconf);
		CompilerConfig cconf = OptimizerUtils.constructCompilerConfig(dmlconf);
		ConfigurationManager.setGlobalConfig(cconf);
		LOG.debug("\nDML config: \n" + dmlconf.getConfigInfo());
		
		setGlobalFlags(dmlconf);

		//Step 2: set local/remote memory if requested (for compile in AM context) 
		if( dmlconf.getBooleanValue(DMLConfig.YARN_APPMASTER) ){
			DMLAppMasterUtils.setupConfigRemoteMaxMemory(dmlconf); 
		}
		
		//Step 3: parse dml script
		Statistics.startCompileTimer();
		ParserWrapper parser = ParserFactory.createParser(scriptType);
		DMLProgram prog = parser.parse(DML_FILE_PATH_ANTLR_PARSER, dmlScriptStr, argVals);
		
		//Step 4: construct HOP DAGs (incl LVA, validate, and setup)
		DMLTranslator dmlt = new DMLTranslator(prog);
		dmlt.liveVariableAnalysis(prog);
		dmlt.validateParseTree(prog);
		dmlt.constructHops(prog);
		
		//init working directories (before usage by following compilation steps)
		initHadoopExecution( dmlconf );
	
		//Step 5: rewrite HOP DAGs (incl IPA and memory estimates)
		dmlt.rewriteHopsDAG(prog);
		
		//Step 6: construct lops (incl exec type and op selection)
		dmlt.constructLops(prog);

		if (LOG.isDebugEnabled()) {
			LOG.debug("\n********************** LOPS DAG *******************");
			dmlt.printLops(prog);
			dmlt.resetLopsDAGVisitStatus(prog);
		}
		
		//Step 7: generate runtime program, incl codegen
		Program rtprog = dmlt.getRuntimeProgram(prog, dmlconf);
		
		//launch SystemML appmaster (if requested and not already in launched AM)
		if( dmlconf.getBooleanValue(DMLConfig.YARN_APPMASTER) ){
			if( !isActiveAM() && DMLYarnClientProxy.launchDMLYarnAppmaster(dmlScriptStr, dmlconf, allArgs, rtprog) )
				return; //if AM launch unsuccessful, fall back to normal execute
			if( isActiveAM() ) //in AM context (not failed AM launch)
				DMLAppMasterUtils.setupProgramMappingRemoteMaxMemory(rtprog);
		}
		
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
			ScriptExecutorUtils.executeRuntimeProgram(rtprog, ec, dmlconf, ConfigurationManager.isStatistics() ? ConfigurationManager.getDMLOptions().getStatisticsMaxHeavyHitters() : 0, null);
		}
		finally {
			if(ec != null && ec instanceof SparkExecutionContext)
				((SparkExecutionContext) ec).close();
			LOG.info("END DML run " + getDateTime() );
			//cleanup scratch_space and all working dirs
			cleanupHadoopExecution( dmlconf );
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
		
		CacheableData.CACHING_BUFFER_SIZE = dmlconf.getDoubleValue(DMLConfig.CACHING_BUFFER_SIZE);
		if(CacheableData.CACHING_BUFFER_SIZE < 0 || CacheableData.CACHING_BUFFER_SIZE > 1) 
			throw new RuntimeException("Incorrect value (" + CacheableData.CACHING_BUFFER_SIZE + ") for the configuration " + DMLConfig.CACHING_BUFFER_SIZE);
				
		NativeHelper.initialize(dmlconf.getTextValue(DMLConfig.NATIVE_BLAS_DIR), dmlconf.getTextValue(DMLConfig.NATIVE_BLAS).trim());
		
		DMLScript.FLOATING_POINT_PRECISION = dmlconf.getTextValue(DMLConfig.FLOATING_POINT_PRECISION);
		org.apache.sysml.runtime.matrix.data.LibMatrixCUDA.resetFloatingPointPrecision();
	}
	
	/**
	 * Launcher for DML debugger. This method should be called after 
	 * execution and debug properties have been correctly set, and customized parameters
	 * 
	 * @param dmlScriptStr DML script contents (including new lines)
	 * @param fnameOptConfig Full path of configuration file for SystemML
	 * @param argVals Key-value pairs defining arguments of DML script
	 * @param scriptType type of script (DML or PyDML)
	 * @throws IOException if IOException occurs
	 */
	private static void launchDebugger(String dmlScriptStr, String fnameOptConfig, Map<String,String> argVals, ScriptType scriptType)
		throws IOException
	{		
		DMLDebuggerProgramInfo dbprog = new DMLDebuggerProgramInfo();
		
		//Step 1: parse configuration files
		DMLConfig conf = DMLConfig.readConfigurationFile(fnameOptConfig);
		ConfigurationManager.setGlobalConfig(conf);

		//Step 2: parse dml script

		ParserWrapper parser = ParserFactory.createParser(scriptType);
		DMLProgram prog = parser.parse(DML_FILE_PATH_ANTLR_PARSER, dmlScriptStr, argVals);
		
		//Step 3: construct HOP DAGs (incl LVA and validate)
		DMLTranslator dmlt = new DMLTranslator(prog);
		dmlt.liveVariableAnalysis(prog);
		dmlt.validateParseTree(prog);
		dmlt.constructHops(prog);

		//Step 4: rewrite HOP DAGs (incl IPA and memory estimates)
		dmlt.rewriteHopsDAG(prog);

		//Step 5: construct LOP DAGs
		dmlt.constructLops(prog);
	
		//Step 6: generate runtime program
		dbprog.rtprog = dmlt.getRuntimeProgram(prog, conf);
		
		try {
			//set execution environment
			initHadoopExecution(conf);
		
			//initialize an instance of SystemML debugger
			DMLDebugger SystemMLdb = new DMLDebugger(dbprog, dmlScriptStr);
			//run SystemML debugger
			SystemMLdb.runSystemMLDebugger();
		}
		finally {
			//cleanup scratch_space and all working dirs
			cleanupHadoopExecution(conf);
		}
	}

	public static void initHadoopExecution( DMLConfig config ) 
		throws IOException, ParseException, DMLRuntimeException
	{
		//check security aspects
		checkSecuritySetup( config );
		
		//create scratch space with appropriate permissions
		String scratch = config.getTextValue(DMLConfig.SCRATCH_SPACE);
		MapReduceTool.createDirIfNotExistOnHDFS(scratch, DMLConfig.DEFAULT_SHARED_DIR_PERMISSION);
		
		//cleanup working dirs from previous aborted runs with same pid in order to prevent conflicts
		cleanupHadoopExecution(config); 
		
		//init caching (incl set active)
		LocalFileUtils.createWorkingDirectory();
		CacheableData.initCaching();
		
		//reset statistics (required if multiple scripts executed in one JVM)
		Statistics.resetNoOfExecutedJobs();
		if( ConfigurationManager.isStatistics() )
			Statistics.reset();
	}
	
	private static void checkSecuritySetup(DMLConfig config) 
		throws IOException, DMLRuntimeException
	{
		//analyze local configuration
		String userName = System.getProperty( "user.name" );
		HashSet<String> groupNames = new HashSet<>();
		try{
			//check existence, for backwards compatibility to < hadoop 0.21
			if( UserGroupInformation.class.getMethod("getCurrentUser") != null ){
				String[] groups = UserGroupInformation.getCurrentUser().getGroupNames();
				Collections.addAll(groupNames, groups);
			}
		}catch(Exception ex){}
		
		//analyze hadoop configuration
		JobConf job = ConfigurationManager.getCachedJobConf();
		boolean localMode     = InfrastructureAnalyzer.isLocalMode(job);
		String taskController = job.get(MRConfigurationNames.MR_TASKTRACKER_TASKCONTROLLER, "org.apache.hadoop.mapred.DefaultTaskController");
		String ttGroupName    = job.get(MRConfigurationNames.MR_TASKTRACKER_GROUP,"null");
		String perm           = job.get(MRConfigurationNames.DFS_PERMISSIONS_ENABLED,"null"); //note: job.get("dfs.permissions.supergroup",null);
		URI fsURI             = FileSystem.getDefaultUri(job);

		//determine security states
		boolean flagDiffUser = !(   taskController.equals("org.apache.hadoop.mapred.LinuxTaskController") //runs map/reduce tasks as the current user
							     || localMode  // run in the same JVM anyway
							     || groupNames.contains( ttGroupName) ); //user in task tracker group 
		boolean flagLocalFS = fsURI==null || fsURI.getScheme().equals("file");
		boolean flagSecurity = perm.equals("yes"); 
		
		LOG.debug("SystemML security check: "
				+ "local.user.name = " + userName + ", "
				+ "local.user.groups = " + Arrays.toString(groupNames.toArray()) + ", "
				+ MRConfigurationNames.MR_JOBTRACKER_ADDRESS + " = " + job.get(MRConfigurationNames.MR_JOBTRACKER_ADDRESS) + ", "
				+ MRConfigurationNames.MR_TASKTRACKER_TASKCONTROLLER + " = " + taskController + ","
				+ MRConfigurationNames.MR_TASKTRACKER_GROUP + " = " + ttGroupName + ", "
				+ MRConfigurationNames.FS_DEFAULTFS + " = " + ((fsURI!=null) ? fsURI.getScheme() : "null") + ", "
				+ MRConfigurationNames.DFS_PERMISSIONS_ENABLED + " = " + perm );

		//print warning if permission issues possible
		if( flagDiffUser && ( flagLocalFS || flagSecurity ) ) {
			LOG.warn("Cannot run map/reduce tasks as user '"+userName+"'. Using tasktracker group '"+ttGroupName+"'.");
		}
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
		
		//1) cleanup scratch space (everything for current uuid) 
		//(required otherwise export to hdfs would skip assumed unnecessary writes if same name)
		MapReduceTool.deleteFileIfExistOnHDFS( config.getTextValue(DMLConfig.SCRATCH_SPACE) + dirSuffix );
		
		//2) cleanup hadoop working dirs (only required for LocalJobRunner (local job tracker), because
		//this implementation does not create job specific sub directories)
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		if( InfrastructureAnalyzer.isLocalMode(job) ) {
			try {
				LocalFileUtils.deleteFileIfExists( DMLConfig.LOCAL_MR_MODE_STAGING_DIR + dirSuffix );
				LocalFileUtils.deleteFileIfExists( MRJobConfiguration.getLocalWorkingDirPrefix(job) + dirSuffix );
				MapReduceTool.deleteFileIfExistOnHDFS( MRJobConfiguration.getSystemWorkingDirPrefix(job) + dirSuffix );
				MapReduceTool.deleteFileIfExistOnHDFS( MRJobConfiguration.getStagingWorkingDirPrefix(job) + dirSuffix );
			}
			catch(Exception ex) {
				//we give only a warning because those directories are written by the mapred deamon 
				//and hence, execution can still succeed
				LOG.warn("Unable to cleanup hadoop working dirs: "+ex.getMessage());
			}
		}
		
		//3) cleanup systemml-internal working dirs
		CacheableData.cleanupCacheDir(); //might be local/hdfs
		LocalFileUtils.cleanupWorkingDirectory();
	}

	
	///////////////////////////////
	// private internal helper functionalities
	////////

	private static void printInvocationInfo(String fnameScript, String fnameOptConfig, Map<String,String> argVals) {
		LOG.debug("****** args to DML Script ******\n" + "UUID: " + getUUID() + "\n" + "SCRIPT PATH: " + fnameScript + "\n" 
			+ "RUNTIME: " + ConfigurationManager.getExecutionMode() + "\n" + "BUILTIN CONFIG: " + DMLConfig.DEFAULT_SYSTEMML_CONFIG_FILEPATH + "\n"
			+ "OPTIONAL CONFIG: " + fnameOptConfig + "\n");
		if( !argVals.isEmpty() ) {
			LOG.debug("Script arguments are: \n");
			for (int i=1; i<= argVals.size(); i++)
				LOG.debug("Script argument $" + i + " = " + argVals.get("$" + i) );
		}
	}
	
	private static void printStartExecInfo(String dmlScriptString) {
		LOG.info("BEGIN DML run " + getDateTime());
		LOG.debug("DML script: \n" + dmlScriptString);
		if (ConfigurationManager.getExecutionMode() == RUNTIME_PLATFORM.HADOOP || ConfigurationManager.getExecutionMode() == RUNTIME_PLATFORM.HYBRID) {
			String hadoop_home = System.getenv("HADOOP_HOME");
			LOG.info("HADOOP_HOME: " + hadoop_home);
		}
	}
	
	private static String getDateTime() {
		DateFormat dateFormat = new SimpleDateFormat("MM/dd/yyyy HH:mm:ss");
		Date date = new Date();
		return dateFormat.format(date);
	}

	private static void cleanSystemMLWorkspace() {
		try {
			//read the default config
			DMLConfig conf = DMLConfig.readConfigurationFile(null);
			
			//run cleanup job to clean remote local tmp dirs
			CleanupMR.runJob(conf);
			
			//cleanup scratch space (on HDFS)
			String scratch = conf.getTextValue(DMLConfig.SCRATCH_SPACE);
			if( scratch != null )
				MapReduceTool.deleteFileIfExistOnHDFS(scratch);
			
			//cleanup local working dir
			String localtmp = conf.getTextValue(DMLConfig.LOCAL_TMP_DIR);
			if( localtmp != null )
				LocalFileUtils.cleanupRcWorkingDirectory(localtmp);
		}
		catch(Exception ex) {
			throw new DMLException("Failed to run SystemML workspace cleanup.", ex);
		}
	}
}
