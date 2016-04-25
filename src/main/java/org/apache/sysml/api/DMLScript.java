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
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Scanner;

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
import org.apache.sysml.conf.CompilerConfig;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.debug.DMLDebugger;
import org.apache.sysml.debug.DMLDebuggerException;
import org.apache.sysml.debug.DMLDebuggerProgramInfo;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.hops.OptimizerUtils.OptimizationLevel;
import org.apache.sysml.hops.globalopt.GlobalOptimizerWrapper;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.lops.LopsException;
import org.apache.sysml.parser.AParserWrapper;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.DMLTranslator;
import org.apache.sysml.parser.LanguageException;
import org.apache.sysml.parser.ParseException;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLScriptException;
import org.apache.sysml.runtime.controlprogram.Program;
import org.apache.sysml.runtime.controlprogram.caching.CacheStatistics;
import org.apache.sysml.runtime.controlprogram.caching.CacheableData;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.controlprogram.parfor.ProgramConverter;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.controlprogram.parfor.util.IDHandler;
import org.apache.sysml.runtime.matrix.CleanupMR;
import org.apache.sysml.runtime.matrix.mapred.MRConfigurationNames;
import org.apache.sysml.runtime.matrix.mapred.MRJobConfiguration;
import org.apache.sysml.runtime.util.LocalFileUtils;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.utils.Explain;
import org.apache.sysml.utils.Explain.ExplainCounts;
import org.apache.sysml.utils.Explain.ExplainType;
import org.apache.sysml.utils.Statistics;
import org.apache.sysml.yarn.DMLAppMasterUtils;
import org.apache.sysml.yarn.DMLYarnClientProxy;


public class DMLScript 
{	
	public enum RUNTIME_PLATFORM { 
		HADOOP, 	    // execute all matrix operations in MR
		SINGLE_NODE,    // execute all matrix operations in CP
		HYBRID,         // execute matrix operations in CP or MR
		HYBRID_SPARK,   // execute matrix operations in CP or Spark   
		SPARK			// execute matrix operations in Spark
	}
	
	public static RUNTIME_PLATFORM rtplatform = RUNTIME_PLATFORM.HYBRID; //default exec mode
	public static boolean STATISTICS = false; //default statistics
	public static boolean ENABLE_DEBUG_MODE = false; //default debug mode
	public static boolean USE_LOCAL_SPARK_CONFIG = false; //set default local spark configuration - used for local testing
	public static String DML_FILE_PATH_ANTLR_PARSER = null;
	public static ExplainType EXPLAIN = ExplainType.NONE; //default explain
	
	// flag that indicates whether or not to suppress any prints to stdout
	public static boolean _suppressPrint2Stdout = false;
	
	public static String _uuid = IDHandler.createDistributedUniqueID(); 
	public static boolean _activeAM = false;
	
	private static final Log LOG = LogFactory.getLog(DMLScript.class.getName());
	
	public static String USAGE = 
			"Usage is " + DMLScript.class.getCanonicalName() + " -f <filename>" 
	        //+ " (-exec <mode>)?" + " (-explain <type>)?" + " (-stats)?" + " (-clean)?" + " (-config=<config_filename>)? 
			+ " [-options] ([-args | -nvargs] <args-list>)? \n" 
			+ "   -f: <filename> will be interpreted as a filename path (if <filename> is prefixed\n"
			+ "         with hdfs or gpfs it is read from DFS, otherwise from local file system)\n" 
			//undocumented feature in beta 08/2014 release
			//+ "   -s: <filename> will be interpreted as a DML script string \n"
			+ "   -python: (optional) parses Python-like DML\n"
			+ "   -debug: (optional) run in debug mode\n"
			// Later add optional flags to indicate optimizations turned on or off. Currently they are turned off.
			//+ "   -debug: <flags> (optional) run in debug mode\n"
			//+ "			Optional <flags> that is supported for this mode is optimize=(on|off)\n"
			+ "   -exec: <mode> (optional) execution mode (hadoop, singlenode, [hybrid], hybrid_spark)\n"
			+ "   -explain: <type> (optional) explain plan (hops, [runtime], recompile_hops, recompile_runtime)\n"
			+ "   -stats: (optional) monitor and report caching/recompilation statistics\n"
			+ "   -clean: (optional) cleanup all SystemML working directories (FS, DFS).\n"
			+ "         All other flags are ignored in this mode. \n"
			+ "   -config: (optional) use config file <config_filename> (default: use parameter\n"
			+ "         values in default SystemML-config.xml config file; if <config_filename> is\n" 
			+ "         prefixed with hdfs or gpfs it is read from DFS, otherwise from local file system)\n"
			+ "   -args: (optional) parameterize DML script with contents of [args list], ALL args\n"
			+ "         after -args flag, each argument must be an unnamed-argument, where 1st value\n"
			+ "         after -args will replace $1 in DML script, 2nd value will replace $2, etc.\n"
			+ "   -nvargs: (optional) parameterize DML script with contents of [args list], ALL args\n"
			+ "         after -nvargs flag, each argument must be be named-argument of form argName=argValue,\n"
			+ "         where value will replace $argName in DML script, argName must be a valid DML variable\n"
			+ "         name (start with letter, contain only letters, numbers, or underscores).\n"
			+ "   <args-list>: (optional) args to DML script \n" 
			+ "   -? | -help: (optional) show this help message \n";
	
	
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
	 * @param uuid
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
	 * Default DML script invocation (e.g., via 'hadoop jar SystemML.jar -f Test.dml')
	 * 
	 * @param args
	 * @throws IOException
	 * @throws DMLException
	 */
	public static void main(String[] args) 
		throws IOException, DMLException
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

	public static boolean executeScript( String[] args ) 
		throws DMLException
	{
		Configuration conf = new Configuration(ConfigurationManager.getCachedJobConf());
		return executeScript( conf, args );
	}
	
	/**
	 * This version of executeScript() is invoked from RJaqlUdf (from BigR).
	 *  
	 * @param conf
	 * @param args
	 * @param suppress
	 * @return
	 * @throws DMLException
	 * @throws ParseException
	 */
	public static String executeScript( Configuration conf, String[] args, boolean suppress) 
		throws DMLException
	{
		_suppressPrint2Stdout = suppress;
		try {
			boolean ret = executeScript(conf, args);
			return Boolean.toString(ret);
		} catch(DMLScriptException e) {
			return e.getMessage();
		}
	}
	
	/**
	 * Single entry point for all public invocation alternatives (e.g.,
	 * main, executeScript, JaqlUdf etc)
	 * 
	 * @param conf
	 * @param args
	 * @return
	 * @throws DMLException 
	 * @throws ParseException 
	 */
	public static boolean executeScript( Configuration conf, String[] args ) 
		throws DMLException
	{
		//Step 1: parse arguments 
		//check for help 
		if( args.length==0 || (args.length==1 && (args[0].equalsIgnoreCase("-help")|| args[0].equalsIgnoreCase("-?"))) ){
			System.err.println( USAGE );
			return true;
		}
		
		//check for clean
		else if( args.length==1 && args[0].equalsIgnoreCase("-clean") ){
			cleanSystemMLWorkspace();
			return true;
		}
			
		//check number of args - print usage if incorrect
		if( args.length < 2 ){
			System.err.println( "ERROR: Unrecognized invocation arguments." );
			System.err.println( USAGE );
			return false;
		}
				
		//check script arg - print usage if incorrect
		if (!(args[0].equals("-f") || args[0].equals("-s"))){
			System.err.println("ERROR: First argument must be either -f or -s");
			System.err.println( USAGE );
			return false;
		}
		
		//parse arguments and set execution properties
		RUNTIME_PLATFORM oldrtplatform = rtplatform; //keep old rtplatform
		ExplainType oldexplain = EXPLAIN; //keep old explain	
		
		// Reset global flags to avoid errors in test suite
		ENABLE_DEBUG_MODE = false;
		
		boolean parsePyDML = false;
		try
		{
			String fnameOptConfig = null; //optional config filename
			String[] scriptArgs = null; //optional script arguments
			boolean namedScriptArgs = false;
			
			for( int i=2; i<args.length; i++ )
			{
				if( args[i].equalsIgnoreCase("-explain") ) { 
					EXPLAIN = ExplainType.RUNTIME;
					if( args.length > (i+1) && !args[i+1].startsWith("-") )
						EXPLAIN = Explain.parseExplainType(args[++i]);
				}
				else if( args[i].equalsIgnoreCase("-stats") )
					STATISTICS = true;
				else if ( args[i].equalsIgnoreCase("-exec")) {
					rtplatform = parseRuntimePlatform(args[++i]);
					if( rtplatform==null ) 
						return false;
				}
				else if (args[i].startsWith("-config=")) // legacy
					fnameOptConfig = args[i].substring(8).replaceAll("\"", "");
				else if (args[i].equalsIgnoreCase("-config"))
					fnameOptConfig = args[++i];
				else if( args[i].equalsIgnoreCase("-debug") ) {					
					ENABLE_DEBUG_MODE = true;
				}
				else if( args[i].equalsIgnoreCase("-python") ) {
					parsePyDML = true;
				}
				else if (args[i].startsWith("-args") || args[i].startsWith("-nvargs")) {
					namedScriptArgs = args[i].startsWith("-nvargs"); i++;
					scriptArgs = new String[args.length - i];
					System.arraycopy(args, i, scriptArgs, 0, scriptArgs.length); 
					break;
				}
				else{
					System.err.println("ERROR: Unknown argument: " + args[i]);
					return false;
				}
			}
			
			//set log level
			if (!ENABLE_DEBUG_MODE)
				setLoggingProperties( conf );
		
			//Step 2: prepare script invocation
			String dmlScriptStr = readDMLScript(args[0], args[1]);
			Map<String, String> argVals = createArgumentsMap(namedScriptArgs, scriptArgs);
			
			DML_FILE_PATH_ANTLR_PARSER = args[1];
			
			//Step 3: invoke dml script
			printInvocationInfo(args[1], fnameOptConfig, argVals);
			if (ENABLE_DEBUG_MODE) {
				// inner try loop is just to isolate the debug exception, which will allow to manage the bugs from debugger v/s runtime
				launchDebugger(dmlScriptStr, fnameOptConfig, argVals, parsePyDML);
			}
			else {
				execute(dmlScriptStr, fnameOptConfig, argVals, args, parsePyDML);
			}

		}
		catch (ParseException pe) {
			throw pe;
		}
		catch (DMLScriptException e) {
			//rethrow DMLScriptException to propagate stop call
			throw e;
		}
		catch(Exception ex)
		{
			LOG.error("Failed to execute DML script.", ex);
			throw new DMLException(ex);
		}
		finally
		{
			//reset runtime platform and visualize flag
			rtplatform = oldrtplatform;
			EXPLAIN = oldexplain;
		}
		
		return true;
	}
	
	///////////////////////////////
	// private internal utils (argument parsing)
	////////

	/**
	 * 
	 * @param hasNamedArgs
	 * @param args
	 * @throws LanguageException
	 */
	protected static Map<String,String> createArgumentsMap(boolean hasNamedArgs, String[] args)
		throws LanguageException
	{			
		Map<String, String> argMap = new HashMap<String,String>();
		
		if (args == null)
			return argMap;
		
		for(int i=1; i<=args.length; i++)
		{
			String arg = args[i-1];
			
			if (arg.equalsIgnoreCase("-l") || arg.equalsIgnoreCase("-log") ||
				arg.equalsIgnoreCase("-v") || arg.equalsIgnoreCase("-visualize")||
				arg.equalsIgnoreCase("-explain") || 
				arg.equalsIgnoreCase("-debug") || 
				arg.equalsIgnoreCase("-stats") || 
				arg.equalsIgnoreCase("-exec") ||
				arg.equalsIgnoreCase("-debug") ||
				arg.startsWith("-config="))
			{
					throw new LanguageException("-args or -nvargs must be the final argument for DMLScript!");
			}
			
			//parse arguments (named args / args by position)
			if(hasNamedArgs)
			{
				// CASE: named argument argName=argValue -- must add <argName, argValue> pair to _argVals
				String[] argPieces = arg.split("=");
				if(argPieces.length < 2)
					throw new LanguageException("for -nvargs option, elements in arg list must be named and have form argName=argValue");
				String argName = argPieces[0];
				StringBuilder sb = new StringBuilder();
				for (int jj=1; jj < argPieces.length; jj++){
					sb.append(argPieces[jj]); 
				}
				
				String varNameRegex = "^[a-zA-Z]([a-zA-Z0-9_])*$";
				if (!argName.matches(varNameRegex))
					throw new LanguageException("argName " + argName + " must be a valid variable name in DML. Valid variable names in DML start with upper-case or lower-case letter, and contain only letters, digits, or underscores");
					
				argMap.put("$"+argName,sb.toString());
			}
			else 
			{
				// CASE: unnamed argument -- use position in arg list for name
				argMap.put("$"+i ,arg);
			}
		}
		
		return argMap;
	}
	
	/**
	 * 
	 * @param argname
	 * @param script
	 * @return
	 * @throws IOException 
	 * @throws LanguageException 
	 */
	protected static String readDMLScript( String argname, String script ) 
		throws IOException, LanguageException
	{
		boolean fromFile = argname.equals("-f");
		String dmlScriptStr;
		
		if( fromFile )
		{
			//read DML script from file
			if(script == null)
				throw new LanguageException("DML script path was not specified!");
			
			StringBuilder sb = new StringBuilder();
			BufferedReader in = null;
			try 
			{
				//read from hdfs or gpfs file system
				if(    script.startsWith("hdfs:") 
					|| script.startsWith("gpfs:") ) 
				{ 
					if( !LocalFileUtils.validateExternalFilename(script, true) )
						throw new LanguageException("Invalid (non-trustworthy) hdfs filename.");
					FileSystem fs = FileSystem.get(ConfigurationManager.getCachedJobConf());
					Path scriptPath = new Path(script);
					in = new BufferedReader(new InputStreamReader(fs.open(scriptPath)));
				}
				// from local file system
				else 
				{ 
					if( !LocalFileUtils.validateExternalFilename(script, false) )
						throw new LanguageException("Invalid (non-trustworthy) local filename.");
					in = new BufferedReader(new FileReader(script));
				}
				
				//core script reading
				String tmp = null;
				while ((tmp = in.readLine()) != null)
				{
					sb.append( tmp );
					sb.append( "\n" );
				}
			}
			catch (IOException ex)
			{
				LOG.error("Failed to read the script from the file system", ex);
				throw ex;
			}
			finally 
			{
				if( in != null )
					in.close();
			}
			
			dmlScriptStr = sb.toString();
		}
		else
		{
			//parse given script string 
			if(script == null)
				throw new LanguageException("DML script was not specified!");
			
			InputStream is = new ByteArrayInputStream(script.getBytes());
			Scanner scan = new Scanner(is);
			dmlScriptStr = scan.useDelimiter("\\A").next();	
			scan.close();
		}
		
		return dmlScriptStr;
	}
	
	/**
	 * 
	 * @param platform
	 * @return
	 */
	private static RUNTIME_PLATFORM parseRuntimePlatform( String platform )
	{
		RUNTIME_PLATFORM lrtplatform = null;
		
		if ( platform.equalsIgnoreCase("hadoop")) 
			lrtplatform = RUNTIME_PLATFORM.HADOOP;
		else if ( platform.equalsIgnoreCase("singlenode"))
			lrtplatform = RUNTIME_PLATFORM.SINGLE_NODE;
		else if ( platform.equalsIgnoreCase("hybrid"))
			lrtplatform = RUNTIME_PLATFORM.HYBRID;
		else if ( platform.equalsIgnoreCase("spark"))
			lrtplatform = RUNTIME_PLATFORM.SPARK;
		else if ( platform.equalsIgnoreCase("hybrid_spark"))
			lrtplatform = RUNTIME_PLATFORM.HYBRID_SPARK;
		else 
			System.err.println("ERROR: Unknown runtime platform: " + platform);
		
		return lrtplatform;
	}
	
	/**
	 * 
	 * @param conf
	 */
	private static void setLoggingProperties( Configuration conf )
	{
		String debug = conf.get("systemml.logging");
		
		if (debug == null)
			debug = System.getProperty("systemml.logging");
		
		if (debug != null){
			if (debug.equalsIgnoreCase("debug")){
				Logger.getLogger("org.apache.sysml").setLevel((Level) Level.DEBUG);
			}
			else if (debug.equalsIgnoreCase("trace")){
				Logger.getLogger("org.apache.sysml").setLevel((Level) Level.TRACE);
			}
		}
	}
	
	
	///////////////////////////////
	// private internal interface 
	// (core compilation and execute)
	////////

	/**
	 * run: The running body of DMLScript execution. This method should be called after execution properties have been correctly set,
	 * and customized parameters have been put into _argVals
	 * @throws ParseException 
	 * @throws IOException 
	 * @throws DMLRuntimeException 
	 * @throws HopsException 
	 * @throws LanguageException 
	 * @throws LopsException 
	 */
	private static void execute(String dmlScriptStr, String fnameOptConfig, Map<String,String> argVals, String[] allArgs, boolean parsePyDML)
		throws ParseException, IOException, DMLRuntimeException, LanguageException, HopsException, LopsException 
	{	
		//print basic time and environment info
		printStartExecInfo( dmlScriptStr );
		
		//Step 1: parse configuration files
		DMLConfig dmlconf = DMLConfig.readConfigurationFile(fnameOptConfig);
		ConfigurationManager.setGlobalConfig(dmlconf);		
		CompilerConfig cconf = OptimizerUtils.constructCompilerConfig(dmlconf);
		ConfigurationManager.setGlobalConfig(cconf);
		LOG.debug("\nDML config: \n" + dmlconf.getConfigInfo());

		//Step 2: set local/remote memory if requested (for compile in AM context) 
		if( dmlconf.getBooleanValue(DMLConfig.YARN_APPMASTER) ){
			DMLAppMasterUtils.setupConfigRemoteMaxMemory(dmlconf); 
		}
		
		//Step 3: parse dml script
		Statistics.startCompileTimer();
		AParserWrapper parser = AParserWrapper.createParser(parsePyDML);
		DMLProgram prog = parser.parse(DML_FILE_PATH_ANTLR_PARSER, dmlScriptStr, argVals);
		
		//Step 4: construct HOP DAGs (incl LVA and validate)
		DMLTranslator dmlt = new DMLTranslator(prog);
		dmlt.liveVariableAnalysis(prog);			
		dmlt.validateParseTree(prog);
		dmlt.constructHops(prog);
		
		if (LOG.isDebugEnabled()) {
			LOG.debug("\n********************** HOPS DAG (Before Rewrite) *******************");
			dmlt.printHops(prog);
			DMLTranslator.resetHopsDAGVisitStatus(prog);
		}
	
		//Step 5: rewrite HOP DAGs (incl IPA and memory estimates)
		dmlt.rewriteHopsDAG(prog);
		
		if (LOG.isDebugEnabled()) {
			LOG.debug("\n********************** HOPS DAG (After Rewrite) *******************");
			dmlt.printHops(prog);
			DMLTranslator.resetHopsDAGVisitStatus(prog);
		
			LOG.debug("\n********************** OPTIMIZER *******************\n" + 
			          "Level = " + OptimizerUtils.getOptLevel() + "\n"
					 +"Available Memory = " + ((double)InfrastructureAnalyzer.getLocalMaxMemory()/1024/1024) + " MB" + "\n"
					 +"Memory Budget = " + ((double)OptimizerUtils.getLocalMemBudget()/1024/1024) + " MB" + "\n");
		}

		//Step 6: construct lops (incl exec type and op selection)
		dmlt.constructLops(prog);

		if (LOG.isDebugEnabled()) {
			LOG.debug("\n********************** LOPS DAG *******************");
			dmlt.printLops(prog);
			dmlt.resetLopsDAGVisitStatus(prog);
		}
		
		//Step 7: generate runtime program
		Program rtprog = prog.getRuntimeProgram(dmlconf);

		if (LOG.isDebugEnabled()) {
			LOG.info("********************** Instructions *******************");
			rtprog.printMe();
			LOG.info("*******************************************************");
		}

		//Step 8: [optional global data flow optimization]
		if(OptimizerUtils.isOptLevel(OptimizationLevel.O4_GLOBAL_TIME_MEMORY) ) 
		{
			LOG.warn("Optimization level '" + OptimizationLevel.O4_GLOBAL_TIME_MEMORY + "' " +
					"is still in experimental state and not intended for production use.");
			rtprog = GlobalOptimizerWrapper.optimizeProgram(prog, rtprog);
		}
		
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
		if( EXPLAIN != ExplainType.NONE ) {
			LOG.info("EXPLAIN ("+EXPLAIN.toString()+"):\n" 
					 + Explain.explainMemoryBudget(counts)+"\n"
					 + Explain.explainDegreeOfParallelism(counts)
					 + Explain.explain(prog, rtprog, EXPLAIN));
		}
		
		Statistics.stopCompileTimer();
		
		//double costs = CostEstimationWrapper.getTimeEstimate(rtprog, ExecutionContextFactory.createContext());
		//System.out.println("Estimated costs: "+costs);
		
		
		//Step 10: execute runtime program
		Statistics.startRunTimer();
		ExecutionContext ec = null;
		try 
		{  
			initHadoopExecution( dmlconf );
			
			//run execute (w/ exception handling to ensure proper shutdown)
			ec = ExecutionContextFactory.createContext(rtprog);
			rtprog.execute( ec );  
			
		}
		finally //ensure cleanup/shutdown
		{	
			if(ec != null && ec instanceof SparkExecutionContext) {
				((SparkExecutionContext) ec).close();
			}
			
			//display statistics (incl caching stats if enabled)
			Statistics.stopRunTimer();
			LOG.info(Statistics.display());
			LOG.info("END DML run " + getDateTime() );
			
			//cleanup scratch_space and all working dirs
			cleanupHadoopExecution( dmlconf );		
		}	
	}		
	
	/**
	 * launchDebugger: Launcher for DML debugger. This method should be called after 
	 * execution and debug properties have been correctly set, and customized parameters 
	 * have been put into _argVals
	 * @param  dmlScriptStr DML script contents (including new lines)
	 * @param  fnameOptConfig Full path of configuration file for SystemML
	 * @param  argVals Key-value pairs defining arguments of DML script
	 * @throws ParseException 
	 * @throws IOException 
	 * @throws DMLRuntimeException 
	 * @throws DMLDebuggerException
	 * @throws HopsException 
	 * @throws LanguageException 
	 * @throws LopsException
	 */
	private static void launchDebugger(String dmlScriptStr, String fnameOptConfig, Map<String,String> argVals, boolean parsePyDML)
		throws ParseException, IOException, DMLRuntimeException, DMLDebuggerException, LanguageException, HopsException, LopsException 
	{		
		DMLDebuggerProgramInfo dbprog = new DMLDebuggerProgramInfo();
		
		//Step 1: parse configuration files
		DMLConfig conf = DMLConfig.readConfigurationFile(fnameOptConfig);
		ConfigurationManager.setGlobalConfig(conf);

		//Step 2: parse dml script
		AParserWrapper parser = AParserWrapper.createParser(parsePyDML);
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
		dbprog.rtprog = prog.getRuntimeProgram(conf);
		
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

	/**
	 * @throws ParseException 
	 * @throws IOException 
	 * @throws DMLRuntimeException 
	 * 
	 */
	static void initHadoopExecution( DMLConfig config ) 
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
		Statistics.resetNoOfExecutedJobs( 0 );
		if( STATISTICS ) {
			CacheStatistics.reset();
			Statistics.reset();
		}
	}
	
	/**
	 * 
	 * @param config 
	 * @throws IOException
	 * @throws DMLRuntimeException 
	 */
	private static void checkSecuritySetup(DMLConfig config) 
		throws IOException, DMLRuntimeException
	{
		//analyze local configuration
		String userName = System.getProperty( "user.name" );
		HashSet<String> groupNames = new HashSet<String>();
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
				+ "local.user.groups = " + ProgramConverter.serializeStringCollection(groupNames) + ", "
				+ MRConfigurationNames.MR_JOBTRACKER_ADDRESS + " = " + job.get(MRConfigurationNames.MR_JOBTRACKER_ADDRESS) + ", "
				+ MRConfigurationNames.MR_TASKTRACKER_TASKCONTROLLER + " = " + taskController + ","
				+ MRConfigurationNames.MR_TASKTRACKER_GROUP + " = " + ttGroupName + ", "
				+ MRConfigurationNames.FS_DEFAULTFS + " = " + ((fsURI!=null) ? fsURI.getScheme() : "null") + ", "
				+ MRConfigurationNames.DFS_PERMISSIONS_ENABLED + " = " + perm );

		//print warning if permission issues possible
		if( flagDiffUser && ( flagLocalFS || flagSecurity ) )
		{
			LOG.warn("Cannot run map/reduce tasks as user '"+userName+"'. Using tasktracker group '"+ttGroupName+"'."); 		 
		}
		
		//validate external filenames working directories
		String localtmpdir = config.getTextValue(DMLConfig.LOCAL_TMP_DIR);
		String hdfstmpdir = config.getTextValue(DMLConfig.SCRATCH_SPACE);
		if( !LocalFileUtils.validateExternalFilename(localtmpdir, false) )
			throw new DMLRuntimeException("Invalid (non-trustworthy) local working directory.");
		if( !LocalFileUtils.validateExternalFilename(hdfstmpdir, true) )
			throw new DMLRuntimeException("Invalid (non-trustworthy) hdfs working directory.");
	}
	
	/**
	 * 
	 * @param config
	 * @throws IOException
	 * @throws ParseException
	 */
	private static void cleanupHadoopExecution( DMLConfig config ) 
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
			try 
			{
				LocalFileUtils.deleteFileIfExists( DMLConfig.LOCAL_MR_MODE_STAGING_DIR + //staging dir (for local mode only) 
					                                   dirSuffix  );	
				LocalFileUtils.deleteFileIfExists( MRJobConfiguration.getLocalWorkingDirPrefix(job) + //local dir
		                                               dirSuffix );
				MapReduceTool.deleteFileIfExistOnHDFS( MRJobConfiguration.getSystemWorkingDirPrefix(job) + //system dir
													   dirSuffix  );
				MapReduceTool.deleteFileIfExistOnHDFS( MRJobConfiguration.getStagingWorkingDirPrefix(job) + //staging dir
								                       dirSuffix  );
			}
			catch(Exception ex)
			{
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

	/**
	 * 
	 * @param fnameScript
	 * @param fnameOptConfig
	 * @param argVals
	 */
	private static void printInvocationInfo(String fnameScript, String fnameOptConfig, Map<String,String> argVals)
	{		
		LOG.debug("****** args to DML Script ******\n" + "UUID: " + getUUID() + "\n" + "SCRIPT PATH: " + fnameScript + "\n" 
	                + "RUNTIME: " + rtplatform + "\n" + "BUILTIN CONFIG: " + DMLConfig.DEFAULT_SYSTEMML_CONFIG_FILEPATH + "\n"
	                + "OPTIONAL CONFIG: " + fnameOptConfig + "\n");

		if( !argVals.isEmpty() ) {
			LOG.debug("Script arguments are: \n");
			for (int i=1; i<= argVals.size(); i++)
				LOG.debug("Script argument $" + i + " = " + argVals.get("$" + i) );
		}
	}
	
	private static void printStartExecInfo(String dmlScriptString)
	{
		LOG.info("BEGIN DML run " + getDateTime());
		LOG.debug("DML script: \n" + dmlScriptString);
		
		if (rtplatform == RUNTIME_PLATFORM.HADOOP || rtplatform == RUNTIME_PLATFORM.HYBRID) {
			String hadoop_home = System.getenv("HADOOP_HOME");
			LOG.info("HADOOP_HOME: " + hadoop_home);
		}
	}
	
	/**
	 * 
	 * @return
	 */
	private static String getDateTime() 
	{
		DateFormat dateFormat = new SimpleDateFormat("MM/dd/yyyy HH:mm:ss");
		Date date = new Date();
		return dateFormat.format(date);
	}

	/**
	 * 
	 * @throws DMLException
	 */
	private static void cleanSystemMLWorkspace() 
		throws DMLException
	{
		try
		{
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
		catch(Exception ex)
		{
			throw new DMLException("Failed to run SystemML workspace cleanup.", ex);
		}
	}
}  
