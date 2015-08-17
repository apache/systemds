/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.api;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URI;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Scanner;

import javax.xml.parsers.ParserConfigurationException;

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
import org.xml.sax.SAXException;

import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.debug.DMLDebugger;
import com.ibm.bi.dml.debug.DMLDebuggerException;
import com.ibm.bi.dml.debug.DMLDebuggerProgramInfo;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.hops.OptimizerUtils.OptimizationLevel;
import com.ibm.bi.dml.hops.globalopt.GlobalOptimizerWrapper;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.parser.DMLProgram;
import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.parser.LanguageException;
import com.ibm.bi.dml.parser.ParseException;
import com.ibm.bi.dml.parser.antlr4.DMLParserWrapper;
import com.ibm.bi.dml.parser.python.PyDMLParserWrapper;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLScriptException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.caching.CacheStatistics;
import com.ibm.bi.dml.runtime.controlprogram.caching.CacheableData;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContextFactory;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ProgramConverter;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDHandler;
import com.ibm.bi.dml.runtime.matrix.CleanupMR;
import com.ibm.bi.dml.runtime.matrix.mapred.MRConfigurationNames;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.util.LocalFileUtils;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.utils.Explain;
import com.ibm.bi.dml.utils.Explain.ExplainType;
import com.ibm.bi.dml.utils.Statistics;
import com.ibm.bi.dml.yarn.DMLAppMasterUtils;
// import com.ibm.bi.dml.utils.visualize.DotGraph;
import com.ibm.bi.dml.yarn.DMLYarnClientProxy;


public class DMLScript 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public enum RUNTIME_PLATFORM { 
		HADOOP, 	    // execute all matrix operations in MR
		SINGLE_NODE,    // execute all matrix operations in CP
		HYBRID,         // execute matrix operations in CP or MR
		HYBRID_SPARK,   // execute matrix operations in CP or Spark   
		SPARK			// execute matrix operations in Spark
	};
	
	public static RUNTIME_PLATFORM rtplatform = RUNTIME_PLATFORM.HYBRID; //default exec mode
	public static boolean VISUALIZE = false; //default visualize
	public static boolean STATISTICS = false; //default statistics
	public static boolean ENABLE_DEBUG_MODE = false; //default debug mode
	public static boolean USE_LOCAL_SPARK_CONFIG = false; //set default local spark configuration - used for local testing
	public static String DML_FILE_PATH_ANTLR_PARSER = null;
	public static ExplainType EXPLAIN = ExplainType.NONE; //default explain
	
	public static boolean PARSER_TREAT_UDF_AS_EXPRESSIONS = false;

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
			//undocumented feature in beta 08/2014 release
			//+ "   [-v | -visualize]: (optional) use visualization of DAGs \n"
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
	 * @throws ParseException
	 * @throws IOException
	 * @throws SAXException
	 * @throws ParserConfigurationException
	 */
	public static void main(String[] args) 
		throws IOException, DMLException 
	{
		Configuration conf = new Configuration();
		String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
		
		try {
		DMLScript.executeScript(conf, otherArgs); 
		} catch (DMLScriptException e){
			// In case of DMLScriptException, simply print the error message.
			System.err.println(e.getMessage());
		}
	} 

	public static boolean executeScript( String[] args ) 
		throws DMLException
	{
		Configuration conf = new Configuration();
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
	 * @throws LanguageException 
	 */
	public static boolean executeScript( Configuration conf, String[] args ) 
		throws DMLException
	{
		boolean ret = false;
		
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
			return ret;
		}
				
		//check script arg - print usage if incorrect
		if (!(args[0].equals("-f") || args[0].equals("-s"))){
			System.err.println("ERROR: First argument must be either -f or -s");
			System.err.println( USAGE );
			return ret;
		}
		
		//parse arguments and set execution properties
		RUNTIME_PLATFORM oldrtplatform = rtplatform; //keep old rtplatform
		boolean oldvisualize = VISUALIZE; //keep old visualize	
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
				if (args[i].equalsIgnoreCase("-v") || args[i].equalsIgnoreCase("-visualize"))
					VISUALIZE = true;
				else if( args[i].equalsIgnoreCase("-explain") ) { 
					EXPLAIN = ExplainType.RUNTIME;
					if( args.length > (i+1) && !args[i+1].startsWith("-") )
						EXPLAIN = Explain.parseExplainType(args[++i]);
				}
				else if( args[i].equalsIgnoreCase("-stats") )
					STATISTICS = true;
				else if ( args[i].equalsIgnoreCase("-exec")) {
					rtplatform = parseRuntimePlatform(args[++i]);
					if( rtplatform==null ) 
						return ret;
				}
				else if (args[i].startsWith("-config="))
					fnameOptConfig = args[i].substring(8).replaceAll("\"", ""); 
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
					return ret;
				}
			}
			
			//set log level
			if (!ENABLE_DEBUG_MODE)
				setLoggingProperties( conf );
		
			//Step 2: prepare script invocation
			String dmlScriptStr = readDMLScript(args[0], args[1]);
			HashMap<String, String> argVals = createArgumentsMap(namedScriptArgs, scriptArgs);		
			
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
			
			ret = true;
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
			VISUALIZE = oldvisualize;
			EXPLAIN = oldexplain;
		}
		
		return ret;
	}
	
	///////////////////////////////
	// private internal utils (argument parsing)
	////////

	/**
	 * 
	 * @param hasNamedArgs
	 * @param scriptArguments
	 * @throws LanguageException
	 */
	public static HashMap<String,String> createArgumentsMap(boolean hasNamedArgs, String[] args) 
		throws LanguageException
	{			
		HashMap<String, String> argMap = new HashMap<String,String>();
		
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
	 * @param arg
	 * @return
	 * @throws IOException 
	 * @throws LanguageException 
	 */
	protected static String readDMLScript( String argname, String script ) 
		throws IOException, LanguageException
	{
		boolean fromFile = argname.equals("-f") ? true : false;
		String dmlScriptStr = null;
		
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
	public static RUNTIME_PLATFORM parseRuntimePlatform( String platform )
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
				Logger.getLogger("com.ibm.bi.dml").setLevel((Level) Level.DEBUG);
			}
			else if (debug.equalsIgnoreCase("trace")){
				Logger.getLogger("com.ibm.bi.dml").setLevel((Level) Level.TRACE);
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
	 * @throws DMLUnsupportedOperationException 
	 * @throws LopsException 
	 * @throws DMLException 
	 */
	private static void execute(String dmlScriptStr, String fnameOptConfig, HashMap<String,String> argVals, String[] allArgs, boolean parsePyDML)
		throws ParseException, IOException, DMLRuntimeException, LanguageException, HopsException, LopsException, DMLUnsupportedOperationException 
	{				
		//print basic time and environment info
		printStartExecInfo( dmlScriptStr );
		
		//Step 1: parse configuration files
		DMLConfig conf = DMLConfig.readAndMergeConfigurationFiles(fnameOptConfig);
		ConfigurationManager.setConfig(conf);
		LOG.debug("\nDML config: \n" + conf.getConfigInfo());
		
		//Step 2: set local/remote memory if requested (for compile in AM context) 
		if( conf.getBooleanValue(DMLConfig.YARN_APPMASTER) ){
			DMLAppMasterUtils.setupConfigRemoteMaxMemory(conf); 
		}
		
		//Step 3: parse dml script
		DMLProgram prog = null;
		if(parsePyDML) {
			PyDMLParserWrapper parser = new PyDMLParserWrapper();
			prog = parser.parse(DML_FILE_PATH_ANTLR_PARSER, dmlScriptStr, argVals);
		}
		else {
			DMLParserWrapper parser = new DMLParserWrapper();
			prog = parser.parse(DML_FILE_PATH_ANTLR_PARSER, dmlScriptStr, argVals);
		}
		
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
		}
		
		if( VISUALIZE ) { // HOPs before rewrite
//			DotGraph gt = new DotGraph();
//			gt.drawHopsDAG(prog, "HopsDAG After Rewrite", 100, 100, PATH_TO_SRC, VISUALIZE);
//			dmlt.resetHopsDAGVisitStatus(prog);
		}
	
		//Step 6: backend-specific compile and execute
		switch( rtplatform )
		{
			case HADOOP:
			case SINGLE_NODE:
			case HYBRID:
			case SPARK:
			case HYBRID_SPARK:
				executeHadoop(dmlt, prog, conf, dmlScriptStr, allArgs);
				break;
				
			default:
				throw new DMLRuntimeException("Unsupported runtime platform: "+rtplatform);
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
	 * @throws DMLUnsupportedOperationException 
	 * @throws LopsException 
	 * @throws DMLException 
	 */
	private static void launchDebugger(String dmlScriptStr, String fnameOptConfig, HashMap<String,String> argVals, boolean parsePyDML)
		throws ParseException, IOException, DMLRuntimeException, DMLDebuggerException, LanguageException, HopsException, LopsException, DMLUnsupportedOperationException 
	{		
		//produce debugging information (parse, compile and generate runtime program for a given DML script)
		DMLDebuggerProgramInfo p = compileForDebug(dmlScriptStr, fnameOptConfig, argVals, parsePyDML);
		
		try {
			//set execution environment
			initHadoopExecution(p.conf);
		
			//initialize an instance of SystemML debugger
			DMLDebugger SystemMLdb = new DMLDebugger(p, dmlScriptStr, argVals);
			//run SystemML debugger
			SystemMLdb.runSystemMLDebugger();
		}
		finally {
			//cleanup scratch_space and all working dirs
			cleanupHadoopExecution(p.conf);
		}
	}
	
	/**
	 * compile: Compile DML script and generate hops, lops and runtime program for debugger. 
	 * This method should be called after execution and debug properties have been set, and 
	 * customized parameters have been put into _argVals
	 * @param  dmlScriptStr DML script contents (including new lines)
	 * @param  fnameOptConfig Full path of configuration file for SystemML
	 * @param  argVals Key-value pairs defining arguments of DML script
	 * @return dbprog Class containing parsed and compiled DML script w/ hops, lops and runtime program   
	 * @throws ParseException
	 * @throws IOException
	 * @throws DMLRuntimeException
	 * @throws LanguageException
	 * @throws HopsException
	 * @throws LopsException
	 * @throws DMLUnsupportedOperationException
	 */
	//TODO: MB: remove this redundant compile and execute (or at least remove from DMLScript)
	//TODO: This method should be private once debugger infrastructure is on top of the programmatic API  
	public static DMLDebuggerProgramInfo compileForDebug(String dmlScriptStr, String fnameOptConfig, HashMap<String,String> argVals, boolean parsePyDML)
			throws ParseException, IOException, DMLRuntimeException, LanguageException, HopsException, LopsException, DMLUnsupportedOperationException
	{					
		DMLDebuggerProgramInfo dbprog = new DMLDebuggerProgramInfo();
		
		//Step 1: parse configuration files
		dbprog.conf = DMLConfig.readAndMergeConfigurationFiles(fnameOptConfig);
		ConfigurationManager.setConfig(dbprog.conf);
	
		//Step 2: parse dml script
		if(parsePyDML) {
			PyDMLParserWrapper parser = new PyDMLParserWrapper();
			dbprog.prog = parser.parse(DML_FILE_PATH_ANTLR_PARSER, dmlScriptStr, argVals);
		}
		else {
			DMLParserWrapper parser = new DMLParserWrapper();
			dbprog.prog = parser.parse(DML_FILE_PATH_ANTLR_PARSER, dmlScriptStr, argVals);
		}

		//Step 3: construct HOP DAGs (incl LVA and validate)
		dbprog.dmlt = new DMLTranslator(dbprog.prog);
		dbprog.dmlt.liveVariableAnalysis(dbprog.prog);
		dbprog.dmlt.validateParseTree(dbprog.prog);
		dbprog.dmlt.constructHops(dbprog.prog);

		//Step 4: rewrite HOP DAGs (incl IPA and memory estimates)
		dbprog.dmlt.rewriteHopsDAG(dbprog.prog);

		//Step 5: construct LOP DAGs
		dbprog.dmlt.constructLops(dbprog.prog);
	
		//Step 6: generate runtime program
		dbprog.rtprog = dbprog.prog.getRuntimeProgram(dbprog.conf);
		
		return dbprog;
	}
	
	/**
	 * executeHadoop: Handles execution on the Hadoop Map-reduce runtime
	 * 
	 * @param dmlt DML Translator 
	 * @param prog DML Program object from parsed DML script
	 * @param config read from provided configuration file (e.g., config.xml)
	 * @param out writer for log output 
	 * @throws ParseException 
	 * @throws IOException 
	 * @throws LopsException 
	 * @throws HopsException 
	 * @throws LanguageException 
	 * @throws DMLUnsupportedOperationException 
	 * @throws DMLRuntimeException 
	 * @throws DMLException 
	 */
	private static void executeHadoop(DMLTranslator dmlt, DMLProgram prog, DMLConfig conf, String dmlScriptStr, String[] allArgs) 
		throws ParseException, IOException, LanguageException, HopsException, LopsException, DMLRuntimeException, DMLUnsupportedOperationException 
	{	
		LOG.debug("\n********************** OPTIMIZER *******************\n" + 
		          "Level = " + OptimizerUtils.getOptLevel() + "\n"
				 +"Available Memory = " + ((double)InfrastructureAnalyzer.getLocalMaxMemory()/1024/1024) + " MB" + "\n"
				 +"Memory Budget = " + ((double)OptimizerUtils.getLocalMemBudget()/1024/1024) + " MB" + "\n");
		
		/////////////////////// construct the lops ///////////////////////////////////
		dmlt.constructLops(prog);

		if (LOG.isDebugEnabled()) {
			LOG.debug("\n********************** LOPS DAG *******************");
			dmlt.printLops(prog);
			dmlt.resetLopsDAGVisitStatus(prog);
		}

		// lops plan visualization 
		if(VISUALIZE){
//			DotGraph gt = new DotGraph();
//			gt.drawLopsDAG(prog, "LopsDAG", 150, 150, PATH_TO_SRC, VISUALIZE);
//			dmlt.resetLopsDAGVisitStatus(prog);
		}
		
		////////////////////// generate runtime program ///////////////////////////////
		Program rtprog = prog.getRuntimeProgram(conf);

		if (LOG.isDebugEnabled()) {
			LOG.info("********************** Instructions *******************");
			rtprog.printMe();
			LOG.info("*******************************************************");
		}

		//optional global data flow optimization
		if(OptimizerUtils.isOptLevel(OptimizationLevel.O4_GLOBAL_TIME_MEMORY) ) 
		{
			LOG.warn("Optimization level '" + OptimizationLevel.O4_GLOBAL_TIME_MEMORY + "' " +
					"is still in experimental state and not intended for production use.");
			rtprog = GlobalOptimizerWrapper.optimizeProgram(prog, rtprog);
		}
		
		//launch SystemML appmaster (if requested and not already in launched AM)
		if( conf.getBooleanValue(DMLConfig.YARN_APPMASTER) ){
			if( !isActiveAM() && DMLYarnClientProxy.launchDMLYarnAppmaster(dmlScriptStr, conf, allArgs, rtprog) )
				return; //if AM launch unsuccessful, fall back to normal execute
			if( isActiveAM() ) //in AM context (not failed AM launch)
				DMLAppMasterUtils.setupProgramMappingRemoteMaxMemory(rtprog);
		}
		
		//count number compiled MR jobs / SP instructions	
		int jobCount = OptimizerUtils.isSparkExecutionMode() ?
				Explain.countCompiledSPInst(rtprog) : Explain.countCompiledMRJobs(rtprog);
		Statistics.resetNoOfCompiledJobs( jobCount );				
		
		//explain plan of program (hops or runtime)
		if( EXPLAIN != ExplainType.NONE ) {
			LOG.info("EXPLAIN ("+EXPLAIN.toString()+"):\n" 
					 + Explain.explainMemoryBudget()+"\n"
					 + Explain.explainDegreeOfParallelism()
					 + Explain.explain(prog, rtprog, EXPLAIN));
		}
				
		//double costs = CostEstimationWrapper.getTimeEstimate(rtprog, ExecutionContextFactory.createContext());
		//System.out.println("Estimated costs: "+costs);
		
		
		/////////////////////////// execute program //////////////////////////////////////
		Statistics.startRunTimer();
		ExecutionContext ec = null;
		try 
		{  
			initHadoopExecution( conf );
			
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
			cleanupHadoopExecution( conf );		
		}
	} 
	

	/**
	 * @throws ParseException 
	 * @throws IOException 
	 * @throws DMLRuntimeException 
	 * 
	 */
	private static void initHadoopExecution( DMLConfig config ) 
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
				for( String g : groups )
					groupNames.add( g );
			}
		}catch(Exception ex){}
		
		//analyze hadoop configuration
		JobConf job = ConfigurationManager.getCachedJobConf();
		boolean localMode     = InfrastructureAnalyzer.isLocalMode(job);
		String taskController = job.get("mapred.task.tracker.task-controller", "org.apache.hadoop.mapred.DefaultTaskController");
		String ttGroupName    = job.get("mapreduce.tasktracker.group","null");
		String perm           = job.get(MRConfigurationNames.DFS_PERMISSIONS,"null"); //note: job.get("dfs.permissions.supergroup",null);
		URI fsURI             = FileSystem.getDefaultUri(job);

		//determine security states
		boolean flagDiffUser = !(   taskController.equals("org.apache.hadoop.mapred.LinuxTaskController") //runs map/reduce tasks as the current user
							     || localMode  // run in the same JVM anyway
							     || groupNames.contains( ttGroupName) ); //user in task tracker group 
		boolean flagLocalFS = fsURI==null || fsURI.getScheme().equals("file");
		boolean flagSecurity = perm.equals("yes"); 
		
		LOG.debug("SystemML security check: " + "local.user.name = " + userName + ", " + "local.user.groups = " + ProgramConverter.serializeStringCollection(groupNames) + ", "
				        + "mapred.job.tracker = " + job.get("mapred.job.tracker") + ", " + "mapred.task.tracker.task-controller = " + taskController + "," + "mapreduce.tasktracker.group = " + ttGroupName + ", "
				        + "fs.default.name = " + ((fsURI!=null)?fsURI.getScheme():"null") + ", " + MRConfigurationNames.DFS_PERMISSIONS+" = " + perm );

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
		JobConf job = new JobConf();
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
	private static void printInvocationInfo(String fnameScript, String fnameOptConfig, HashMap<String,String> argVals)
	{		
		LOG.debug("****** args to DML Script ******\n" + "UUID: " + getUUID() + "\n" + "SCRIPT PATH: " + fnameScript + "\n" 
	                + "VISUALIZE: "  + VISUALIZE + "\n" 
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
			DMLConfig conf = DMLConfig.readAndMergeConfigurationFiles(null);
			
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
