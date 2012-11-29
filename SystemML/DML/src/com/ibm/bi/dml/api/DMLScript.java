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
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Properties;
import java.util.Scanner;

import javax.xml.parsers.ParserConfigurationException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.security.UserGroupInformation;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.nimble.configuration.NimbleConfig;
import org.nimble.control.DAGQueue;
import org.nimble.control.PMLDriver;
import org.w3c.dom.Element;
import org.xml.sax.SAXException;

import com.ibm.bi.dml.hops.Hops;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.packagesupport.PackageRuntimeException;
import com.ibm.bi.dml.parser.DMLProgram;
import com.ibm.bi.dml.parser.DMLQLParser;
import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.parser.ParseException;
import com.ibm.bi.dml.runtime.controlprogram.LocalVariableMap;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.caching.CacheableData;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ProgramConverter;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.ConfigurationManager;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDHandler;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.util.LocalFileUtils;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.sql.sqlcontrolprogram.ExecutionContext;
import com.ibm.bi.dml.sql.sqlcontrolprogram.NetezzaConnector;
import com.ibm.bi.dml.sql.sqlcontrolprogram.SQLProgram;
import com.ibm.bi.dml.utils.DMLException;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;
import com.ibm.bi.dml.utils.HopsException;
import com.ibm.bi.dml.utils.LanguageException;
import com.ibm.bi.dml.utils.LopsException;
import com.ibm.bi.dml.utils.Statistics;
import com.ibm.bi.dml.utils.configuration.DMLConfig;
import com.ibm.bi.dml.utils.visualize.DotGraph;


public class DMLScript {
	//TODO: VISUALIZE option should be updated
	public enum EXECUTION_PROPERTIES {VISUALIZE, RUNTIME_PLATFORM, CONFIG};
	public static boolean VISUALIZE = false;
	public enum RUNTIME_PLATFORM { HADOOP, SINGLE_NODE, HYBRID, NZ, INVALID };
	// We should assume the default value is HYBRID
	public static RUNTIME_PLATFORM rtplatform = RUNTIME_PLATFORM.HYBRID;
	public static String _uuid = IDHandler.createDistributedUniqueID(); 
	
	private String _dmlScriptString;
	// stores name of the OPTIONAL config file
	private String _optConfig;
	// stores optional args to parameterize DML script 
	private HashMap<String, String> _argVals;
	
	private static final Log LOG = LogFactory.getLog(DMLScript.class.getName());
	
	public static final String DEFAULT_SYSTEMML_CONFIG_FILEPATH = "./SystemML-config.xml";

	// stores the path to the source
	private static final String PATH_TO_SRC = "./";
	
	public static String USAGE = "Usage is " + DMLScript.class.getCanonicalName() 
			+ " [-f | -s] <filename>" + " -exec <mode>" +  /*" (-nz)?" + */ " (-config=<config_filename>)? (-args)? <args-list>? \n" 
			+ " -f: <filename> will be interpreted as a filename path + \n"
			+ "     <filename> prefixed with hdfs: is hdfs file, otherwise it is local file + \n" 
			+ " -s: <filename> will be interpreted as a DML script string \n"
			+ " -exec: <mode> (optional) execution mode (hadoop, singlenode, hybrid)\n"
			+ " [-v | -visualize]: (optional) use visualization of DAGs \n"
			+ " -config: (optional) use config file <config_filename> (default: use parameter values in default SystemML-config.xml config file) \n" 
			+ "          <config_filename> prefixed with hdfs: is hdfs file, otherwise it is local file + \n"
			+ " -args: (optional) parameterize DML script with contents of [args list], ALL args after -args flag \n"
			+ "    1st value after -args will replace $1 in DML script, 2nd value will replace $2 in DML script, and so on."
			+ "<args-list>: (optional) args to DML script \n" ;
			
	public DMLScript (){
		
	}
	
	public DMLScript(String dmlScript, boolean visualize, RUNTIME_PLATFORM rt, String config, HashMap<String, String> argVals){
		_dmlScriptString = dmlScript;
		VISUALIZE = visualize;
		rtplatform = rt;
		_optConfig = config;
		_argVals = argVals;
		
	}
	
	
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
	private void run()throws ParseException, IOException, DMLRuntimeException, LanguageException, HopsException, LopsException, DMLUnsupportedOperationException {
				
		LOG.info("BEGIN DML run " + getDateTime());
		LOG.debug("DML script: \n" + _dmlScriptString);
		
		if (rtplatform == RUNTIME_PLATFORM.HADOOP || rtplatform == RUNTIME_PLATFORM.HYBRID) {
			String hadoop_home = System.getenv("HADOOP_HOME");
			LOG.info("HADOOP_HOME: " + hadoop_home);
		}
		
		// optional config specified overwrites/merge into the default config
		DMLConfig defaultConfig = null;
		DMLConfig optionalConfig = null;
		
		if (_optConfig != null) { // the optional config is specified
			try { // try to get the default config first 
				defaultConfig = new DMLConfig(DEFAULT_SYSTEMML_CONFIG_FILEPATH);
			} catch (Exception e) { // it is ok to not have the default
				defaultConfig = null;
				LOG.warn("Default config file " + DEFAULT_SYSTEMML_CONFIG_FILEPATH + " not provided ");
			}
			try { // try to get the optional config next
				optionalConfig = new DMLConfig(_optConfig);	
			} 
			catch (ParseException e) { // it is not ok as the specification is wrong
				optionalConfig = null;
				throw e;
			}
			if (defaultConfig != null) {
				try {
					defaultConfig.merge(optionalConfig);
				}
				catch(ParseException e){
					defaultConfig = null;
					throw e;
				}
			}
			else {
				defaultConfig = optionalConfig;
			}
		}
		else { // the optional config is not specified
			try { // try to get the default config 
				defaultConfig = new DMLConfig(DEFAULT_SYSTEMML_CONFIG_FILEPATH);
			} catch (ParseException e) { // it is not OK to not have the default
				defaultConfig = null;
				throw e;
			}
		}
		
		ConfigurationManager.setConfig(defaultConfig);
		
		
		////////////////print config file parameters /////////////////////////////
		LOG.debug("\nDML config for this run: \n" + defaultConfig.getConfigInfo());
		
		
		///////////////////////////////////// parse script ////////////////////////////////////////////
		DMLProgram prog = null;
		DMLQLParser parser = new DMLQLParser(_dmlScriptString, _argVals);
	
		prog = parser.parse();


		if (prog == null){
			throw new ParseException("DMLQLParser parsing returns a NULL object");
		}
		
		/////////////////////////// construct HOPS ///////////////////////////////
		DMLTranslator dmlt = new DMLTranslator(prog);

		dmlt.liveVariableAnalysis(prog);			

		
		dmlt.validateParseTree(prog);
	
		//TODO: Doug will work on the format of prog.toString()
		LOG.debug("\nCOMPILER: \n" + prog.toString());

		dmlt.constructHops(prog);
		
		/*
		if(LOG.isDebugEnabled()){
			System.out.println("********************** HOPS DAG (Before Rewrite) *******************");
			// print
			dmlt.printHops(prog);
			dmlt.resetHopsDAGVisitStatus(prog);
		}*/
		
		// plan visualization (hops before rewrite)
		if( VISUALIZE ) {
			DotGraph gt = new DotGraph();
			gt.drawHopsDAG(prog, "HopsDAG Before Rewrite", 50, 50, PATH_TO_SRC, VISUALIZE);
			dmlt.resetHopsDAGVisitStatus(prog);
		}
	
		// rewrite HOPs DAGs
		// defaultConfig contains reconciled information for config
		dmlt.rewriteHopsDAG(prog, defaultConfig);
		dmlt.resetHopsDAGVisitStatus(prog);
		
		if (LOG.isDebugEnabled()) {
			LOG.debug("\n********************** HOPS DAG (After Rewrite) *******************");
			// print
			dmlt.printHops(prog);
			dmlt.resetHopsDAGVisitStatus(prog);
		}
		
		// plan visualization (hops after rewrite)
		if( VISUALIZE ) {
			DotGraph gt = new DotGraph();
			gt.drawHopsDAG(prog, "HopsDAG After Rewrite", 100, 100, PATH_TO_SRC, VISUALIZE);
			dmlt.resetHopsDAGVisitStatus(prog);
		}
		

		executeHadoop(dmlt, prog, defaultConfig);
	}
	
	
	/**
	 * executeScript: Execute a DML script, which is provided by the user as a file path to the script file.
	 * @param scriptPathName Path to the DML script file
	 * @param scriptArguments Variant arguments provided by the user to run with the DML Script
	 * @throws ParseException 
	 * @throws IOException 
	 * @throws DMLException 
	 */
	public boolean executeScript (String scriptPathName, String... scriptArguments) throws IOException, ParseException, DMLException{
		boolean success = executeScript(scriptPathName, new Configuration(), (Properties)null, scriptArguments);
		return success;
	}
	/**
	 * executeScript: Execute a DML script, which is provided by the user as a file path to the script file.
	 * @param scriptPathName Path to the DML script file
	 * @param executionProperties DMLScript runtime and debug settings
	 * @param scriptArguments Variant arguments provided by the user to run with the DML Script
	 * @throws ParseException 
	 * @throws IOException 
	 * @throws HopsException 
	 * @throws LanguageException 
	 * @throws DMLRuntimeException 
	 * @throws DMLUnsupportedOperationException 
	 * @throws LopsException 
	 */

	public boolean executeScript (String scriptPathName, Configuration conf, Properties executionProperties, String... scriptArguments) throws IOException, ParseException, DMLException{
		
		VISUALIZE = false;
		
		String debug = conf.get("systemml.logging");
		
		if (debug == null){
			debug = System.getProperty("systemml.logging");
		}
		
		if (debug != null){
			if (debug.toLowerCase().equals("debug")){
				Logger.getLogger("com.ibm.bi.dml").setLevel((Level) Level.DEBUG);
			}
			else if (debug.toLowerCase().equals("trace")){
				Logger.getLogger("com.ibm.bi.dml").setLevel((Level) Level.TRACE);
			}
		}
			
		_dmlScriptString = null;
		_optConfig = null;
		_argVals = new HashMap<String, String>();
		
		//Process the script path, get the content of the script
		StringBuilder dmlScriptString = new StringBuilder();
		
		if (scriptPathName == null){
			throw new LanguageException("DML script path was not provided by the user");
		}
		else {
			String s1 = null;
			BufferedReader in = null;
			//TODO: update this hard coded line
			try {
				if (scriptPathName.startsWith("hdfs:")){ 
					FileSystem hdfs = FileSystem.get(new Configuration());
					Path scriptPath = new Path(scriptPathName);
					in = new BufferedReader(new InputStreamReader(hdfs.open(scriptPath)));
				}
				else { // from local file system
					in = new BufferedReader(new FileReader(scriptPathName));
				}
				while ((s1 = in.readLine()) != null)
					dmlScriptString.append(s1 + "\n");
			}
			catch (IOException ex){
				LOG.error("Failed to read the script from the file system", ex);
				throw ex;
			}
			finally {
				in.close();
			}
		}
		_dmlScriptString=dmlScriptString.toString();
		
		try {
			processExecutionProperties(executionProperties);
			processOptionalScriptArgs(scriptArguments);
		
			LOG.debug("****** args to DML Script ******\n" + "UUID: " + getUUID() + "\n" + "SCRIPT PATH: " + scriptPathName + "\n" 
		                + "VISUALIZE: "  + VISUALIZE + "\n" 
		                + "RUNTIME: " + rtplatform + "\n" + "BUILTIN CONFIG: " + DEFAULT_SYSTEMML_CONFIG_FILEPATH + "\n"
		                + "OPTIONAL CONFIG: " + _optConfig + "\n");
		
			if (_argVals.size() > 0) {
				LOG.debug("Script arguments are: \n");
				for (int i=1; i<= _argVals.size(); i++)
					LOG.debug("Script argument $" + i + " = " + _argVals.get("$" + i) );
			}
		
			run();
		}
		catch (IOException e){
			LOG.error("Failed in executing DML script with SystemML engine, IO failure detected", e);
			throw e;
		}
		catch (ParseException e){
			LOG.error("Failed in executing DML script with SystemML engine, parsing failure detected", e);
			throw e;
		}
		catch (DMLException e){
			LOG.error("Failed in executing DML script with SystemML engine, DML exception detected", e);
			throw e;
		}
		finally{
			resetExecutionOptions();	
		}
		
		return true;
	}
	
	
	/**
	 * executeScript: Execute a DML script. The content of the script is provided by the user as an input stream.
	 * @param script InputStream as the DML script
	 * @param executionProperties DMLScript runtime and debug settings
	 * @param scriptArguments Variant arguments provided by the user to run with the DML Script
	 * @throws ParseException 
	 * @throws IOException 
	 * @throws LanguageException 
	 * @throws HopsException 
	 * @throws DMLRuntimeException 
	 * @throws DMLUnsupportedOperationException 
	 * @throws LopsException 
	 */
	public boolean executeScript (InputStream script,  Configuration conf, Properties executionProperties, String... scriptArguments) throws IOException, ParseException, DMLException{
		VISUALIZE = false;
		
		String debug = conf.get("systemml.logging");
		
		if (debug == null){
			debug = System.getProperty("systemml.logging");
		}
		
		if (debug != null){
			if (debug.toLowerCase().equals("debug")){
				Logger.getLogger("com.ibm.bi.dml").setLevel((Level) Level.DEBUG);
			}
			else if (debug.toLowerCase().equals("trace")){
				Logger.getLogger("com.ibm.bi.dml").setLevel((Level) Level.TRACE);
			}
		}
		
		_dmlScriptString = null;
		_optConfig = null;
		_argVals = new HashMap<String, String>();
		
		if (script == null){
			LOG.error("DML script path was not provided by the user");
			throw new LanguageException("DML script path was not provided by the user");
		}
		else {		
			_dmlScriptString = new Scanner(script).useDelimiter("\\A").next();
		}
		
		try {
			processExecutionProperties(executionProperties);
			processOptionalScriptArgs(scriptArguments);
			LOG.debug("****** args to DML Script ******\n" + "UUID: " + getUUID() + "\n" + "SCRIPT PATH: " + _dmlScriptString + "\n" 
                + "VISUALIZE: "  + VISUALIZE + "\n" 
                + "RUNTIME: " + rtplatform + "\n" + "BUILTIN CONFIG: " + DEFAULT_SYSTEMML_CONFIG_FILEPATH + "\n"
                + "OPTIONAL CONFIG: " + _optConfig + "\n");

			if (_argVals.size() > 0) {
				LOG.debug("Script arguments are: \n");
				for (int i=1; i<= _argVals.size(); i++)
					LOG.debug("Script argument $" + i + " = " + _argVals.get("$" + i) );
			}

			run();
		}
		catch (IOException e){
			LOG.error("Failed in executing DML script with SystemML engine, IO failure detected", e);
			throw e;
		}
		catch (ParseException e){
			LOG.error("Failed in executing DML script with SystemML engine, parsing failure detected", e);
			throw e;
		}
		catch (DMLException e){
			LOG.error("Failed in executing DML script with SystemML engine, DML exception detected", e);
			throw e;
		}
		finally{
			resetExecutionOptions();	
		}
		
		return true;
	}
	
	private void resetExecutionOptions(){
		VISUALIZE = false;
		rtplatform = RUNTIME_PLATFORM.HYBRID;
		_optConfig = null;
	}
	
	//Process execution properties
	private void processExecutionProperties(Properties executionProperties) throws LanguageException {
		
		if (executionProperties != null){
			//Make sure that the properties are in the defined property list that can be handled
			@SuppressWarnings("unchecked")
			Enumeration<String> e = (Enumeration<String>) executionProperties.propertyNames();

			while (e.hasMoreElements()){
				String key = e.nextElement();
				boolean validProperty = false;
				for (EXECUTION_PROPERTIES p : EXECUTION_PROPERTIES.values()){
					if (p.name().equals(key)){
						validProperty = true;
						break;
					}
				}
				if (!validProperty){
					resetExecutionOptions();
					throw new LanguageException("Unknown execution property: " + key);
				}

			}

			VISUALIZE = Boolean.valueOf(executionProperties.getProperty(EXECUTION_PROPERTIES.VISUALIZE.toString(), "false"));

			String runtime_pt = executionProperties.getProperty(EXECUTION_PROPERTIES.RUNTIME_PLATFORM.toString(), "hybrid");
			if (runtime_pt.equalsIgnoreCase("hadoop"))
				rtplatform = RUNTIME_PLATFORM.HADOOP;
			else if ( runtime_pt.equalsIgnoreCase("singlenode"))
				rtplatform = RUNTIME_PLATFORM.SINGLE_NODE;
			else if ( runtime_pt.equalsIgnoreCase("hybrid"))
				rtplatform = RUNTIME_PLATFORM.HYBRID;
			else if ( runtime_pt.equalsIgnoreCase("nz"))
				rtplatform = RUNTIME_PLATFORM.NZ;

			_optConfig = executionProperties.getProperty(EXECUTION_PROPERTIES.CONFIG.toString(), null);
		}
		else {
			resetExecutionOptions();
		}
		
	}
	
	//Process the optional script arguments provided by the user to run with the DML script
	private void processOptionalScriptArgs(String... scriptArguments) throws LanguageException{
		
		if (scriptArguments != null){
			int index = 1;
			for (String arg : scriptArguments){
				if (arg.equalsIgnoreCase("-l") || arg.equalsIgnoreCase("-log") ||
					arg.equalsIgnoreCase("-v") || arg.equalsIgnoreCase("-visualize")||
					arg.equalsIgnoreCase("-exec") ||
					arg.startsWith("-config=")){
					resetExecutionOptions();
					throw new LanguageException("-args must be the final argument for DMLScript!");
			}
						
				_argVals.put("$"+index ,arg);
				index++;
			}

		}
	}
	
	
	/**
	 * @param args
	 * @throws ParseException
	 * @throws IOException
	 * @throws SAXException
	 * @throws ParserConfigurationException
	 */
	public static void main(String[] args) throws IOException, ParseException, DMLException {
		// This is a show case how to create a DMLScript object to accept a DML script provided by the user,
		// and how to run it.
		
		 Configuration conf = new Configuration();
		 String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
		/////////// if the args is incorrect, print usage /////////////
		if (otherArgs.length < 2){
			//System.err.println(USAGE);
			return;
		}
		////////////process -f | -s to set dmlScriptString ////////////////
		else if (!(otherArgs[0].equals("-f") || otherArgs[0].equals("-s"))){
			System.err.println("ERROR: First argument must be either -f or -s");
			//System.err.println(USAGE);
			return;
		}
		
		DMLScript d = new DMLScript();
		boolean fromFile = (otherArgs[0].equals("-f")) ? true : false;
		boolean success = false;
		String script = otherArgs[1];	
		Properties executionProperties = new Properties();
		String[] scriptArgs = null;
		int i = 2;
		while (i<otherArgs.length){
			if (otherArgs[i].equalsIgnoreCase("-v") || otherArgs[i].equalsIgnoreCase("-visualize")) {
				executionProperties.put(EXECUTION_PROPERTIES.VISUALIZE.toString(), "true");
			} else if ( otherArgs[i].equalsIgnoreCase("-exec")) {
				i++;
				if ( otherArgs[i].equalsIgnoreCase("hadoop")) 
					executionProperties.put(EXECUTION_PROPERTIES.RUNTIME_PLATFORM.toString(), "hadoop");
				else if ( otherArgs[i].equalsIgnoreCase("singlenode"))
					executionProperties.put(EXECUTION_PROPERTIES.RUNTIME_PLATFORM.toString(), "singlenode");
				else if ( otherArgs[i].equalsIgnoreCase("hybrid"))
					executionProperties.put(EXECUTION_PROPERTIES.RUNTIME_PLATFORM.toString(), "hybrid");
				else if ( otherArgs[i].equalsIgnoreCase("nz"))
					executionProperties.put(EXECUTION_PROPERTIES.RUNTIME_PLATFORM.toString(), "nz");
				else {
					System.err.println("ERROR: Unknown runtime platform: " + otherArgs[i]);
					return;
				}
			// handle config file
			} else if (otherArgs[i].startsWith("-config=")){
				executionProperties.put(EXECUTION_PROPERTIES.CONFIG.toString(), otherArgs[i].substring(8).replaceAll("\"", "")); 
			}
			// handle the args to DML Script -- rest of args will be passed here to 
			else if (otherArgs[i].startsWith("-args")) {
				i++;
				scriptArgs = new String[otherArgs.length - i];
				int j = 0;
				while( i < otherArgs.length){
					scriptArgs[j++]=otherArgs[i++];
				}
			} 
			else {
				System.err.println("ERROR: Unknown argument: " + otherArgs[i]);
				return;
			}
			i++;
		}
		if (fromFile){
			success = d.executeScript(script, conf, executionProperties, scriptArgs);
		}
		else {
			InputStream is = new ByteArrayInputStream(script.getBytes());
			success = d.executeScript(is, conf, executionProperties, scriptArgs);
		}
		
		if (!success){
			System.err.println("ERROR: Script cannot be executed!");
			return;
		}
	} ///~ end main

	
	/**
	 * executeHadoop: Handles execution on the Hadoop Map-reduce runtime
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
	private static void executeHadoop(DMLTranslator dmlt, DMLProgram prog, DMLConfig config) throws ParseException, IOException, LanguageException, HopsException, LopsException, DMLRuntimeException, DMLUnsupportedOperationException {
		
		LOG.debug("\n********************** OPTIMIZER *******************\n" + 
		          "Type = " + OptimizerUtils.getOptType() + "\n"
				 +"Mode = " + OptimizerUtils.getOptMode() + "\nAvailable Memory = " + ((double)InfrastructureAnalyzer.getLocalMaxMemory()/1024/1024) + " MB" + "\n"
				 +"Memory Budget = " + ((double)Hops.getMemBudget(true)/1024/1024) + " MB" + "\n"
				 +"Defaults: mem util " + OptimizerUtils.MEM_UTIL_FACTOR + ", sparsity " + OptimizerUtils.DEF_SPARSITY + ", def mem " +  + OptimizerUtils.DEF_MEM_FACTOR);
			
		/////////////////////// construct the lops ///////////////////////////////////
		dmlt.constructLops(prog);

		if (LOG.isDebugEnabled()) {
			LOG.debug("\n********************** LOPS DAG *******************");
			dmlt.printLops(prog);
			dmlt.resetLopsDAGVisitStatus(prog);
		}

		// plan visualization (hops after rewrite)
		if(VISUALIZE){
			DotGraph gt = new DotGraph();
			gt.drawLopsDAG(prog, "LopsDAG", 150, 150, PATH_TO_SRC, VISUALIZE);
			dmlt.resetLopsDAGVisitStatus(prog);
		}
		

		////////////////////// generate runtime program ///////////////////////////////
		Program rtprog = null;

			rtprog = prog.getRuntimeProgram(config);
		
		//setup nimble queue (external package support)
		DAGQueue dagQueue = setupNIMBLEQueue(config);
		if (dagQueue == null)
			LOG.warn("dagQueue is not set");
		rtprog.setDAGQueue(dagQueue);
		
		//count number compiled MR jobs	
		int jobCount = DMLProgram.countCompiledMRJobs(rtprog);
		Statistics.setNoOfCompiledMRJobs( jobCount );
		
		/*
		if (DEBUG) {
			System.out.println("********************** Instructions *******************");
			//TODO: Leo this should be deleted
			System.out.println(rtprog.toString());
			rtprog.printMe();

			// visualize
			DotGraph gt = new DotGraph();
         	gt.drawInstructionsDAG(rtprog, "InstructionsDAG", 200, 200, PATH_TO_SRC, VISUALIZE);

			System.out.println("********************** Execute *******************");
		}
		
		LOG.info("********************** Instructions *******************");
		rtprog.printMe();
		LOG.info("*******************************************************");*/

		LOG.trace("Compile Status for executeHadoop is OK ");

		/////////////////////////// execute program //////////////////////////////////////
		Statistics.startRunTimer();		
		try 
		{   
			initHadoopExecution( config );
			
			//run execute (w/ exception handling to ensure proper shutdown)
			rtprog.execute (new LocalVariableMap (), null);  
		}
		finally //ensure cleanup/shutdown
		{			
			//TODO: Should statistics being turned on at info level?
			Statistics.stopRunTimer();
	
			//TODO: System.out is for running with JAQL shell, eventually we hope JAQL shell will not use
			// its own log4j.properties
			LOG.info(Statistics.display());
			
			LOG.info("END DML run " + getDateTime() );
			//cleanup all nimble threads
			if(rtprog.getDAGQueue() != null)
		  	    rtprog.getDAGQueue().forceShutDown();
			
			//cleanup scratch_space and all working dirs
			cleanupHadoopExecution( config );
		}
	} // end executeHadoop

	/**
	 * executeNetezza: handles execution on Netezza runtime
	 * @param dmlt DML Translator
	 * @param prog DML program from parsed DML script
	 * @param config from parsed config file (e.g., config.xml)
	 * @throws ParseException 
	 * @throws HopsException 
	 * @throws DMLRuntimeException 
	 */
	private static void executeNetezza(DMLTranslator dmlt, DMLProgram prog, DMLConfig config, String fileName)
		throws HopsException, LanguageException, ParseException, DMLRuntimeException
	{
	
		dmlt.constructSQLLops(prog);
	
		SQLProgram sqlprog = dmlt.getSQLProgram(prog);
		String[] split = fileName.split("/");
		String name = split[split.length-1].split("\\.")[0];
		sqlprog.set_name(name);
		dmlt.resetSQLLopsDAGVisitStatus(prog);

		// plan visualization (hops after rewrite)
		if(VISUALIZE){
			DotGraph g = new DotGraph();
			g.drawSQLLopsDAG(prog, "SQLLopsDAG", 100, 100, PATH_TO_SRC, VISUALIZE);
			dmlt.resetSQLLopsDAGVisitStatus(prog);
		}
	
		String sql = sqlprog.generateSQLString();
	
		Program pr = sqlprog.getProgram();
		pr.printMe();
	
		if (true) {
			System.out.println(sql);
		}
	
		NetezzaConnector con = new NetezzaConnector();
		try
		{
			ExecutionContext ec = new ExecutionContext(con);
			ec.setDebug(false);
	
			LocalVariableMap vm = new LocalVariableMap ();
			ec.set_variables (vm);
			con.connect();
			long time = System.currentTimeMillis();
			pr.execute (vm, ec);
			long end = System.currentTimeMillis() - time;
			System.out.println("Control program took " + ((double)end / 1000) + " seconds");
			con.disconnect();
			System.out.println("Done");
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
		/*
		// Code to execute the stored procedure version of the DML script
		try
		{
			con.connect();
			con.executeSQL(sql);
			long time = System.currentTimeMillis();
			con.callProcedure(name);
			long end = System.currentTimeMillis() - time;
			System.out.println("Stored procedure took " + ((double)end / 1000) + " seconds");
			System.out.println(String.format("Procedure %s was executed on Netezza", name));
			con.disconnect();
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}*/

	} // end executeNetezza
	/**
	 * Method to setup the NIMBLE task queue. 
	 * This will be used in future external function invocations
	 * @param dmlCfg DMLConfig object
	 * @return NIMBLE task queue
	 */
	static DAGQueue setupNIMBLEQueue(DMLConfig dmlCfg) {

		//config not provided
		if (dmlCfg == null) 
			return null;
		
		// read in configuration files
		NimbleConfig config = new NimbleConfig();

		try {
			config.parseSystemDocuments(dmlCfg.getConfig_file_name());
			
			//ensure unique working directory for nimble output
			StringBuffer sb = new StringBuffer();
			sb.append( dmlCfg.getTextValue(DMLConfig.SCRATCH_SPACE) );
			sb.append( Lops.FILE_SEPARATOR );
			sb.append( Lops.PROCESS_PREFIX );
			sb.append( getUUID() );
			sb.append( Lops.FILE_SEPARATOR  );
			sb.append( dmlCfg.getTextValue(DMLConfig.NIMBLE_SCRATCH) );			
			((Element)config.getSystemConfig().getParameters().getElementsByTagName(DMLConfig.NIMBLE_SCRATCH).item(0))
			                .setTextContent( sb.toString() );						
		} catch (Exception e) {
			throw new PackageRuntimeException ("Error parsing Nimble configuration files", e);
		}

		// get threads configuration and validate
		int numSowThreads = 1;
		int numReapThreads = 1;

		numSowThreads = Integer.parseInt
				(NimbleConfig.getTextValue(config.getSystemConfig().getParameters(), DMLConfig.NUM_SOW_THREADS));
		numReapThreads = Integer.parseInt
				(NimbleConfig.getTextValue(config.getSystemConfig().getParameters(), DMLConfig.NUM_REAP_THREADS));
		
		if (numSowThreads < 1 || numReapThreads < 1){
			throw new PackageRuntimeException("Illegal values for thread count (must be > 0)");
		}

		// Initialize an instance of the driver.
		PMLDriver driver = null;
		try {
			driver = new PMLDriver(numSowThreads, numReapThreads, config);
			driver.startEmptyDriver(config);
		} catch (Exception e) {
			throw new PackageRuntimeException("Problem starting nimble driver", e);
		} 

		return driver.getDAGQueue();
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
		checkSecuritySetup();
		
		//create scratch space with appropriate permissions
		String scratch = config.getTextValue(DMLConfig.SCRATCH_SPACE);
		MapReduceTool.createDirIfNotExistOnHDFS(scratch, DMLConfig.DEFAULT_SHARED_DIR_PERMISSION);
		
		//cleanup working dirs from previous aborted runs with same pid in order to prevent conflicts
		cleanupHadoopExecution(config); 
		
		//init caching (incl set active)
		LocalFileUtils.createWorkingDirectory();
		CacheableData.initCaching();
						
		//reset statistics (required if multiple scripts executed in one JVM)
		Statistics.setNoOfExecutedMRJobs( 0 );
	
	}
	
	/**
	 * 
	 * @throws IOException
	 * @throws DMLRuntimeException 
	 */
	private static void checkSecuritySetup() 
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
		}catch(Exception ex){
			LOG.warn("Failed in checking current user and group security info: " + ex.getStackTrace());
		}
		
		//analyze hadoop configuration
		JobConf job = new JobConf();
		String jobTracker     = job.get("mapred.job.tracker", "local");
		String taskController = job.get("mapred.task.tracker.task-controller", "org.apache.hadoop.mapred.DefaultTaskController");
		String ttGroupName    = job.get("mapreduce.tasktracker.group","null");
		String perm           = job.get("dfs.permissions","null"); //note: job.get("dfs.permissions.supergroup",null);
		URI fsURI             = FileSystem.getDefaultUri(job);

		//determine security states
		boolean flagDiffUser = !(   taskController.equals("org.apache.hadoop.mapred.LinuxTaskController") //runs map/reduce tasks as the current user
							     || jobTracker.equals("local")  // run in the same JVM anyway
							     || groupNames.contains( ttGroupName) ); //user in task tracker group 
		boolean flagLocalFS = fsURI==null || fsURI.getScheme().equals("file");
		boolean flagSecurity = perm.equals("yes"); 
		
		//TODO format should stay the same
		LOG.debug("SystemML security check: " + "local.user.name = " + userName + ", " + "local.user.groups = " + ProgramConverter.serializeStringHashSet(groupNames) + ", "
				        + "mapred.job.tracker = " + jobTracker + ", " + "mapred.task.tracker.task-controller = " + taskController + "," + "mapreduce.tasktracker.group = " + ttGroupName + ", "
				        + "fs.default.name = " + fsURI.getScheme() + ", " + "dfs.permissions = " + perm );

		//print warning if permission issues possible
		if( flagDiffUser && ( flagLocalFS || flagSecurity ) )
		{
			LOG.warn("Cannot run map/reduce tasks as user '"+userName+"'. Using tasktracker group '"+ttGroupName+"'."); 		 
		}
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
		sb.append(Lops.FILE_SEPARATOR);
		sb.append(Lops.PROCESS_PREFIX);
		sb.append(DMLScript.getUUID());
		String dirSuffix = sb.toString();
		
		//1) cleanup scratch space (everything for current uuid) 
		//(required otherwise export to hdfs would skip assumed unnecessary writes if same name)
		MapReduceTool.deleteFileIfExistOnHDFS( config.getTextValue(DMLConfig.SCRATCH_SPACE) + dirSuffix );
		
		//2) cleanup hadoop working dirs (only required for LocalJobRunner (local job tracker), because
		//this implementation does not create job specific sub directories)
		if( MRJobConfiguration.isLocalJobTracker() ) {
			try 
			{
				LocalFileUtils.deleteFileIfExists( DMLConfig.LOCAL_MR_MODE_STAGING_DIR + //staging dir (for local mode only) 
					                                   dirSuffix  );	
				LocalFileUtils.deleteFileIfExists( MRJobConfiguration.getLocalWorkingDirPrefix() + //local dir
		                                               dirSuffix );
				MapReduceTool.deleteFileIfExistOnHDFS( MRJobConfiguration.getSystemWorkingDirPrefix() + //system dir
													   dirSuffix  );
				MapReduceTool.deleteFileIfExistOnHDFS( MRJobConfiguration.getStagingWorkingDirPrefix() + //staging dir
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
	
	
	private static String getDateTime() {
		DateFormat dateFormat = new SimpleDateFormat("MM/dd/yyyy HH:mm:ss");
		Date date = new Date();
		return dateFormat.format(date);
	}

		
	public void setDMLScriptString(String dmlScriptString){
		_dmlScriptString = dmlScriptString;
	}
	
	public String getDMLScriptString (){
		return _dmlScriptString;
	}
	
	public void setOptConfig (String optConfig){
		_optConfig = optConfig;
	}
	
	public String getOptConfig (){
		return _optConfig;
	}
	
	public void setArgVals (HashMap<String, String> argVals){
		_argVals = argVals;
	}
	
	public HashMap<String, String> getArgVals() {
		return _argVals;
	}
	
	public static String getUUID()
	{
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
	
	
}  ///~ end class
