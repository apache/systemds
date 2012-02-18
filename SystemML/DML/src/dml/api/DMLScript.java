package dml.api;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;

import javax.xml.parsers.ParserConfigurationException;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.nimble.configuration.NimbleConfig;
import org.nimble.control.DAGQueue;
import org.nimble.control.PMLDriver;
import org.nimble.exception.ConfigurationException;
import org.xml.sax.SAXException;

import dml.packagesupport.PackageRuntimeException;
import dml.parser.DMLProgram;
import dml.parser.DMLQLParser;
import dml.parser.DMLTranslator;
import dml.parser.ParseException;
import dml.runtime.controlprogram.Program;
import dml.runtime.controlprogram.ProgramBlock;
import dml.runtime.controlprogram.WhileProgramBlock;
import dml.runtime.instructions.CPInstructions.Data;
import dml.runtime.instructions.Instruction.INSTRUCTION_TYPE;
import dml.sql.sqlcontrolprogram.ExecutionContext;
import dml.sql.sqlcontrolprogram.NetezzaConnector;
import dml.sql.sqlcontrolprogram.SQLProgram;
import dml.utils.DMLException;
import dml.utils.HopsException;
import dml.utils.LanguageException;
import dml.utils.Statistics;
import dml.utils.configuration.DMLConfig;
import dml.utils.visualize.DotGraph;


public class DMLScript {

	public static String USAGE = "Usage is " + DMLScript.class.getCanonicalName() 
			+ " [-f | -s] <filename>" + /*" (-nz)?" + */ " [-d | -debug]?" + " [-v | -visualize]?"  + " [-l | -log]?" + " (-config=<config_filename>)? (-args)? <args-list>? \n" 
			+ " -f: <filename> will be interpreted as a filename path \n"
			+ " -s: <filename> will be interpreted as a DML script string \n"
			// COMMENT OUT FOR BIRelease
			//+ " -nz: (optional) use Netezza runtime (default: use Hadoop runtime) \n"
			+ " [-d | -debug]: (optional) output debug info \n"
			// COMMENT OUT -v option before RELEASE
			+ " [-v | -visualize]: (optional) use visualization of DAGs \n"
			+ " [-l | -log]: (optional) output log info \n"
			+ " -config: (optional) use config file <config_filename> (default: use default SystemML-config.xml config file) \n" 
			+ " -args: (optional) parameterize DML script with contents of [args list], ALL args after -args flag \n"
			+ "    1st value after -args will replace $1 in DML script, 2nd value will replace $2 in DML script, and so on."
			+ "<args-list>: (optional) args to DML script; use double-quotes to denote string args. \n" 
			+ "NOTE: Double-quotes will have to be escaped with slash -- i.e., \"string-arg\" ";
					
	public static boolean DEBUG = false;
	public static boolean VISUALIZE = false;
	public static boolean LOG = false;	
	public enum RUNTIME_PLATFORM { HADOOP, NZ, INVALID };
	public static RUNTIME_PLATFORM rtplatform = RUNTIME_PLATFORM.HADOOP;
	public static String DEFAULT_SYSTEMML_CONFIG_FILEPATH = "./SystemML-config.xml";
	
	// stores the path to the source
	public static final String path_to_src = "./";
	
	/**
	 * @param args
	 * @throws ParseException
	 * @throws IOException
	 * @throws SAXException
	 * @throws ParserConfigurationException
	 */
	public static void main(String[] args) throws IOException, ParseException, DMLException {

		Logger mapredLogger = Logger.getLogger("org.apache.hadoop.mapred");
		Logger mmcjLogger = Logger.getLogger("dml.runtime.matrix.MMCJMR");
		// Logger jvmLogger = Logger.getLogger("jvm.JvmMetrics");			
		final String logFileName = "SystemML.log";
		
		// stores runtime platform
		rtplatform = RUNTIME_PLATFORM.HADOOP;
		
		// stores the (filename | DMLScript string) passed
		String fileName = null;
		
		// stores if <filename> arg is file (if true) or is a string (if false)
		boolean fromFile = false;
	
		// stores name of the OPTIONAL config file
		String optionalConfigurationFileName = null;
		
		//////////////// for DEBUG, dump arguments /////////////////////////////
		System.out.println("INFO: Value for args passed to DMLScript: ");
		for (int i=0; i< args.length; i++)
			System.out.println("INFO: arg " + i + " = " + args[i] );
		
	
		// print usage
		if (args.length < 2){
			System.err.println(USAGE);
			return;
		}
		
		// stores optional args to parameterize DML script 
		HashMap<String, String> argVals = new HashMap<String,String>();
		
		//////////// process -f | -s to set dmlScriptString ////////////////
		if (!((args[0].equals("-f") || args[0].equals("-s")))){
			System.err.println("first argument must be either -f or -s");
			System.err.println(USAGE);
			return;
		}
		fromFile = (args[0].equals("-f")) ? true : false;
		fileName = args[1];
		String dmlScriptString = new String();
		if (fromFile){
			BufferedReader in = new BufferedReader(new FileReader(fileName));
			String s1 = null;
			while ((s1 = in.readLine()) != null)
				dmlScriptString += s1 + "\n";
			in.close();	
		}
		else {
			dmlScriptString = fileName;
		}
		
		////////////////// process rest of args list ////////////////////////
		int argid = 2;
		while (argid < args.length) {
			
			if (args[argid].equalsIgnoreCase("-d") || args[argid].equalsIgnoreCase("-debug")) {
				DEBUG = true;
			} else if (args[argid].equalsIgnoreCase("-l") || args[argid].equalsIgnoreCase("-log")) {
				LOG = true;
			} else if (args[argid].equalsIgnoreCase("-v") || args[argid].equalsIgnoreCase("-visualize")) {
				VISUALIZE = true;
			// handle config file
			//} else if(args[argid].equalsIgnoreCase("-nz")){
			//	rtplatform = RUNTIME_PLATFORM.NZ;
		
			} else if (args[argid].startsWith("-config=")){
				optionalConfigurationFileName = args[argid].substring(8).replaceAll("\"", "");
				optionalConfigurationFileName = args[argid].substring(8).replaceAll("\'", "");	
				
			// handle the args to DML Script -- rest of args will be passed here to 
			} else if (args[argid].startsWith("-args")) {
			
				argid++;
				int index = 1;
				while( argid < args.length){
					argVals.put("$"+index ,args[argid]);
					argid++;
					index++;
				}
			}
			
			// increment counter
			argid++;
		} // while (argid < args.length) {

		//////////////// for DEBUG, dump arguments /////////////////////////////
		if (DEBUG){
			System.out.println("INFO: ****** args to DML Script ****** ");
			System.out.println("INFO: FROM-FILE: " + fromFile);
			System.out.println("INFO: SCRIPT: " + fileName);
			System.out.println("INFO: DEBUG: "  + DEBUG);
			System.out.println("INFO: VISUALIZE: "  + VISUALIZE);
			System.out.println("INFO: BUILTIN CONFIG: " + DEFAULT_SYSTEMML_CONFIG_FILEPATH);
			System.out.println("INFO: OPTIONAL CONFIG: " + optionalConfigurationFileName);
			System.out.println("INFO: RUNTIME: " + rtplatform);
			System.out.println("INFO: LOG: "  + LOG);
			
			if (argVals.size() > 0)
				System.out.println("INFO: Value for script parameter args: ");
			for (int i=1; i<= argVals.size(); i++)
				System.out.println("INFO: $" + i + " = " + argVals.get("$" + i) );
		}
	
		/////////////// set logger level //////////////////////////////////////
		if (rtplatform == RUNTIME_PLATFORM.HADOOP){
			if (DEBUG)
				mapredLogger.setLevel(Level.WARN);
			else {
				mapredLogger.setLevel(Level.WARN);
				mmcjLogger.setLevel(Level.WARN);
				// jvmLogger.setLevel(Level.FATAL);
			}
		}
		////////////// handle log output //////////////////////////
		BufferedWriter out = null;
		if (LOG && rtplatform == RUNTIME_PLATFORM.HADOOP) {
			// copy the input DML script to ./log folder
			String hadoop_home = System.getenv("HADOOP_HOME");
			File logfile = new File(hadoop_home + "/" + logFileName);
			if (!logfile.exists()) {
				boolean success = logfile.createNewFile(); // creates the file
				// if it does not
				if (success == false) 
					System.out.println("Failed to create log file: " + hadoop_home + "/" + logFileName);	
			}
		
			out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(logfile, true)));
			out.write("BEGIN DMLRun " + getDateTime() + "\n");
			out.write("BEGIN DMLScript\n");
			BufferedReader script = new BufferedReader(new InputStreamReader(new FileInputStream(fileName)));
			String s = null;
			while ((s = script.readLine()) != null) 
				out.write(s + "\n");
			out.write("END DMLScript\n");
			script.close();
		}

		//////////////////// parse the builtin config file and optional config file //////////////////////// 
		// try to read
		DMLConfig defaultConfig = null;
		try {
			defaultConfig = new DMLConfig(DEFAULT_SYSTEMML_CONFIG_FILEPATH);

		} catch (Exception e) {
			System.out.println("Error parsing default configuration file: " + DEFAULT_SYSTEMML_CONFIG_FILEPATH);
			e.printStackTrace();
			System.exit(-1);
		}
		
		// read the optional config file to overwrite changed parameter values
		DMLConfig optionalConfig = null;
		if (optionalConfigurationFileName != null) {
			try {
				optionalConfig = new DMLConfig(optionalConfigurationFileName);
					
			} catch (Exception e) {
				System.out.println("Error parsing optional configuration file: " + optionalConfigurationFileName);
				System.exit(-1);
			}
		}
		
		// update default config file with values from optional config file
		try {
			defaultConfig.merge(optionalConfig);
		}
		catch(Exception e){
			System.out.println("ERROR: failed to merge default ");
			return;
		}
		
		//////////////////////////// parse script ///////////////////////////////
		DMLProgram prog = null;
		DMLQLParser parser = new DMLQLParser(dmlScriptString,argVals);
		prog = parser.parse();
		
		if (prog == null){
			System.out.println("ERROR: Parsing failed");
			return;
		}
		
		if (DEBUG) {
			System.out.println("********************** PARSER *******************");
			System.out.println(prog.toString());
		}

		///////////////////////////////////// construct HOPS ///////////////////////////////

		DMLTranslator dmlt = new DMLTranslator(prog);
		dmlt.validateParseTree(prog);
		dmlt.liveVariableAnalysis(prog);

		if (DEBUG) {
			System.out.println("********************** COMPILER *******************");
			System.out.println(prog.toString());
		}
		dmlt.constructHops(prog);
		
		if (DEBUG) {
			System.out.println("********************** HOPS DAG (Before Rewrite) *******************");
			// print
			dmlt.printHops(prog);
			dmlt.resetHopsDAGVisitStatus(prog);

			// visualize
			DotGraph gt = new DotGraph();
			//
			// last parameter: the path of DML source directory. If dml source
			// is at /path/to/dml/src then it should be /path/to/dml.
			//
			gt.drawHopsDAG(prog, "HopsDAG Before Rewrite", 50, 50, path_to_src, VISUALIZE);
			dmlt.resetHopsDAGVisitStatus(prog);
		}

		// rewrite HOPs DAGs
		if (DEBUG) {
			System.out.println("********************** Rewriting HOPS DAG *******************");
		}

		// defaultConfig contains reconciled information for config
		dmlt.rewriteHopsDAG(prog, defaultConfig);
		dmlt.resetHopsDAGVisitStatus(prog);

		if (DEBUG) {
			System.out.println("********************** HOPS DAG (After Rewrite) *******************");
			// print
			dmlt.printHops(prog);
			dmlt.resetHopsDAGVisitStatus(prog);

			// visualize
			DotGraph gt = new DotGraph();
			gt.drawHopsDAG(prog, "HopsDAG After Rewrite", 100, 100, path_to_src, VISUALIZE);
			dmlt.resetHopsDAGVisitStatus(prog);
		}

		//if ( rtplatform == RUNTIME_PLATFORM.NZ ) {
		//	executeNetezza(dmlt, prog, config, fileName);
			
		//} else if (rtplatform == RUNTIME_PLATFORM.HADOOP ) {
			executeHadoop(dmlt, prog, defaultConfig, out);
		//}			
	} // end main

	
	/**
	 * executeHadoop: Handles execution on the Hadoop Map-reduce runtime
	 * @param dmlt DML Translator 
	 * @param prog DML Program object from parsed DML script
	 * @param config read from provided configuration file (e.g., config.xml)
	 * @param out writer for log output 
	 * @throws ParseException 
	 * @throws IOException 
	 * @throws DMLException 
	 */
	private static void executeHadoop(DMLTranslator dmlt, DMLProgram prog, DMLConfig config, BufferedWriter out) throws ParseException, IOException, DMLException{
		
		/////////////////////// construct the lops ///////////////////////////////////
		dmlt.constructLops(prog);

		if (DEBUG) {
			System.out.println("********************** LOPS DAG *******************");
			dmlt.printLops(prog);
			dmlt.resetLopsDAGVisitStatus(prog);

			DotGraph gt = new DotGraph();
			gt.drawLopsDAG(prog, "LopsDAG", 150, 150, path_to_src, VISUALIZE);
			dmlt.resetLopsDAGVisitStatus(prog);
		}

		////////////////////// generate runtime program ///////////////////////////////
		Program rtprog = prog.getRuntimeProgram(DEBUG, config);
		DAGQueue dagQueue = setupNIMBLEQueue(config);
		if (DEBUG && config == null){
			System.out.println("INFO: config is null -- you may need to verify config file path");
		}
		if (DEBUG && dagQueue == null){
			System.out.println("INFO: dagQueue is not set");
		}
		
		
		rtprog.setDAGQueue(dagQueue);

		
		// Count number compiled MR jobs
		int jobCount = 0;
		for (ProgramBlock blk : rtprog.getProgramBlocks()) 
			jobCount += countCompiledJobs(blk); 		
		Statistics.setNoOfCompiledMRJobs(jobCount);

		
		// TODO: DRB --- graph is definitely broken; printMe() is okay
		if (DEBUG) {
			System.out.println("********************** Instructions *******************");
			System.out.println(rtprog.toString());
			rtprog.printMe();

			// visualize
			//DotGraph gt = new DotGraph();
			//gt.drawInstructionsDAG(rtprog, "InstructionsDAG", 200, 200, path_to_src);

			System.out.println("********************** Execute *******************");
		}

		if (LOG) 
			out.write("Compile Status OK\n");
		

		/////////////////////////// execute program //////////////////////////////////////
		Statistics.startRunTimer();		
		rtprog.execute(new HashMap<String,Data>(), null);  
		Statistics.stopRunTimer();

		if (DEBUG) 
			System.out.println(Statistics.display());
		
		if (LOG) {
			out.write("END DMLRun " + getDateTime() + "\n");
			out.close();
		}

		if(rtprog.getDAGQueue() != null)
	  	    rtprog.getDAGQueue().forceShutDown();
		
	} // end executeHadoop

	
	/**
	 * executeNetezza: handles execution on Netezza runtime
	 * @param dmlt DML Translator
	 * @param prog DML program from parsed DML script
	 * @param config from parsed config file (e.g., config.xml)
	 * @throws ParseException 
	 * @throws HopsException 
	 */
	private static void executeNetezza(DMLTranslator dmlt, DMLProgram prog, DMLConfig config, String fileName) throws HopsException, LanguageException, ParseException{
	
		dmlt.constructSQLLops(prog);
	
		SQLProgram sqlprog = dmlt.getSQLProgram(prog);
		String[] split = fileName.split("/");
		String name = split[split.length-1].split("\\.")[0];
		sqlprog.set_name(name);
	
		dmlt.resetSQLLopsDAGVisitStatus(prog);
		DotGraph g = new DotGraph();
		g.drawSQLLopsDAG(prog, "SQLLops DAG", 100, 100, path_to_src, VISUALIZE);
		dmlt.resetSQLLopsDAGVisitStatus(prog);
	
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
	
			HashMap<String,Data> hm = new HashMap<String,Data>();
			ec.set_variables(hm);
			con.connect();
			long time = System.currentTimeMillis();
			pr.execute(hm, ec);
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
		if(dmlCfg == null)
			return null;
		
		// read in configuration files
		NimbleConfig config = new NimbleConfig();


		if(dmlCfg.getTextValue("NimbleSystemConfigFile") == null)
			return null;


		try {
			config.parseSystemDocuments(dmlCfg.getTextValue("NimbleSystemConfigFile"));
		} catch (ConfigurationException e) {
			throw new PackageRuntimeException ("Error parsing Nimble configuration files");
		}

		// get threads configuration and validate
		int numSowThreads = 1;
		int numReapThreads = 1;

		numSowThreads = Integer.parseInt
				(NimbleConfig.getTextValue(config.getSystemConfig().getParameters(), "NumberOfSowThreads"));
		numReapThreads = Integer.parseInt
				(NimbleConfig.getTextValue(config.getSystemConfig().getParameters(), "NumberOfReapThreads"));

		if (numSowThreads < 1 || numReapThreads < 1){
			throw new PackageRuntimeException("Illegal values for thread count (must be > 0)");
		}

		// Initialize an instance of the driver.
		PMLDriver driver = null;
		try {
			driver = new PMLDriver(numSowThreads, numReapThreads, config);
			driver.startEmptyDriver(config);
		} catch (Exception e) {

			throw new PackageRuntimeException("Problem starting nimble driver");
		} 

		return driver.getDAGQueue();
	}

	private static String getDateTime() {
		DateFormat dateFormat = new SimpleDateFormat("MM/dd/yyyy HH:mm:ss");
		Date date = new Date();
		return dateFormat.format(date);
	}

	private static int countCompiledJobs(ProgramBlock blk) {

		int jobCount = 0;

		if (blk instanceof WhileProgramBlock){	
			ArrayList<ProgramBlock> childBlocks = ((WhileProgramBlock) blk).getChildBlocks();
			for (ProgramBlock pb : childBlocks){
				jobCount += countCompiledJobs(pb);
			}

			if (blk.getNumInstructions() > 0){
				System.out.println("error:  while programBlock should not have instructions ");
			}
		}
		else {

			for (int i = 0; i < blk.getNumInstructions(); i++)
				if (blk.getInstruction(i).getType() == INSTRUCTION_TYPE.MAPREDUCE_JOB)
					jobCount++;
		}
		return jobCount;
	}
	
}  // end class
