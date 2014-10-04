/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.debugger;

import java.io.IOException;
import java.io.PrintStream;
import java.util.HashMap;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.ParseException;
import org.apache.commons.lang.math.IntRange;

import com.ibm.bi.dml.debugger.DMLDebuggerFunctions;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.parser.DMLQLParser;
import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.parser.LanguageException;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.BreakPointInstruction.BPINSTRUCTION_STATUS;

/** 
 * This class implements a debugger control module for DML scripts.
 * Note: ONLY USED FOR DEBUGGING PURPOSES
 */
public class DMLDebugger
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";


	// This will be supported in subsquent release. turning off the -debug 'optimize' feature for current release.
	//public static boolean ENABLE_DEBUG_OPTIMIZER = false; //default debug mode
	//public static boolean ENABLE_SERVER_SIDE_DEBUG_MODE = false; //default debug mode
		
	//public static final String DEBUGGER_SYSTEMML_CONFIG_FILEPATH = "./DebuggerSystemML-config.xml";
	
	
	private DMLDebuggerProgramInfo dbprog; //parsed and compiled DML script w/ hops, lops and runtime program
	public DMLDebuggerInterface debuggerUI; //debugger command line interface
	private DMLDebuggerFunctions dbFunctions; //debugger functions interface
	private CommandLine cmd; //debugger function command
	
	//support for obtaining STDOUT/STDERR streams of DML program running in debug mode 
	private PrintStream originalOut = null;
	private PrintStream originalErr = null; 

	private String dmlScriptStr; //DML script contents (including new lines)
	HashMap<String,String> argVals; //key-value pairs defining arguments of DML script
	ExecutionContext preEC = null;
	ExecutionContext currEC = null;
	String [] lines;
	volatile boolean quit=false;
	
	/**
	 * Constructor for DML debugger CLI
	 */
	public DMLDebugger(DMLDebuggerProgramInfo p, String dmlScript, HashMap<String,String> args) 
	{
		dbprog = p;
		dmlScriptStr = dmlScript;
		lines = dmlScriptStr.split("\n");
		argVals = args;		
		debuggerUI = new DMLDebuggerInterface();
		dbFunctions = new DMLDebuggerFunctions();		
		preEC = new ExecutionContext(dbprog.rtprog);
		preEC.initDebugState();
		setupDMLRuntime();
	}
	
	/**
	 * Sets up DML runtime with DML script and instructions information
	 */
	private void setupDMLRuntime() 
	{
		dbprog.setDMLInstMap();
		preEC.getDebugState().setDMLScript(lines);
	}

	/**
	 * Sets STDOUT stream of a DML program running in debug mode
	 */
	@SuppressWarnings("unused")
	private void setStdOut() 
	{
		originalOut = System.out;
		System.setOut(originalOut);
	}

	/**
	 * Gets STDOUT stream of a DML program running in debug mode
	 * @return STDOUT stream of DML program 
	 */
	@SuppressWarnings("unused")
	private PrintStream getStdOut() 
	{
		System.out.flush();
		return originalOut;
	}
	
	/**
	 * Sets STDERR stream of a DML program running in debug mode
	 */
	@SuppressWarnings("unused")
	private void setStdErr() 
	{
		originalErr = System.err;
		System.setOut(originalErr);
	}

	/** 
	 * Gets STDERR stream of a DML program running in debug mode
	 * @return STDERR stream of DML program
	 */
	@SuppressWarnings("unused")
	private PrintStream getStdErr() 
	{
		System.err.flush();
		return originalErr;
	}
	
	/** 
	 * Get debug function command from debugger CLI
	 * @throws DMLDebuggerException
	 */
	private void getCommand() throws DMLDebuggerException
	{
		cmd = debuggerUI.getDebuggerCommand();
	}
	
	/**
	 * Class for running the DML runtime as a thread
	 */
	Runnable DMLRuntime = new Runnable() 
	{
		public void run() {
			try {
				// System.out.println("Starting DML script ...");
				dbprog.rtprog.execute(currEC);
				// System.out.println("DML script has finished execution.");
				synchronized(DMLDebugger.class) {
					quit = true;
				}
			}
			catch (Exception e) {
				System.err.println("Exception raised by DML runtime:" + e);		
			}
		}
	};
	
	/**
	 * Controls the communication between debugger CLI and DML runtime.  
	 */
	@SuppressWarnings("deprecation")
	public synchronized void runSystemMLDebugger()
	{
		debuggerUI.setOptions();
		debuggerUI.getDebuggerCLI();
		Thread runtime = new Thread(DMLRuntime);
		boolean isRuntimeInstruction = false;
		
		while (!quit) {
			try {
				//get debugger function from CLI
				getCommand();
				if(cmd != null) {
					isRuntimeInstruction = false;
					//check for help
					if(cmd.hasOption("h")) {
						debuggerUI.getDebuggerCLI();
					}
					//check for exit
					else if (cmd.hasOption("q")) {
						synchronized(DMLDebugger.class) {
							quit = true;    	    	
						}
						runtime.stop();
					}    	    	
					else if (cmd.hasOption("r")) {						
						if (currEC != null) {
							System.out.println("Runtime has already started. Try \"s\" to go to next line, or \"c\" to continue running your DML script.");
						}
						else {
							currEC = preEC;
							runtime.start();
							isRuntimeInstruction = true;
						}
    	    		}
					else if (cmd.hasOption("c")) {
						if (currEC == null)
							System.out.println("Runtime has not been started. Try \"r\" to start DML runtime execution.");
						else if (!runtime.isAlive()) {
							System.err.println("Invalid debug state.");
							//System.out.println("Runtime terminated. Try \"-c\" to recompile followed by \"r\" to restart DML runtime execution.");
						}
						else {
							System.out.println("Resuming DML script execution ...");
							preEC.getDebugState().setCommand(null);
							runtime.resume();
							isRuntimeInstruction = true;
						}
    	    		}
    	    		else if (cmd.hasOption("si")) {
    	    			if (!runtime.isAlive()) {
    	    				currEC = preEC;
							runtime.start();
							isRuntimeInstruction = true;
    	    				// System.out.println("Runtime must be started before single stepping can be enabled. Try \"r\" to start DML runtime execution.");
    	    			}
    	    			//else {
    	    				preEC.getDebugState().setCommand("step_instruction");
    	    				runtime.resume();
    	    				isRuntimeInstruction = true;
    	    			//}
    	    		}
    	    		else if (cmd.hasOption("s")) {
    	    			if (!runtime.isAlive()) {
    	    				currEC = preEC;
							runtime.start();
							isRuntimeInstruction = true;
    	    				//System.out.println("Runtime must be started before step over can be enabled. Try \"r\" to start DML runtime execution.");
    	    			}
    	    			//else {
    	    				preEC.getDebugState().setCommand("step_line");
    	    				runtime.resume();
    	    				isRuntimeInstruction = true;
    	    			//}
    	    		}
//    	    		else if (cmd.hasOption("step_return")) {
//    	    			if (!runtime.isAlive()) {
//    	    				System.out.println("Runtime must be started before step return can be enabled. Try \"r\" to start DML runtime execution.");
//    	    			}
//    	    			else {
//    	    				String fname = dbFunctions.getValue(cmd.getOptionValues("step_return"));
//    	    				dbprog.rtprog.setCommand("step return");
//    	    				if (fname != null) {
//    	    					dbprog.rtprog.setCommandArg(fname);
//    	    				}
//    	    				runtime.resume();
//    	    				isRuntimeInstruction = true;
//    	    			}
//    	    		}
    	    		else if (cmd.hasOption("b")) {
       	    			int lineNumber = dbFunctions.getValue(cmd.getOptionValues("b"), lines.length);
    	    			if (lineNumber > 0) {
    	    				if (DMLBreakpointManager.getBreakpoint(lineNumber) == null)
    	    					System.out.println("Sorry, a breakpoint cannot be inserted at line " + lineNumber + ". Please try a different line number.");
    	    				else {
    	    					if (DMLBreakpointManager.getBreakpoint(lineNumber).getBPInstructionStatus() != BPINSTRUCTION_STATUS.INVISIBLE) {
    	    						System.out.format("Breakpoint at line %d already exists.\n", lineNumber);
    	    					}
    	    					else {
    	    						dbprog.accessBreakpoint(lineNumber, 0, BPINSTRUCTION_STATUS.ENABLED);
    	    					}
    	    				}
    	    			}
    	    		}
    	    		else if (cmd.hasOption("d")) {
    	    			int lineNumber = dbFunctions.getValue(cmd.getOptionValues("d"), lines.length);
    	    			if (lineNumber > 0 && DMLBreakpointManager.getBreakpoint(lineNumber) != null && 
	    						DMLBreakpointManager.getBreakpoint(lineNumber).getBPInstructionStatus() != BPINSTRUCTION_STATUS.INVISIBLE) {
    	    				dbprog.accessBreakpoint(lineNumber, 1, BPINSTRUCTION_STATUS.INVISIBLE);
    	    				//dbprog.accessBreakpoint(lineNumber, 1, BPINSTRUCTION_STATUS.DISABLED);
	    				}
    	    			else {
    	    				System.out.println("Sorry, a breakpoint cannot be deleted at line " + lineNumber + ". Please try a different line number.");
    	    			}
    	    			
    	    		}
    	    		else if (cmd.hasOption("i")) {
    	    			String [] infoOptions = cmd.getOptionValues("i");
    	    			if(infoOptions == null || infoOptions.length == 0) {
    	    				System.err.println("The command \"info\" requires option. Try \"info break\" or \"info frame\".");
    	    			}
    	    			else if(infoOptions[0].trim().compareTo("break") == 0) {
    	    				dbFunctions.listBreakpoints(DMLBreakpointManager.getBreakpoints());
    	    			}
    	    			else if(infoOptions[0].trim().compareTo("frame") == 0) {
    	    				if (!runtime.isAlive())
        	    				System.err.println("Runtime has not been started. Try \"r\" or \"s\" to start DML runtime execution.");
        	    			else 
        	    				dbFunctions.printCallStack(currEC.getDebugState().getCurrentFrame(), currEC.getDebugState().getCallStack());
    	    			}
    	    			else {
    	    				System.err.println("Invalid option for command \"info\".  Try \"info break\" or \"info frame\".");
    	    			}
    	    		}
    	    		else if (cmd.hasOption("p")) {
    	    			String [] pOptions = cmd.getOptionValues("p");
    	    			if(pOptions == null || pOptions.length != 1) {
    	    				System.err.println("Incorrect options for command \"print\"");
    	    			}
    	    			else {
    	    				String varName = pOptions[0].trim();
    	    				if (runtime.isAlive()) {
    	    					if(varName.contains("[")) {
        	    					// matrix with index: can be cell or column or row
    	    						try {
    	    							//System.out.println("" + varName);
    	    							String variableNameWithoutIndices = varName.split("\\[")[0].trim();
    	    							//System.out.println("" + variableNameWithoutIndices);
    	    							String indexString = (varName.split("\\[")[1].trim()).split("\\]")[0].trim();
    	    							//System.out.println(">>" + indexString + "<<");
    	    							String rowIndexStr = "";
    	    							String colIndexStr = "";
    	    							if(indexString.startsWith(",")) {
    	    								
    	    								colIndexStr = indexString.split(",")[1].trim();
    	    							}
    	    							else if(indexString.endsWith(",")) {
    	    								rowIndexStr = indexString.split(",")[0].trim();
    	    							}
    	    							else {
    	    								rowIndexStr = indexString.split(",")[0].trim();
        	    							colIndexStr = indexString.split(",")[1].trim();
    	    							}
    	    							int rowIndex = -1;
    	    							int colIndex = -1;
    	    							if(rowIndexStr.compareTo("") != 0) {
    	    								rowIndex = Integer.parseInt(rowIndexStr);
    	    							}
    	    							if(colIndexStr.compareTo("") != 0) {
    	    								colIndex = Integer.parseInt(colIndexStr);
    	    							}
    	    							//System.out.println("" + rowIndex + " " + colIndex);
    	    							dbFunctions.print(currEC.getDebugState().getVariables(), variableNameWithoutIndices, "value", rowIndex, colIndex);
    	    						}
    	    						catch(Exception indicesException) {
    	    							indicesException.printStackTrace();
    	    							System.err.println("Incorrect fomat for \"p\". If you are trying to print matrix variable M, you can use M[1,] or M[,1] or M[1,1] (without spaces).");
    	    						}
        	    				}
        	    				else {
        	    					// Print entire matrix
        	    					dbFunctions.print(currEC.getDebugState().getVariables(), varName, "value", -1, -1);
        	    				}
    	    				}
        	    			else
        	    				System.err.println("Runtime has not been started. Try \"r\" or \"s\" to start DML runtime execution.");
    	    			}
    	    		}
    	    		else if (cmd.hasOption("whatis")) {
    	    			String [] pOptions = cmd.getOptionValues("whatis");
    	    			if(pOptions == null || pOptions.length != 1) {
    	    				System.err.println("Incorrect options for command \"whatis\"");
    	    			}
    	    			else {
    	    				String varName = pOptions[0].trim();
    	    				dbFunctions.print(currEC.getDebugState().getVariables(), varName, "metadata", -1, -1);
    	    			}
    	    		}
    	    		else if (cmd.hasOption("set")) {
    	    			String [] pOptions = cmd.getOptionValues("set");
    	    			if(pOptions == null || pOptions.length != 2) {
    	    				System.err.println("Incorrect options for command \"set\"");
    	    			}
    	    			else {
    	    				try {
    	    					if(pOptions[0].contains("[")) {
		    	    				String [] paramsToSetMatrix = new String[4];
		    	    				paramsToSetMatrix[0] = pOptions[0].split("\\[")[0].trim();
		    	    				String indexString =  (pOptions[0].split("\\[")[1].trim()).split("\\]")[0].trim();
		    	    				paramsToSetMatrix[1] = indexString.split(",")[0].trim();
		    	    				paramsToSetMatrix[2] = indexString.split(",")[1].trim();
		    	    				paramsToSetMatrix[3] = pOptions[1].trim();
		    	    				dbFunctions.setMatrixCell(currEC.getDebugState().getVariables(), paramsToSetMatrix);
    	    					}
    	    					else {
    	    						dbFunctions.setScalarValue(currEC.getDebugState().getVariables(), pOptions);
    	    					}
    	    				}
    	    				catch(Exception exception1) {
    	    					System.out.println("Only scalar variable or a matrix cell available in current frame can be set in current version.");
    	    				}
    	    			}
    	    		}
    	    		else if (cmd.hasOption("l")) {
    	    			
    	    			String [] pOptions = cmd.getOptionValues("l");
    	    			String [] argsForRange = new String[2];
    	    			int currentPC = 1;
    	    			
    	    			if(runtime.isAlive()) { 
    	    				currentPC = currEC.getDebugState().getPC().getLineNumber();
    	    			}
    	    			
    	    			IntRange range = null;
    	    			if(pOptions == null) {
    	    				// Print first 10 lines
    	    				range = new IntRange(currentPC, Math.min(lines.length, currentPC+10));
    	    			}
    	    			else if(pOptions.length == 1 && pOptions[0].trim().toLowerCase().compareTo("all") == 0) {
    	    				// Print entire program
    	    				range = new IntRange(1, lines.length);
    	    			}
    	    			else if(pOptions.length == 2 && pOptions[0].trim().toLowerCase().compareTo("next") == 0) {
    	    				int numLines = 10;
    	    				try {
    	    					numLines = Integer.parseInt(pOptions[1]);
    	    				}
    	    				catch(Exception e1) {}
    	    				
    	    				argsForRange[0] = "" + currentPC;
    	    				argsForRange[1] = "" + Math.min(lines.length, numLines + currentPC);
    	    				range = dbFunctions.getRange(argsForRange, lines.length);
    	    				
    	    			}
    	    			else if(pOptions.length == 2 && pOptions[0].trim().toLowerCase().compareTo("prev") == 0) {
    	    				int numLines = 10;
    	    				try {
    	    					numLines = Integer.parseInt(pOptions[1]);
    	    				}
    	    				catch(Exception e1) {}
    	    				
    	    				argsForRange[0] = "" + Math.max(1, currentPC - numLines);
    	    				argsForRange[1] = "" + currentPC;
    	    				range = dbFunctions.getRange(argsForRange, lines.length);
    	    			}
    	    			
    	    			
    	    			if(range == null) {
    	    				System.err.println("Incorrect usage of command \"l\". Try \"l\" or \"l all\" or \"l next 5\" or \"l prev 5\".");
    	    			}
    	    			else {
    	    				if (range.getMinimumInteger() > 0) {
    	    					dbFunctions.printLines(lines, range);
    	    	    		}
    	    				else {
    	    					System.err.println("Sorry no lines that can be printed. Try \"l\" or \"l all\" or \"l next 5\" or \"l prev 5\".");
    	    				}
    	    			}
    	    			
    	    			
    	    			
    	    			// Old code:
    	    			// IntRange range = dbFunctions.getRange(cmd.getOptionValues("p"), lines.length);
    	    			//if (range.getMinimumInteger() > 0) {
    	    			//	dbFunctions.printLines(lines, range);
    	    			// }
    	    		}
    	    		else if (cmd.hasOption("li")) {
    	    			
    	    			String [] pOptions = cmd.getOptionValues("li");
    	    			String [] argsForRange = new String[2];
    	    			int currentPC = 1;
    	    			
    	    			if(runtime.isAlive()) { 
    	    				currentPC = currEC.getDebugState().getPC().getLineNumber();
    	    			}
    	    			
    	    			IntRange range = null;
    	    			if(pOptions == null) {
    	    				// Print first 10 lines
    	    				range = new IntRange(currentPC, Math.min(lines.length, currentPC+10));
    	    			}
    	    			else if(pOptions.length == 1 && pOptions[0].trim().toLowerCase().compareTo("all") == 0) {
    	    				// Print entire program
    	    				range = new IntRange(1, lines.length);
    	    			}
    	    			else if(pOptions.length == 2 && pOptions[0].trim().toLowerCase().compareTo("next") == 0) {
    	    				int numLines = 10;
    	    				try {
    	    					numLines = Integer.parseInt(pOptions[1]);
    	    				}
    	    				catch(Exception e1) {}
    	    				
    	    				argsForRange[0] = "" + currentPC;
    	    				argsForRange[1] = "" + Math.min(lines.length, numLines + currentPC);
    	    				range = dbFunctions.getRange(argsForRange, lines.length);
    	    				
    	    			}
    	    			else if(pOptions.length == 2 && pOptions[0].trim().toLowerCase().compareTo("prev") == 0) {
    	    				int numLines = 10;
    	    				try {
    	    					numLines = Integer.parseInt(pOptions[1]);
    	    				}
    	    				catch(Exception e1) {}
    	    				
    	    				argsForRange[0] = "" + Math.max(1, currentPC - numLines);
    	    				argsForRange[1] = "" + currentPC;
    	    				range = dbFunctions.getRange(argsForRange, lines.length);
    	    			}
    	    			
    	    			
    	    			if(range == null) {
    	    				System.err.println("Incorrect usage of command \"li\". Try \"li\" or \"li all\" or \"li next 5\" or \"li prev 5\".");
    	    			}
    	    			else {
    	    				if (range.getMinimumInteger() > 0) {
    	    					dbFunctions.printInstructions(lines, dbprog.getDMLInstMap(), range, false);
    	    	    		}
    	    				else {
    	    					System.err.println("Sorry no lines that can be printed. Try \"li\" or \"li all\" or \"li next 5\" or \"li prev 5\".");
    	    				}
    	    			}
    	    			
    	    			
    	    			
    	    			// Old code:
    	    			// IntRange range = dbFunctions.getRange(cmd.getOptionValues("p"), lines.length);
    	    			//if (range.getMinimumInteger() > 0) {
    	    			//	dbFunctions.printLines(lines, range);
    	    			// }
    	    		}
    	    		else if (cmd.hasOption("set_scalar")) {
						if (!runtime.isAlive())
							System.err.println("Runtime has not been started. Try \"r\" to start DML runtime execution.");
						else
							dbFunctions.setScalarValue(currEC.getDebugState().getVariables(), cmd.getOptionValues("set_scalar"));
    	    		}
    	    		else if (cmd.hasOption("m")) {
    	    			String varname = dbFunctions.getValue(cmd.getOptionValues("m"));
    	    			if (runtime.isAlive())
    	    				dbFunctions.printMatrixVariable(currEC.getDebugState().getVariables(), varname);
    	    			else
    	    				System.err.println("Runtime has not been started. Try \"r\" to start DML runtime execution.");
    	    		}
    	    		else if (cmd.hasOption("x")) {
    	    			if (!runtime.isAlive())
    	    				System.err.println("Runtime has not been started. Try \"r\" to start DML runtime execution.");
    	    			else {
    	    				dbFunctions.printMatrixCell(currEC.getDebugState().getVariables(), cmd.getOptionValues("x"));
    	    			}
    	    		}
    	    		else if (cmd.hasOption("set_cell")) {
    	    			if (!runtime.isAlive())
    	    				System.err.println("Runtime has not been started. Try \"r\" to start DML runtime execution.");
    	    			else {
    	    				dbFunctions.setMatrixCell(currEC.getDebugState().getVariables(), cmd.getOptionValues("set_cell"));
    	    			}
    	    		}
    	    		else {
    	    			System.err.println("Undefined command. Try \"help\".");
    	    		}    
					//block until runtime suspends execution or terminates 
					//while(runtime.isAlive() && !currEC.getProgram().isStopped()) {
					wait(300); // To avoid race condition between submitting job and
					//System.out.println(">> Before while");
					while(isRuntimeInstruction && !currEC.getDebugState().canAcceptNextCommand()) {
						if(quit){
							break;
						}
						else {
							wait(300); //wait
						}
					}
					//System.out.println(">> After while");
				}
				wait(300);
			} catch (Exception e) {
    			System.err.println("Error processing debugger command. Try \"help\".");
    		}
		}
	}
	    
	
	/**
	 * compile: Compile DML script and generate hops, lops and runtime program for debugger. 
	 * @param  dmlScriptStr DML script contents (including new lines)
	 * @param  argVals Key-value pairs defining arguments of DML script  
	 * @throws ParseException
	 * @throws IOException
	 * @throws DMLRuntimeException
	 * @throws LanguageException
	 * @throws HopsException
	 * @throws LopsException
	 * @throws DMLUnsupportedOperationException
	 * @throws Exception 
	 */
	private void recompile(String dmlScriptStr, HashMap<String,String> argVals)
			throws ParseException, IOException, DMLRuntimeException, LanguageException, HopsException, LopsException, DMLUnsupportedOperationException, Exception
	{
		dbprog = new DMLDebuggerProgramInfo();
			
		//Step 1: parse dml script
		DMLQLParser parser = new DMLQLParser(dmlScriptStr, argVals);
		dbprog.prog = parser.parse();		
			
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
		
		//Set debug mode flag
		setupDMLRuntime();
	}
}
