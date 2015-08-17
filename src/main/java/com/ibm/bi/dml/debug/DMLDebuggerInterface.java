/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.debug;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
//import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;


/**
 * Class implements a command line interface (CLI) for the DML language debugger 
 */
public class DMLDebuggerInterface 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//SystemML debugger functionality options
	private Options options; 
	
	/**
	 * Constructor for a DML debugger CLI 
	 */
	public DMLDebuggerInterface() {
		options = new Options();
	}
	
	/**
	 * Returns the debugger CLI options
	 * @return options CLI options
	 */
	public Options getOptions()
	{
		return options;
	}
	
	/**
	 * Set DML debugger CLI functionality menu
	 */
	@SuppressWarnings("static-access")
	public void setOptions()
	{
		//add help option
		options.addOption("h", "help", false, "list debugger functions");
		
		//add run option
		options.addOption("r", "run", false, "start your DML script");
		
		//add quit option
		options.addOption("q", "quit", false, "exit debug mode");
		
		//add resume option
		options.addOption("c", "continue", false, "continue running your DML script");
		
		//add step over
		//options.addOption("n", "next", false, "next line, stepping over function calls");
  	
		//add single-stepping
		options.addOption("s", "step", false, "next line, stepping into function calls");
		
		//add single-stepping
		options.addOption("si", "stepi", false, "next runtime instruction rather than DML source lines (for advanced users)");
		
		// No step return for now
		//add step return
//		Option stepReturn = OptionBuilder.withArgName( "function-name" )
//                .hasOptionalArg()
//                .withDescription( "execute instructions associated with current function as single step")
//                .create( "step_return" );
//		options.addOption(stepReturn);
		
		//add set breakpoint option
		Option setBreakpoint = OptionBuilder.withLongOpt( "break" )
					.withArgName( "line-number" )
					.hasArg()
		            .withDescription( "set breakpoint at given line number" )
		            .create( "b" );
		options.addOption(setBreakpoint);

		// The key assumption here is that user doesnot keep toggling breakpoints too often
		//add delete breakpoint option
		Option disableBreakpoint = OptionBuilder.withLongOpt( "delete" )
					.withArgName( "line-number" )
					.hasArg()
				    .withDescription( "delete breakpoint at given line number" )
				    .create( "d" );
		options.addOption(disableBreakpoint);

		//add list breakpoints option
		Option infoOption = OptionBuilder.withLongOpt( "info" )
				.withArgName( "[break | frame]" )
				.hasOptionalArgs(1)
			    .withDescription( "show all breakpoints or frames (info <break | frame>)" )
			    .create( "i" );
		options.addOption(infoOption);
		
		//add display DML script option
		Option displayScript = OptionBuilder.withLongOpt( "list" )
				.withArgName( "[next numlines] | [prev numlines] | [all]" )
                .hasOptionalArgs(2)
                .withValueSeparator(' ')
                .withDescription( "display DML script source lines. Default: numlines = 10" )
                .create( "l" );
		options.addOption(displayScript);

		//add display DML script interspersed with runtime instructions option
		Option displayInst = OptionBuilder.withLongOpt( "listi" )
				.withArgName( "[next numlines] | [prev numlines] | [all]" )
				.hasOptionalArgs(2)
				.withValueSeparator(' ')
				.withDescription( "display corresponding instructions for DML script source lines. Default: numlines = 10  (for advanced users)" )
				.create( "li" );
		options.addOption(displayInst);
		
		//add set value of DML scalar variable option
		Option setVar = OptionBuilder.withArgName( "varName value" )
				.hasArgs(2)
				.withValueSeparator(' ')
				.withDescription( "set value of a scalar or specified cell of a matrix variable. (Eg: \'set alpha 0.1\' or \'set A[1,2] 20\')" )
				.create( "set" );
		options.addOption(setVar);
		
		//add display DML matrix (or vector) variable option
		Option displayMatrix = OptionBuilder.withLongOpt( "print" )
				.withArgName( "varName" )
				.hasArg()
				.withDescription( "display contents of a scalar or matrix variable or rows/columns/cell of matrix. (Eg: \'p alpha\' or \'p A\' or \'p A[1,]\')" )
				.create( "p" );
		options.addOption(displayMatrix);
				
		Option displayTypeMatrix = OptionBuilder //.withLongOpt( "whatis" )
				.withArgName( "varName" )
				.hasArg()
				.withDescription( "display the type (and metadata) of a variable. (Eg: \'whatis alpha\' or \'whatis A\')" )
				.create( "whatis" );
		options.addOption(displayTypeMatrix);
	}
	
	/**
	 * Add new function to debugger CLI functions menu
	 * @param opt New debugging function option
	 */
	public void setOption(Option opt)
	{
		this.options.addOption(opt);
	}
	
	public void writeToStandardOutput(String outputStr) {
		System.out.print(outputStr);
		System.out.flush();
//		if(!DMLScript.ENABLE_SERVER_SIDE_DEBUG_MODE) {
//			System.out.print(outputStr);
//			System.out.flush();
//		}
//		else {
//			// TODO: Write to client socket that was created by server socket
//		}
	}
	
	public void writelnToStandardOutput(String outputStr) {
		System.out.println(outputStr);
		System.out.flush();
//		if(!DMLScript.ENABLE_SERVER_SIDE_DEBUG_MODE) {
//			System.out.println(outputStr);
//			System.out.flush();
//		}
//		else {
//			// TODO: Write to client socket that was created by server socket
//		}
	}
	
	public void writeToStandardError(String errStr) {
		System.err.print(errStr);
		System.err.flush();
//		if(!DMLScript.ENABLE_SERVER_SIDE_DEBUG_MODE) {
//			System.err.print(errStr);
//			System.err.flush();
//		}
//		else {
//			// TODO: Write to client socket that was created by server socket
//		}
	}

	
	public void writelnToStandardError(String errStr) {
		System.err.println(errStr);
		System.err.flush();
//		if(!DMLScript.ENABLE_SERVER_SIDE_DEBUG_MODE) {
//			System.err.println(errStr);
//			System.err.flush();
//		}
//		else {
//			// TODO: Write to client socket that was created by server socket
//		}
	}
	
	
	/**
	 * Display debugger usage/help info
	 */
	public void getDebuggerCLI()
	{
		// if(!DMLScript.ENABLE_SERVER_SIDE_DEBUG_MODE) {
			// Not using formatter because it outputs -h, -s commands. But, I still want to use GnuParser
			// HelpFormatter debuggerUsage = new HelpFormatter();
			// debuggerUsage.setLongOptPrefix("-"); //Eliminates the use of "--" for alternate commands
			// debuggerUsage.setWidth(125); //Enables readability of commands description
			// debuggerUsage.printHelp( "SystemMLdb <command> ", "\nSystemMLdb commands:\n", options, 
			// "\n\nSystemMLdb is a prototype debugger for SystemML. There is NO warranty as "
			//		+ "it is still in experimental state.\n\n" );
			String helpString = "SystemMLdb commands:"
					// "usage: SystemMLdb <command>\n\nSystemMLdb commands:\n"
					 + "\nh,help                                                 list debugger functions"
					 + "\nr,run                                                  start your DML script"
					 + "\nq,quit                                                 exit debug mode"
					 + "\nc,continue                                             continue running your DML script"
					 //
					 + "\nl,list <[next numlines] | [prev numlines] | [all]>     display DML script source lines. Default: numlines = 10"
					 + "\nb,break <line-number>                                  set breakpoint at given line number"
					 + "\nd,delete <line-number>                                 delete breakpoint at given line number"
					 + "\ns,step                                                 next line, stepping into function calls"
					 + "\ni,info <break | frame>                                 show all breakpoints or frames (info <break | frame>)"
					 //
					 + "\np,print <varName>                                      display contents of a scalar or matrix variable or"
					 + "\n                                                       rows/columns/cell of matrix. (Eg: \'p alpha\' or \'p A\' or \'p A[1,]\')"
					 + "\nset <varName value>                                    set value of a scalar or specified cell of a matrix variable. (Eg:"
					 + "\n                                                       \'set alpha 0.1\' or \'set A[1,2] 20\')"
					 + "\nwhatis <varName>                                       display the type (and metadata) of a variable. (Eg: \'whatis alpha\'"
					 + "\n                                                       or \'whatis A\')"
					 + "\nli,listi <[next numlines] | [prev numlines] | [all]>   display corresponding instructions for DML script source lines."
					 + "\n                                                       Default: numlines = 10  (for advanced users)"
					 + "\nsi,stepi                                               next runtime instruction rather than DML source lines (for advanced"
					 + "\n                                                       users)"
					 //
					+ "\n"
					;
			writelnToStandardOutput(helpString);
			
//		}
//		else {
//			// TODO: Write to client socket that was created by server socket
//			// printHelp(PrintWriter pw, int width, String cmdLineSyntax, String header, Options options, int leftPad, int descPad, String footer)
//		}
		
	}
	
	/**
	 * Read, process and return command from debugger CLI
	 * @return CommandLine Current debug command (enter by user)
	 * @throws DMLDebuggerException
	 */
	public CommandLine getDebuggerCommand()
		throws DMLDebuggerException
	{		
		CommandLine cmd = null;
		String [] args = null;
		
//		if(!DMLScript.ENABLE_SERVER_SIDE_DEBUG_MODE) {
			//Display input prompt
	        writeToStandardOutput("(SystemMLdb) ");
			BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
			try {
				//read command line argument(s)
				// To add 'up/down'-feature, use jline library
				String line = br.readLine();
				if(line != null && !line.isEmpty() ) {
					args = line.split(" ");
					if(args[0].startsWith("-")) {
						// So as to avoid parsing '-i' command
						writelnToStandardError("Error reading command line arguments. Try \"help\".");
						return cmd;
					}
					args[0] = "-" + args[0];
				}
			} catch (IOException ae) {
				writelnToStandardError("Error reading command line arguments. Try \"help\".");
				return cmd;
	    	}
//	    	
//		}
//		else {
//			// TODO: Read commands from Process that was created by server socket
//		}
		
		CommandLineParser CLIparser = new GnuParser();		
		try {
			//parse current command
			cmd = CLIparser.parse(getOptions(), args);				
		} catch (ParseException pe) {
			System.err.println("Undefined command (or command arguments). Try \"help\".");
		}
		return cmd;
	}	
}
