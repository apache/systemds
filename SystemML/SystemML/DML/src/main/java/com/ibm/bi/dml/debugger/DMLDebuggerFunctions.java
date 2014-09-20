/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.debugger;

import java.util.ArrayList;
import java.util.Stack;
import java.util.TreeMap;

import org.apache.commons.lang.math.IntRange;

import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.LocalVariableMap;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.instructions.BPInstruction;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.instructions.BPInstruction.BPINSTRUCTION_STATUS;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.BooleanObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.CPInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.DoubleObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.IntObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.ScalarObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.StringObject;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;

public class DMLDebuggerFunctions {
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//Maximum values of rows and columns when displaying a DML matrix/vector 
	public static final int DISPLAY_MAX_ROWS = 30;
	public static final int DISPLAY_MAX_COLUMNS = 8;
	
	/////////////////////////////////////////////
	// public interface for debugger functions //
	/////////////////////////////////////////////
	
	/**
	 * Print all breakpoints along with current status (i.e. enabled or disabled)
	 * @param breakpoints Contains all existing breakpoints
	 */
	public void listBreakpoints(TreeMap<Integer, BPInstruction> breakpoints) 
	{	
		//Display all breakpoints 
		if (breakpoints == null) 
		{
			System.out.println("No breakpoints are set for this program.");
			return;
		}
		int currBreakpoint = 1; //active breakpoint ids
		int numVisibleBreakpoints = 0;
		for (Integer lineNumber : breakpoints.keySet()) 
		{
			if (breakpoints.get(lineNumber).getBPInstructionStatus() == BPINSTRUCTION_STATUS.ENABLED) {
				System.out.format("Breakpoint %2d, at line %4d (%s)\n", currBreakpoint++, lineNumber, "enabled");
				numVisibleBreakpoints++;
			}
			else if (breakpoints.get(lineNumber).getBPInstructionStatus() == BPINSTRUCTION_STATUS.DISABLED) {
				System.out.format("Breakpoint %2d, at line %4d (%s)\n", currBreakpoint++, lineNumber, "disabled");
				numVisibleBreakpoints++;
			}
		}
		
		if(numVisibleBreakpoints == 0) {
			System.out.println("No breakpoints are set for this program.");
		}
	}
	
	/**
	 * Print range of DML program lines
	 * @param lines DML script lines of code
	 * @param range Range of lines of DML code to be displayed
	 */
	public void printLines(String [] lines, IntRange range) 
	{		
		//Display all lines of DML script
		for (int lineNumber=range.getMinimumInteger() ; lineNumber<=range.getMaximumInteger() ; lineNumber++)			
			System.out.format("line %4d: %s\n", lineNumber, lines[lineNumber-1]);
	}
	
	/**
	 * Print range of DML program lines interspersed with corresponding runtime instructions
	 * @param lines DML script lines of code
	 * @param DMLInstMap Mapping between source code line number and corresponding runtime instruction(s)
	 * @param range Range of lines of DML code to be displayed
	 * @param debug Flag for displaying instructions in debugger test integration
	 */
	public void printInstructions(String [] lines, TreeMap<Integer, ArrayList<Instruction>> DMLInstMap, IntRange range, boolean debug) 
	{
		//Display instructions with corresponding DML line numbers
		for (int lineNumber=range.getMinimumInteger() ; lineNumber<=range.getMaximumInteger() ; lineNumber++)  
		{
			System.out.format("line %4d: %s\n", lineNumber, lines[lineNumber-1]);
			if (DMLInstMap.get(lineNumber) != null) 
			{
				for (Instruction currInst : DMLInstMap.get(lineNumber))  
				{
					if (currInst instanceof CPInstruction) {
						if (!debug)
							System.out.format("\t\t id %4d: %s\n", currInst.getInstID(), prepareInstruction(currInst.toString()));
						else {
							String [] instStr = prepareInstruction(currInst.toString()).split(" ");
							System.out.format("\t\t id %4d: %s %s\n", currInst.getInstID(), instStr[0], instStr[1]);
						}
					}
					else if (currInst instanceof MRJobInstruction)  
					{
						MRJobInstruction currMRInst = (MRJobInstruction) currInst;
						System.out.format("\t\t id %4d: %s\n", currInst.getInstID(), prepareInstruction(currMRInst.getMRString(debug)));
					}
					else if (currInst instanceof BPInstruction) {
						BPInstruction currBPInst = (BPInstruction) currInst;
						System.out.format("\t\t id %4d: %s\n", currInst.getInstID(), currBPInst.toString());
					}
				}
			}
		}
	}

	/**
	 * Print range of program runtime instructions
	 * @param DMLInstMap Mapping between source code line number and corresponding runtime instruction(s)
	 * @param range Range of lines of DML code to be displayed
	 */
	public void printRuntimeInstructions(TreeMap<Integer, ArrayList<Instruction>> DMLInstMap, IntRange range) 
	{
		//Display instructions
		for (int lineNumber=range.getMinimumInteger() ; lineNumber<=range.getMaximumInteger() ; lineNumber++)  
		{
			if (DMLInstMap.get(lineNumber) != null) 
			{
				for (Instruction currInst : DMLInstMap.get(lineNumber))  
				{
					if (currInst instanceof CPInstruction)
						System.out.format("\t\t id %4d: %s\n", currInst.getInstID(), prepareInstruction(currInst.toString()));								
					else if (currInst instanceof MRJobInstruction)  
					{
						MRJobInstruction currMRInst = (MRJobInstruction) currInst;
						System.out.format("\t\t id %4d: %s\n", currInst.getInstID(), prepareInstruction(currMRInst.getMRString(false)));
					}
					else if (currInst instanceof BPInstruction) {
						BPInstruction currBPInst = (BPInstruction) currInst;
						System.out.format("\t\t id %4d: %s\n", currInst.getInstID(), currBPInst.toString());
					}
				}
			}
		}
	}	
	
	/**
	 * Print current program counter
	 * @param pc Current stack frame program counter
	 */
	public void printPC(DMLProgramCounter pc) 
	{		
		//Display program counter
		if (pc != null)
			System.out.println("  Current program counter at " + pc.toString());			
		else
			System.out.println("DML runtime is currently inactive.");
	}
	
	/**
	 * Print current stack frame variables
	 * @param frameVariables Current frame variables  
	 */
	public void printFrameVariables(LocalVariableMap frameVariables) 
	{
		//Display local stack frame variables
		if (frameVariables != null && !frameVariables.keySet().isEmpty()) {
			System.out.println("  Local variables:");
			System.out.format("\t%-40s %-40s", "Variable name", "Variable value" );
			for( String varname : frameVariables.keySet())
				System.out.format("\n\t%-40s %-40s", varname, frameVariables.get(varname).toString());
			System.out.println();
		}
		else
			System.out.println("\tSymbol table for current frame is empty");
	}
	
	/**
	 * Print current stack frame information
	 * @param frame Current stack frame which contains pc and local variables 
	 */
	public void printFrame(DMLFrame frame) {
		if (frame != null && frame.getPC() != null) {
			printPC(frame.getPC());
			printFrameVariables(frame.getVariables());
		}
		else
			System.out.println("DML runtime is currently inactive.");
	}

	/**
	 * Print current call stack
	 * @param currentEC Current stack frame
 	 * @param callStack Saved stack frames  
	 */
	public void printCallStack(DMLFrame currFrame, Stack<DMLFrame> callStack) {
		int frameID = 0;
		if (currFrame == null)
			System.out.println("DML runtime is currently inactive.");
		else {
			if (callStack != null) {
				for(DMLFrame frame : callStack) {
					System.out.println("Frame id: " + frameID++);
					printFrame(frame);
				}			
			}
			System.out.println("Current frame id: " + frameID++);
			printFrame(currFrame);
		}
	}
	
	/**
	 * Print DML scalar variable in current frame (if existing)
	 * @param variables Current frame variables
 	 * @param varname Variable name  
	 */
	public void printScalarVariable(LocalVariableMap variables, String varname) {
		if (varname == null) {
			System.err.println("No scalar variable name entered.");
			return;
		}		
		if (variables != null && !variables.keySet().isEmpty()) {
			if (variables.get(varname) != null) {
				if (variables.get(varname).getDataType() == DataType.SCALAR)
					System.out.println(varname + " = " + variables.get(varname).toString());
				else
					System.out.println("Variable \""+varname+"\" is not scalar variable.");
			}
			else
				System.out.println("DML scalar variable \""+varname+"\" is not in the current frame scope. Try \"a\" to list all variables within current frame scope.");
		}
		else
			System.out.println("Symbol table for current frame is empty");
	}
	
	/**
	 * Set value of specified DML scalar variable in current frame (if existing)
	 * @param variables Current frame variables
 	 * @param args CLI arguments for current debugger function (Variable name, and value) 
	 */
	public void setScalarValue(LocalVariableMap variables, String [] args) {
		String varname = args[0];
		if (variables != null && !variables.keySet().isEmpty()) {
			if (variables.get(varname) != null) {
				if (variables.get(varname).getDataType() == DataType.SCALAR) {
					Data value;
					//try {
						switch(variables.get(varname).getValueType()) {
							case DOUBLE:
								double d = Double.parseDouble(args[1]);
								value = (ScalarObject) new DoubleObject(d);
								break;
							case INT:
								long i = Long.parseLong(args[1]);
								value = (ScalarObject) new IntObject(i);
								break;
							case BOOLEAN:
								boolean b = Boolean.parseBoolean(args[1]);
								value = (ScalarObject) new BooleanObject(b);
								break;
							case STRING:
								value = (ScalarObject) new StringObject(args[1]);
								break;
							default:
								System.err.println("Invalid scalar value type.");
								return;
						}
					//} 
//					catch (Exception e) {
//						System.err.println("Error processing set scalar value command for variable"+varname+".");
//						return;
//					}
					variables.put(varname, value);
					System.out.println(varname + " = " + variables.get(varname).toString());
				}
				else
					System.out.println("Variable \""+varname+"\" is not scalar variable.");
			}
			else
				System.out.println("DML scalar variable \""+varname+"\" is not in the current frame scope. Try \"a\" to list all variables within current frame scope.");
		}
		else
			System.out.println("Symbol table for current frame is empty");
	}
	
	public void print(LocalVariableMap variables, String varname, String displayFunction, int rowIndex, int colIndex) throws DMLRuntimeException {
		if (varname == null) {
			System.err.println("No matrix variable name entered.");
			return;
		}
		if (variables != null && !variables.keySet().isEmpty()) {
			if (variables.get(varname) != null) {
				if (variables.get(varname).getDataType() == DataType.MATRIX) {
					try {
						MatrixObject mo = (MatrixObject) variables.get(varname);
						if (mo.getStatusAsString() == "EMPTY" && (OptimizerUtils.estimateSizeExactSparsity(mo.getNumRows(), mo.getNumColumns(), mo.getSparsity()) > OptimizerUtils.getLocalMemBudget())) {
							//TODO @jlugoma Need to add functionality to bring and display a block. 
							System.err.println("ERROR: Matrix dimensions are too large to fit in main memory.");
							return;
						}
						MatrixBlock mb = mo.acquireRead();
						
						if(displayFunction.compareTo("value") == 0) {
							prettyPrintMatrixBlock(mb, rowIndex, colIndex);
						}
						else if(displayFunction.compareTo("metadata") == 0) {
							System.out.println("Metadata of " + varname + ": matrix"+variables.get(varname).getMetaData().toString());
						}
						mo.release();
						/*
						if(displayFunction.compareTo("value") == 0) {
							if (mb.getNumRows() > DISPLAY_MAX_ROWS || mb.getNumColumns() > DISPLAY_MAX_COLUMNS) {
								System.out.format("WARNING: DML matrix/vector is too large to display on the screen."
										+ "\nOnly a snapshot of %d row(s) and %d column(s) is being displayed.\n", 
										min(mb.getNumRows(), DISPLAY_MAX_ROWS), min(mb.getNumColumns(), DISPLAY_MAX_COLUMNS));
							}
						}
						*/
												
					} catch (Exception e) {
						System.err.println("Error processing \'print\' command for variable "+varname+".");
						return;
					}
				}
				else if (variables.get(varname).getDataType() == DataType.SCALAR) {
					if(displayFunction.compareTo("value") == 0) {
						System.out.println(varname + " = " + variables.get(varname).toString());
					}
					else if(displayFunction.compareTo("metadata") == 0) {
						System.out.println("Metadata of " + varname + ": DataType.SCALAR");
					}
				}
				else
					System.out.println("Variable \""+varname+"\" is not a matrix or vector or scalar variable.");
			}
			else
				System.out.println("DML matrix variable \""+varname+"\" is not in the current frame scope. Try \"info frame\" to list all variables within current frame scope.");
		}
		else
			System.out.println("Symbol table for current frame is empty");
	}
	
	
			
	/**
	 * Print DML matrix variable in current frame (if existing)
	 * @param variables Current frame variables
 	 * @param varname Variable name  
	 * @throws DMLRuntimeException 
	 */
	public void printMatrixVariable(LocalVariableMap variables, String varname) throws DMLRuntimeException {
		if (varname == null) {
			System.err.println("No matrix variable name entered.");
			return;
		}
		if (variables != null && !variables.keySet().isEmpty()) {
			if (variables.get(varname) != null) {
				if (variables.get(varname).getDataType() == DataType.MATRIX) {
					try {
						MatrixObject mo = (MatrixObject) variables.get(varname);
						if (mo.getStatusAsString() == "EMPTY" && (OptimizerUtils.estimateSizeExactSparsity(mo.getNumRows(), mo.getNumColumns(), mo.getSparsity()) > OptimizerUtils.getLocalMemBudget())) {
							//TODO @jlugoma Need to add functionality to bring and display a block. 
							System.err.println("ERROR: DML matrix/vector dimensions are too large to fit in main memory.");
							return;
						}
						MatrixBlock mb = mo.acquireRead();						
						prettyPrintMatrixBlock(mb, -1, -1);
						mo.release();
						if (mb.getNumRows() > DISPLAY_MAX_ROWS || mb.getNumColumns() > DISPLAY_MAX_COLUMNS) {
							System.out.format("WARNING: DML matrix/vector is too large to display on the screen."
									+ "\nOnly a snapshot of %d row(s) and %d column(s) is being displayed.\n", 
									min(mb.getNumRows(), DISPLAY_MAX_ROWS), min(mb.getNumColumns(), DISPLAY_MAX_COLUMNS));
						}						
						System.out.println("Metadata: "+variables.get(varname).getMetaData().toString());						
					} catch (Exception e) {
						System.err.println("Error processing display DML matrix command for variable "+varname+".");
						return;
					}
				}
				else
					System.out.println("Variable \""+varname+"\" is not a matrix or vector variable.");
			}
			else
				System.out.println("DML matrix variable \""+varname+"\" is not in the current frame scope. Try \"a\" to list all variables within current frame scope.");
		}
		else
			System.out.println("Symbol table for current frame is empty");
	}
	
	/**
	 * Print DML matrix cell contents in current frame (if existing)
	 * @param variables Current frame variables
 	 * @param args CLI arguments for current debugger function (Variable name, and row and column indexes)  
	 */
	public void printMatrixCell(LocalVariableMap variables, String [] args) {
		String varname = args[0];
		int rowIndex, columnIndex;
		try {
			rowIndex = Integer.parseInt(args[1]);
			columnIndex = Integer.parseInt(args[2]);
		} catch (Exception e) {
			System.err.print("Invalid display cell arguments.");
			return;
		}
		if (variables != null && !variables.keySet().isEmpty()) {
			if (variables.get(varname) != null) {
				if (variables.get(varname).getDataType() == DataType.MATRIX) {
					double cellValue;
					try {
						MatrixObject mo = (MatrixObject) variables.get(varname);
						if (mo.getStatusAsString() == "EMPTY" && (OptimizerUtils.estimateSizeExactSparsity(mo.getNumRows(), mo.getNumColumns(), mo.getSparsity()) > OptimizerUtils.getLocalMemBudget())) {
							//TODO @jlugoma Need to add functionality to bring and display a block. 
							System.err.println("ERROR: DML matrix/vector dimensions are too large to fit in main memory.");
							return;
						}						
						MatrixBlock mb = mo.acquireRead();
						cellValue = mb.getValue(rowIndex, columnIndex);
						mo.release();
					} catch (Exception e) {
						System.err.println("Error processing DML matrix variable "+varname+". Certain matrix operations are disabled due to memory constraints or read-only restrictions.");
						return;
					}
					System.out.println(varname+"["+rowIndex+","+columnIndex+"] = "+cellValue);
				}
				else
					System.out.println("Variable \""+varname+"\" is not a matrix or vector variable.");
			}
			else
				System.out.println("DML matrix variable \""+varname+"\" is not in the current frame scope. Try \"a\" to list all variables within current frame scope.");
		}
		else
			System.out.println("Symbol table for current frame is empty");
	}

	/**
	 * Set DML matrix cell contents in current frame (if existing)
	 * @param variables Current frame variables
 	 * @param args CLI arguments for current debugger function (Variable name, and row and column indexes)  
	 */
	public void setMatrixCell(LocalVariableMap variables, String [] args) {
		String varname = args[0];
		int rowIndex, columnIndex;
		double value;
		try {
			rowIndex = Integer.parseInt(args[1]) - 1;
			columnIndex = Integer.parseInt(args[2]) - 1; 
			value = Double.parseDouble(args[3]);
		} catch (Exception e) {
			System.err.print("Invalid set cell arguments.");
			return;
		}
		if (variables != null && !variables.keySet().isEmpty()) {
			if (variables.get(varname) != null) {
				if (variables.get(varname).getDataType() == DataType.MATRIX) {
					double updatedCellValue;
					try {
						MatrixObject mo = (MatrixObject) variables.get(varname);
						if (mo.getStatusAsString() == "EMPTY" && (OptimizerUtils.estimateSizeExactSparsity(mo.getNumRows(), mo.getNumColumns(), mo.getSparsity()) > OptimizerUtils.getLocalMemBudget())) {
							//TODO @jlugoma Need to add functionality to bring and display a block. 
							System.err.println("ERROR: DML matrix/vector dimensions are too large to fit in main memory.");
							return;
						}						
						MatrixBlock mb = mo.acquireModify();
						mb.setValue(rowIndex, columnIndex, value);
						updatedCellValue = mb.getValue(rowIndex, columnIndex);
						mo.release();
					} catch (Exception e) {
						System.err.println("Error processing DML matrix variable "+varname+". Certain matrix operations are disabled due to memory constraints or read-only restrictions.");
						return;
					}
					System.out.println(varname+"["+ (rowIndex+1) +","+ (columnIndex+1) +"] = "+updatedCellValue);
				}
				else
					System.out.println("Variable \""+varname+"\" is not a matrix or vector variable.");
			}
			else
				System.out.println("DML matrix variable \""+varname+"\" is not in the current frame scope. Try \"a\" to list all variables within current frame scope.");
		}
		else
			System.out.println("Symbol table for current frame is empty");
	}

	/////////////////////////////////////////
	// internal debugger parsing functions //
	/////////////////////////////////////////
	
	/**
	 * Parse command line argument and return valid value
	 * @param args CLI arguments for current debugger function
	 * @return Validated value for current debug command  
	 */
	protected String getValue(String[] args) {
		String value = null;
		if (args != null) {
			if (args.length > 1)
				System.err.println("Invalid number of argument values for this command. Try \"help\".");
			value = args[0]; 
		}
		return value;
	}	

	/**
	 * Parse command line argument and return valid integer   
	 * @param args CLI arguments for current debugger function
	 * @param length Maximum value for input parameter  
	 * @return Validated integer value for current debug command  
	 */
	protected int getValue(String[] args, int length) {
		int lineNum = 0;
		if (args == null || args.length > 1) {
			System.err.print("Invalid argument value for this command. Parameter must be a positive integer <= "+length);
		}
		else {
			try {
				lineNum = Integer.parseInt(args[0]);				
				if (lineNum <= 0 || lineNum > length) 
				{
					System.err.println("Invalid argument value for this command. Parameter must be a positive integer <= "+length);
					lineNum = 0;
				}
			} catch (NumberFormatException e) {
				System.err.println("Invalid integer format. Parameter must be a positive integer <= "+length);
				lineNum = 0;
			}
		}
		return lineNum;
	}
	
	/**
	 * Parse command line arguments and return valid IntRange variable 
	 * @param args CLI arguments for range of debugger display functionality 
	 * @param size Size (number of lines of code) of DML script
	 * @return Validated range of lines within DML script to be displayed  
	 * @throws DMLDebuggerException Invalid range 
	 */
	protected IntRange getRange(String [] args, int length) throws DMLDebuggerException {
		IntRange range = new IntRange(1, length);
		if (args == null)
			return range;
		if (args.length == 2) {
			try {
				range = new IntRange(Integer.parseInt(args[0]), Integer.parseInt(args[1]));
				if (range.getMinimumInteger() <= 0 || range.getMaximumInteger() > length) {
					System.err.println("Invalid range values. Parameters <start end> must be two positive integers.");
					range = new IntRange(0, 0);
				}
			} catch (NumberFormatException e) {
				System.err.println("Invalid integer range format. Parameter must be a positive integer <= "+length);
				range = new IntRange(0, 0);
			}				
		}
		else { 
			System.err.println("Invalid range values. Parameters <start end> must be two positive integers.");
			range = new IntRange(0, 0);
		}		
		return range;
	}
	
	/**
	 * Returns minimum between two integers
	 * @param a Integer value 
	 * @param b Integer value
	 * @return Minimum between a and b.
	 */
	private int min(int a, int b) {
		if (a < b)
			return a;
		return b;
	}
	
	/**
	 * Displays a pretty-printed version of a DML matrix or vector variable. 
	 * @param mb Current matrix block
	 * @param rowIndex if rowIndex == -1, then prints all rows
	 * @param colIndex if colIndex == -1, then prints all columns
	 */
	private void prettyPrintMatrixBlock(MatrixBlock mb, int rowIndex, int colIndex) {
		if(rowIndex <= 0 && colIndex <= 0) {
			// Print entire matrix
			for(int i=0; i<min(mb.getNumRows(), DISPLAY_MAX_ROWS); i++) {
				for(int j=0; j<min(mb.getNumColumns(), DISPLAY_MAX_COLUMNS); j++) {
					System.out.format("%.4f\t", mb.quickGetValue(i, j));
				}
				System.out.println();
			}
			if (mb.getNumRows() > DISPLAY_MAX_ROWS || mb.getNumColumns() > DISPLAY_MAX_COLUMNS) {
				System.out.format("WARNING: DML matrix/vector is too large to display on the screen."
						+ "\nOnly a snapshot of %d row(s) and %d column(s) is being displayed.\n", 
						min(mb.getNumRows(), DISPLAY_MAX_ROWS), min(mb.getNumColumns(), DISPLAY_MAX_COLUMNS));
			}
		}
		else if(rowIndex >= 0 && colIndex >= 0) {
			// Print a cell
			System.out.format("%.4f\n", mb.quickGetValue(rowIndex-1, colIndex-1));
		}
		else if(rowIndex >= 0) {
			// Print a row
			//for(int i=0; i<min(mb.getNumRows(), DISPLAY_MAX_ROWS); i++) {
				for(int j=0; j<min(mb.getNumColumns(), DISPLAY_MAX_COLUMNS); j++) {
					System.out.format("%.4f\t", mb.quickGetValue(rowIndex-1, j));
				}
				System.out.println();
				if (mb.getNumColumns() > DISPLAY_MAX_COLUMNS) {
					System.out.format("WARNING: the row of given DML matrix/vector is too large to display on the screen."
							+ "\nOnly a snapshot of %d column(s) is being displayed.\n", 
							min(mb.getNumColumns(), DISPLAY_MAX_COLUMNS));
				}
			//}
		}
		else if(colIndex >= 0) {
			// Print a column
			int numItemsPrinted = 0;
			for(int i=0; i<min(mb.getNumRows(), DISPLAY_MAX_ROWS); i++) {
				System.out.format("%.4f\t", mb.quickGetValue(i, colIndex-1));
				System.out.println();
			}
			if (mb.getNumRows() > DISPLAY_MAX_ROWS) {
				System.out.format("WARNING: the column of given DML matrix/vector is too large to display on the screen."
						+ "\nOnly a snapshot of %d row(s) is being displayed.\n", 
						min(mb.getNumRows(), DISPLAY_MAX_ROWS));
			}
		}
	}
	
	/**
	 * Prepare current instruction for printing
	 * by removing internal delimiters.  
	 * @param inst Instruction to be displayed 
	 * @return Post-processed instruction in string format
	 */
	private static String prepareInstruction(String inst) {
		String tmp = inst;
		tmp = tmp.replaceAll(Lop.OPERAND_DELIMITOR, " ");
		tmp = tmp.replaceAll(Lop.DATATYPE_PREFIX, ".");
		tmp = tmp.replaceAll(Lop.INSTRUCTION_DELIMITOR, ", ");

		return tmp;
	}
}
