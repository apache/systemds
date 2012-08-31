package com.ibm.bi.dml.runtime.controlprogram;

import java.util.ArrayList;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.lops.runtime.RunMRJobs;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.instructions.CPInstructionParser;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.BooleanObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.CPInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.DoubleObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.IntObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.MatrixObjectNew;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.ScalarObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.StringObject;
import com.ibm.bi.dml.runtime.instructions.SQLInstructions.SQLInstructionBase;
import com.ibm.bi.dml.runtime.matrix.JobReturn;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixDimensionsMetaData;
import com.ibm.bi.dml.runtime.matrix.MetaData;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.sql.sqlcontrolprogram.ExecutionContext;
import com.ibm.bi.dml.utils.CacheException;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;
import com.ibm.bi.dml.utils.Statistics;


public class ProgramBlock {
	
	protected Program _prog;		// pointer to Program this ProgramBlock is part of
	protected ArrayList<Instruction> _inst;
	protected LocalVariableMap _variables;
	
	public ProgramBlock(Program prog) throws DMLRuntimeException {
		
		_prog = prog;
		_variables = new LocalVariableMap ();
		_inst = new ArrayList<Instruction>();
	}
    
	public void setVariables (LocalVariableMap vars)
	{
		_variables.putAll (vars);
	}

	public Program getProgram(){
		return _prog;
	}
	
	public void setProgram(Program prog){
		_prog = prog;
	}
	
	public void setMetaData(String fname, MetaData md) throws DMLRuntimeException {
		_variables.get(fname).setMetaData(md);
	}
	
	public MetaData getMetaData(String varname) throws DMLRuntimeException {
		return _variables.get(varname).getMetaData();
	}
	
	public void removeMetaData(String varname) throws DMLRuntimeException {
		 _variables.get(varname).removeMetaData();
	}
	
	public void execute(ExecutionContext ec) throws DMLRuntimeException, DMLUnsupportedOperationException {
		execute(_inst, ec);
	}

	public void removeVariable(String name) {
		_variables.remove(name);
	}
	
	private void printSymbolTable() {
		// print _variables map
		System.out.println ("____________________________________");
		System.out.println ("___ Variables ____");
		System.out.print   (_variables.toString ());
		
/*		System.out.println("___ Matrices ____");
		Iterator<Entry<String, MetaData>> mit = _matrices.entrySet().iterator();
		while (mit.hasNext()) {
			Entry<String,MetaData> pairs = mit.next();
		    System.out.println("  " + pairs.getKey() + " = " + pairs.getValue().toString());
		}
*/		System.out.println("____________________________________");
	}
	
	protected void execute(ArrayList<Instruction> inst, ExecutionContext ec) throws DMLRuntimeException, DMLUnsupportedOperationException {
		if ( DMLScript.DEBUG ) {
			printSymbolTable();
		}

		for (int i = 0; i < inst.size(); i++) 
		{
			//indexed access required due to dynamic add
			Instruction currInst = inst.get(i);
			
			if (currInst instanceof MRJobInstruction) 
			{
				if ( DMLScript.DEBUG ) 
					printSymbolTable();
				
				long begin = System.currentTimeMillis();
				MRJobInstruction currMRInst = (MRJobInstruction) currInst;
				
				JobReturn jb = RunMRJobs.submitJob(currMRInst, this);
				
				if ( currMRInst.getJobType() == JobType.SORT ) {
					if ( jb.getMetaData().length > 0 ) {
						/* Populate returned stats into symbol table of matrices */
						for ( int index=0; index < jb.getMetaData().length; index++) {
							String varname = currMRInst.getOutputVars()[index];
							_variables.get(varname).setMetaData(jb.getMetaData()[index]); // updateMatrixCharacteristics(mc);
						}
					}
				}
				else {
					if ( jb.getMetaData().length > 0 ) {
						/* Populate returned stats into symbol table of matrices */
						for ( int index=0; index < jb.getMetaData().length; index++) {
							String varname = currMRInst.getOutputVars()[index];
							MatrixCharacteristics mc = ((MatrixDimensionsMetaData)jb.getMetaData(index)).getMatrixCharacteristics();
							_variables.get(varname).updateMatrixCharacteristics(mc);
						}
					}
				}
				if ( DMLScript.DEBUG )
					System.out.println("MRJob\t" + currMRInst.getJobType() + "\t" + (System.currentTimeMillis()-begin));
				
				Statistics.setNoOfExecutedMRJobs(Statistics.getNoOfExecutedMRJobs() + 1);
			} 
			else if (currInst instanceof CPInstruction) 
			{
				if( currInst.requiresLabelUpdate() ) //update labels only if required
				{
					String currInstStr = currInst.toString();
					String updInst = RunMRJobs.updateLabels(currInstStr, _variables);
					if ( DMLScript.DEBUG )
						System.out.println("Processing CPInstruction: " + updInst);
					
					CPInstruction si = CPInstructionParser.parseSingleInstruction(updInst);
					si.processInstruction(this);
					
					//note: no exchange of updated instruction as labels might change in the general case
				}
				else 
				{
					if ( DMLScript.DEBUG )
						System.out.println("Processing CPInstruction: " + currInst.toString());
					((CPInstruction) currInst).processInstruction(this); 
				}
				if ( DMLScript.DEBUG )
					System.out.println("    memory stats = [" + (Runtime.getRuntime().freeMemory()/(double)(1024*1024)) + ", " + (Runtime.getRuntime().totalMemory()/(double)(1024*1024)) + "].");
			} 
			else if(currInst instanceof SQLInstructionBase)
			{
				try{
				((SQLInstructionBase)currInst).execute(ec);
				}
				catch(Exception e)
				{
					e.printStackTrace();
				}
			}
			/*
			else if(currInst instanceof SQLInstruction)
				((SQLInstruction)currInst).execute(ec);
			else if(currInst instanceof SQLScalarAssignInstruction)
				((SQLScalarAssignInstruction)currInst).execute(ec);
			else if(currInst instanceof SQLPrintInstruction)
				((SQLPrintInstruction)currInst).execute(ec);
				*/
		}
	}
	
	public MatrixBlock getMatrixInput(String varName) throws DMLRuntimeException {
		try {
			MatrixObjectNew mobj = (MatrixObjectNew) this.getVariable(varName);
			return mobj.acquireRead();
		} catch (CacheException e) {
			throw new DMLRuntimeException(e);
		}
	}
	
	public void releaseMatrixInput(String varName) throws DMLRuntimeException {
		try {
			((MatrixObjectNew)this.getVariable(varName)).release();
		} catch (CacheException e) {
			throw new DMLRuntimeException(e);
		}
	}
	
	public Data getVariable(String name) {
		return _variables.get(name);
	}
	
	public ScalarObject getScalarInput(String name, ValueType vt) {
		Data obj = getVariable(name);
		if (obj == null) {
			try {
				switch (vt) {
				case INT:
					int intVal = Integer.parseInt(name);
					IntObject intObj = new IntObject(intVal);
					return intObj;
				case DOUBLE:
					double doubleVal = Double.parseDouble(name);
					DoubleObject doubleObj = new DoubleObject(doubleVal);
					return doubleObj;
				case BOOLEAN:
					Boolean boolVal = Boolean.parseBoolean(name);
					BooleanObject boolObj = new BooleanObject(boolVal);
					return boolObj;
				case STRING:
					StringObject stringObj = new StringObject(name);
					return stringObj;
				default:
					throw new DMLRuntimeException("Unknown variable: " + name + ", or unknown value type: " + vt);
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		return (ScalarObject) obj;
	}

	public LocalVariableMap getVariables() {
		return _variables;
	}

	public void setVariable(String name, Data val) throws DMLRuntimeException
	{
		_variables.put(name, val);
	}
	
	public void setScalarOutput(String varName, ScalarObject so) throws DMLRuntimeException {
		this.setVariable(varName, so);
	}
	
	public void setMatrixOutput(String varName, MatrixBlock outputData) throws DMLRuntimeException {
		MatrixObjectNew sores = (MatrixObjectNew) this.getVariable (varName);
        
		try {
			sores.acquireModify (outputData);
	        
	        // Output matrix is stored in cache and it is written to HDFS, on demand, at a later point in time. 
	        // downgrade() and exportData() need to be called only if we write to HDFS.
	        //sores.downgrade();
	        //sores.exportData();
	        sores.release();
	        
	        this.setVariable (varName, sores);
		
		} catch ( CacheException e ) {
			throw new DMLRuntimeException(e);
		}
	}

	public int getNumInstructions() {
		return _inst.size();
	}

	public void addInstruction(Instruction inst) {
		_inst.add(inst);
	}

	public void addVariables (LocalVariableMap vars) {
		_variables.putAll (vars);
	}
	public Instruction getInstruction(int i) {
		return _inst.get(i);
	}
	
	public  ArrayList<Instruction> getInstructions() 
	{
		return _inst;
	}
	
	public  void setInstructions( ArrayList<Instruction> inst ) 
	{
		_inst = inst;
	}
	
	public void printMe() {
		//System.out.println("***** INSTRUCTION BLOCK *****");
		for (Instruction i : this._inst) {
			i.printMe();
		}
	}
}
