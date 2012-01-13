package dml.runtime.controlprogram;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map.Entry;

import dml.api.DMLScript;
import dml.lops.runtime.RunMRJobs;
import dml.parser.Expression.ValueType;
import dml.runtime.instructions.CPInstructionParser;
import dml.runtime.instructions.Instruction;
import dml.runtime.instructions.MRJobInstruction;
import dml.runtime.instructions.CPInstructions.BooleanObject;
import dml.runtime.instructions.CPInstructions.CPInstruction;
import dml.runtime.instructions.CPInstructions.Data;
import dml.runtime.instructions.CPInstructions.DoubleObject;
import dml.runtime.instructions.CPInstructions.FileObject;
import dml.runtime.instructions.CPInstructions.IntObject;
import dml.runtime.instructions.CPInstructions.ScalarObject;
import dml.runtime.instructions.CPInstructions.StringObject;
import dml.runtime.instructions.Instruction.INSTRUCTION_TYPE;
import dml.runtime.instructions.SQLInstructions.SQLInstructionBase;
import dml.runtime.matrix.JobReturn;
import dml.runtime.matrix.MatrixCharacteristics;
import dml.runtime.matrix.MatrixDimensionsMetaData;
import dml.runtime.matrix.MetaData;
import dml.sql.sqlcontrolprogram.ExecutionContext;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;
import dml.utils.Statistics;

public class ProgramBlock {
	
	protected Program _prog;		// pointer to Program this ProgramBlock is part of
	protected ArrayList<Instruction> _inst;
	protected HashMap<String, Data> _variables;
	protected HashMap<String, MetaData> _matrices;
	
	public ProgramBlock(Program prog) {
		
		_prog = prog;
		_variables = new HashMap<String, Data>();
		_matrices = new HashMap<String, MetaData>();
		_inst = new ArrayList<Instruction>();
		//_mapList = new ArrayList<OutputPair>();
	}
    
	public void setVariables(HashMap<String, Data> vars) {
		_variables.putAll(vars);
	}

	public Program getProgram(){
		return _prog;
	}
	
	public void setProgram(Program prog){
		_prog = prog;
	}
	
	/*
	 * Methods to manipulate _matrices structure
	 */
	public void setMetaData(HashMap<String, MetaData> mdmap) {
		_matrices.putAll(mdmap);
	}

	public HashMap<String, MetaData> getMetaData() {
		return _matrices;
	}

	public void setMetaData(String fname, MetaData md) {
		_matrices.put(fname, md);
	}
	
	public MetaData getMetaData(String fname) {
		return _matrices.get(fname);
	}
	
	public void removeMetaData(String fname) {
		_matrices.remove(fname);
	}
	
	public void execute(ExecutionContext ec) throws DMLRuntimeException, DMLUnsupportedOperationException {
		execute(_inst, ec);
	}

	public void removeVariable(String name) {
		_variables.remove(name);
	}
	
	protected void execute(ArrayList<Instruction> inst, ExecutionContext ec) throws DMLRuntimeException, DMLUnsupportedOperationException {
		if ( DMLScript.DEBUG ) {
			// print _variables map
			System.out.println("____________________________________");
			System.out.println("___ Variables ____");
			Iterator<Entry<String, Data>> it = _variables.entrySet().iterator();
			while (it.hasNext()) {
				Entry<String,Data> pairs = it.next();
			    System.out.println("  " + pairs.getKey() + " = " + pairs.getValue());
			}
			System.out.println("___ Matrices ____");
			Iterator<Entry<String, MetaData>> mit = _matrices.entrySet().iterator();
			while (mit.hasNext()) {
				Entry<String,MetaData> pairs = mit.next();
			    System.out.println("  " + pairs.getKey() + " = " + pairs.getValue().toString());
			}
			System.out.println("____________________________________");
		}
		updateMatrixLabels();

		for (int i = 0; i < inst.size(); i++) {
			Instruction currInst = inst.get(i);
			if (currInst instanceof MRJobInstruction) {
				MRJobInstruction currMRInst = (MRJobInstruction) currInst;
				
				currMRInst.setInputLabelValueMapping(_variables);
				currMRInst.setOutputLabelValueMapping(_variables);
				
				JobReturn jb = RunMRJobs.submitJob(currMRInst, this);
				
				/* Populate returned stats into symbol table of matrices */
				for ( int index=0; index < jb.getMetaData().length; index++) {
					_matrices.put(currMRInst.getIv_outputs()[index], jb.getMetaData(index));
				}
				
				Statistics.setNoOfExecutedMRJobs(Statistics.getNoOfExecutedMRJobs() + 1);
			} else if (currInst instanceof CPInstruction) {
				String updInst = RunMRJobs.updateLabels(currInst.toString(), _variables);
				CPInstruction si = CPInstructionParser.parseSingleInstruction(updInst);
				si.processInstruction(this);
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
	

	/*
	 * _matrices metadata is initialized in initInputMatrixMetada().
	 * Since initInputMatrixMetada() relies on instructions generated by piggybacking, 
	 * some input matrix labels may be placeholders (i.e., ##<label>##).
	 * Such labels must be removed from _matrices data structure.. by invoking updateMatrixLabels().
	 */
	public void updateMatrixLabels() throws DMLRuntimeException {
		String varName = null, fileName = null;
		MatrixCharacteristics matchar = null;
		ArrayList<String> deleteFileLabels = new ArrayList<String>();
		HashMap<String,MetaData> insertFileLabels = new HashMap<String,MetaData>();
		for ( String flabel : _matrices.keySet()) {
			if ( flabel.startsWith("##") && flabel.endsWith("##")) {
				varName = flabel.replaceAll("##", "");
				if(DMLScript.DEBUG)
					System.out.println("flabel=" + flabel + ", varname=" + varName + ", fobject=" + _variables.get(varName));
				
				if (_variables.get(varName) instanceof FileObject){
				
					FileObject fobj = (FileObject)(_variables.get(varName));
					if ( fobj == null ) {
						throw new DMLRuntimeException("Unexpected error: variable '" + varName + "' is not present in _variables map.");
					}
					fileName = fobj.getFilePath();
					
					matchar = ((MatrixDimensionsMetaData)_matrices.get(fileName)).getMatrixCharacteristics();
					if ( matchar == null ) 
						throw new DMLRuntimeException("Could not locate metadata for HDFS file: " + fileName);
					else 
						deleteFileLabels.add(flabel);
					
					
				}
				else if (_variables.get(varName) instanceof StringObject){
					fileName = ((StringObject)_variables.get(varName)).getStringValue();
					if (insertFileLabels.containsKey(fileName))
						System.out.println("inserting " + fileName + "in _matrices multiple times");
					insertFileLabels.put(fileName,_matrices.get(flabel));
					deleteFileLabels.add(flabel);
				}
				else {
					throw new DMLRuntimeException("Unexpected error: variable '" + varName + "' is not of FileObject or StringObject type.");
				}
					
			
			}
		}
		
		for ( int i=0; i < deleteFileLabels.size(); i++ ) {
			_matrices.remove(deleteFileLabels.get(i));
		}
		// insert filename labels for filenames passed as variables (IOStatements)
		for (String key : insertFileLabels.keySet()) {
			_matrices.put(key, insertFileLabels.get(key));
		}
		
	}
	
	/*
	 * Method to initialize the hashmap that stores matrix metadata (_matrices).
	 * It is invoked for each statement block (runtime program block), immediately after all instructions are generated. 
	 */
	public void initInputMatrixMetadata() throws DMLRuntimeException {
		for (Instruction inst : _inst ) {
			if ( inst.getType() == INSTRUCTION_TYPE.MAPREDUCE_JOB ) {
				MRJobInstruction jobinst = (MRJobInstruction) inst;
				if ( jobinst.getIv_inputs() != null ) {
					for ( int i=0; i < jobinst.getIv_inputs().length; i++ ) {
						MatrixCharacteristics matchar = new MatrixCharacteristics(); 
						matchar.numRows = jobinst.getIv_rows()[i];
						matchar.numColumns = jobinst.getIv_cols()[i];
						matchar.numRowsPerBlock = jobinst.getIv_num_rows_per_block()[i];
						matchar.numColumnsPerBlock = jobinst.getIv_num_cols_per_block()[i];
						MatrixDimensionsMetaData mdmd = new MatrixDimensionsMetaData(matchar);
						
						if ( _matrices.get(jobinst.getIv_inputs()[i]) == null ) {
							_matrices.put(jobinst.getIv_inputs()[i], mdmd);
						}
					}
				}
			}
		}
		
	}
	
	public Data getVariable(String name, ValueType vt) {
		Data obj = _variables.get(name);
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
		return obj;
	}

	public ScalarObject getScalarVariable(String name, ValueType vt) {
		return (ScalarObject) getVariable(name, vt);
	}

	public HashMap<String, Data> getVariables() {
		return _variables;
	}

	public void setVariable(String name, Data val) {
		_variables.put(name, val);
	}

	public int getNumInstructions() {
		return _inst.size();
	}

	public void addInstruction(Instruction inst) {
		_inst.add(inst);
	}

	public void addVariables(HashMap<String, Data> vars) {
		_variables.putAll(vars);
	}
	public Instruction getInstruction(int i) {
		return _inst.get(i);
	}
	
	public void printMe() {
		//System.out.println("***** INSTRUCTION BLOCK *****");
		for (Instruction i : this._inst) {
			i.printMe();
		}
	}
}
