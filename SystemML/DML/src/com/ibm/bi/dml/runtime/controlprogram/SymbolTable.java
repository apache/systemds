package com.ibm.bi.dml.runtime.controlprogram;

import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.BooleanObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.DoubleObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.IntObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.ScalarObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.StringObject;
import com.ibm.bi.dml.runtime.matrix.MetaData;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.utils.CacheException;
import com.ibm.bi.dml.utils.DMLRuntimeException;

/**
 * Data structure to hold the state of different variables during the course of
 * program execution. It is a hierarchical structure that holds:
 * 
 * 1) top-level variable map for an entire statement block, which primarily used
 * for passing around variables across different child statement blocks.
 * 
 * 2) array list of symbol tables for all the child statement blocks
 * 
 */

public class SymbolTable {
	private LocalVariableMap _variableMap;
	private ArrayList<SymbolTable> _childTables;

	public LocalVariableMap get_variableMap() {
		return _variableMap;
	}

	public void set_variableMap(LocalVariableMap varMap) {
		_variableMap = varMap;
	}

	public void copy_variableMap(LocalVariableMap varMap) {
		_variableMap.putAll(varMap);
	}

	public ArrayList<SymbolTable> get_childTables() {
		return _childTables;
	}

	public void set_childTables(ArrayList<SymbolTable> childTables) {
		_childTables = childTables;
	}

	public SymbolTable getChildTable(int i) {
		if (_childTables == null || _childTables.size() <= i)
			return null;
		return _childTables.get(i);
	}

	public SymbolTable() {
		_variableMap = null;
		_childTables = null;
	}

	public SymbolTable(boolean allocateMap) {
		if (allocateMap)
			_variableMap = new LocalVariableMap();
		else
			_variableMap = null;
		_childTables = null; // new ArrayList<SymbolTable>();
	}

	SymbolTable(int numChildBlocks) {
		_variableMap = new LocalVariableMap();
		_childTables = new ArrayList<SymbolTable>(numChildBlocks);
	}

	public void addChildTable(SymbolTable childTable) {
		if (_childTables == null) {
			_childTables = new ArrayList<SymbolTable>();
		}
		_childTables.add(childTable);
	}

	/* -------------------------------------------------------
	 * Methods to handle variables and associated data objects
	 * -------------------------------------------------------
	 */
	
	public Data getVariable(String name) {
		return _variableMap.get(name);
	}
	
	public void setVariable(String name, Data val) throws DMLRuntimeException{
		_variableMap.put(name, val);
	}

	public void removeVariable(String name) {
		_variableMap.remove(name);
	}

	public String getVariableString(String name, boolean forSQL)
	{
		Data obj = _variableMap.get(name);
		if(obj != null)
		{
			String s = ((ScalarObject)obj).getStringValue();
			if(obj instanceof StringObject)
				s = "'" + s + "'";
			else if (obj instanceof DoubleObject && forSQL)
				s = s + "::double precision";
 			return s;
		}
		else return name;
	}

	public void setMetaData(String fname, MetaData md) throws DMLRuntimeException {
		_variableMap.get(fname).setMetaData(md);
	}
	
	public MetaData getMetaData(String varname) throws DMLRuntimeException {
		return _variableMap.get(varname).getMetaData();
	}
	
	public void removeMetaData(String varname) throws DMLRuntimeException {
		_variableMap.get(varname).removeMetaData();
	}
	
	public MatrixBlock getMatrixInput(String varName) throws DMLRuntimeException {
		try {
			MatrixObject mobj = (MatrixObject) this.getVariable(varName);
			return mobj.acquireRead();
		} catch (CacheException e) {
			throw new DMLRuntimeException(this.printBlockErrorLocation() , e);
		}
	}
	
	public void releaseMatrixInput(String varName) throws DMLRuntimeException {
		try {
			((MatrixObject)this.getVariable(varName)).release();
		} catch (CacheException e) {
			throw new DMLRuntimeException(this.printBlockErrorLocation() , e);
		}
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
					throw new DMLRuntimeException(this.printBlockErrorLocation() + "Unknown variable: " + name + ", or unknown value type: " + vt);
				}
			} 
			catch (Exception e) 
			{	
				e.printStackTrace();
			}
		}
		return (ScalarObject) obj;
	}

	
	public void setScalarOutput(String varName, ScalarObject so) throws DMLRuntimeException {
		this.setVariable(varName, so);
	}
	
	public void setMatrixOutput(String varName, MatrixBlock outputData) throws DMLRuntimeException {
		MatrixObject sores = (MatrixObject) this.getVariable (varName);
        
		try {
			sores.acquireModify (outputData);
	        
	        // Output matrix is stored in cache and it is written to HDFS, on demand, at a later point in time. 
	        // downgrade() and exportData() need to be called only if we write to HDFS.
	        //sores.downgrade();
	        //sores.exportData();
	        sores.release();
	        
	        this.setVariable (varName, sores);
		
		} catch ( CacheException e ) {
			throw new DMLRuntimeException(this.printBlockErrorLocation() , e);
		}
	}
	
	/**
	 * Pin a given list of variables i.e., set the "clean up" state in 
	 * corresponding matrix objects, so that the cached data inside these
	 * objects is not cleared and the corresponding HDFS files are not 
	 * deleted (through rmvar instructions). 
	 * 
	 * The function returns the OLD "clean up" state of matrix objects.
	 */
	public HashMap<String,Boolean> pinVariables(ArrayList<String> varList) 
	{
		HashMap<String, Boolean> varsState = new HashMap<String,Boolean>();
		
		for( String var : varList )
		{
			Data dat = _variableMap.get(var);
			if( dat instanceof MatrixObject )
			{
				//System.out.println("pin ("+_ID+") "+var);
				MatrixObject mo = (MatrixObject)dat;
				varsState.put( var, mo.isCleanupEnabled() );
				mo.enableCleanup(false); 
			}
		}
		return varsState;
	}
	
	/**
	 * Unpin the a given list of variables by setting their "cleanup" status
	 * to the values specified by <code>varsStats</code>.
	 * 
	 * Typical usage:
	 *    <code> 
	 *    oldStatus = pinVariables(varList);
	 *    ...
	 *    unpinVariables(varList, oldStatus);
	 *    </code>
	 * 
	 * i.e., a call to unpinVariables() is preceded by pinVariables(). 
	 */
	public void unpinVariables(ArrayList<String> varList, HashMap<String,Boolean> varsState)
	{
		for( String var : varList)
		{
			//System.out.println("unpin ("+_ID+") "+var);
			
			Data dat = _variableMap.get(var);
			if( dat instanceof MatrixObject )
				((MatrixObject)dat).enableCleanup(varsState.get(var));
		}
	}


	///////////////////////////////////////////////////////////////////////////
	// store position information for program blocks
	///////////////////////////////////////////////////////////////////////////
	
	public int _beginLine, _beginColumn;
	public int _endLine, _endColumn;
	
	public void setBeginLine(int passed)    { _beginLine = passed;   }
	public void setBeginColumn(int passed) 	{ _beginColumn = passed; }
	public void setEndLine(int passed) 		{ _endLine = passed;   }
	public void setEndColumn(int passed)	{ _endColumn = passed; }
	
	public void setAllPositions(int blp, int bcp, int elp, int ecp){
		_beginLine	 = blp; 
		_beginColumn = bcp; 
		_endLine 	 = elp;
		_endColumn 	 = ecp;
	}

	public int getBeginLine()	{ return _beginLine;   }
	public int getBeginColumn() { return _beginColumn; }
	public int getEndLine() 	{ return _endLine;   }
	public int getEndColumn()	{ return _endColumn; }
	
	public String printErrorLocation(){
		return "ERROR: line " + _beginLine + ", column " + _beginColumn + " -- ";
	}
	
	public String printBlockErrorLocation(){
		return "ERROR: Runtime error in program block generated from statement block between lines " + _beginLine + " and " + _endLine + " -- ";
	}
	
	public String printWarningLocation(){
		return "WARNING: line " + _beginLine + ", column " + _beginColumn + " -- ";
	}

}
