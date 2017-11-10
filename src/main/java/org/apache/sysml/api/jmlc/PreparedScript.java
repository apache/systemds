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

package org.apache.sysml.api.jmlc;

import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.api.DMLException;
import org.apache.sysml.conf.CompilerConfig;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.conf.CompilerConfig.ConfigType;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.hops.ipa.FunctionCallGraph;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.Program;
import org.apache.sysml.runtime.controlprogram.caching.FrameObject;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysml.runtime.instructions.cp.BooleanObject;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.instructions.cp.DoubleObject;
import org.apache.sysml.runtime.instructions.cp.IntObject;
import org.apache.sysml.runtime.instructions.cp.ScalarObject;
import org.apache.sysml.runtime.instructions.cp.StringObject;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MetaDataFormat;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.utils.Explain;

/**
 * Representation of a prepared (precompiled) DML/PyDML script.
 */
public class PreparedScript 
{
	private static final Log LOG = LogFactory.getLog(PreparedScript.class.getName());
	
	//input/output specification
	private final HashSet<String> _inVarnames;
	private final HashSet<String> _outVarnames;
	private final HashMap<String,Data> _inVarReuse;
	
	//internal state (reused)
	private final Program _prog;
	private final LocalVariableMap _vars;
	private final DMLConfig _dmlconf;
	private final CompilerConfig _cconf;
	
	private PreparedScript(PreparedScript that) {
		//shallow copy, except for a separate symbol table
		//and related meta data of reused inputs
		_prog = that._prog.clone(false);
		_vars = new LocalVariableMap();
		for(Entry<String, Data> e : that._vars.entrySet())
			_vars.put(e.getKey(), e.getValue());
		_vars.setRegisteredOutputs(that._outVarnames);
		_inVarnames = that._inVarnames;
		_outVarnames = that._outVarnames;
		_inVarReuse = new HashMap<>(that._inVarReuse);
		_dmlconf = that._dmlconf;
		_cconf = that._cconf;
	}
	
	/**
	 * Meant to be invoked only from Connection.
	 * 
	 * @param prog the DML/PyDML program
	 * @param inputs input variables to register
	 * @param outputs output variables to register
	 * @param dmlconf dml configuration 
	 * @param cconf compiler configuration
	 */
	protected PreparedScript( Program prog, String[] inputs, String[] outputs, DMLConfig dmlconf, CompilerConfig cconf ) 
	{
		_prog = prog;
		_vars = new LocalVariableMap();
		
		//populate input/output vars
		_inVarnames = new HashSet<>();
		Collections.addAll(_inVarnames, inputs);
		_outVarnames = new HashSet<>();
		Collections.addAll(_outVarnames, outputs);
		_inVarReuse = new HashMap<>();
		
		//attach registered outputs (for dynamic recompile)
		_vars.setRegisteredOutputs(_outVarnames);
		
		//keep dml and compiler configuration to be set as thread-local config
		//on execute, which allows different threads creating/executing the script
		_dmlconf = dmlconf;
		_cconf = cconf;
	}
	
	/**
	 * Binds a scalar boolean to a registered input variable.
	 * 
	 * @param varname input variable name
	 * @param scalar boolean value
	 * @throws DMLException if DMLException occurs
	 */
	public void setScalar(String varname, boolean scalar) throws DMLException {
		setScalar(varname, scalar, false);
	}
	
	/**
	 * Binds a scalar boolean to a registered input variable.
	 * 
	 * @param varname input variable name
	 * @param scalar boolean value
	 * @param reuse if {@code true}, preserve value over multiple {@code executeScript} calls
	 * @throws DMLException if DMLException occurs
	 */
	public void setScalar(String varname, boolean scalar, boolean reuse) throws DMLException {
		setScalar(varname, new BooleanObject(scalar), reuse);
	}
	
	/**
	 * Binds a scalar long to a registered input variable.
	 * 
	 * @param varname input variable name
	 * @param scalar long value
	 * @throws DMLException if DMLException occurs
	 */
	public void setScalar(String varname, long scalar) throws DMLException {
		setScalar(varname, scalar, false);
	}
	
	/**
	 * Binds a scalar long to a registered input variable.
	 * 
	 * @param varname input variable name
	 * @param scalar long value
	 * @param reuse if {@code true}, preserve value over multiple {@code executeScript} calls
	 * @throws DMLException if DMLException occurs
	 */
	public void setScalar(String varname, long scalar, boolean reuse) throws DMLException {
		setScalar(varname, new IntObject(scalar), reuse);
	}
	
	/** Binds a scalar double to a registered input variable.
	 * 
	 * @param varname input variable name
	 * @param scalar double value
	 * @throws DMLException if DMLException occurs
	 */
	public void setScalar(String varname, double scalar) throws DMLException {
		setScalar(varname, scalar, false);
	}
	
	/**
	 * Binds a scalar double to a registered input variable.
	 * 
	 * @param varname input variable name
	 * @param scalar double value
	 * @param reuse if {@code true}, preserve value over multiple {@code executeScript} calls
	 * @throws DMLException if DMLException occurs
	 */
	public void setScalar(String varname, double scalar, boolean reuse) throws DMLException {
		setScalar(varname, new DoubleObject(scalar), reuse);
	}
	
	/**
	 * Binds a scalar string to a registered input variable.
	 * 
	 * @param varname input variable name
	 * @param scalar string value
	 * @throws DMLException if DMLException occurs
	 */
	public void setScalar(String varname, String scalar) throws DMLException {
		setScalar(varname, scalar, false);
	}
	
	/**
	 * Binds a scalar string to a registered input variable.
	 * 
	 * @param varname input variable name
	 * @param scalar string value
	 * @param reuse if {@code true}, preserve value over multiple {@code executeScript} calls
	 * @throws DMLException if DMLException occurs
	 */
	public void setScalar(String varname, String scalar, boolean reuse) throws DMLException {
		setScalar(varname, new StringObject(scalar), reuse);
	}

	/**
	 * Binds a scalar object to a registered input variable. 
	 * If reuse requested, then the input is guaranteed to be 
	 * preserved over multiple <code>executeScript</code> calls. 
	 * 
	 * @param varname input variable name
	 * @param scalar scalar object
	 * @param reuse if {@code true}, preserve value over multiple {@code executeScript} calls
	 * @throws DMLException if DMLException occurs
	 */
	public void setScalar(String varname, ScalarObject scalar, boolean reuse) 
		throws DMLException
	{
		if( !_inVarnames.contains(varname) )
			throw new DMLException("Unspecified input variable: "+varname);
		
		_vars.put(varname, scalar);
	}

	/**
	 * Binds a matrix object to a registered input variable.
	 * 
	 * @param varname input variable name
	 * @param matrix two-dimensional double array matrix representation
	 * @throws DMLException if DMLException occurs
	 */
	public void setMatrix(String varname, double[][] matrix) throws DMLException {
		setMatrix(varname, matrix, false);
	}
	
	/**
	 * Binds a matrix object to a registered input variable.
	 * 
	 * @param varname input variable name
	 * @param matrix two-dimensional double array matrix representation
	 * @param reuse if {@code true}, preserve value over multiple {@code executeScript} calls
	 * @throws DMLException if DMLException occurs
	 */
	public void setMatrix(String varname, double[][] matrix, boolean reuse) throws DMLException {
		setMatrix(varname, DataConverter.convertToMatrixBlock(matrix), reuse);
	}
	
	/**
	 * Binds a matrix object to a registered input variable. 
	 * If reuse requested, then the input is guaranteed to be 
	 * preserved over multiple <code>executeScript</code> calls. 
	 * 
	 * @param varname input variable name
	 * @param matrix matrix represented as a MatrixBlock
	 * @param reuse if {@code true}, preserve value over multiple {@code executeScript} calls
	 * @throws DMLException if DMLException occurs
	 */
	public void setMatrix(String varname, MatrixBlock matrix, boolean reuse)
		throws DMLException
	{
		if( !_inVarnames.contains(varname) )
			throw new DMLException("Unspecified input variable: "+varname);
				
		int blocksize = ConfigurationManager.getBlocksize();
		
		//create new matrix object
		MatrixCharacteristics mc = new MatrixCharacteristics(matrix.getNumRows(), matrix.getNumColumns(), blocksize, blocksize);
		MetaDataFormat meta = new MetaDataFormat(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);
		MatrixObject mo = new MatrixObject(ValueType.DOUBLE, OptimizerUtils.getUniqueTempFileName(), meta);
		mo.acquireModify(matrix); 
		mo.release();
		
		//put create matrix wrapper into symbol table
		_vars.put(varname, mo);
		if( reuse ) {
			mo.enableCleanup(false); //prevent cleanup
			_inVarReuse.put(varname, mo);
		}
	}

	/**
	 * Binds a frame object to a registered input variable.
	 * 
	 * @param varname input variable name
	 * @param frame two-dimensional string array frame representation
	 * @throws DMLException if DMLException occurs
	 */
	public void setFrame(String varname, String[][] frame) throws DMLException {
		setFrame(varname, frame, false);
	}
	
	/**
	 * Binds a frame object to a registered input variable.
	 * 
	 * @param varname input variable name
	 * @param frame two-dimensional string array frame representation
	 * @param schema list representing the types of the frame columns
	 * @throws DMLException if DMLException occurs
	 */
	public void setFrame(String varname, String[][] frame, List<ValueType> schema) throws DMLException {
		setFrame(varname, frame, schema, false);
	}
	
	/**
	 * Binds a frame object to a registered input variable.
	 * 
	 * @param varname input variable name
	 * @param frame two-dimensional string array frame representation
	 * @param schema list representing the types of the frame columns
	 * @param colnames frame column names
	 * @throws DMLException if DMLException occurs
	 */
	public void setFrame(String varname, String[][] frame, List<ValueType> schema, List<String> colnames) throws DMLException {
		setFrame(varname, frame, schema, colnames, false);
	}
	
	/**
	 * Binds a frame object to a registered input variable.
	 * 
	 * @param varname input variable name
	 * @param frame two-dimensional string array frame representation
	 * @param reuse if {@code true}, preserve value over multiple {@code executeScript} calls
	 * @throws DMLException if DMLException occurs
	 */
	public void setFrame(String varname, String[][] frame, boolean reuse) throws DMLException {
		setFrame(varname, DataConverter.convertToFrameBlock(frame), reuse);
	}
	
	/**
	 * Binds a frame object to a registered input variable.
	 * 
	 * @param varname input variable name
	 * @param frame two-dimensional string array frame representation
	 * @param schema list representing the types of the frame columns
	 * @param reuse if {@code true}, preserve value over multiple {@code executeScript} calls
	 * @throws DMLException if DMLException occurs
	 */
	public void setFrame(String varname, String[][] frame, List<ValueType> schema, boolean reuse) throws DMLException {
		setFrame(varname, DataConverter.convertToFrameBlock(frame, schema.toArray(new ValueType[0])), reuse);
	}
	
	/**
	 * Binds a frame object to a registered input variable.
	 * 
	 * @param varname input variable name
	 * @param frame two-dimensional string array frame representation
	 * @param schema list representing the types of the frame columns
	 * @param colnames frame column names
	 * @param reuse if {@code true}, preserve value over multiple {@code executeScript} calls
	 * @throws DMLException if DMLException occurs
	 */
	public void setFrame(String varname, String[][] frame, List<ValueType> schema, List<String> colnames, boolean reuse) throws DMLException {
		setFrame(varname, DataConverter.convertToFrameBlock( frame, 
				schema.toArray(new ValueType[0]), colnames.toArray(new String[0])), reuse);
	}
	
	/**
	 * Binds a frame object to a registered input variable. 
	 * If reuse requested, then the input is guaranteed to be 
	 * preserved over multiple <code>executeScript</code> calls. 
	 * 
	 * @param varname input variable name
	 * @param frame frame represented as a FrameBlock
	 * @param reuse if {@code true}, preserve value over multiple {@code executeScript} calls
	 * @throws DMLException if DMLException occurs
	 */
	public void setFrame(String varname, FrameBlock frame, boolean reuse)
		throws DMLException
	{
		if( !_inVarnames.contains(varname) )
			throw new DMLException("Unspecified input variable: "+varname);
		
		//create new frame object
		MatrixCharacteristics mc = new MatrixCharacteristics(frame.getNumRows(), frame.getNumColumns(), -1, -1);
		MetaDataFormat meta = new MetaDataFormat(mc, OutputInfo.BinaryCellOutputInfo, InputInfo.BinaryCellInputInfo);
		FrameObject fo = new FrameObject(OptimizerUtils.getUniqueTempFileName(), meta);
		fo.acquireModify(frame);
		fo.release();
		
		//put create matrix wrapper into symbol table
		_vars.put(varname, fo);
		if( reuse ) {
			fo.enableCleanup(false); //prevent cleanup
			_inVarReuse.put(varname, fo);
		}
	}
	
	/**
	 * Remove all current values bound to input or output variables.
	 * 
	 */
	public void clearParameters() {
		_vars.removeAll();
	}
	
	/**
	 * Executes the prepared script over the bound inputs, creating the
	 * result variables according to bound and registered outputs.
	 * 
	 * @return ResultVariables object encapsulating output results
	 * @throws DMLException if DMLException occurs
	 */
	public ResultVariables executeScript() 
		throws DMLException
	{
		//add reused variables
		_vars.putAll(_inVarReuse);
		
		//set thread-local configurations
		ConfigurationManager.setLocalConfig(_dmlconf);
		ConfigurationManager.setLocalConfig(_cconf);
		
		//create and populate execution context
		ExecutionContext ec = ExecutionContextFactory.createContext(_vars, _prog);
		
		//core execute runtime program
		_prog.execute(ec);
		
		//cleanup unnecessary outputs
		_vars.removeAllNotIn(_outVarnames);
		
		//construct results
		ResultVariables rvars = new ResultVariables();
		for( String ovar : _outVarnames ) {
			Data tmpVar = _vars.get(ovar);
			if( tmpVar != null )
				rvars.addResult(ovar, tmpVar);
		}
		
		//clear thread-local configurations
		ConfigurationManager.clearLocalConfigs();
		
		return rvars;
	}
	
	/**
	 * Explain the DML/PyDML program and view result as a string.
	 * 
	 * @return string results of explain
	 * @throws DMLException if DMLException occurs
	 */
	public String explain() throws DMLException {
		return Explain.explain(_prog);
	}
	
	/**
	 * Enables function recompilation, selectively for the given functions. 
	 * If dynamic recompilation is globally enabled this has no additional 
	 * effect; otherwise the given functions are dynamically recompiled once
	 * on every entry but not at the granularity of individually last-level 
	 * program blocks. Use this fine-grained recompilation option for important
	 * functions in small-data scenarios where dynamic recompilation overheads 
	 * might not be amortized.  
	 * 
	 * @param fnamespace function namespace, null for default namespace
	 * @param fnames function name
	 * @throws DMLException if any compilation error occurs
	 */
	public void enableFunctionRecompile(String fnamespace, String... fnames) 
		throws DMLException 
	{
		//handle default name space
		if( fnamespace == null )
			fnamespace = DMLProgram.DEFAULT_NAMESPACE;
		
		//enable dynamic recompilation (note that this does not globally enable
		//dynamic recompilation because the program has been compiled already)
		CompilerConfig cconf = ConfigurationManager.getCompilerConfig();
		cconf.set(ConfigType.ALLOW_DYN_RECOMPILATION, true);
		ConfigurationManager.setLocalConfig(cconf);
		
		//build function call graph (to probe for recursive functions)
		FunctionCallGraph fgraph = _prog.getProgramBlocks().isEmpty() ? null :
			new FunctionCallGraph(_prog.getProgramBlocks().get(0).getStatementBlock().getDMLProg());
		
		//enable requested functions for recompile once
		for( String fname : fnames ) {
			String fkey = DMLProgram.constructFunctionKey(fnamespace, fname);
			if( fgraph != null && !fgraph.isRecursiveFunction(fkey) ) {
				FunctionProgramBlock fpb = _prog.getFunctionProgramBlock(fnamespace, fname);
				if( fpb != null )
					fpb.setRecompileOnce(true);
				else
					LOG.warn("Failed to enable function recompile for non-existing '"+fkey+"'.");		
			}
			else if( fgraph != null ) {
				LOG.warn("Failed to enable function recompile for recursive '"+fkey+"'.");
			}
		}
	}
	
	/**
	 * Creates a cloned instance of the prepared script, which
	 * allows for concurrent execution without side effects.
	 * 
	 * @param deep indicator if a deep copy needs to be created;
	 *   if false, only a shallow (i.e., by reference) copy of the 
	 *   program and read-only meta data is created. 
	 * @return an equivalent prepared script
	 */
	public PreparedScript clone(boolean deep) {
		if( deep )
			throw new NotImplementedException();
		return new PreparedScript(this);
	}
	
	@Override
	public Object clone() {
		return clone(true);
	}
}
