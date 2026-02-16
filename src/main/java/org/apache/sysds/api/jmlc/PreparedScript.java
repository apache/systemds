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

package org.apache.sysds.api.jmlc;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.ConfigurableAPI;
import org.apache.sysds.api.DMLException;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.conf.CompilerConfig.ConfigType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.ipa.FunctionCallGraph;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysds.runtime.controlprogram.LocalVariableMap;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.instructions.cp.BooleanObject;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.cp.IntObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.cp.StringObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.meta.MetaDataFormat;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.utils.Explain;
import org.apache.sysds.utils.Statistics;

import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;

/**
 * Representation of a prepared (precompiled) DML/PyDML script.
 */
public class PreparedScript implements ConfigurableAPI
{
	private static final Log LOG = LogFactory.getLog(PreparedScript.class.getName());
	
	//input/output specification
	private final HashSet<String> _inVarnames;
	private final HashSet<String> _outVarnames;
	private final LocalVariableMap _inVarReuse;
	
	//internal state (reused)
	private final Program _prog;
	private final LocalVariableMap _vars;
	private final DMLConfig _dmlconf;
	private final CompilerConfig _cconf;
	private HashMap<String, String> _outVarLineage;
	
	private PreparedScript(PreparedScript that) {
		//shallow copy, except for a separate symbol table
		//and related meta data of reused inputs
		_prog = (Program)that._prog.clone();
		_vars = new LocalVariableMap();
		for(Entry<String, Data> e : that._vars.entrySet())
			_vars.put(e.getKey(), e.getValue());
		_vars.setRegisteredOutputs(that._outVarnames);
		_inVarnames = that._inVarnames;
		_outVarnames = that._outVarnames;
		_inVarReuse = new LocalVariableMap(that._inVarReuse);
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
	protected PreparedScript( Program prog, String[] inputs, String[] outputs, DMLConfig dmlconf, CompilerConfig cconf ) {
		_prog = prog;
		_vars = new LocalVariableMap();
		_outVarLineage = new HashMap<>();
		
		//populate input/output vars
		_inVarnames = new HashSet<>();
		Collections.addAll(_inVarnames, inputs);
		_outVarnames = new HashSet<>();
		Collections.addAll(_outVarnames, outputs);
		_inVarReuse = new LocalVariableMap();
		
		//attach registered outputs (for dynamic recompile)
		_vars.setRegisteredOutputs(_outVarnames);
		
		//keep dml and compiler configuration to be set as thread-local config
		//on execute, which allows different threads creating/executing the script
		_dmlconf = dmlconf;
		_cconf = cconf;
	}
	
	@Override
	public void resetConfig() {
		_dmlconf.set(new DMLConfig());
	}

	@Override
	public void setConfigProperty(String propertyName, String propertyValue) {
		try {
			_dmlconf.setTextValue(propertyName, propertyValue);
		} 
		catch( DMLRuntimeException e ) {
			throw new RuntimeException(e);
		}
	}
	
	/**
	 * Get the dml configuration object associated with
	 * the prepared script instance.
	 * 
	 * @return dml configuration
	 */
	public DMLConfig getDMLConfig() {
		return _dmlconf;
	}
	
	/**
	 * Get the compiler configuration object associated with
	 * the prepared script instance.
	 * 
	 * @return compiler configuration
	 */
	public CompilerConfig getCompilerConfig() {
		return _cconf;
	}
	
	/**
	 * Binds a scalar boolean to a registered input variable.
	 * 
	 * @param varname input variable name
	 * @param scalar boolean value
	 */
	public void setScalar(String varname, boolean scalar) {
		setScalar(varname, scalar, false);
	}
	
	/**
	 * Binds a scalar boolean to a registered input variable.
	 * 
	 * @param varname input variable name
	 * @param scalar boolean value
	 * @param reuse if {@code true}, preserve value over multiple {@code executeScript} calls
	 */
	public void setScalar(String varname, boolean scalar, boolean reuse) {
		setScalar(varname, new BooleanObject(scalar), reuse);
	}
	
	/**
	 * Binds a scalar long to a registered input variable.
	 * 
	 * @param varname input variable name
	 * @param scalar long value
	 */
	public void setScalar(String varname, long scalar) {
		setScalar(varname, scalar, false);
	}
	
	/**
	 * Binds a scalar long to a registered input variable.
	 * 
	 * @param varname input variable name
	 * @param scalar long value
	 * @param reuse if {@code true}, preserve value over multiple {@code executeScript} calls
	 */
	public void setScalar(String varname, long scalar, boolean reuse) {
		setScalar(varname, new IntObject(scalar), reuse);
	}
	
	/** Binds a scalar double to a registered input variable.
	 * 
	 * @param varname input variable name
	 * @param scalar double value
	 */
	public void setScalar(String varname, double scalar) {
		setScalar(varname, scalar, false);
	}
	
	/**
	 * Binds a scalar double to a registered input variable.
	 * 
	 * @param varname input variable name
	 * @param scalar double value
	 * @param reuse if {@code true}, preserve value over multiple {@code executeScript} calls
	 */
	public void setScalar(String varname, double scalar, boolean reuse) {
		setScalar(varname, new DoubleObject(scalar), reuse);
	}
	
	/**
	 * Binds a scalar string to a registered input variable.
	 * 
	 * @param varname input variable name
	 * @param scalar string value
	 */
	public void setScalar(String varname, String scalar) {
		setScalar(varname, scalar, false);
	}
	
	/**
	 * Binds a scalar string to a registered input variable.
	 * 
	 * @param varname input variable name
	 * @param scalar string value
	 * @param reuse if {@code true}, preserve value over multiple {@code executeScript} calls
	 */
	public void setScalar(String varname, String scalar, boolean reuse) {
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
	 */
	public void setScalar(String varname, ScalarObject scalar, boolean reuse) {
		if( !_inVarnames.contains(varname) )
			throw new DMLException("Unspecified input variable: "+varname);
		_vars.put(varname, scalar);
	}

	/**
	 * Binds a matrix object to a registered input variable.
	 * 
	 * @param varname input variable name
	 * @param matrix two-dimensional double array matrix representation
	 */
	public void setMatrix(String varname, double[][] matrix) {
		setMatrix(varname, matrix, false);
	}
	
	/**
	 * Binds a matrix object to a registered input variable.
	 * 
	 * @param varname input variable name
	 * @param matrix two-dimensional double array matrix representation
	 * @param reuse if {@code true}, preserve value over multiple {@code executeScript} calls
	 */
	public void setMatrix(String varname, double[][] matrix, boolean reuse) {
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
	 */
	public void setMatrix(String varname, MatrixBlock matrix, boolean reuse) {
		if( !_inVarnames.contains(varname) )
			throw new DMLException("Unspecified input variable: "+varname);
				
		int blocksize = ConfigurationManager.getBlocksize();
		
		//create new matrix object
		MatrixCharacteristics mc = new MatrixCharacteristics(matrix.getNumRows(), matrix.getNumColumns(), blocksize, blocksize);
		MetaDataFormat meta = new MetaDataFormat(mc, FileFormat.BINARY);
		MatrixObject mo = new MatrixObject(ValueType.FP64, OptimizerUtils.getUniqueTempFileName(), meta);
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
	 */
	public void setFrame(String varname, String[][] frame) {
		setFrame(varname, frame, false);
	}
	
	/**
	 * Binds a frame object to a registered input variable.
	 * 
	 * @param varname input variable name
	 * @param frame two-dimensional string array frame representation
	 * @param schema list representing the types of the frame columns
	 */
	public void setFrame(String varname, String[][] frame, List<ValueType> schema) {
		setFrame(varname, frame, schema, false);
	}
	
	/**
	 * Binds a frame object to a registered input variable.
	 * 
	 * @param varname input variable name
	 * @param frame two-dimensional string array frame representation
	 * @param schema list representing the types of the frame columns
	 * @param colnames frame column names
	 */
	public void setFrame(String varname, String[][] frame, List<ValueType> schema, List<String> colnames) {
		setFrame(varname, frame, schema, colnames, false);
	}
	
	/**
	 * Binds a frame object to a registered input variable.
	 * 
	 * @param varname input variable name
	 * @param frame two-dimensional string array frame representation
	 * @param reuse if {@code true}, preserve value over multiple {@code executeScript} calls
	 */
	public void setFrame(String varname, String[][] frame, boolean reuse) {
		setFrame(varname, DataConverter.convertToFrameBlock(frame), reuse);
	}
	
	/**
	 * Binds a frame object to a registered input variable.
	 * 
	 * @param varname input variable name
	 * @param frame two-dimensional string array frame representation
	 * @param schema list representing the types of the frame columns
	 * @param reuse if {@code true}, preserve value over multiple {@code executeScript} calls
	 */
	public void setFrame(String varname, String[][] frame, List<ValueType> schema, boolean reuse) {
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
	 */
	public void setFrame(String varname, String[][] frame, List<ValueType> schema, List<String> colnames, boolean reuse) {
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
	 */
	public void setFrame(String varname, FrameBlock frame, boolean reuse) {
		if( !_inVarnames.contains(varname) )
			throw new DMLException("Unspecified input variable: "+varname);
		
		//create new frame object
		MatrixCharacteristics mc = new MatrixCharacteristics(frame.getNumRows(), frame.getNumColumns(), -1, -1);
		MetaDataFormat meta = new MetaDataFormat(mc, FileFormat.BINARY);
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
	 * Remove all references to pinned variables from this script.
	 * Note: this *does not* remove the underlying data. It merely
	 * removes a reference to it from this prepared script. This is
	 * useful if you want to maintain an independent cache of weights
	 * and allow the JVM to garbage collect under memory pressure.
	 */
	public void clearPinnedData() { _inVarReuse.removeAll(); }

	/**
	 * Executes the prepared script over the bound inputs, creating the
	 * result variables according to bound and registered outputs.
	 * 
	 * @return ResultVariables object encapsulating output results
	 */
	public ResultVariables executeScript() {
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
			if( tmpVar != null ) {
				rvars.addResult(ovar, tmpVar);
				if (ec.getLineage() != null)
					_outVarLineage.put(ovar, Explain.explain(ec.getLineage().get(ovar)));
			}
		}
		
		//clear thread-local configurations
		ConfigurationManager.clearLocalConfigs();

		return rvars;
	}
	
	/**
	 * Explain the DML/PyDML program and view result as a string.
	 * 
	 * @return string results of explain
	 */
	public String explain() {
		return Explain.explain(_prog);
	}

	/**
	 * Capture lineage of the DML/PyDML program and view result as a string.
	 *
	 * @param var the output variable name on which lineage trace is sought
	 *
	 * @return string results of lineage trace
	 *
	 */
	public String getLineageTrace(String var) {
		return _outVarLineage.get(var);
	}
	
	/**
	 * Return a string containing runtime statistics. Note: these are not thread local
	 * and will reflect execution in all threads
	 * @return string containing statistics
	 */
	public String statistics() { return Statistics.display(); }
	
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
	 */
	public void enableFunctionRecompile(String fnamespace, String... fnames) {
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
