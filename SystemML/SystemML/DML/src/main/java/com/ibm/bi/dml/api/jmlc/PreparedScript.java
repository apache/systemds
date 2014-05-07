/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.api.jmlc;

import java.util.HashSet;

import com.ibm.bi.dml.api.DMLException;
import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.LocalVariableMap;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.BooleanObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.DoubleObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.IntObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.ScalarObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.StringObject;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.util.DataConverter;

/**
 * JMLC (Java Machine Learning Connector) API:
 * 
 * NOTE: Currently fused API and implementation in order to reduce complexity. 
 */
public class PreparedScript 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//input/output specification
	private HashSet<String> _inVarnames = null;
	private HashSet<String> _outVarnames = null;
	
	//internal state (reused)
	private Program _prog = null;
	private LocalVariableMap _vars = null; 
	
	/**
	 * Meant to be invoked only from Connection
	 */
	protected PreparedScript( Program prog, String[] inputs, String[] outputs )
	{
		_prog = prog;
		_vars = new LocalVariableMap();
		
		//populate input/output vars
		_inVarnames = new HashSet<String>();
		for( String var : inputs )
			_inVarnames.add( var );
		_outVarnames = new HashSet<String>();
		for( String var : outputs )
			_outVarnames.add( var );
	}
	
	/**
	 * 
	 * @param varname
	 * @param scalar
	 * @throws DMLException 
	 */
	public void setScalar(String varname, ScalarObject scalar) 
		throws DMLException
	{
		if( !_inVarnames.contains(varname) )
			throw new DMLException("Unspecified input variable: "+varname);
		
		_vars.put(varname, scalar);
	}
	
	/**
	 * 
	 * @param varname
	 * @param scalar
	 * @throws DMLException 
	 */
	public void setScalar(String varname, boolean scalar) 
		throws DMLException
	{
		if( !_inVarnames.contains(varname) )
			throw new DMLException("Unspecified input variable: "+varname);
		
		BooleanObject bo = new BooleanObject(varname, scalar);
		_vars.put(varname, bo);
	}
	
	/**
	 * 
	 * @param varname
	 * @param scalar
	 * @throws DMLException 
	 */
	public void setScalar(String varname, int scalar) 
		throws DMLException
	{
		if( !_inVarnames.contains(varname) )
			throw new DMLException("Unspecified input variable: "+varname);
		
		IntObject io = new IntObject(varname, scalar);
		_vars.put(varname, io);
	}
	
	/**
	 * 
	 * @param varname
	 * @param scalar
	 * @throws DMLException 
	 */
	public void setScalar(String varname, double scalar) 
		throws DMLException
	{
		if( !_inVarnames.contains(varname) )
			throw new DMLException("Unspecified input variable: "+varname);
		
		DoubleObject doo = new DoubleObject(varname, scalar);
		_vars.put(varname, doo);	
	}
	
	/**
	 * 
	 * @param varname
	 * @param scalar
	 * @throws DMLException 
	 */
	public void setScalar(String varname, String scalar) 
		throws DMLException
	{
		if( !_inVarnames.contains(varname) )
			throw new DMLException("Unspecified input variable: "+varname);
		
		StringObject so = new StringObject(varname, scalar);
		_vars.put(varname, so);
	}
	
	/**
	 * 
	 * @param varname
	 * @param matrix
	 * @throws DMLException
	 */
	public void setMatrix(String varname, MatrixBlock matrix)
		throws DMLException
	{
		if( !_inVarnames.contains(varname) )
			throw new DMLException("Unspecified input variable: "+varname);
		
		
		DMLConfig conf = ConfigurationManager.getConfig();
		String scratch_space = conf.getTextValue(DMLConfig.SCRATCH_SPACE);
		int blocksize = conf.getIntValue(DMLConfig.DEFAULT_BLOCK_SIZE);
		
		//create new matrix object
		MatrixCharacteristics mc = new MatrixCharacteristics(matrix.getNumRows(), matrix.getNumColumns(), blocksize, blocksize);
		MatrixFormatMetaData meta = new MatrixFormatMetaData(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);
		MatrixObject mo = new MatrixObject(ValueType.DOUBLE, scratch_space+"/"+varname, meta);
		mo.acquireModify(matrix); 
		mo.release();
		
		//put create matrix wrapper into symbol table
		_vars.put(varname, mo);
	}
	
	/**
	 * 
	 * @param varname
	 * @param matrix
	 * @throws DMLException
	 */
	public void setMatrix(String varname, double[][] matrix)
		throws DMLException
	{
		if( !_inVarnames.contains(varname) )
			throw new DMLException("Unspecified input variable: "+varname);
		
		MatrixBlock mb = DataConverter.convertToMatrixBlock(matrix);
		setMatrix(varname, mb);
	}
	
	
	/**
	 * 
	 */
	public void clearParameters()
	{
		_vars.removeAll();
	}
	
	/**
	 * 
	 * @return
	 * @throws DMLException 
	 */
	public ResultVariables executeScript() 
		throws DMLException
	{
		//create and populate execution context
		ExecutionContext ec = new ExecutionContext(_prog);	
		ec.setVariables(_vars);
		
		//core execute runtime program	
		_prog.execute( ec );  
		
		//construct results
		ResultVariables rvars = new ResultVariables();
		for( String ovar : _outVarnames )
			if( _vars.keySet().contains(ovar) )
				rvars.addResult(ovar, _vars.get(ovar));
			
		return rvars;
	}
}
