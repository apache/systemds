/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.cp;

import java.util.HashMap;

import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.parser.Statement;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.functionobjects.ParameterizedBuiltin;
import com.ibm.bi.dml.runtime.functionobjects.ValueFunction;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.mr.GroupedAggregateInstruction;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.SimpleOperator;


public class ParameterizedBuiltinCPInstruction extends ComputationCPInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private int arity;
	protected HashMap<String,String> params;
	
	public ParameterizedBuiltinCPInstruction(Operator op, HashMap<String,String> paramsMap, CPOperand out, String opcode, String istr )
	{
		super(op, null, null, out, opcode, istr);
		_cptype = CPINSTRUCTION_TYPE.ParameterizedBuiltin;
		params = paramsMap;
	}

	public int getArity() {
		return arity;
	}
	
	public static HashMap<String, String> constructParameterMap(String[] params) {
		// process all elements in "params" except first(opcode) and last(output)
		HashMap<String,String> paramMap = new HashMap<String,String>();
		
		// all parameters are of form <name=value>
		String[] parts;
		for ( int i=1; i <= params.length-2; i++ ) {
			parts = params[i].split(Lop.NAME_VALUE_SEPARATOR);
			paramMap.put(parts[0], parts[1]);
		}
		
		return paramMap;
	}
	
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		// first part is always the opcode
		String opcode = parts[0];
		// last part is always the output
		CPOperand out = new CPOperand( parts[parts.length-1] ); 

		// process remaining parts and build a hash map
		HashMap<String,String> paramsMap = constructParameterMap(parts);

		// determine the appropriate value function
		ValueFunction func = null;
		if ( opcode.equalsIgnoreCase("cdf") ) {
			if ( paramsMap.get("dist") == null ) 
				throw new DMLRuntimeException("Probability distribution must to be specified to compute cumulative probability. (e.g., q = cumulativeProbability(1.5, dist=\"chisq\", df=20))");
			func = ParameterizedBuiltin.getParameterizedBuiltinFnObject(opcode, paramsMap.get("dist") );
			// Determine appropriate Function Object based on opcode
			return new ParameterizedBuiltinCPInstruction(new SimpleOperator(func), paramsMap, out, opcode, str);
		}
		else if ( opcode.equalsIgnoreCase("groupedagg")) {
			// check for mandatory arguments
			String fnStr = paramsMap.get("fn");
			if ( fnStr == null ) 
				throw new DMLRuntimeException("Function parameter is missing in groupedAggregate.");
			if ( fnStr.equalsIgnoreCase("centralmoment") ) {
				if ( paramsMap.get("order") == null )
					throw new DMLRuntimeException("Mandatory \"order\" must be specified when fn=\"centralmoment\" in groupedAggregate.");
			}
			
			Operator op = GroupedAggregateInstruction.parseGroupedAggOperator(fnStr, paramsMap.get("order"));
			return new ParameterizedBuiltinCPInstruction(op, paramsMap, out, opcode, str);
		}
		else if(   opcode.equalsIgnoreCase("rmempty") 
				|| opcode.equalsIgnoreCase("replace") ) 
		{
			func = ParameterizedBuiltin.getParameterizedBuiltinFnObject(opcode);
			return new ParameterizedBuiltinCPInstruction(new SimpleOperator(func), paramsMap, out, opcode, str);
		}
		else {
			throw new DMLRuntimeException("Unknown opcode (" + opcode + ") for ParameterizedBuiltin Instruction.");
		}

	}

	@Override 
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		
		String opcode = getOpcode();
		ScalarObject sores = null;
		
		if ( opcode.equalsIgnoreCase("cdf")) {
			SimpleOperator op = (SimpleOperator) _optr;
			double result =  op.fn.execute(params);
			sores = new DoubleObject(result);
			ec.setScalarOutput(output.getName(), sores);
		} 
		else if ( opcode.equalsIgnoreCase("groupedagg") ) {
			// acquire locks
			MatrixBlock target = ec.getMatrixInput(params.get(Statement.GAGG_TARGET));
			MatrixBlock groups = ec.getMatrixInput(params.get(Statement.GAGG_GROUPS));
			MatrixBlock weights= null;
			if ( params.get(Statement.GAGG_WEIGHTS) != null )
				weights = ec.getMatrixInput(params.get(Statement.GAGG_WEIGHTS));
			
			int ngroups = -1;
			if ( params.get(Statement.GAGG_NUM_GROUPS) != null) {
				ngroups = (int) Double.parseDouble(params.get(Statement.GAGG_NUM_GROUPS));
			}
			
			// compute the result
			MatrixBlock soresBlock = (MatrixBlock) (groups.groupedAggOperations(target, weights, new MatrixBlock(), ngroups, _optr));
			
			ec.setMatrixOutput(output.getName(), soresBlock);
			// release locks
			target = groups = weights = null;
			ec.releaseMatrixInput(params.get(Statement.GAGG_TARGET));
			ec.releaseMatrixInput(params.get(Statement.GAGG_GROUPS));
			if ( params.get(Statement.GAGG_WEIGHTS) != null )
				ec.releaseMatrixInput(params.get(Statement.GAGG_WEIGHTS));
			
		}
		else if ( opcode.equalsIgnoreCase("rmempty") ) {
			// acquire locks
			MatrixBlock target = ec.getMatrixInput(params.get("target"));
			
			// compute the result
			String margin = params.get("margin");
			MatrixBlock soresBlock = null;
			if( margin.equals("rows") )
				soresBlock = (MatrixBlock) target.removeEmptyOperations(new MatrixBlock(), true);
			else if( margin.equals("cols") ) 
				soresBlock = (MatrixBlock) target.removeEmptyOperations(new MatrixBlock(), false);
			else
				throw new DMLRuntimeException("Unspupported margin identifier '"+margin+"'.");
			
			//release locks
			ec.setMatrixOutput(output.getName(), soresBlock);
			ec.releaseMatrixInput(params.get("target"));
		}
		else if ( opcode.equalsIgnoreCase("replace") ) {
			// acquire locks
			MatrixBlock target = ec.getMatrixInput(params.get("target"));
			
			// compute the result
			double pattern = Double.parseDouble( params.get("pattern") );
			double replacement = Double.parseDouble( params.get("replacement") );
			MatrixBlock ret = (MatrixBlock) target.replaceOperations(new MatrixBlock(), pattern, replacement);
			
			//release locks
			ec.setMatrixOutput(output.getName(), ret);
			ec.releaseMatrixInput(params.get("target"));
		}
		else {
			throw new DMLRuntimeException("Unknown opcode : " + opcode);
		}
		
	}
	

}
