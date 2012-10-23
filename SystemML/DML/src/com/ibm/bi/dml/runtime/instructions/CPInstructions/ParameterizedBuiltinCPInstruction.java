package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import java.util.HashMap;

import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.functionobjects.ParameterizedBuiltin;
import com.ibm.bi.dml.runtime.functionobjects.ValueFunction;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.GroupedAggregateInstruction;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.SimpleOperator;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


public class ParameterizedBuiltinCPInstruction extends ComputationCPInstruction {
	int arity;
	protected HashMap<String,String> params;
	
	public ParameterizedBuiltinCPInstruction(Operator op, HashMap<String,String> paramsMap, CPOperand out, String istr )
	{
		super(op, null, null, out);
		cptype = CPINSTRUCTION_TYPE.ParameterizedBuiltin;
		params = paramsMap;
		instString = istr;
	}

	public int getArity() {
		return arity;
	}
	
	protected static HashMap<String, String> constructParameterMap(String[] params) {
		// process all elements in "params" except first(opcode) and last(output)
		HashMap<String,String> paramMap = new HashMap<String,String>();
		
		// all parameters are of form <name=value>
		String[] parts;
		for ( int i=1; i <= params.length-2; i++ ) {
			parts = params[i].split(Lops.NAME_VALUE_SEPARATOR);
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
			return new ParameterizedBuiltinCPInstruction(new SimpleOperator(func), paramsMap, out, str);
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
			return new ParameterizedBuiltinCPInstruction(op, paramsMap, out, str);
		}
		else if ( opcode.equalsIgnoreCase("rmempty") ) {
			func = ParameterizedBuiltin.getParameterizedBuiltinFnObject(opcode);
			return new ParameterizedBuiltinCPInstruction(new SimpleOperator(func), paramsMap, out, str);
		}
		else {
			throw new DMLRuntimeException("Unknown opcode (" + opcode + ") for ParameterizedBuiltin Instruction.");
		}

	}
	
	@Override 
	public void processInstruction(ProgramBlock pb) throws DMLRuntimeException, DMLUnsupportedOperationException {
		
		String opcode = InstructionUtils.getOpCode(instString);
		ScalarObject sores = null;
		
		if ( opcode.equalsIgnoreCase("cdf")) {
			SimpleOperator op = (SimpleOperator) optr;
			double result =  op.fn.execute(params);
			sores = new DoubleObject(result);
			pb.setScalarOutput(output.get_name(), sores);
		} 
		else if ( opcode.equalsIgnoreCase("groupedagg") ) {
			// acquire locks
			MatrixBlock target = pb.getMatrixInput(params.get("target"));
			MatrixBlock groups = pb.getMatrixInput(params.get("groups"));
			MatrixBlock weights= null;
			if ( params.get("weights") != null )
				weights = pb.getMatrixInput(params.get("weights"));
			
			// compute the result
			MatrixBlock soresBlock = (MatrixBlock) (groups.groupedAggOperations(target, weights, new MatrixBlock(), optr));
			
			pb.setMatrixOutput(output.get_name(), soresBlock);
			// release locks
			target = groups = weights = null;
			pb.releaseMatrixInput(params.get("target"));
			pb.releaseMatrixInput(params.get("groups"));
			if ( params.get("weights") != null )
				pb.releaseMatrixInput(params.get("weights"));
			
		}
		else if ( opcode.equalsIgnoreCase("rmempty") ) {
			// acquire locks
			MatrixBlock target = pb.getMatrixInput(params.get("target"));
			
			// compute the result
			String margin = params.get("margin");
			MatrixBlock soresBlock = null;
			if( margin.equals("rows") )
				soresBlock = (MatrixBlock) target.removeEmptyRows(new MatrixBlock());
			else if( margin.equals("cols") ) 
				soresBlock = (MatrixBlock) target.removeEmptyColumns(new MatrixBlock());
			else
				throw new DMLRuntimeException("Unspupported margin identifier '"+margin+"'.");
			
			//release locks
			pb.setMatrixOutput(output.get_name(), soresBlock);
			pb.releaseMatrixInput(params.get("target"));
		}
		else {
			throw new DMLRuntimeException("Unknown opcode : " + opcode);
		}
		
	}
	

}
