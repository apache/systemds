package org.apache.sysml.runtime.instructions.spark;

import java.util.HashMap;

import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.matrix.operators.Operator;

public class ParamservBuiltinSPInstruction extends ParameterizedBuiltinSPInstruction {

	ParamservBuiltinSPInstruction(Operator op, HashMap<String, String> paramsMap, CPOperand out, String opcode,
			String istr, boolean bRmEmptyBC) {
		super(op, paramsMap, out, opcode, istr, bRmEmptyBC);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		// bobo's TODO implement the spark runtime
	}
}
