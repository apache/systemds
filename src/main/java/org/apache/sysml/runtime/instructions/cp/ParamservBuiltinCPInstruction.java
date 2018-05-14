package org.apache.sysml.runtime.instructions.cp;

import java.util.LinkedHashMap;

import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.matrix.operators.Operator;

public class ParamservBuiltinCPInstruction extends ParameterizedBuiltinCPInstruction {

	protected ParamservBuiltinCPInstruction(Operator op, LinkedHashMap<String, String> paramsMap, CPOperand out, String opcode, String istr) {
		super(op, paramsMap, out, opcode, istr);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		// bobo's TODO runtime development
	}
}
