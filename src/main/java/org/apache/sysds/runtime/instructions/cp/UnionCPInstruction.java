package org.apache.sysds.runtime.instructions.cp;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.MultiThreadedOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;

public class UnionCPInstruction extends BinaryCPInstruction {

	private UnionCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr) {
		super(CPType.Union, op, in1, in2, out, opcode, istr);
	}

	public static UnionCPInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];

		if(!opcode.equalsIgnoreCase(Opcodes.UNION_DISTINCT.toString()))
			throw new DMLRuntimeException("Invalid opcode for UNION_DISTINCT: " + opcode);

		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand out = new CPOperand(parts[parts.length - 2]);
		MultiThreadedOperator operator = new MultiThreadedOperator();
		return new UnionCPInstruction(operator, in1, in2, out, opcode, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		MatrixBlock matBlock1 = ec.getMatrixInput(input1.getName());
		MatrixBlock matBlock2 = ec.getMatrixInput(input2.getName());
		MatrixBlock out = matBlock1.unionOperations(matBlock1, matBlock2);
		ec.releaseMatrixInput(input1.getName());
		ec.releaseMatrixInput(input2.getName());
		ec.setMatrixOutput(output.getName(), out);
		System.out.println("Processing UNION CP instruction");
	}

}
