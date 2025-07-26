package org.apache.sysds.runtime.instructions.ooc;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

public class UnaryOOCInstruction extends ComputationOOCInstruction {
    private UnaryOperator _uop = null;

    protected UnaryOOCInstruction(OOCType type, Operator op, CPOperand in1, CPOperand out, String opcode, String istr) {
        super(type, op, in1, out, opcode, istr);
    }

    public static UnaryOOCInstruction parseInstruction(String str) {
        String[] parts = InstructionUtils.getInstructionParts(str);
        InstructionUtils.checkNumFields(parts, 2);
        String opcode = parts[0];
        CPOperand in1 = new CPOperand(parts[1]);
        CPOperand out = new CPOperand(parts[2]);

        UnaryOperator uopcode = InstructionUtils.parseUnaryOperator(opcode);
        return new UnaryOOCInstruction(OOCType.Unary, uopcode, in1, out, str, str);
    }

    public void processInstruction( ExecutionContext ec ) {

    }
}
