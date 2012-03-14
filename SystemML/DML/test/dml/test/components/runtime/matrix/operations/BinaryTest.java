package dml.test.components.runtime.matrix.operations;

import static org.junit.Assert.*;

import org.junit.Test;

import dml.runtime.instructions.MRInstructions.BinaryInstruction;
import dml.utils.DMLException;

public class BinaryTest {

    @Test
    public void testParseOperation() {
       /*
        try {
            assertEquals(Binary.SupportedOperation.BINARY_ADDITION, Binary.parseOperation("b+"));
            assertEquals(Binary.SupportedOperation.BINARY_SUBSTRACTION, Binary.parseOperation("b-"));
            assertEquals(Binary.SupportedOperation.BINARY_MAXIMIZATION, Binary.parseOperation("bmax"));
            assertEquals(Binary.SupportedOperation.BINARY_MINIMIZATION, Binary.parseOperation("bmin"));
            assertEquals(Binary.SupportedOperation.BINARY_MULTIPLICATION, Binary.parseOperation("b*"));
            assertEquals(Binary.SupportedOperation.BINARY_DIVIDE, Binary.parseOperation("b/"));
        } catch(DMLUnsupportedOperationException e) {
            fail("Operation parsing failed");
        }
        try {
            Binary.parseOperation("wrong");
            fail("Wrong operation gets parsed");
        } catch(DMLUnsupportedOperationException e) { }
        */
    }

    @Test
    public void testParseInstruction() {
  /*      try {
            BinaryInstruction instType = (BinaryInstruction) BinaryInstruction.parseInstruction("b+ 0 1 2");
            //assertEquals(Binary.SupportedOperation.BINARY_ADDITION, instType.operation);
            assertEquals(0, instType.input1);
            assertEquals(1, instType.input2);
            assertEquals(2, instType.output);

            instType = (BinaryInstruction) BinaryInstruction.parseInstruction("b- 0 1 2");
            //assertEquals(Binary.SupportedOperation.BINARY_SUBSTRACTION, instType.operation);
            assertEquals(0, instType.input1);
            assertEquals(1, instType.input2);
            assertEquals(2, instType.output);

            instType = (BinaryInstruction) BinaryInstruction.parseInstruction("bmax 0 1 2");
            //assertEquals(Binary.SupportedOperation.BINARY_MAXIMIZATION, instType.operation);
            assertEquals(0, instType.input1);
            assertEquals(1, instType.input2);
            assertEquals(2, instType.output);

            instType = (BinaryInstruction) BinaryInstruction.parseInstruction("bmin 0 1 2");
            //assertEquals(Binary.SupportedOperation.BINARY_MINIMIZATION, instType.operation);
            assertEquals(0, instType.input1);
            assertEquals(1, instType.input2);
            assertEquals(2, instType.output);

            instType = (BinaryInstruction) BinaryInstruction.parseInstruction("b* 0 1 2");
            //assertEquals(Binary.SupportedOperation.BINARY_MULTIPLICATION, instType.operation);
            assertEquals(0, instType.input1);
            assertEquals(1, instType.input2);
            assertEquals(2, instType.output);

            instType = (BinaryInstruction) BinaryInstruction.parseInstruction("b/ 0 1 2");
            //assertEquals(Binary.SupportedOperation.BINARY_DIVIDE, instType.operation);
            assertEquals(0, instType.input1);
            assertEquals(1, instType.input2);
            assertEquals(2, instType.output);
        } catch (DMLException e) {
            fail("Instruction parsing failed");
        }*/
    }

}
