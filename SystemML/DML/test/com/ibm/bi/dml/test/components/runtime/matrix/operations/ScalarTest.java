package com.ibm.bi.dml.test.components.runtime.matrix.operations;


import org.junit.Test;


public class ScalarTest {

    @Test
    public void testParseOperation() {
        /*
    	try {
            assertEquals(Scalar.SupportedOperation.SCALAR_ADDITION, Scalar.parseOperation("s+"));
            assertEquals(Scalar.SupportedOperation.SCALAR_SUBSTRACTION, Scalar.parseOperation("s-"));
            assertEquals(Scalar.SupportedOperation.SCALAR_SUBSTRACTION_RIGHT, Scalar.parseOperation("s-r"));
            assertEquals(Scalar.SupportedOperation.SCALAR_MAXIMIZATION, Scalar.parseOperation("smax"));
            assertEquals(Scalar.SupportedOperation.SCALAR_MINIMIZATION, Scalar.parseOperation("smin"));
            assertEquals(Scalar.SupportedOperation.SCALAR_MULTIPLICATION, Scalar.parseOperation("s*"));
            assertEquals(Scalar.SupportedOperation.SCALAR_DIVIDE, Scalar.parseOperation("s/"));
            assertEquals(Scalar.SupportedOperation.SCALAR_OVER, Scalar.parseOperation("so"));
            assertEquals(Scalar.SupportedOperation.SCALAR_LOG, Scalar.parseOperation("sl"));
            assertEquals(Scalar.SupportedOperation.SCALAR_POWER, Scalar.parseOperation("s^"));
        } catch(DMLUnsupportedOperationException e) {
            fail("Operation parsing failed");
        }
        try {
            Scalar.parseOperation("wrong");
            fail("Wrong operation gets parsed");
        } catch(DMLUnsupportedOperationException e) { }
        */
    }

    @Test
    public void testParseInstruction() {
    /*    try {
            ScalarInstruction instType = (ScalarInstruction) ScalarInstruction.parseInstruction("s+ 0 1.0 1");
            //assertEquals(Scalar.SupportedOperation.SCALAR_ADDITION, instType.operation);
            assertEquals(0, instType.input);
            //assertEquals(1.0, instType.constant, 0);
            assertEquals(1, instType.output);
            
            instType = (ScalarInstruction) ScalarInstruction.parseInstruction("s- 0 1.0 1");
           // assertEquals(Scalar.SupportedOperation.SCALAR_SUBSTRACTION, instType.operation);
            assertEquals(0, instType.input);
            //assertEquals(1.0, instType.constant, 0);
            assertEquals(1, instType.output);
            
            instType = (ScalarInstruction) ScalarInstruction.parseInstruction("s-r 0 1.0 1");
            //assertEquals(Scalar.SupportedOperation.SCALAR_SUBSTRACTION_RIGHT, instType.operation);
            assertEquals(0, instType.input);
            //assertEquals(1.0, instType.constant, 0);
            assertEquals(1, instType.output);
            
            instType = (ScalarInstruction) ScalarInstruction.parseInstruction("smax 0 1.0 1");
            //assertEquals(Scalar.SupportedOperation.SCALAR_MAXIMIZATION, instType.operation);
            assertEquals(0, instType.input);
            //assertEquals(1.0, instType.constant, 0);
            assertEquals(1, instType.output);
            
            instType = (ScalarInstruction) ScalarInstruction.parseInstruction("smin 0 1.0 1");
            //assertEquals(Scalar.SupportedOperation.SCALAR_MINIMIZATION, instType.operation);
            assertEquals(0, instType.input);
            //assertEquals(1.0, instType.constant, 0);
            assertEquals(1, instType.output);
            
            instType = (ScalarInstruction) ScalarInstruction.parseInstruction("s* 0 1.0 1");
            //assertEquals(Scalar.SupportedOperation.SCALAR_MULTIPLICATION, instType.operation);
            assertEquals(0, instType.input);
            //assertEquals(1.0, instType.constant, 0);
            assertEquals(1, instType.output);
            
            instType = (ScalarInstruction) ScalarInstruction.parseInstruction("s/ 0 1.0 1");
            //assertEquals(Scalar.SupportedOperation.SCALAR_DIVIDE, instType.operation);
            assertEquals(0, instType.input);
            //assertEquals(1.0, instType.constant, 0);
            assertEquals(1, instType.output);
            
            instType = (ScalarInstruction) ScalarInstruction.parseInstruction("so 0 1.0 1");
            //assertEquals(Scalar.SupportedOperation.SCALAR_OVER, instType.operation);
            assertEquals(0, instType.input);
            //assertEquals(1.0, instType.constant, 0);
            assertEquals(1, instType.output);
            
            instType = (ScalarInstruction) ScalarInstruction.parseInstruction("sl 0 1.0 1");
            //assertEquals(Scalar.SupportedOperation.SCALAR_LOG, instType.operation);
            assertEquals(0, instType.input);
            //assertEquals(1.0, instType.constant, 0);
            assertEquals(1, instType.output);
            
            instType = (ScalarInstruction) ScalarInstruction.parseInstruction("s^ 0 1.0 1");
            //assertEquals(Scalar.SupportedOperation.SCALAR_POWER, instType.operation);
            assertEquals(0, instType.input);
            //assertEquals(1.0, instType.constant, 0);
            assertEquals(1, instType.output);
        } catch (DMLException e) {
            fail("Instruction parsing failed");
        }*/
    }

}
