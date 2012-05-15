package dml.test.components.lops;

import org.junit.Test;

import dml.lops.Rand;
import dml.parser.DataIdentifier;
import dml.parser.Expression.DataType;
import dml.parser.Expression.FormatType;
import dml.parser.Expression.ValueType;

public class RandTest {

    private static final long NUM_ROWS = 10;
    private static final long NUM_COLS = 11;
    private static final long NUM_ROWS_IN_BLOCK = 12;
    private static final long NUM_COLS_IN_BLOCK = 13;
    private static final double MIN_VALUE = 0.0;
    private static final double MAX_VALUE = 1.0;
    private static final double SPARSITY = 0.5;
    private static final String PDF = "uniform";
    private static final String DIR = "./in/";
    
    
    @Test
    public void testGetInstructionsIntInt() {
        Rand randLop = getRandInstance();
        //assertEquals("Rand 0 1 min=" + MIN_VALUE + " max=" + MAX_VALUE + " sparsity=" + SPARSITY + " pdf=" + PDF + " dir=" + DIR ,
        //        randLop.getInstructions(0, 1));
    }
    
    private Rand getRandInstance() {
        DataIdentifier id = new DataIdentifier("A");
        id.setFormatType(FormatType.BINARY);
        id.setValueType(ValueType.DOUBLE);
        id.setDataType(DataType.MATRIX);
        id.setDimensions(NUM_ROWS, NUM_COLS);
        id.setBlockDimensions(NUM_ROWS_IN_BLOCK, NUM_COLS_IN_BLOCK);
        return new Rand(id, MIN_VALUE, MAX_VALUE, SPARSITY, PDF, DIR,
        		DataType.MATRIX, ValueType.DOUBLE);
    }

}
