package dml.test.components.hops;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.junit.Test;

import dml.hops.RandOp;
import dml.lops.Lops;
import dml.lops.Rand;
import dml.lops.OutputParameters.Format;
import dml.parser.DataIdentifier;
import dml.parser.Expression.DataType;
import dml.parser.Expression.FormatType;
import dml.parser.Expression.ValueType;
import dml.utils.LopsException;

public class RandOpTest {
    
    private static final long NUM_ROWS = 10;
    private static final long NUM_COLS = 11;
    private static final long NUM_ROWS_IN_BLOCK = 12;
    private static final long NUM_COLS_IN_BLOCK = 13;
    private static final double MIN_VALUE = 0.0;
    private static final double MAX_VALUE = 1.0;
    private static final double SPARSITY = 0.5;
    private static final String PDF = "uniform";
    

    @Test
    public void testConstructLops() {
        RandOp ro = getRandOpInstance();
        Lops lop = ro.constructLops();
        if(!(lop instanceof Rand))
            fail("Lop is not instance of Rand LOP");
        assertEquals(NUM_ROWS, lop.getOutputParameters().getNum_rows().longValue());
        assertEquals(NUM_COLS, lop.getOutputParameters().getNum_cols().longValue());
        assertEquals(NUM_ROWS_IN_BLOCK, lop.getOutputParameters().getNum_rows_per_block().longValue());
        assertEquals(NUM_COLS_IN_BLOCK, lop.getOutputParameters().getNum_cols_per_block().longValue());
        assertTrue(lop.getOutputParameters().isBlocked_representation());
        assertEquals(Format.BINARY, lop.getOutputParameters().getFormat());
        try {
            assertEquals("Rand 0 1 min=" + MIN_VALUE + " max=" + MAX_VALUE + " sparsity=" + SPARSITY + " pdf=" + PDF,
                    lop.getInstructions(0, 1));
        } catch(LopsException e)
        {
            fail("failed to get instructions: " + e.getMessage());
        }
    }
    
    private RandOp getRandOpInstance() {
        DataIdentifier id = new DataIdentifier("A");
        id.setFormatType(FormatType.BINARY);
        id.setValueType(ValueType.DOUBLE);
        id.setDataType(DataType.MATRIX);
        id.setDimensions(NUM_ROWS, NUM_COLS);
        id.setBlockDimensions(NUM_ROWS_IN_BLOCK, NUM_COLS_IN_BLOCK);
        return new RandOp(id, MIN_VALUE, MAX_VALUE, SPARSITY, PDF);
    }

}
