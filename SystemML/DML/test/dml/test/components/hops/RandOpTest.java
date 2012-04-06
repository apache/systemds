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
import dml.utils.HopsException;
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
	public void testConstructLops() throws HopsException {
		RandOp ro = getRandOpInstance();
		Lops lop = ro.constructLops();
		if (!(lop instanceof Rand))
			fail("Lop is not instance of Rand LOP");
		assertEquals(NUM_ROWS, lop.getOutputParameters().getNum_rows()
				.longValue());
		assertEquals(NUM_COLS, lop.getOutputParameters().getNum_cols()
				.longValue());
		assertEquals(NUM_ROWS_IN_BLOCK, lop.getOutputParameters()
				.getNum_rows_per_block().longValue());
		assertEquals(NUM_COLS_IN_BLOCK, lop.getOutputParameters()
				.getNum_cols_per_block().longValue());
		assertTrue(lop.getOutputParameters().isBlocked_representation());
		assertEquals(Format.BINARY, lop.getOutputParameters().getFormat());
		try {
			assertEquals("MR" + dml.lops.Lops.OPERAND_DELIMITOR + "Rand" + dml.lops.Lops.OPERAND_DELIMITOR + "0"
					+ dml.lops.Lops.OPERAND_DELIMITOR + "1"
					+ dml.lops.Lops.OPERAND_DELIMITOR + "rows=" + NUM_ROWS
					+ dml.lops.Lops.OPERAND_DELIMITOR + "cols=" + NUM_COLS
					+ dml.lops.Lops.OPERAND_DELIMITOR + "min=" + MIN_VALUE
					+ dml.lops.Lops.OPERAND_DELIMITOR + "max=" + MAX_VALUE
					+ dml.lops.Lops.OPERAND_DELIMITOR + "sparsity=" + SPARSITY
					+ dml.lops.Lops.OPERAND_DELIMITOR + "pdf=" + PDF, lop
					.getInstructions(0, 1));
		} catch (LopsException e) {
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
		RandOp rand = new RandOp(id, MIN_VALUE, MAX_VALUE, SPARSITY, PDF);
		rand.set_dim1(id.getDim1());
		rand.set_dim2(id.getDim2());
		rand.set_rows_per_block((int) id.getRowsInBlock());
		rand.set_cols_per_block((int) id.getColumnsInBlock());
		return rand;
	}

}
