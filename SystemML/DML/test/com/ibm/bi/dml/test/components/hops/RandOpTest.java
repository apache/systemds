package com.ibm.bi.dml.test.components.hops;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.hops.Hops;
import com.ibm.bi.dml.hops.LiteralOp;
import com.ibm.bi.dml.hops.RandOp;
import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.lops.Rand;
import com.ibm.bi.dml.lops.OutputParameters.Format;
import com.ibm.bi.dml.parser.DataIdentifier;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.FormatType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.ConfigurationManager;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDHandler;
import com.ibm.bi.dml.utils.HopsException;
import com.ibm.bi.dml.utils.LopsException;
import com.ibm.bi.dml.utils.configuration.DMLConfig;

public class RandOpTest {

	private static final long NUM_ROWS = 10;
	private static final long NUM_COLS = 11;
	private static final long NUM_ROWS_IN_BLOCK = 12;
	private static final long NUM_COLS_IN_BLOCK = 13;
	private static final double MIN_VALUE = 0.0;
	private static final double MAX_VALUE = 1.0;
	private static final double SPARSITY = 0.5;
	private static final long SEED = 7;
	private static final String PDF = "uniform";
	private static final String DIR = "scratch_space/_p"+IDHandler.createDistributedUniqueID()+"//_t0/";
	
	@Test
	public void testConstructLops() throws HopsException, LopsException {
		setupConfiguration();
		RandOp ro = getRandOpInstance();
		Lops lop = ro.constructLops();
		if (!(lop instanceof Rand))
			fail("Lop is not instance of Rand LOP");
		assertEquals(NUM_ROWS, lop.getOutputParameters().getNum_rows()
				.longValue());
		assertEquals(NUM_COLS, lop.getOutputParameters().getNum_cols()
				.longValue());
		assertEquals(NUM_ROWS_IN_BLOCK, lop.getOutputParameters()
				.get_rows_in_block().longValue());
		assertEquals(NUM_COLS_IN_BLOCK, lop.getOutputParameters()
				.get_cols_in_block().longValue());
		assertTrue(lop.getOutputParameters().isBlocked_representation());
		assertEquals(Format.BINARY, lop.getOutputParameters().getFormat());
		try {
			assertEquals("CP" + com.ibm.bi.dml.lops.Lops.OPERAND_DELIMITOR + "Rand" + com.ibm.bi.dml.lops.Lops.OPERAND_DELIMITOR + "0"
					+ com.ibm.bi.dml.lops.Lops.OPERAND_DELIMITOR + "1"
					+ com.ibm.bi.dml.lops.Lops.OPERAND_DELIMITOR + NUM_ROWS
					+ com.ibm.bi.dml.lops.Lops.OPERAND_DELIMITOR + NUM_COLS
					+ com.ibm.bi.dml.lops.Lops.OPERAND_DELIMITOR + NUM_ROWS_IN_BLOCK
					+ com.ibm.bi.dml.lops.Lops.OPERAND_DELIMITOR + NUM_COLS_IN_BLOCK
					+ com.ibm.bi.dml.lops.Lops.OPERAND_DELIMITOR + MIN_VALUE
					+ com.ibm.bi.dml.lops.Lops.OPERAND_DELIMITOR + MAX_VALUE
					+ com.ibm.bi.dml.lops.Lops.OPERAND_DELIMITOR + SPARSITY
					+ com.ibm.bi.dml.lops.Lops.OPERAND_DELIMITOR + SEED
					+ com.ibm.bi.dml.lops.Lops.OPERAND_DELIMITOR + PDF
					+ com.ibm.bi.dml.lops.Lops.OPERAND_DELIMITOR + DIR, lop
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
		LiteralOp min = new LiteralOp(String.valueOf(MIN_VALUE), MIN_VALUE);
		LiteralOp max = new LiteralOp(String.valueOf(MAX_VALUE), MAX_VALUE);
		LiteralOp sparsity = new LiteralOp(String.valueOf(SPARSITY), SPARSITY);
		LiteralOp seed = new LiteralOp(String.valueOf(SEED), SEED);
		LiteralOp pdf = new LiteralOp(String.valueOf(PDF), PDF);
		LiteralOp rows = new LiteralOp(String.valueOf(NUM_ROWS), NUM_ROWS);
		LiteralOp cols = new LiteralOp(String.valueOf(NUM_COLS), NUM_COLS);
		HashMap<String, Hops> inputParameters = new HashMap<String, Hops>();
		inputParameters.put("min", min);
		inputParameters.put("max", max);
		inputParameters.put("sparsity", sparsity);
		inputParameters.put("seed", seed);
		inputParameters.put("pdf", pdf);
		inputParameters.put("rows", rows);
		inputParameters.put("cols", cols);
		RandOp rand = new RandOp(id, inputParameters);
		rand.set_dim1(id.getDim1());
		rand.set_dim2(id.getDim2());
		rand.setNnz(id.getNnz());
		rand.set_rows_in_block((int) id.getRowsInBlock());
		rand.set_cols_in_block((int) id.getColumnsInBlock());
		return rand;
	}

	private void setupConfiguration(){
		DMLConfig defaultConfig = null;
		try 
		{
			defaultConfig = new DMLConfig(DMLScript.DEFAULT_SYSTEMML_CONFIG_FILEPATH);
		} 
		catch (Exception e) 
		{
			e.printStackTrace();
		}
		ConfigurationManager.setConfig(defaultConfig);
	}
	
}