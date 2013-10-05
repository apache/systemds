/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.components.hops;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.LiteralOp;
import com.ibm.bi.dml.hops.DataGenOp;
import com.ibm.bi.dml.hops.Hop.DataGenMethod;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.DataGen;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.OutputParameters.Format;
import com.ibm.bi.dml.parser.DataIdentifier;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.FormatType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDHandler;

public class RandOpTest 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
		DataGenOp ro = getRandOpInstance();
		Lop lop = ro.constructLops();
		if (!(lop instanceof DataGen))
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
			assertEquals("CP" + Lop.OPERAND_DELIMITOR + "Rand" + Lop.OPERAND_DELIMITOR + "0"
					+ Lop.OPERAND_DELIMITOR + "1"
					+ Lop.OPERAND_DELIMITOR + NUM_ROWS
					+ Lop.OPERAND_DELIMITOR + NUM_COLS
					+ Lop.OPERAND_DELIMITOR + NUM_ROWS_IN_BLOCK
					+ Lop.OPERAND_DELIMITOR + NUM_COLS_IN_BLOCK
					+ Lop.OPERAND_DELIMITOR + MIN_VALUE
					+ Lop.OPERAND_DELIMITOR + MAX_VALUE
					+ Lop.OPERAND_DELIMITOR + SPARSITY
					+ Lop.OPERAND_DELIMITOR + SEED
					+ Lop.OPERAND_DELIMITOR + PDF
					+ Lop.OPERAND_DELIMITOR + DIR, lop
					.getInstructions(0, 1));
		} catch (LopsException e) {
			fail("failed to get instructions: " + e.getMessage());
		}
	}

	private DataGenOp getRandOpInstance() {
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
		HashMap<String, Hop> inputParameters = new HashMap<String, Hop>();
		inputParameters.put("min", min);
		inputParameters.put("max", max);
		inputParameters.put("sparsity", sparsity);
		inputParameters.put("seed", seed);
		inputParameters.put("pdf", pdf);
		inputParameters.put("rows", rows);
		inputParameters.put("cols", cols);
		DataGenOp rand = new DataGenOp(DataGenMethod.RAND, id, inputParameters);
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