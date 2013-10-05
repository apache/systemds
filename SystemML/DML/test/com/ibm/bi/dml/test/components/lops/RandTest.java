/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.components.lops;

import static org.junit.Assert.*;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.hops.Hop.DataGenMethod;
import com.ibm.bi.dml.lops.Data;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.DataGen;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.parser.DataIdentifier;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.FormatType;
import com.ibm.bi.dml.parser.Expression.ValueType;


public class RandTest 
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
    private static final String DIR = "./in/";
    
    
    @Test
    public void testGetInstructionsIntInt() throws LopsException {
        DataGen randLop = getRandInstance();

        assertEquals("MR"+ Lop.OPERAND_DELIMITOR + "Rand" + 
        		Lop.OPERAND_DELIMITOR + "0" + 
        		Lop.OPERAND_DELIMITOR + "1" + 
        		Lop.OPERAND_DELIMITOR + "pREAD"+ String.valueOf(NUM_ROWS) +
        		Lop.OPERAND_DELIMITOR + "pREAD"+ String.valueOf(NUM_COLS) +
        		Lop.OPERAND_DELIMITOR + String.valueOf(NUM_ROWS_IN_BLOCK) +
        		Lop.OPERAND_DELIMITOR + String.valueOf(NUM_COLS_IN_BLOCK) +
        		Lop.OPERAND_DELIMITOR + "pREAD"+ String.valueOf(MIN_VALUE) +
        		Lop.OPERAND_DELIMITOR + "pREAD"+ String.valueOf(MAX_VALUE) +
        		Lop.OPERAND_DELIMITOR + "pREAD"+ String.valueOf(SPARSITY) +
        		Lop.OPERAND_DELIMITOR + "pREAD"+ String.valueOf(SEED) +
        		Lop.OPERAND_DELIMITOR + "pREAD"+ String.valueOf(PDF) +
        		Lop.OPERAND_DELIMITOR + String.valueOf(DIR),
                randLop.getInstructions(0, 1));
    }
    
    private DataGen getRandInstance() throws LopsException {
        DataIdentifier id = new DataIdentifier("A");
        id.setFormatType(FormatType.BINARY);
        id.setValueType(ValueType.DOUBLE);
        id.setDataType(DataType.MATRIX);
        id.setDimensions(NUM_ROWS, NUM_COLS);
        id.setBlockDimensions(NUM_ROWS_IN_BLOCK, NUM_COLS_IN_BLOCK);
        Data min = new Data("", Data.OperationTypes.READ, String.valueOf(MIN_VALUE), "TEXT", DataType.SCALAR, ValueType.DOUBLE, false);
        Data max = new Data("", Data.OperationTypes.READ, String.valueOf(MAX_VALUE), "TEXT", DataType.SCALAR, ValueType.DOUBLE, false);
        Data sparsity = new Data("", Data.OperationTypes.READ, String.valueOf(SPARSITY), "TEXT", DataType.SCALAR, ValueType.DOUBLE, false);
        Data seed = new Data("", Data.OperationTypes.READ, String.valueOf(SEED), "TEXT", DataType.SCALAR, ValueType.DOUBLE, false);
        Data pdf = new Data("", Data.OperationTypes.READ, String.valueOf(PDF), "TEXT", DataType.SCALAR, ValueType.DOUBLE, false);
        Data rows = new Data("", Data.OperationTypes.READ, String.valueOf(NUM_ROWS), "TEXT", DataType.SCALAR, ValueType.DOUBLE, false);
        Data cols = new Data("", Data.OperationTypes.READ, String.valueOf(NUM_COLS), "TEXT", DataType.SCALAR, ValueType.DOUBLE, false);
        HashMap<String, Lop> 
    	inputParametersLops = new HashMap<String, Lop>();
        inputParametersLops.put("min", min);
        inputParametersLops.put("max", max);
        inputParametersLops.put("sparsity", sparsity);
        inputParametersLops.put("seed", seed);
        inputParametersLops.put("pdf", pdf);
        inputParametersLops.put("rows", rows);
        inputParametersLops.put("cols", cols);
        return new DataGen(DataGenMethod.RAND, id, inputParametersLops, DIR,
        		DataType.MATRIX, ValueType.DOUBLE, ExecType.MR);
    }

}
