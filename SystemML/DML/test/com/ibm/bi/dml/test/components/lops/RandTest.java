package com.ibm.bi.dml.test.components.lops;

import static org.junit.Assert.*;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.hops.Hops.DataGenMethod;
import com.ibm.bi.dml.lops.Data;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.lops.DataGen;
import com.ibm.bi.dml.parser.DataIdentifier;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.FormatType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.utils.LopsException;


public class RandTest {

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

        assertEquals("MR"+ com.ibm.bi.dml.lops.Lops.OPERAND_DELIMITOR + "Rand" + 
        		com.ibm.bi.dml.lops.Lops.OPERAND_DELIMITOR + "0" + 
        		com.ibm.bi.dml.lops.Lops.OPERAND_DELIMITOR + "1" + 
        		com.ibm.bi.dml.lops.Lops.OPERAND_DELIMITOR + "pREAD"+ String.valueOf(NUM_ROWS) +
        		com.ibm.bi.dml.lops.Lops.OPERAND_DELIMITOR + "pREAD"+ String.valueOf(NUM_COLS) +
        		com.ibm.bi.dml.lops.Lops.OPERAND_DELIMITOR + String.valueOf(NUM_ROWS_IN_BLOCK) +
        		com.ibm.bi.dml.lops.Lops.OPERAND_DELIMITOR + String.valueOf(NUM_COLS_IN_BLOCK) +
        		com.ibm.bi.dml.lops.Lops.OPERAND_DELIMITOR + "pREAD"+ String.valueOf(MIN_VALUE) +
        		com.ibm.bi.dml.lops.Lops.OPERAND_DELIMITOR + "pREAD"+ String.valueOf(MAX_VALUE) +
        		com.ibm.bi.dml.lops.Lops.OPERAND_DELIMITOR + "pREAD"+ String.valueOf(SPARSITY) +
        		com.ibm.bi.dml.lops.Lops.OPERAND_DELIMITOR + "pREAD"+ String.valueOf(SEED) +
        		com.ibm.bi.dml.lops.Lops.OPERAND_DELIMITOR + "pREAD"+ String.valueOf(PDF) +
        		com.ibm.bi.dml.lops.Lops.OPERAND_DELIMITOR + String.valueOf(DIR),
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
        HashMap<String, Lops> 
    	inputParametersLops = new HashMap<String, Lops>();
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
