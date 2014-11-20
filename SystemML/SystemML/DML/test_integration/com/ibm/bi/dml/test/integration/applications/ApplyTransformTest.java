/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.applications;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;

@RunWith(value = Parameterized.class)
public class ApplyTransformTest extends AutomatedTestBase{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_DIR = "applications/apply-transform/";
	private final static String TEST_APPLY_TRANSFORM = "apply-transform";
	
	private String X, missing_value_maps, binning_maps, dummy_coding_maps, normalization_maps;
    
	public ApplyTransformTest(String X,
							  String missing_value_maps, 
							  String binning_maps, 
							  String dummy_coding_maps,
							  String normalization_maps) {
		this.X = X;
		this.missing_value_maps = missing_value_maps;
		this.binning_maps = binning_maps;
		this.dummy_coding_maps = dummy_coding_maps;
		this.normalization_maps = normalization_maps;
	}
    
	@Parameters
	 public static Collection<Object[]> data() {
	   Object[][] data = new Object[][] { 
			   {"newX.mtx", "missing_value_map.mtx", "bindefns.mtx", "dummy_code_maps.mtx", "normalization_maps.mtx"}
			   ,{"newX.mtx", "missing_value_map.mtx", " ", " ", " "},
			   {"newX.mtx", "missing_value_map.mtx", " ", " ", "normalization_maps.mtx"},
			   {"newX.mtx", "missing_value_map.mtx", "bindefns.mtx", " ", "normalization_maps.mtx"}
			   ,{"newX_nomissing.mtx", " ", "bindefns.mtx", " ", " "}
			   ,{"newX_nomissing.mtx", " ", "bindefns.mtx", "dummy_code_maps.mtx", " "}
			   ,{"newX_nomissing.mtx", " ", " ", " ", "normalization_maps.mtx"}
			   };
	   
	   return Arrays.asList(data);
	 }

	 @Override
		public void setUp() {
			setUpBase();
	    	addTestConfiguration(TEST_APPLY_TRANSFORM, new TestConfiguration(TEST_DIR, "apply-transform",
	                new String[] {"transformed_X.mtx"}));
		}

	 @Test
	    public void testApplyTransform() {
		 String APPLY_TRANSFORM_HOME = SCRIPT_DIR + TEST_DIR;
		 
		 TestConfiguration config = getTestConfiguration(TEST_APPLY_TRANSFORM);
		 
		 /* This is for running the junit test by constructing the arguments directly */
		 fullDMLScriptName = APPLY_TRANSFORM_HOME + TEST_APPLY_TRANSFORM + ".dml";
		 programArgs = new String[]{"-stats", "-nvargs", 
				 					"X=" + APPLY_TRANSFORM_HOME + X,
		               				"missing_value_maps=" + (missing_value_maps.equals(" ") ? " " : APPLY_TRANSFORM_HOME + missing_value_maps),
		               				"bin_defns=" + (binning_maps.equals(" ") ? " " : APPLY_TRANSFORM_HOME + binning_maps),
		               				"dummy_code_maps=" + (dummy_coding_maps.equals(" ") ? " " : APPLY_TRANSFORM_HOME + dummy_coding_maps),
		               				"normalization_maps=" + (normalization_maps.equals(" ") ? " " : APPLY_TRANSFORM_HOME + normalization_maps),
		               				"transformed_X=" + APPLY_TRANSFORM_HOME + OUTPUT_DIR + "transformed_X.mtx",
		               				"Log=" + APPLY_TRANSFORM_HOME + OUTPUT_DIR + "log.csv"};
		 
		 loadTestConfiguration(config);
		 runTest(true, false, null, -1);
		 
		 HashMap<CellIndex, Double> XDML= readDMLMatrixFromHDFS("transformed_X.mtx");
		 
		 Iterator<Map.Entry<CellIndex, Double>> iter = XDML.entrySet().iterator();
		 while(iter.hasNext()){
			 Map.Entry<CellIndex, Double> elt = iter.next();
			 int row = elt.getKey().row;
			 int col = elt.getKey().column;
			 double val = elt.getValue();
			 
			 System.out.println("[" + row + "," + col + "]->" + val);
		 }
		 
		 boolean success = true;
		 
		 if(missing_value_maps != " " && normalization_maps != " "){
			 CellIndex cell;
			 if(dummy_coding_maps != " ") cell = new CellIndex(3,3);
			 else cell = new CellIndex(3,2);
			 
			 if(XDML.containsKey(cell)){
				 double val = XDML.get(cell).doubleValue();
				 success = success && (Math.abs(val) < 0.0000001);
			 }
		 }else if(missing_value_maps != " "){
			 CellIndex cell;
			 if(dummy_coding_maps != " ") cell = new CellIndex(3,3);
			 else cell = new CellIndex(3,2);
			 
			 if(XDML.containsKey(cell)){
				 double val = XDML.get(cell).doubleValue();
				 success = success && (Math.abs(-0.2/3 - val) < 0.0000001);
			 }else success = false;
		 }else if(normalization_maps != " "){
			 CellIndex cell;
			 if(dummy_coding_maps != " ") cell = new CellIndex(3,3);
			 else cell = new CellIndex(3,2);
			 
			 if(XDML.containsKey(cell)){
				 double val = XDML.get(cell).doubleValue();
				 success = success && (Math.abs(0.2/3 - val) < 0.0000001);
			 }else success = false;
		 }else{
			 CellIndex cell;
			 if(dummy_coding_maps != " ") cell = new CellIndex(3,3);
			 else cell = new CellIndex(3,2);
			 
			 if(XDML.containsKey(cell)){
				 double val = XDML.get(cell).doubleValue();
				 success = success && (Math.abs(val) < 0.0000001);
			 }
		 }
	 
		 if(binning_maps != " "){
			 CellIndex cell1, cell2, cell3, cell4;
			 if(dummy_coding_maps != " "){
				 cell1 = new CellIndex(1,1);
				 cell2 = new CellIndex(2,1);
				 cell3 = new CellIndex(3,2);
				 cell4 = new CellIndex(4,2);
			 }else{
				 cell1 = new CellIndex(1,1);
				 cell2 = new CellIndex(2,1);
				 cell3 = new CellIndex(3,1);
				 cell4 = new CellIndex(4,1);
			 }
		 
			 if(!XDML.containsKey(cell1)) success = false;
			 else success = success && (XDML.get(cell1).doubleValue() == 1);
		 
			 if(!XDML.containsKey(cell2)) success = false;
			 else success = success && (XDML.get(cell2).doubleValue() == 1);
			 
			 if(!XDML.containsKey(cell3)) success = false;
			 else success = success && (dummy_coding_maps != " ") ? (XDML.get(cell3).doubleValue() == 1) : (XDML.get(cell3).doubleValue() == 2);
			 
			 if(!XDML.containsKey(cell4)) success = false;
			 else success = success && (dummy_coding_maps != " ") ? (XDML.get(cell4).doubleValue() == 1) : (XDML.get(cell4).doubleValue() == 2);
		 }
		 
		 System.out.println("SUCCESS: " + success);
	 }
}
