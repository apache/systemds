/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.test.applications;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class ApplyTransformTest extends AutomatedTestBase{
	
	protected final static String TEST_DIR = "applications/apply-transform/";
	protected final static String TEST_NAME = "apply-transform";
	protected String TEST_CLASS_DIR = TEST_DIR + ApplyTransformTest.class.getSimpleName() + "/";
	
	protected String X, missing_value_maps, binning_maps, dummy_coding_maps, normalization_maps;

	public ApplyTransformTest(String X, String missing_value_maps, String binning_maps,
		String dummy_coding_maps, String normalization_maps)
	{
		this.X = X;
		this.missing_value_maps = missing_value_maps;
		this.binning_maps = binning_maps;
		this.dummy_coding_maps = dummy_coding_maps;
		this.normalization_maps = normalization_maps;
	}

	@Parameters
	 public static Collection<Object[]> data() {
	   Object[][] data = new Object[][] { 
			   {"newX.mtx", "missing_value_map.mtx", "bindefns.mtx", "dummy_code_maps.mtx", "normalization_maps.mtx"},
			   {"newX.mtx", "missing_value_map.mtx", " ", " ", " "},
			   {"newX.mtx", "missing_value_map.mtx", " ", " ", "normalization_maps.mtx"},
			   {"newX.mtx", "missing_value_map.mtx", "bindefns.mtx", " ", "normalization_maps.mtx"},
			   {"newX_nomissing.mtx", " ", "bindefns.mtx", " ", " "},
			   {"newX_nomissing.mtx", " ", "bindefns.mtx", "dummy_code_maps.mtx", " "},
			   {"newX_nomissing.mtx", " ", " ", " ", "normalization_maps.mtx"}
		};
	   return Arrays.asList(data);
	 }

	@Override
	public void setUp() {
		addTestConfiguration(TEST_CLASS_DIR, TEST_NAME);
	}

	@Test
	public void testApplyTransform() {
		System.out.println("------------ BEGIN " + TEST_NAME + " TEST WITH {" + X + ", " + missing_value_maps
					+ ", " + binning_maps + ", " + dummy_coding_maps + ", " + normalization_maps + "} ------------");
		
		getAndLoadTestConfiguration(TEST_NAME);
		
		List<String> proArgs = new ArrayList<>();
		proArgs.add("-stats");
		proArgs.add("-ngrams");
		proArgs.add("1,2,3,4,5,6,7,8,9,10");
		proArgs.add("10");
		proArgs.add("-nvargs");
		proArgs.add("X=" + sourceDirectory + X);
		proArgs.add("missing_value_maps=" + (missing_value_maps.equals(" ") ? " " : sourceDirectory + missing_value_maps));
		proArgs.add("bin_defns=" + (binning_maps.equals(" ") ? " " : sourceDirectory + binning_maps));
		proArgs.add("dummy_code_maps=" + (dummy_coding_maps.equals(" ") ? " " : sourceDirectory + dummy_coding_maps));
		proArgs.add("normalization_maps=" + (normalization_maps.equals(" ") ? " " : sourceDirectory + normalization_maps));
		proArgs.add("transformed_X=" + output("transformed_X.mtx"));
		proArgs.add("Log=" + output("log.csv"));
		programArgs = proArgs.toArray(new String[proArgs.size()]);

		fullDMLScriptName = getScript();
		 
		runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
		 
		HashMap<CellIndex, Double> XDML= readDMLMatrixFromOutputDir("transformed_X.mtx");
		 
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
	 }
}
