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

package org.apache.sysds.test.component.compress.workload;

import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import org.apache.sysds.api.DMLOptions;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.DMLTranslator;
import org.apache.sysds.parser.ParserFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.cost.CostEstimatorBuilder;
import org.apache.sysds.runtime.compress.cost.InstructionTypeCounter;
import org.apache.sysds.runtime.compress.workload.WTreeRoot;
import org.apache.sysds.runtime.compress.workload.WorkloadAnalyzer;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class WorkloadTest {

	private static final String basePath = "src/test/scripts/component/compress/workload/";
	private static final String testFile = "src/test/resources/component/compress/1-1.csv";
	private static final String yFile = "src/test/resources/component/compress/1-1_y.csv";

	@Parameterized.Parameter(0)
	public int scans;
	@Parameterized.Parameter(1)
	public int decompressions;
	@Parameterized.Parameter(2)
	public int overlappingDecompressions;
	@Parameterized.Parameter(3)
	public int leftMultiplications;
	@Parameterized.Parameter(4)
	public int rightMultiplications;
	@Parameterized.Parameter(5)
	public int compressedMultiplications;
	@Parameterized.Parameter(6)
	public int dictionaryOps;
	@Parameterized.Parameter(7)
	public int indexing;
	@Parameterized.Parameter(8)
	public boolean shouldCompress;
	@Parameterized.Parameter(9)
	public boolean withRewrites;
	@Parameterized.Parameter(10)
	public String scriptName;
	@Parameterized.Parameter(11)
	public Map<String, String> args;

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		Map<String, String> args = new HashMap<>();
		args.put("$1", testFile);

		// Simple tests no loops verifying basic behavior
		tests.add(new Object[] {0, 0, 0, 0, 0, 0, 1, 0, false, false, "sum.dml", args});
		tests.add(new Object[] {0, 0, 0, 0, 0, 0, 1, 0, false, false, "mean.dml", args});
		tests.add(new Object[] {0, 0, 0, 0, 0, 0, 1, 1, false, false, "plus.dml", args});
		tests.add(new Object[] {0, 0, 0, 0, 0, 0, 2, 0, false, false, "sliceCols.dml", args});
		tests.add(new Object[] {0, 0, 0, 0, 0, 0, 2, 0, false, false, "sliceIndex.dml", args});
		tests.add(new Object[] {0, 0, 0, 1, 0, 0, 0, 0, false, false, "leftMult.dml", args});
		tests.add(new Object[] {0, 0, 0, 0, 1, 0, 1, 0, false, false, "rightMult.dml", args});
		tests.add(new Object[] {0, 0, 0, 1, 0, 0, 0, 0, false, false, "TLeftMult.dml", args});

		tests.add(new Object[] {0, 0, 1, 0, 1, 0, 0, 0, false, false, "TRightMult.dml", args});

		// Loops:
		tests.add(new Object[] {0, 0, 0, 11, 0, 0, 0, 0, true, false, "loop/leftMult.dml", args});
		tests.add(new Object[] {0, 0, 0, 101, 0, 0, 0, 0, true, false, "loop/leftMultStaticLoop.dml", args});
		tests.add(new Object[] {0, 0, 0, 10, 0, 0, 0, 0, true, false, "loop/leftMultWhile.dml", args});

		// functions:

		// Builtins:
		// nr 11:
		tests.add(new Object[] {0, 0, 0, 0, 0, 0, 6, 0, true, false, "functions/scale.dml", args});
		tests.add(new Object[] {0, 0, 0, 0, 0, 0, 5, 0, true, true, "functions/scale.dml", args});
		tests.add(new Object[] {0, 0, 0, 0, 0, 0, 6, 0, true, false, "functions/scale_continued.dml", args});
		tests.add(new Object[] {0, 0, 0, 0, 0, 0, 6, 0, true, true, "functions/scale_continued.dml", args});
		tests.add(new Object[] {0, 0, 0, 0, 0, 0, 2, 0, false, true, "functions/scale_onlySide.dml", args});
		tests.add(new Object[] {0, 0, 0, 0, 0, 0, 6, 0, true, false, "functions/scale_onlySide.dml", args});

//		TODO these tests are failing
//		tests.add(new Object[] {0, 0, 0, 0, 1, 1, 8, 0, true, false, "functions/pca.dml", args});
//		tests.add(new Object[] {0, 0, 0, 0, 1, 1, 5, 0, true, true, "functions/pca.dml", args});

		args = new HashMap<>();
		args.put("$1", testFile);
		args.put("$2", "FALSE");
		args.put("$3", "0");

		// no recompile
		tests.add(new Object[] {0, 1, 1, 1, 1, 1, 5, 0, true, false, "functions/lmDS.dml", args});
		// with recompile
		tests.add(new Object[] {0, 0, 0, 1, 0, 1, 0, 0, true, true, "functions/lmDS.dml", args});
		tests.add(new Object[] {0, 0, 0, 1, 10, 10, 1, 0, true, true, "functions/lmCG.dml", args});

		args = new HashMap<>();
		args.put("$1", testFile);
		args.put("$2", "TRUE");
		args.put("$3", "0");
		tests.add(new Object[] {0, 1, 1, 1, 1, 1, 0, 0, true, true, "functions/lmDS.dml", args});
		tests.add(new Object[] {0, 0, 1, 1, 11, 10, 1, 0, true, true, "functions/lmCG.dml", args});

		// args = new HashMap<>();
		// args.put("$1", testFile);
		// args.put("$2", "TRUE");
		// args.put("$3", "1");
		// tests.add(new Object[] {0, 2, 1, 1, 1, 1, 1, 0, true, true, "functions/lmDS.dml", args});
		// tests.add(new Object[] {0, 1, 1, 1, 11, 10, 2, 0, true, true, "functions/lmCG.dml", args});

		// args = new HashMap<>();
		// args.put("$1", testFile);
		// args.put("$2", "TRUE");
		// args.put("$3", "2");
		// tests.add(new Object[] {0, 2, 1, 1, 1, 1, 3, 0, true, true, "functions/lmDS.dml", args});
		// tests.add(new Object[] {0, 1, 1, 1, 11, 10, 4, 0, true, true, "functions/lmCG.dml", args});

		args = new HashMap<>();
		args.put("$1", testFile);
		args.put("$2", "FALSE");
		tests.add(new Object[] {0, 0, 10, 11, 10, 0, 1, 0, true, true, "functions/l2svm.dml", args});

		args = new HashMap<>();
		args.put("$1", yFile);
		args.put("$2", "FALSE");
		tests.add(new Object[] {0, 1, 0, 1, 0, 0, 10, 0, true, true, "functions/l2svm_Y.dml", args});

		args = new HashMap<>();
		args.put("$1", testFile);
		args.put("$2", "100");
		args.put("$3", "16");
		tests.add(new Object[] {0, 0, 100, 0, 100, 0, 100, 0, true, true, "mmrbem+.dml", args});

		return tests;
	}

	@Test
	public void runWorkloadAnalysisTest() {
		try {

			DMLProgram prog = parse(scriptName);

			DMLTranslator dmlt = new DMLTranslator(prog);
			dmlt.liveVariableAnalysis(prog);
			dmlt.validateParseTree(prog);
			dmlt.constructHops(prog);
			if(withRewrites)
				dmlt.rewriteHopsDAG(prog);

			WTreeRoot wtr = getWorkloadTree(prog);
			CostEstimatorBuilder ceb = new CostEstimatorBuilder(wtr);
			InstructionTypeCounter itc = ceb.getCounter();

			verify(wtr, itc, ceb, scriptName, args);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail();
		}
	}

	private void verify(WTreeRoot wtr, InstructionTypeCounter itc, CostEstimatorBuilder ceb, String name,
		Map<String, String> args) {

		String errorString = wtr + "\n" + itc + " \n " + name + "  -- " + args + "\n";
		Assert.assertEquals(errorString + "scans:", scans, itc.getScans());
		Assert.assertEquals(errorString + "decompressions", decompressions, itc.getDecompressions());
		Assert.assertEquals(errorString + "overlappingDecompressions", overlappingDecompressions,
			itc.getOverlappingDecompressions());
		Assert.assertEquals(errorString + "leftMultiplications", leftMultiplications, itc.getLeftMultipications());
		Assert.assertEquals(errorString + "rightMultiplications", rightMultiplications, itc.getRightMultiplications());
		Assert.assertEquals(errorString + "compressedMultiplications", compressedMultiplications,
			itc.getCompressedMultiplications());
		Assert.assertEquals(errorString + "dictionaryOps", dictionaryOps, itc.getDictionaryOps());
		Assert.assertEquals(errorString + "lookup", indexing, itc.getIndexing());
		Assert.assertEquals(errorString + "Should Compresss", shouldCompress, ceb.shouldTryToCompress());
	}

	private static WTreeRoot getWorkloadTree(DMLProgram prog) {
		Map<Long, WTreeRoot> c = WorkloadAnalyzer.getAllCandidateWorkloads(prog);
		Assert.assertEquals(c.size(), 1);
		for(long k : c.keySet())
			return c.get(k);
		throw new DMLRuntimeException("There was no Workload");
	}

	private DMLProgram parse(String name) {
		try {
			boolean isFile = true;
			String filePath = basePath + name;
			String dmlScript = DMLScript.readDMLScript(isFile, filePath);
			return ParserFactory.createParser().parse(DMLOptions.defaultOptions.filePath, dmlScript, args);
		}
		catch(Exception e) {
			throw new DMLRuntimeException("Error in parsing", e);
		}
	}
}
