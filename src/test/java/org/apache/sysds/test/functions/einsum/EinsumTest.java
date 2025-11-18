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

package org.apache.sysds.test.functions.einsum;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.After;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RunWith(Parameterized.class)
public class EinsumTest extends AutomatedTestBase
{
    final private static List<Config> TEST_CONFIGS = List.of(
            new Config("ij,jk->ik", List.of(shape(5, 6), shape(6, 5))), // mm
            new Config("ji,jk->ik", List.of(shape(6, 5), shape(6, 10))),
            new Config("ji,kj->ik", List.of(shape(6, 5), shape(10, 6))),
            new Config("ij,kj->ik", List.of(shape(5, 6), shape(10, 6))),
			new Config("ij,jk->ki", List.of(shape(5, 6), shape(6, 5))), // mm t
			new Config("ji,jk->ki", List.of(shape(6, 5), shape(6, 10))),
			new Config("ji,kj->ki", List.of(shape(6, 5), shape(10, 6))),
			new Config("ij,kj->ki", List.of(shape(5, 6), shape(10, 6))),
			new Config("ij,kp,pj->ki", List.of(shape(5,6), shape(5,4), shape(4, 6))), // reordering
			new Config("ab,bc,cd,de->ae", List.of(shape(5, 6), shape(6, 5),shape(5, 6), shape(6, 5))), // mm chain

            new Config("ji,jk->i", List.of(shape(6, 5), shape(6, 4))),
            new Config("ij,jk->i", List.of(shape(5, 6), shape(6, 4))),
            new Config("ji,jk->k", List.of(shape(6, 5), shape(6, 4))),
            new Config("ij,jk->k", List.of(shape(5, 6), shape(6, 4))),
            new Config("ji,jk->j", List.of(shape(6, 5), shape(6, 4))),

            new Config("ji,ji->ji", List.of(shape(60, 5), shape(60, 5))), // elemwise mult
            new Config("ji,ij->ji", List.of(shape(60, 5), shape(5, 60))), // elemwise mult

            new Config("ij,i->ij",   List.of(shape(10, 5), shape(10))), // col mult
            new Config("ji,i->ij",   List.of(shape(5, 10), shape(10))), // row mult
            new Config("ij,i->i",   List.of(shape(10, 5), shape(10))),
            new Config("ij,i->j",   List.of(shape(10, 5), shape(10))),

            new Config("i,i->",     List.of(shape(5), shape(5))), // dot
            new Config("i,j->",     List.of(shape(5), shape(80))), // sum
            new Config("i,j->ij",     List.of(shape(5), shape(80))), // outer vect mult
            new Config("i,j->ji",     List.of(shape(5), shape(80))), // outer vect mult

            new Config("ij->",     List.of(shape(10, 5))), // sum
			new Config("i->",     List.of(shape(10))), // sum
            new Config("ij->i",     List.of(shape(10, 5))), // sum(1)
            new Config("ij->j",     List.of(shape(10, 5))), // sum(0)
			new Config("ij->ji",     List.of(shape(10, 5))), // T
			new Config("ij->ij",     List.of(shape(10, 5))),
			new Config("i->i",     List.of(shape(10))),
			new Config("ii->i",     List.of(shape(10, 10))), // Diag
			new Config("ii,i->i",     List.of(shape(10, 10),shape(10))), // Diag*vec

            new Config("ab,cd->ba",     List.of(shape( 6, 10), shape(6, 5))), // sum cd to scalar and multiply ab

            new Config("fx,fg,fz,xg,zx,zg->g", // each idx 3 times (cell tpl fallback)
                    List.of(shape(5, 6), shape(5, 3), shape(5, 10), shape(6, 3), shape(10, 6), shape(10, 3))),

            // test fused:
			new Config("ij,ij,ji->ij",   List.of(shape(10, 5), shape(10, 5), shape(5, 10))),
			new Config("ij,ij,ji,i,j->ij",   List.of(shape(10, 5), shape(10, 5),shape(5, 10),shape(10),shape(5))),
			new Config("ij,ij,ji,i->ij",   List.of(shape(10, 5), shape(10, 5),shape(5, 10),shape(10))),
			new Config("ij,ij,ji,j->ij",   List.of(shape(10, 5), shape(10, 5),shape(5, 10),shape(5))),

			new Config("ij,ij,ji->i",   List.of(shape(10, 5), shape(10, 5), shape(5, 10))),
			new Config("ij,ij,ji,i,j->i",   List.of(shape(10, 5), shape(10, 5),shape(5, 10),shape(10),shape(5))),
			new Config("ij,ij,ji,i->i",   List.of(shape(10, 5), shape(10, 5),shape(5, 10),shape(10))),
			new Config("ij,ij,ji,j->i",   List.of(shape(10, 5), shape(10, 5),shape(5, 10),shape(5))),

			new Config("ij,ij,ji->j",   List.of(shape(10, 5), shape(10, 5), shape(5, 10))),
			new Config("ij,ij,ji,i,j->j",   List.of(shape(10, 5), shape(10, 5),shape(5, 10),shape(10),shape(5))),
			new Config("ij,ij,ji,i->j",   List.of(shape(10, 5), shape(10, 5),shape(5, 10),shape(10))),
			new Config("ij,ij,ji,j->j",   List.of(shape(10, 5), shape(10, 5),shape(5, 10),shape(5))),

			new Config("ij,ij,ji->",   List.of(shape(10, 5), shape(10, 5), shape(5, 10))),
			new Config("ij,ij,ji,i,j->",   List.of(shape(10, 5), shape(10, 5),shape(5, 10),shape(10),shape(5))),
			new Config("ij,ij,ji,i->",   List.of(shape(10, 5), shape(10, 5),shape(5, 10),shape(10))),
			new Config("ij,ij,ji,j->",   List.of(shape(10, 5), shape(10, 5),shape(5, 10),shape(5))),

			new Config("ij,ij,ij,i,j,iz->z",  List.of(shape(5, 5), shape(5, 5), shape(5, 5),shape(5),shape(5),shape(5, 6))),
			new Config("ij,ij,ij,i,j,iz,z->z",  List.of(shape(5, 5), shape(5, 5), shape(5, 5),shape(5),shape(5),shape(5, 6),shape(6))),

			new Config("ij,i,j,iz->z",  List.of(shape(10, 5),shape(10),shape(5),shape(10, 51))),
			new Config("ij,i,j,iz->z",  List.of(shape(20, 10),shape(20),shape(10),shape(20, 10))),

            new Config("ij,ij,ji,j,i, ab,ba,ab,a,b->jb",   Map.of('i',10, 'j',5, 'a', 11, 'b', 6)),
            new Config("ij,ij,ji,j,i,iz->jz",   Map.of('i',600, 'j',10,'z', 6)),  // //skinny right with outer mm
            new Config("ij,ij,ji,j,i,iz->jz",   Map.of('i',10, 'j',10,'z', 10)),  // // no skinny right
            new Config("ij,ij,ji,j,i,iz->zj",   Map.of('i',60, 'j',10,'z', 6)),  // //skinny right with outer mm
			new Config("ij,ij,ij,jk->ik",   List.of(shape(10, 5), shape(10, 5),shape(10, 5),shape(5, 10))),
			new Config("ab,ab,a,ag,gz->bz",   List.of(shape(10, 5), shape(10, 5),shape(10),shape(10,200),shape(200,7)))
    );
	private final int id;
	private final String einsumStr;
	//private final List<int[]> shapes;
	private final File dmlFile;
	private final File rFile;
	private final boolean outputScalar;

	public EinsumTest(String einsumStr, List<int[]> shapes, File dmlFile, File rFile, boolean outputScalar, int id){
		this.id = id;
		this.einsumStr = einsumStr;
		//this.shapes = shapes;
		this.dmlFile = dmlFile;
		this.rFile = rFile;
		this.outputScalar = outputScalar;
	}

	@Parameterized.Parameters(name = "{index}: einsum={0}")
	public static Collection<Object[]> data() throws IOException {
		List<Object[]> parameters = new ArrayList<>();

		int counter = 1;

		for (Config config : TEST_CONFIGS) {
			//List<File> files = new ArrayList<>();
			String fullDMLScriptName = "SystemDS_einsum_test" + counter;

			File dmlFile = File.createTempFile(fullDMLScriptName, ".dml");
			dmlFile.deleteOnExit();

			boolean outputScalar = config.einsumStr.trim().endsWith("->");

			StringBuilder sb = createDmlFile(config, outputScalar);

			Files.writeString(dmlFile.toPath(), sb.toString());

			File rFile = File.createTempFile(fullDMLScriptName, ".R");
			rFile.deleteOnExit();

			sb = createRFile(config, outputScalar);

			Files.writeString(rFile.toPath(), sb.toString());

			parameters.add(new Object[]{config.einsumStr, config.shapes, dmlFile, rFile, outputScalar, counter});

			counter++;
		}

		return parameters;
	}

	private static StringBuilder createDmlFile(Config config, boolean outputScalar) {
		StringBuilder sb = new StringBuilder();

		for (int i = 0; i < config.shapes.size(); i++) {
			int[] dims = config.shapes.get(i);

			double factor = config.factors != null ? config.factors.get(i) : 0.0001;
			sb.append("A");
			sb.append(i);

			if (dims.length == 1) { // A1 = seq(1,100) * 0.0001
				sb.append(" = seq(1,");
				sb.append(dims[0]);
				sb.append(") * ");
				sb.append(factor);
			} else { // A0 = matrix(seq(1,5000), 100, 5) * 0.0001
				sb.append(" = matrix(seq(1, ");
				sb.append(dims[0]*dims[1]);
				sb.append("), ");
				sb.append(dims[0]);
				sb.append(", ");
				sb.append(dims[1]);

				sb.append(") * ");
				sb.append(factor);
			}
			sb.append("\n");
		}
		sb.append("\n");
		sb.append("R = einsum(\"");
		sb.append(config.einsumStr);
		sb.append("\", ");

		for (int i = 0; i < config.shapes.size() - 1; i++) {
			sb.append("A");
			sb.append(i);
			sb.append(", ");
		}
		sb.append("A");
		sb.append(config.shapes.size() - 1);
		sb.append(")");

		sb.append("\n\n");
		sb.append("write(R, $1)\n");
		return sb;
	}

	private static StringBuilder createRFile(Config config, boolean outputScalar) {
		StringBuilder sb = new StringBuilder();
		sb.append("args<-commandArgs(TRUE)\n");
		sb.append("options(digits=22)\n");
		sb.append("library(\"Matrix\")\n");
		sb.append("library(\"matrixStats\")\n");
		sb.append("library(\"einsum\")\n\n");


		for (int i = 0; i < config.shapes.size(); i++) {
			int[] dims = config.shapes.get(i);

			double factor = config.factors != null ? config.factors.get(i) : 0.0001;
			sb.append("A");
			sb.append(i);

			if (dims.length == 1) { // A1 = seq(1,100) * 0.0001
				sb.append(" = seq(1,");
				sb.append(dims[0]);
				sb.append(") * ");
				sb.append(factor);
			} else { // A0 = matrix(seq(1,5000), 100, 5, byrow=TRUE) * 0.0001
				sb.append(" = matrix(seq(1, ");
				sb.append(dims[0]*dims[1]);
				sb.append("), ");
				sb.append(dims[0]);
				sb.append(", ");
				sb.append(dims[1]);

				sb.append(", byrow=TRUE) * ");
				sb.append(factor);
			}
			sb.append("\n");
		}
		sb.append("\n");

		sb.append("R = einsum(\"");
		sb.append(config.einsumStr);
		sb.append("\", ");

		for (int i = 0; i < config.shapes.size()-1; i++) {
			sb.append("A");
			sb.append(i);
			sb.append(", ");
		}
		sb.append("A");
		sb.append(config.shapes.size()-1);
		sb.append(")");

		sb.append("\n\n");
		if(outputScalar){
			sb.append("write(R, paste(args[2], \"S\", sep=\"\"))\n");
		}else{
			sb.append("writeMM(as(R, \"CsparseMatrix\"), paste(args[2], \"S\", sep=\"\"))\n");
		}
		return sb;
	}

	@Test
	public void testEinsumWithFiles() {
		System.out.println("Testing einsum: " + this.einsumStr);
		testCodegenIntegration(TEST_NAME_EINSUM+this.id);
	}
	@After
	public void cleanUp() {
		if (this.dmlFile.exists()) {
			boolean deleted = this.dmlFile.delete();
			if (!deleted) {
				System.err.println("Failed to delete temp file: " + this.dmlFile.getAbsolutePath());
			}
		}
		if (this.rFile.exists()) {
			boolean deleted = this.rFile.delete();
			if (!deleted) {
				System.err.println("Failed to delete temp file: " + this.rFile.getAbsolutePath());
			}
		}
	}

	private static class Config {
        public List<Double> factors;
		String einsumStr;
		List<int[]> shapes;

        Config(String einsum, List<int[]> shapes) {
            this(einsum,shapes,null);
        }
        Config(String einsum, Map<Character, Integer> charToSize){
            this(einsum, charToSize, null);
        }

        Config(String einsum, Map<Character, Integer> charToSize, List<Double> factors) {
            this.einsumStr = einsum;
            String leftPart = einsum.split("->")[0];
            List<int[]> shapes = new ArrayList<>();
            for(String op : Arrays.stream(leftPart.split(",")).map(x->x.trim()).toList()){
                if (op.length() == 1) {
                    shapes.add(new int[]{charToSize.get(op.charAt(0))});
                }else{
                    shapes.add(new int[]{charToSize.get(op.charAt(0)),charToSize.get(op.charAt(1))});
                }

            }
            this.shapes = shapes;
            this.factors = factors;
        }
		Config(String einsum, List<int[]> shapes, List<Double> factors) {
			this.einsumStr = einsum;
			this.shapes = shapes;
            this.factors = factors;
		}
	}

	private static int[] shape(int... dims) {
		return dims;
	}
	private static final Log LOG = LogFactory.getLog(EinsumTest.class.getName());

	private static final String TEST_NAME_EINSUM = "einsum";
	private static final String TEST_DIR = "functions/einsum/";
	private static final String TEST_CLASS_DIR = TEST_DIR + EinsumTest.class.getSimpleName() + "/";
	private final static String TEST_CONF = "SystemDS-config-codegen.xml";
	private final static File   TEST_CONF_FILE = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF);

	private static double eps = Math.pow(10, -10);

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		for(int i = 1; i<= TEST_CONFIGS.size(); i++)
			addTestConfiguration( TEST_NAME_EINSUM+i, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_EINSUM+i, new String[] { String.valueOf(i) }) );
	}

	private void testCodegenIntegration( String testname)
	{
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		ExecMode platformOld = setExecMode(ExecType.CP);

		String testnameDml = this.dmlFile.getAbsolutePath();
		String testnameR = this.rFile.getAbsolutePath();
		try
		{
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);

			fullDMLScriptName = testnameDml;
			programArgs = new String[]{"-stats", "-explain", "-args", output("S") };
			fullRScriptName = testnameR;
			rCmd = getRCmd(inputDir(), expectedDir());

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = false;

			runTest(true, false, null, -1);
//			if(true) throw new RuntimeException("aa");
			runRScript(true);

			if(outputScalar){
				HashMap<CellIndex, Double> dmlfile = readDMLScalarFromOutputDir("S");
				HashMap<CellIndex, Double> rfile = readRScalarFromExpectedDir("S");
				TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			}else {
				//compare matrices
				HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("S");
				HashMap<CellIndex, Double> rfile = readRMatrixFromExpectedDir("S");
				TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			}
		}
		finally {
			resetExecMode(platformOld);
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
			OptimizerUtils.ALLOW_AUTO_VECTORIZATION = true;
			OptimizerUtils.ALLOW_OPERATOR_FUSION = true;
		}
	}


	/**
	 * Override default configuration with custom test configuration to ensure
	 * scratch space and local temporary directory locations are also updated.
	 */
	@Override
	protected File getConfigTemplateFile() {
		// Instrumentation in this test's output log to show custom configuration file used for template.
		LOG.debug("This test case overrides default configuration with " + TEST_CONF_FILE.getPath());
		return TEST_CONF_FILE;
	}
}
