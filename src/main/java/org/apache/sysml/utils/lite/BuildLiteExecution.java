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
package org.apache.sysml.utils.lite;

import java.io.File;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.io.FileUtils;
import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.sysml.api.jmlc.Connection;
import org.apache.sysml.api.jmlc.PreparedScript;
import org.apache.sysml.api.jmlc.ResultVariables;
import org.apache.sysml.conf.CompilerConfig.ConfigType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.util.DataConverter;

/**
 * Execute code and conditionally build a lightweight jar file that can support
 * the execution of that code.
 *
 */
public class BuildLiteExecution 
{
	private static Logger log = Logger.getLogger(BuildLiteExecution.class);
	private static final String ROOT = "functions/jmlc/temp/";
	private static String _rootPrefix = null;
	
	public static void setRootPrefix(String prefix) {
		_rootPrefix = prefix;
	}
	
	public static String getRoot() {
		return (_rootPrefix != null) ? _rootPrefix + ROOT : ROOT;
	}
	
	public static void main(String[] args) throws Exception {

		BasicConfigurator.configure();
		Logger.getRootLogger().setLevel(Level.INFO);
		log.setLevel(Level.DEBUG);

		jmlcHelloWorld();
		jmlcScoringExample();
		jmlcUnivariateStatistics();
		jmlcWriteMatrix();
		jmlcReadMatrix();
		jmlcBasics();
		jmlcL2SVM();
		jmlcLinReg();
		jmlcALS();
		jmlcKmeans();
		jmlcTests();

		BuildLite.createLiteJar(true);

	}

	public static void jmlcHelloWorld() throws Exception {
		Connection conn = getConfiguredConnection();
		PreparedScript script = conn.prepareScript("print('hello world');", new String[] {}, new String[] {}, false);
		script.executeScript();
		conn.close();
	}

	public static void jmlcScoringExample() throws Exception {
		String scriptString =
		"X = read(\"./tmp/X\", rows=-1, cols=-1);\n" +
		"W = read(\"./tmp/W\", rows=-1, cols=-1);\n" +
		"\n" +
		"numRows = nrow(X);\n" +
		"numCols = ncol(X);\n" +
		"b = W[numCols+1,]\n" +
		"scores = X %*% W[1:numCols,] + b;\n" +
		"predicted_y = rowIndexMax(scores);\n" +
		"\n" +
		"print('pred:' + toString(predicted_y))\n" +
		"\n" +
		"write(predicted_y, \"./tmp\", format=\"text\");\n";

		File file = new File(getRoot()+"scoring-example.dml");
		System.out.println(file.toString());
		FileUtils.writeStringToFile(file, scriptString);

		Connection conn = getConfiguredConnection();
		String dml = conn.readScript(getRoot()+"scoring-example.dml");
		PreparedScript script = conn.prepareScript(dml,
			new String[] { "W", "X" }, new String[] { "predicted_y" }, false);

		double[][] mtx = matrix(4, 3, new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
		double[][] result = null;

		script.setMatrix("W", mtx);
		script.setMatrix("X", randomMatrix(3, 3, -1, 1, 0.7));
		result = script.executeScript().getMatrix("predicted_y");
		log.debug(displayMatrix(result));

		script.setMatrix("W", mtx);
		script.setMatrix("X", randomMatrix(3, 3, -1, 1, 0.7));
		result = script.executeScript().getMatrix("predicted_y");
		log.debug(displayMatrix(result));

		script.setMatrix("W", mtx);
		script.setMatrix("X", randomMatrix(3, 3, -1, 1, 0.7));
		result = script.executeScript().getMatrix("predicted_y");
		log.debug(displayMatrix(result));

		conn.close();
	}

	public static void jmlcUnivariateStatistics() throws Exception {

		Connection conn = getConfiguredConnection();

		String dml = conn.readScript("scripts/algorithms/Univar-Stats.dml");
		Map<String, String> m = new HashMap<>();
		m.put("$CONSOLE_OUTPUT", "TRUE");

		PreparedScript script = conn.prepareScript(dml,
			m, new String[] { "A", "K" }, new String[] { "baseStats" }, false);

		double[][] data = new double[100][4];
		for (int i = 0; i < 100; i++) {
			int one = ThreadLocalRandom.current().nextInt(0, 101);
			int two = ThreadLocalRandom.current().nextInt(0, 101);
			int three = ThreadLocalRandom.current().nextInt(0, 101);
			int four = ThreadLocalRandom.current().nextInt(1, 3);
			double[] row = new double[] { one, two, three, four };
			data[i] = row;
		}
		log.debug(displayMatrix(data));

		double[][] types = matrix(1, 4, new double[] { 1, 1, 1, 2 });

		script.setMatrix("A", data);
		script.setMatrix("K", types);
		double[][] baseStats = script.executeScript().getMatrix("baseStats");
		log.debug(displayMatrix(baseStats));

		conn.close();
	}

	public static void jmlcWriteMatrix() throws Exception {
		Connection conn = getConfiguredConnection();
		PreparedScript script = conn.prepareScript(
				"x=matrix('1 2 3 4',rows=2,cols=2);write(x,'"+getRoot()+"x.csv',format='csv');", new String[] {},
				new String[] {}, false);
		script.executeScript();

		String scriptString =
		"m = matrix('1 2 3 0 0 0 7 8 9 0 0 0', rows=4, cols=3)\n" +
		"write(m, '"+getRoot()+"m.txt', format='text')\n" +
		"write(m, '"+getRoot()+"m.mm', format='mm')\n" +
		"write(m, '"+getRoot()+"m.csv', format='csv')\n" +
		"write(m, '"+getRoot()+"m.binary', format='binary')\n";

		script = conn.prepareScript(scriptString, new String[] {}, new String[] {}, false);
		script.executeScript();

		conn.close();
	}

	public static void jmlcReadMatrix() throws Exception {
		Connection conn = getConfiguredConnection();
		PreparedScript script = conn.prepareScript("x=read('"+getRoot()+"x.csv',format='csv');y=x*2;print(toString(y));",
				new String[] {}, new String[] {}, false);
		script.executeScript();
		
		String scriptString = "m = read('"+getRoot()+"m.csv',format='csv')\n" +
		"print(toString(m))\n" +
		"print('min:' + min(m))\n" +
		"print('max:' + max(m))\n" +
		"print('sum:' + sum(m))\n" +
		"mRowSums = rowSums(m)\n" +
		"for (i in 1:nrow(mRowSums)) {\n" +
		"    print('row ' + i + ' sum:' + as.scalar(mRowSums[i,1]))\n" +
		"}\n" +
		"mColSums = colSums(m)\n" +
		"for (i in 1:ncol(mColSums)) {\n" +
		"    print('col ' + i + ' sum:' + as.scalar(mColSums[1,i]))\n" +
		"}\n";

		script = conn.prepareScript(scriptString, new String[] {}, new String[] {}, false);
		script.executeScript();

		conn.close();
	}

	public static void jmlcBasics() throws Exception {

		String dml =
		"A = matrix(\"1 2 3 4 5 6\", rows=3, cols=2)\n"+
		"print(toString(A))\n"+
		"B = A + 4\n"+
		"B = t(B)\n"+
		"print(toString(B))\n"+
		"C = A %*% B\n"+
		"print(toString(C))\n"+
		"D = matrix(5, rows=nrow(C), cols=ncol(C))\n"+
		"D = (C - D) / 2\n"+
		"print(toString(D))\n"+
		"\n"+
		"A = matrix(\"1 2 3 4 5 6 7 8 9\", rows=3, cols=3)\n"+
		"print(toString(A))\n"+
		"B = A[3,3]\n"+
		"print(toString(B))\n"+
		"C = A[2,]\n"+
		"print(toString(C))\n"+
		"D = A[,3]\n"+
		"print(toString(D))\n"+
		"E = A[2:3,1:2]\n"+
		"print(toString(E))\n"+
		"\n"+
		"i = 1\n"+
		"while (i <= 3) {\n"+
		" if (i == 1) {\n"+
		" print(\'hello\')\n"+
		" } else if (i == 2) {\n"+
		" print(\'world\')\n"+
		" } else {\n"+
		" print(\'!!!\')\n"+
		" }\n"+
		" i = i + 1\n"+
		"}\n"+
		"\n"+
		"A = matrix(\"1 2 3 4 5 6\", rows=3, cols=2)\n"+
		"\n"+
		"for (i in 1:nrow(A)) {\n"+
		" print(\"for A[\" + i + \",1]:\" + as.scalar(A[i,1]))\n"+
		"}\n"+
		"\n"+
		"parfor(i in 1:nrow(A)) {\n"+
		" print(\"parfor A[\" + i + \",1]:\" + as.scalar(A[i,1]))\n"+
		"}\n"+
		"\n"+
		"doSomething = function(matrix[double] mat) return (matrix[double] ret) {\n"+
		" additionalCol = matrix(1, rows=nrow(mat), cols=1) # 1x3 matrix with 1 values\n"+
		" ret = cbind(mat, additionalCol) # concatenate column to matrix\n"+
		" ret = cbind(ret, seq(0, 2, 1)) # concatenate column (0,1,2) to matrix\n"+
		" ret = cbind(ret, rowMaxs(ret)) # concatenate column of max row values to matrix\n"+
		" ret = cbind(ret, rowSums(ret)) # concatenate column of row sums to matrix\n"+
		"}\n"+
		"\n"+
		"A = rand(rows=3, cols=2, min=0, max=2) # random 3x2 matrix with values 0 to 2\n"+
		"B = doSomething(A)\n"+
		"write(A, \""+getRoot()+"A.csv\", format=\"csv\")\n"+
		"write(B, \""+getRoot()+"B.csv\", format=\"csv\")\n";

		Connection conn = getConfiguredConnection();
		PreparedScript script = conn.prepareScript(dml, new String[] {}, new String[] {}, false);
		script.executeScript();
		conn.close();
	}

	public static void jmlcL2SVM() throws Exception {

		Connection conn = getConfiguredConnection();

		String dml = conn.readScript("scripts/algorithms/l2-svm.dml");
		PreparedScript l2svm = conn.prepareScript(dml, new String[] { "X", "Y", "fmt", "Log" },
				new String[] { "w", "debug_str" }, false);

		double[][] trainData = new double[150][3];
		for (int i = 0; i < 150; i++) {
			int one = ThreadLocalRandom.current().nextInt(0, 101);
			int two = ThreadLocalRandom.current().nextInt(0, 101);
			int three = ThreadLocalRandom.current().nextInt(0, 101);
			double[] row = new double[] { one, two, three };
			trainData[i] = row;
		}

		l2svm.setMatrix("X", trainData);
		log.debug(displayMatrix(trainData));

		double[][] trainLabels = new double[150][1];
		for (int i = 0; i < 150; i++) {
			int one = ThreadLocalRandom.current().nextInt(1, 3);
			double[] row = new double[] { one };
			trainLabels[i] = row;
		}
		l2svm.setMatrix("Y", trainLabels);
		log.debug(displayMatrix(trainLabels));

		l2svm.setScalar("fmt", "csv");

		l2svm.setScalar("Log", "temp/l2-svm-log.csv");

		ResultVariables l2svmResults = l2svm.executeScript();
		double[][] model = l2svmResults.getMatrix("w");
		log.debug("MODEL:");
		log.debug(displayMatrix(model));
		String debugString = l2svmResults.getString("debug_str");
		log.debug("DEBUG STRING:");
		log.debug(debugString);

		String s = conn.readScript("scripts/algorithms/l2-svm-predict.dml");
		Map<String, String> m = new HashMap<>();
		m.put("$Y", "temp/1.csv");
		m.put("$confusion", "temp/2.csv");
		m.put("$scores", "temp/3.csv");

		PreparedScript l2svmPredict = conn.prepareScript(s, m, new String[] { "X", "Y", "w", "fmt" },
				new String[] { "scores", "confusion_mat" }, false);

		double[][] testData = new double[150][3];
		for (int i = 0; i < 150; i++) {
			int one = ThreadLocalRandom.current().nextInt(0, 101);
			int two = ThreadLocalRandom.current().nextInt(0, 101);
			int three = ThreadLocalRandom.current().nextInt(0, 101);
			double[] row = new double[] { one, two, three };
			testData[i] = row;
		}
		l2svmPredict.setMatrix("X", testData);

		double[][] testLabels = new double[150][1];
		for (int i = 0; i < 150; i++) {
			int one = ThreadLocalRandom.current().nextInt(1, 3);
			double[] row = new double[] { one };
			testLabels[i] = row;
		}
		l2svmPredict.setMatrix("Y", testLabels);

		l2svmPredict.setMatrix("w", model);

		l2svmPredict.setScalar("fmt", "csv");

		ResultVariables l2svmPredictResults = l2svmPredict.executeScript();
		double[][] scores = l2svmPredictResults.getMatrix("scores");
		log.debug("SCORES:");
		log.debug(displayMatrix(scores));

		double[][] confusionMatrix = l2svmPredictResults.getMatrix("confusion_mat");
		log.debug("CONFUSION MATRIX:");
		log.debug(displayMatrix(confusionMatrix));

		conn.close();
	}

	public static void jmlcLinReg() throws Exception {

		Connection conn = getConfiguredConnection();

		String linRegDS = conn.readScript("scripts/algorithms/LinearRegDS.dml");
		PreparedScript linRegDSScript = conn.prepareScript(linRegDS, new String[] { "X", "y" },
				new String[] { "beta_out" }, false);

		double[][] trainData = new double[500][3];
		for (int i = 0; i < 500; i++) {
			double one = ThreadLocalRandom.current().nextDouble(0, 100);
			double two = ThreadLocalRandom.current().nextDouble(0, 100);
			double three = ThreadLocalRandom.current().nextDouble(0, 100);
			double[] row = new double[] { one, two, three };
			trainData[i] = row;
		}
		linRegDSScript.setMatrix("X", trainData);
		log.debug(displayMatrix(trainData));

		double[][] trainLabels = new double[500][1];
		for (int i = 0; i < 500; i++) {
			double one = ThreadLocalRandom.current().nextDouble(0, 100);
			double[] row = new double[] { one };
			trainLabels[i] = row;
		}
		linRegDSScript.setMatrix("y", trainLabels);
		log.debug(displayMatrix(trainLabels));

		ResultVariables linRegDSResults = linRegDSScript.executeScript();
		double[][] dsBetas = linRegDSResults.getMatrix("beta_out");
		log.debug("DS BETAS:");
		log.debug(displayMatrix(dsBetas));

		String linRegCG = conn.readScript("scripts/algorithms/LinearRegCG.dml");
		PreparedScript linRegCGScript = conn.prepareScript(linRegCG, new String[] { "X", "y" },
				new String[] { "beta_out" }, false);
		linRegCGScript.setMatrix("X", trainData);
		linRegCGScript.setMatrix("y", trainLabels);
		ResultVariables linRegCGResults = linRegCGScript.executeScript();
		double[][] cgBetas = linRegCGResults.getMatrix("beta_out");
		log.debug("CG BETAS:");
		log.debug(displayMatrix(cgBetas));

		String glmPredict = conn.readScript("scripts/algorithms/GLM-predict.dml");
		PreparedScript glmPredictScript = conn.prepareScript(glmPredict, new String[] { "X", "Y", "B_full" },
				new String[] { "means" }, false);
		double[][] testData = new double[500][3];
		for (int i = 0; i < 500; i++) {
			double one = ThreadLocalRandom.current().nextDouble(0, 100);
			double two = ThreadLocalRandom.current().nextDouble(0, 100);
			double three = ThreadLocalRandom.current().nextDouble(0, 100);
			double[] row = new double[] { one, two, three };
			testData[i] = row;
		}
		glmPredictScript.setMatrix("X", testData);
		double[][] testLabels = new double[500][1];
		for (int i = 0; i < 500; i++) {
			double one = ThreadLocalRandom.current().nextDouble(0, 100);
			double[] row = new double[] { one };
			testLabels[i] = row;
		}
		glmPredictScript.setMatrix("Y", testLabels);
		glmPredictScript.setMatrix("B_full", cgBetas);
		ResultVariables glmPredictResults = glmPredictScript.executeScript();
		double[][] means = glmPredictResults.getMatrix("means");
		log.debug("GLM PREDICT MEANS:");
		log.debug(displayMatrix(means));

		conn.close();
	}

	public static void jmlcALS() throws Exception {
		Connection conn = getConfiguredConnection();

		String dataGen = conn.readScript("scripts/datagen/genRandData4ALS.dml");

		Map<String, String> m = new HashMap<>();
		m.put("$rows", "1000");
		m.put("$cols", "1000");
		m.put("$rank", "100");
		m.put("$nnz", "10000");

		PreparedScript dataGenScript = conn.prepareScript(dataGen, m, new String[] {}, new String[] { "X", "W", "H" },
				false);

		ResultVariables dataGenResults = dataGenScript.executeScript();
		double[][] x = dataGenResults.getMatrix("X");
		log.debug(displayMatrix(x));

		Map<String, String> m2 = new HashMap<>();
		m2.put("$rank", "100");
		String alsCg = conn.readScript("scripts/algorithms/ALS-CG.dml");
		PreparedScript alsCgScript = conn.prepareScript(alsCg, m2, new String[] { "X" }, new String[] { "U", "V" },
				false);
		alsCgScript.setMatrix("X", x);
		ResultVariables alsCgResults = alsCgScript.executeScript();
		double[][] u = alsCgResults.getMatrix("U");
		log.debug("u:" + u);
		log.debug(displayMatrix(u));
		double[][] v = alsCgResults.getMatrix("V");
		log.debug("v:" + v);
		log.debug(displayMatrix(v));

		String alsDs = conn.readScript("scripts/algorithms/ALS-DS.dml");
		PreparedScript alsDsScript = conn.prepareScript(alsDs, m2, new String[] { "V" }, new String[] { "L", "Rt" },
				false);
		alsDsScript.setMatrix("V", x);
		ResultVariables alsDsResults = alsDsScript.executeScript();
		double[][] l = alsDsResults.getMatrix("L");
		log.debug("l:" + l);
		log.debug(displayMatrix(l));
		double[][] rt = alsDsResults.getMatrix("Rt");
		log.debug("rt:" + rt);
		log.debug(displayMatrix(rt));

		conn.close();
	}

	public static void jmlcKmeans() throws Exception {

		Connection conn = getConfiguredConnection();
		Map<String, String> m = new HashMap<>();
		m.put("$k", "5");
		m.put("$isY", "TRUE");
		m.put("$verb", "TRUE");

		String kMeans = conn.readScript("scripts/algorithms/Kmeans.dml");
		PreparedScript kMeansScript = conn.prepareScript(kMeans, m, new String[] { "X" }, new String[] { "C", "Y" },
				false);

		double[][] x = randomMatrix(50, 50, -1, 1, 0.1);
		kMeansScript.setMatrix("X", x);
		log.debug("X:");
		log.debug(displayMatrix(x));

		ResultVariables kMeansResults = kMeansScript.executeScript();
		double[][] c = kMeansResults.getMatrix("C");
		log.debug("C:");
		log.debug(displayMatrix(c));

		double[][] y = kMeansResults.getMatrix("Y");
		log.debug("Y:");
		log.debug(displayMatrix(y));

		conn.close();
	}

	@SuppressWarnings({ "rawtypes", "resource" })
	public static void jmlcTests() {
		try {
			File jmlcTestDir = new File("target/test-classes");
			if (!jmlcTestDir.exists()) {
				log.error("Test class directory could not be found");
				return;
			}
			URL url = jmlcTestDir.toURI().toURL();
			URL[] urls = new URL[] { url };
			ClassLoader cl = new URLClassLoader(urls);

			Class clazz = cl.loadClass("org.apache.sysml.test.integration.functions.jmlc.FrameCastingTest");
			Object obj = clazz.newInstance();
			invokeMethod(clazz, obj, "setUpBase");
			invokeMethod(clazz, obj, "setUp");
			invokeMethod(clazz, obj, "testJMLCTransformDense");
			invokeMethod(clazz, obj, "testJMLCTransformSparse");
			invokeMethod(clazz, obj, "testJMLCTransformDenseReuse");
			invokeMethod(clazz, obj, "testJMLCTransformSparseReuse");
			invokeMethod(clazz, obj, "tearDown");

			clazz = cl.loadClass("org.apache.sysml.test.integration.functions.jmlc.FrameDecodeTest");
			obj = clazz.newInstance();
			invokeMethod(clazz, obj, "setUpBase");
			invokeMethod(clazz, obj, "setUp");
			invokeMethod(clazz, obj, "testJMLCTransformDense");
			invokeMethod(clazz, obj, "testJMLCTransformSparse");
			invokeMethod(clazz, obj, "testJMLCTransformDenseReuse");
			invokeMethod(clazz, obj, "testJMLCTransformSparseReuse");
			invokeMethod(clazz, obj, "tearDown");

			clazz = cl.loadClass("org.apache.sysml.test.integration.functions.jmlc.FrameEncodeTest");
			obj = clazz.newInstance();
			invokeMethod(clazz, obj, "setUpBase");
			invokeMethod(clazz, obj, "setUp");
			invokeMethod(clazz, obj, "testJMLCTransformDense");
			invokeMethod(clazz, obj, "testJMLCTransformSparse");
			invokeMethod(clazz, obj, "testJMLCTransformDenseReuse");
			invokeMethod(clazz, obj, "testJMLCTransformSparseReuse");
			invokeMethod(clazz, obj, "tearDown");

			clazz = cl.loadClass("org.apache.sysml.test.integration.functions.jmlc.FrameIndexingAppendTest");
			obj = clazz.newInstance();
			invokeMethod(clazz, obj, "setUpBase");
			invokeMethod(clazz, obj, "setUp");
			invokeMethod(clazz, obj, "testJMLCTransformDense");
			invokeMethod(clazz, obj, "testJMLCTransformSparse");
			invokeMethod(clazz, obj, "testJMLCTransformDenseReuse");
			invokeMethod(clazz, obj, "testJMLCTransformSparseReuse");
			invokeMethod(clazz, obj, "tearDown");

			clazz = cl.loadClass("org.apache.sysml.test.integration.functions.jmlc.FrameLeftIndexingTest");
			obj = clazz.newInstance();
			invokeMethod(clazz, obj, "setUpBase");
			invokeMethod(clazz, obj, "setUp");
			invokeMethod(clazz, obj, "testJMLCTransformDense");
			invokeMethod(clazz, obj, "testJMLCTransformSparse");
			invokeMethod(clazz, obj, "testJMLCTransformDenseReuse");
			invokeMethod(clazz, obj, "testJMLCTransformSparseReuse");
			invokeMethod(clazz, obj, "tearDown");

			clazz = cl.loadClass("org.apache.sysml.test.integration.functions.jmlc.FrameReadMetaTest");
			obj = clazz.newInstance();
			invokeMethod(clazz, obj, "setUpBase");
			invokeMethod(clazz, obj, "setUp");
			invokeMethod(clazz, obj, "testJMLCTransformDenseSpec");
			invokeMethod(clazz, obj, "testJMLCTransformDenseReuseSpec");
			invokeMethod(clazz, obj, "testJMLCTransformDense");
			invokeMethod(clazz, obj, "testJMLCTransformDenseReuse");
			invokeMethod(clazz, obj, "testJMLCTransformDenseReadFrame");
			invokeMethod(clazz, obj, "testJMLCTransformDenseReuseReadFrame");
			invokeMethod(clazz, obj, "tearDown");

			clazz = cl.loadClass("org.apache.sysml.test.integration.functions.jmlc.FrameTransformTest");
			obj = clazz.newInstance();
			invokeMethod(clazz, obj, "setUpBase");
			invokeMethod(clazz, obj, "setUp");
			invokeMethod(clazz, obj, "testJMLCTransformDense");
			invokeMethod(clazz, obj, "testJMLCTransformSparse");
			invokeMethod(clazz, obj, "testJMLCTransformDenseReuse");
			invokeMethod(clazz, obj, "testJMLCTransformSparseReuse");
			invokeMethod(clazz, obj, "tearDown");

			clazz = cl.loadClass("org.apache.sysml.test.integration.functions.jmlc.JMLCInputOutputTest");
			obj = clazz.newInstance();
			invokeMethod(clazz, obj, "setUpBase");
			invokeMethod(clazz, obj, "setUp");
			invokeMethod(clazz, obj, "testScalarInputInt");
			invokeMethod(clazz, obj, "testScalarInputDouble");
			invokeMethod(clazz, obj, "testScalarInputBoolean");
			invokeMethod(clazz, obj, "testScalarInputLong");
			invokeMethod(clazz, obj, "testScalarInputStringExplicitValueType");
			invokeMethod(clazz, obj, "testScalarOutputLong");
			invokeMethod(clazz, obj, "testScalarOutputDouble");
			invokeMethod(clazz, obj, "testScalarOutputString");
			invokeMethod(clazz, obj, "testScalarOutputBoolean");
			invokeMethod(clazz, obj, "testScalarOutputScalarObject");
			invokeMethod(clazz, obj, "tearDown");

			clazz = cl.loadClass("org.apache.sysml.test.integration.functions.jmlc.JMLCInputStreamReadTest");
			obj = clazz.newInstance();
			invokeMethod(clazz, obj, "setUpBase");
			invokeMethod(clazz, obj, "setUp");
			invokeMethod(clazz, obj, "testInputStreamReadMatrixDenseCSV");
			invokeMethod(clazz, obj, "testInputStreamReadMatrixDenseText");
			invokeMethod(clazz, obj, "testInputStreamReadMatrixSparseCSV");
			invokeMethod(clazz, obj, "testInputStreamReadMatrixSparseText");
			invokeMethod(clazz, obj, "testInputStreamReadFrameDenseCSV");
			invokeMethod(clazz, obj, "testInputStreamReadFrameDenseText");
			invokeMethod(clazz, obj, "testInputStreamReadFrameSparseCSV");
			invokeMethod(clazz, obj, "testInputStreamReadFrameSparseText");
			invokeMethod(clazz, obj, "testInputStreamReadFrameDenseCSVMeta");
			invokeMethod(clazz, obj, "testInputStreamReadFrameDenseTextMeta");
			invokeMethod(clazz, obj, "testInputStreamReadFrameSparseCSVMeta");
			invokeMethod(clazz, obj, "testInputStreamReadFrameSparseTextMeta");
			invokeMethod(clazz, obj, "tearDown");

			clazz = cl.loadClass("org.apache.sysml.test.integration.functions.jmlc.MulticlassSVMScoreTest");
			obj = clazz.newInstance();
			invokeMethod(clazz, obj, "setUpBase");
			invokeMethod(clazz, obj, "setUp");
			invokeMethod(clazz, obj, "testJMLCMulticlassScoreDense");
			invokeMethod(clazz, obj, "testJMLCMulticlassScoreSparse");
			invokeMethod(clazz, obj, "tearDown");

			clazz = cl.loadClass("org.apache.sysml.test.integration.functions.jmlc.ReuseModelVariablesTest");
			obj = clazz.newInstance();
			invokeMethod(clazz, obj, "setUpBase");
			invokeMethod(clazz, obj, "setUp");
			invokeMethod(clazz, obj, "testJMLCScoreGLMDense");
			invokeMethod(clazz, obj, "testJMLCScoreGLMSparse");
			invokeMethod(clazz, obj, "testJMLCScoreGLMDenseReuse");
			invokeMethod(clazz, obj, "testJMLCScoreGLMSparseReuse");
			invokeMethod(clazz, obj, "testJMLCScoreMSVMDense");
			invokeMethod(clazz, obj, "testJMLCScoreMSVMSparse");
			invokeMethod(clazz, obj, "testJMLCScoreMSVMDenseReuse");
			invokeMethod(clazz, obj, "testJMLCScoreMSVMSparseReuse");
			invokeMethod(clazz, obj, "tearDown");

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	@SuppressWarnings({ "unchecked", "rawtypes" })
	private static void invokeMethod(Class clazz, Object obj, String methodName) throws NoSuchMethodException,
			SecurityException, IllegalAccessException, IllegalArgumentException, InvocationTargetException {
		Method m = clazz.getMethod(methodName);
		m.invoke(obj);
	}

	public static double[][] matrix(int rows, int cols, double[] vals) {
		double[][] matrix = new double[rows][cols];
		if ((vals == null) || (vals.length == 0)) {
			return matrix;
		}
		for (int i = 0; i < vals.length; i++) {
			matrix[i / cols][i % cols] = vals[i];
		}
		return matrix;
	}

	public static double[][] randomMatrix(int rows, int cols, double min, double max, double sparsity) {
		double[][] matrix = new double[rows][cols];
		Random random = new Random(System.currentTimeMillis());
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				if (random.nextDouble() > sparsity) {
					continue;
				}
				matrix[i][j] = (random.nextDouble() * (max - min) + min);
			}
		}
		return matrix;
	}

	public static String displayMatrix(double[][] matrix) {
		try {
			return DataConverter.toString(DataConverter.convertToMatrixBlock(matrix));
		} catch (DMLRuntimeException e) {
			return "N/A";
		}
	}

	private static Connection getConfiguredConnection() {
		Connection conn = new Connection(ConfigType.ALLOW_DYN_RECOMPILATION, ConfigType.PARALLEL_LOCAL_OR_REMOTE_PARFOR,
				ConfigType.PARALLEL_CP_MATRIX_OPERATIONS);
		return conn;
	}
}
