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

package org.apache.sysml.api.ml.classification;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Predictor;
import org.apache.spark.ml.param.IntParam;
import org.apache.spark.ml.param.Param;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.StructType;
import org.apache.sysml.api.DMLException;
import org.apache.sysml.api.MLContext;
import org.apache.sysml.api.MLOutput;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.api.ml.param.DecisionTreeClassifierParams;

import scala.Tuple2;

public class DecisionTreeClassifier extends
		Predictor<Vector, DecisionTreeClassifier, DecisionTreeClassificationModel> implements
		DecisionTreeClassifierParams {

	private static final long serialVersionUID = -6304519164821016521L;
	private SparkContext sc = null;
	private DataFrame rDF = null;
	private HashMap<String, String> cmdLineParams = new HashMap<String, String>();
	private HashMap<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>> results =
			new HashMap<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>>();

	private Param<String> impurity = new Param<String>(
			this, "impurity", "Impurity measure: entropy or Gini " + "(the default)");
	private IntParam bins = new IntParam(
			this, "bins", "Number of equiheight bins per scale feature to choose " + "thresholds");
	private IntParam depth = new IntParam(
			this, "depth", "Maximum depth of the learned tree");
	private IntParam numLeaf = new IntParam(
			this, "num_leaf", "Number of samples when splitting stops and a " + "leaf node is added");
	private IntParam numSamples = new IntParam(
			this, "num_samples", "Number of samples at which point we switch "
					+ "to in-memory subtree building");
	private Param<String> rCol = new Param<String>(
			this, "rCol", "Name of the column that has categorical data indices in features vector");

	public DecisionTreeClassifier(SparkContext sc) throws DMLRuntimeException {
		this.sc = sc;
		setAllParameters(20, 25, 10, 3000, "Gini");
	}

	public DecisionTreeClassifier(SparkContext sc, DataFrame rDF) throws DMLRuntimeException {
		this.sc = sc;
		this.rDF = rDF;
		setAllParameters(20, 25, 10, 3000, "Gini");
	}

	public DecisionTreeClassifier(SparkContext sc, int maxBins, int maxDepth, int numOfLeaf, int numOfSamples,
			String imp) throws DMLRuntimeException {
		this.sc = sc;
		setAllParameters(maxBins, maxDepth, numOfLeaf, numOfSamples, imp);
	}

	public DecisionTreeClassifier(SparkContext sc, DataFrame rDF, int maxBins, int maxDepth, int numOfLeaf,
			int numOfSamples, String imp) throws DMLRuntimeException {
		this.sc = sc;
		this.rDF = rDF;
		setAllParameters(maxBins, maxDepth, numOfLeaf, numOfSamples, imp);
	}

	private void setAllParameters(int maxBins, int maxDepth, int numOfLeaf, int numOfSamples, String imp) {
		setDefault(maxBins(), maxBins);
		cmdLineParams.put(bins.name(), Integer.toString(maxBins));
		setDefault(maxDepth(), maxDepth);
		cmdLineParams.put(depth.name(), Integer.toString(maxDepth));
		setDefault(numLeaf(), numOfLeaf);
		cmdLineParams.put(numLeaf.name(), Integer.toString(numOfLeaf));
		setDefault(numSamples(), numOfSamples);
		cmdLineParams.put(numSamples.name(), Integer.toString(numOfSamples));
		setDefault(impurity(), imp);
		cmdLineParams.put(impurity.name(), imp);
		setRCol("catIndices");
	}

	@Override
	public DecisionTreeClassifier copy(ParamMap paramMap) {
		try {
			String strBins = paramMap.getOrElse(bins, getMaxBins()).toString();
			String strDepth = paramMap.getOrElse(depth, getMaxDepth()).toString();
			String strNumLeaf = paramMap.getOrElse(numLeaf, getNumLeaf()).toString();
			String strNumSamples = paramMap.getOrElse(numSamples, getNumSamples()).toString();
			String strImpurity = paramMap.getOrElse(impurity, getImpurity());

			DecisionTreeClassifier dtc = new DecisionTreeClassifier(
					sc,
					rDF,
					Integer.parseInt(strBins),
					Integer.parseInt(strDepth),
					Integer
							.parseInt(strNumLeaf),
					Integer.parseInt(strNumSamples),
					strImpurity);

			dtc.cmdLineParams.put(bins.name(), strBins);
			dtc.cmdLineParams.put(depth.name(), strDepth);
			dtc.cmdLineParams.put(numLeaf.name(), strNumLeaf);
			dtc.cmdLineParams.put(numSamples.name(), strNumSamples);
			dtc.cmdLineParams.put(impurity.name(), strImpurity);
			dtc.setFeaturesCol(getFeaturesCol());
			dtc.setLabelCol(getLabelCol());
			dtc.setRCol(getRCol());

			return dtc;
		} catch (DMLRuntimeException e) {
			e.printStackTrace();
		}

		return null;
	}

	@Override
	public String uid() {
		return Long.toString(serialVersionUID);
	}

	@Override
	public StructType validateAndTransformSchema(StructType arg0, boolean arg1, DataType arg2) {
		return null;
	}

	public DecisionTreeClassifier setMaxBins(int value) {
		cmdLineParams.put(bins.name(), Integer.toString(value));
		return (DecisionTreeClassifier) setDefault(bins, value);
	}

	@Override
	public IntParam maxBins() {
		return bins;
	}

	@Override
	public int getMaxBins() {
		return Integer.parseInt(cmdLineParams.get(bins.name()));
	}

	public DecisionTreeClassifier setMaxDepth(int value) {
		cmdLineParams.put(depth.name(), Integer.toString(value));
		return (DecisionTreeClassifier) setDefault(depth, value);
	}

	@Override
	public IntParam maxDepth() {
		return depth;
	}

	@Override
	public int getMaxDepth() {
		return Integer.parseInt(cmdLineParams.get(depth.name()));
	}

	public DecisionTreeClassifier setNumLeaf(int value) {
		cmdLineParams.put(numLeaf.name(), Integer.toString(value));
		return (DecisionTreeClassifier) setDefault(numLeaf, value);
	}

	@Override
	public IntParam numLeaf() {
		return numLeaf;
	}

	@Override
	public int getNumLeaf() {
		return Integer.parseInt(cmdLineParams.get(numLeaf.name()));
	}

	public DecisionTreeClassifier setNumSamples(int value) {
		cmdLineParams.put(numSamples.name(), Integer.toString(value));
		return (DecisionTreeClassifier) setDefault(numSamples, value);
	}

	@Override
	public IntParam numSamples() {
		return numSamples;
	}

	@Override
	public int getNumSamples() {
		return Integer.parseInt(cmdLineParams.get(numSamples.name()));
	}

	public DecisionTreeClassifier setImpurity(String value) {
		cmdLineParams.put(impurity.name(), value);
		return (DecisionTreeClassifier) setDefault(impurity, value);
	}

	@Override
	public Param<String> impurity() {
		return impurity;
	}

	@Override
	public String getImpurity() {
		return cmdLineParams.get(impurity.name());
	}

	public DecisionTreeClassifier setRCol(String value) {
		cmdLineParams.put(rCol.name(), value);
		return (DecisionTreeClassifier) setDefault(rCol, value);
	}

	@Override
	public Param<String> rCol() {
		return rCol;
	}

	@Override
	public String getRCol() {
		return cmdLineParams.get(rCol.name());
	}

	@Override
	public DecisionTreeClassificationModel train(DataFrame df) {
		MLContext ml = null;
		MLOutput out = null;

		try {
			ml = new MLContext(
					sc);
		} catch (DMLRuntimeException e1) {
			e1.printStackTrace();
			return null;
		}

		// Convert input data to format that SystemML accepts
		MatrixCharacteristics mcXin = new MatrixCharacteristics();
		JavaPairRDD<MatrixIndexes, MatrixBlock> Xin = null;
		Xin = RDDConverterUtils.dataFrameToBinaryBlock(new JavaSparkContext(
				sc), df.select(getFeaturesCol()), mcXin, false, true);

		MatrixCharacteristics mcYin = new MatrixCharacteristics();
		JavaPairRDD<MatrixIndexes, MatrixBlock> yin = null;
		yin = RDDConverterUtils.dataFrameToBinaryBlock(new JavaSparkContext(
				sc), df.select(getLabelCol()), mcYin, false, true);

		MatrixCharacteristics mcRin = new MatrixCharacteristics();
		JavaPairRDD<MatrixIndexes, MatrixBlock> Rin = null;
		if (rDF != null) {
			try {
				Rin = RDDConverterUtils.dataFrameToBinaryBlock(new JavaSparkContext(
						sc), rDF.select(getRCol()), mcXin, false, true);
				ml.registerInput("R", Rin, mcRin);
				cmdLineParams.put("R", "R");
			} catch (DMLRuntimeException e1) {
				e1.printStackTrace();
				return null;
			}
		}

		try {
			// Register the input/output variables of script
			// 'decision-tree.dml'
			ml.registerInput("X", Xin, mcXin);
			ml.registerInput("Y_bin", yin, mcYin);
			ml.registerOutput("M");

//			String systemmlHome = System.getenv("SYSTEMML_HOME");
//			if (systemmlHome == null) {
//				System.err.println("ERROR: The environment variable SYSTEMML_HOME is not set.");
//				return null;
//			}

			// Or add ifdef in decision-tree.dml
			// cmdLineParams.put("M", " ");

//			String dmlFilePath = systemmlHome + File.separator + "scripts" + File.separator + "algorithms" + File.separator + "decision-tree.dml";
			String dmlFilePath = "scripts" + File.separator + "algorithms" + File.separator + "decision-tree.dml";

			synchronized (MLContext.class) {
				// static synchronization is necessary before
				// execute call
				out = ml.execute(dmlFilePath, cmdLineParams);
			}

			results.put("M",
					new Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>(
							out.getBinaryBlockedRDD("M"),
							out.getMatrixCharacteristics("M")));

			return new DecisionTreeClassificationModel(
					results, sc, rDF, getFeaturesCol(), getLabelCol(), getRCol()).setParent(
							this);
		} catch (IOException e) {
			throw new RuntimeException(e);
		} catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		} catch (DMLException e) {
			throw new RuntimeException(e);
		}
	}
}
