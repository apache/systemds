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

package org.apache.sysds.test.functions.federated.paramserv;

import org.apache.sysds.test.TestUtils;

import java.util.Objects;

/**
 * Util class with helper methods for generating and writing
 * features and labels for parameter server tests.
 */
public class ParamServTestUtils {
	/**
	 * Generate features
	 * @param networkType network type
	 * @param numExamples number of input examples
	 * @param C number of channels
	 * @param Hin input height
	 * @param Win input width
	 * @return features
	 */
	public static double[][] generateFeatures(String networkType, int numExamples, int C, int Hin, int Win){
		if (Objects.equals(networkType, "UNet"))
			return generateDummyMedicalImageFeatures(numExamples, C, Hin, Win);
		else
			return generateDummyMNISTFeatures(numExamples, C, Hin, Win);
	}

	/**
	 * Generates an feature matrix that has the same format as the MNIST dataset,
	 * but is completely random and normalized
	 *
	 *  @param numExamples Number of examples to generate
	 *  @param C Channels in the input data
	 *  @param Hin Height in Pixels of the input data
	 *  @param Win Width in Pixels of the input data
	 *  @return a dummy MNIST feature matrix
	 */
	private static double[][] generateDummyMNISTFeatures(int numExamples, int C, int Hin, int Win) {
		// Seed -1 takes the time in milliseconds as a seed
		// Sparsity 1 means no sparsity
		return TestUtils.generateTestMatrix(numExamples, C*Hin*Win, 0, 1, 1, -1);
	}

	/**
	 * Generate dummy medical image features for training UNet.
	 * Input height and input width are padded so that the output
	 * dimensions of UNet matches the label dimensions.
	 * @param numExamples number of input examples
	 * @param C number of channels
	 * @param Hin input height
	 * @param Win input width
	 * @return features
	 */
	private static double[][] generateDummyMedicalImageFeatures(int numExamples, int C, int Hin, int Win) {
		// Pad height and width
		Hin = Hin + 184;
		Win = Win + 184;
		return TestUtils.generateTestMatrix(numExamples, C*Hin*Win, -1024, 4096, 1, -1);
	}

	/**
	 * Generate labels
	 * @param networkType type of network
	 * @param numExamples number of examples to generate labels for
	 * @param numLabels number of labels to generate (except for UNet)
	 * @param numFeatures number of features without padding (only used for UNet)
	 * @param features features for which labels are generated (only used for UNet)
	 * @return labels
	 */
	public static double[][] generateLabels(String networkType, int numExamples, int numLabels, int numFeatures, double[][] features) {
		if (Objects.equals(networkType, "UNet"))
			return generateDummyMedicalImageLabels(features,numFeatures);
		else
			return generateDummyMNISTLabels(numExamples, numLabels);
	}

	/**
	 * Generates a label matrix that has the same format as the MNIST dataset, but is completely random and consists
	 * of one hot encoded vectors as rows
	 *
	 *  @param numExamples Number of examples to generate
	 *  @param numLabels Number of labels to generate
	 *  @return a dummy MNIST lable matrix
	 */
	private static double[][] generateDummyMNISTLabels(int numExamples, int numLabels) {
		// Seed -1 takes the time in milliseconds as a seed
		// Sparsity 1 means no sparsity
		return TestUtils.generateTestMatrix(numExamples, numLabels, 0, 1, 1, -1);
	}

	/**
	 * Return labels as 0 or 1 based on the values in features.
	 * @param features for which labels are generated
	 * @param numFeatures number of features without padding
	 * @return labels
	 */
	private static double[][] generateDummyMedicalImageLabels(double[][] features, int numFeatures) {
		double split = 1000;
		double[][] labels = new double[features.length][numFeatures];
		for ( int i = 0; i < labels.length; i++ ){
			for ( int j = 0; j < labels[0].length; j++ ){
				labels[i][j] = (features[i][j] > split) ? 1 : 0;
			}
		}
		return labels;
	}
}
