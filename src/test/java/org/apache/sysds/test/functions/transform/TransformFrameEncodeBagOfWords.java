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

package org.apache.sysds.test.functions.transform;

import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.runtime.transform.encode.ColumnEncoderBagOfWords;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.nio.file.Files;

import static org.apache.sysds.runtime.transform.encode.ColumnEncoderBagOfWords.tokenize;

public class TransformFrameEncodeBagOfWords extends AutomatedTestBase
{
	private final static String TEST_NAME1 = "TransformFrameEncodeBagOfWords";
	private final static String TEST_NAME2 = "TransformFrameEncodeApplyBagOfWords";
	private final static String TEST_DIR = "functions/transform/";
	private final static String TEST_CLASS_DIR = TEST_DIR + TransformFrameEncodeBagOfWords.class.getSimpleName() + "/";
	// for benchmarking: Digital_Music_Text.csv
	private String DATASET = "amazonReview2023/Digital_Music_Text_Head2k_With_RCD_Col.csv";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2));
	}

	// These tests result in dense output
	@Test
	public void testTransformBagOfWords() {
		runTransformTest(TEST_NAME1, ExecMode.SINGLE_NODE, false, false);
	}
	@Test
	public void testTransformApplyBagOfWords() {
		runTransformTest(TEST_NAME2, ExecMode.SINGLE_NODE, false, false);
	}

	@Test
	public void testTransformApplySeparateStagesBagOfWords() {
		MultiColumnEncoder.APPLY_ENCODER_SEPARATE_STAGES = true;
		runTransformTest(TEST_NAME2, ExecMode.SINGLE_NODE, false, false);
		MultiColumnEncoder.APPLY_ENCODER_SEPARATE_STAGES = false;
	}

	@Test
	public void testTransformBagOfWordsError() {
		runTransformTest(TEST_NAME1, ExecMode.SINGLE_NODE, false, false, false, true, false, false);
	}

	@Test
	public void testTransformBagOfWordsPlusRecode() {
		runTransformTest(TEST_NAME1, ExecMode.SINGLE_NODE, true, false);
	}

	@Test
	public void testTransformApplyBagOfWordsPlusRecode() {
		runTransformTest(TEST_NAME2, ExecMode.SINGLE_NODE, true, false);
	}

	@Test
	public void testTransformBagOfWords2() {
		runTransformTest(TEST_NAME1, ExecMode.SINGLE_NODE, false, true);
	}

	@Test
	public void testTransformApplyBagOfWords2() {
		runTransformTest(TEST_NAME2, ExecMode.SINGLE_NODE, false, true);
	}

	@Test
	public void testTransformBagOfWordsPlusRecode2() {
		runTransformTest(TEST_NAME1, ExecMode.SINGLE_NODE, true, true);
	}

	@Test
	public void testTransformApplyBagOfWordsPlusRecode2() {
		runTransformTest(TEST_NAME2, ExecMode.SINGLE_NODE, true, true);
	}

	// AmazonReviewDataset transformation results in a sparse output
	@Test
	public void testTransformBagOfWordsAmazonReviews() {
		runTransformTest(TEST_NAME1, ExecMode.SINGLE_NODE, false, false, true);
	}

	@Test
	public void testTransformApplyBagOfWordsAmazonReviews() {
		runTransformTest(TEST_NAME2, ExecMode.SINGLE_NODE, false, false, true);
	}

	@Test
	public void testTransformBagOfWordsAmazonReviews2() {
		runTransformTest(TEST_NAME1, ExecMode.SINGLE_NODE, false, true, true);
	}

	@Test
	public void testTransformApplyBagOfWordsAmazonReviews2() {
		runTransformTest(TEST_NAME2, ExecMode.SINGLE_NODE, false, true, true);
	}

	@Test
	public void testTransformBagOfWordsAmazonReviewsAndRandRecode() {
		runTransformTest(TEST_NAME1, ExecMode.SINGLE_NODE, true, false, true);
	}

	@Test
	public void testTransformApplyBagOfWordsAmazonReviewsAndRandRecode() {
		runTransformTest(TEST_NAME2, ExecMode.SINGLE_NODE, true, false, true);
	}

	@Test
	public void testTransformBagOfWordsAmazonReviewsAndDummyCode() {
		// TODO: compare result
		runTransformTest(TEST_NAME1, ExecMode.SINGLE_NODE, true, false, true, false, true, false);
	}

	@Test
	public void testTransformBagOfWordsAmazonReviewsAndPassThrough() {
		// TODO: compare result
		runTransformTest(TEST_NAME1, ExecMode.SINGLE_NODE, false, false, true, false, false, true);
	}

	@Test
	public void testTransformBagOfWordsAmazonReviewsAndRandRecode2() {
		runTransformTest(TEST_NAME1, ExecMode.SINGLE_NODE, true, true, true);
	}

	@Test
	public void testTransformApplyBagOfWordsAmazonReviewsAndRandRecode2() {
		runTransformTest(TEST_NAME2, ExecMode.SINGLE_NODE, true, true, true);
	}

	@Test
	public void testTransformApplySeparateStagesBagOfWordsAmazonReviewsAndRandRecode2() {
		MultiColumnEncoder.APPLY_ENCODER_SEPARATE_STAGES = true;
		runTransformTest(TEST_NAME2, ExecMode.SINGLE_NODE, true, true, true);
		MultiColumnEncoder.APPLY_ENCODER_SEPARATE_STAGES = false;
	}


	@Test
	public void testTransformBagOfWordsSpark() {
		runTransformTest(TEST_NAME1, ExecMode.SPARK, false, false);
	}

	@Test
	public void testTransformBagOfWordsAmazonReviewsSpark() {
		runTransformTest(TEST_NAME1, ExecMode.SPARK, false, false, true);
	}

	@Test
	public void testTransformBagOfWordsAmazonReviews2Spark() {
		runTransformTest(TEST_NAME1, ExecMode.SPARK, false, true, true);
	}

	@Test
	public void testTransformBagOfWordsAmazonReviewsAndRandRecodeSpark() {
		runTransformTest(TEST_NAME1, ExecMode.SPARK, true, false, true);
	}

	@Test
	public void testTransformBagOfWordsAmazonReviewsAndRandRecode2Spark() {
		runTransformTest(TEST_NAME1, ExecMode.SPARK, true, true, true);
	}

	@Test
	public void testBuildPartialBagOfWordsNotApplicable() {
		ColumnEncoderBagOfWords bow = new ColumnEncoderBagOfWords();
		assert bow.getColID() == -1;
		try {
			bow.buildPartial(null); // should run without error
		} catch (Exception e) {
			throw new AssertionError("Test failed: Expected no errors due to early abort (colId = -1). " +
					"Encountered exception:\n" + e + "\nMessage: " + Arrays.toString(e.getStackTrace()));
		}
	}

	private void runTransformTest(String testname, ExecMode rt, boolean recode, boolean dup){
		runTransformTest(testname, rt, recode, dup, false);
	}

	private void runTransformTest(String testname, ExecMode rt, boolean recode, boolean dup, boolean fromFile){
		runTransformTest(testname, rt, recode, dup, fromFile, false, false, false);
	}

	private void runTransformTest(String testname, ExecMode rt, boolean recode, boolean dup, boolean fromFile, boolean error, boolean dc, boolean pt)
	{
		//set runtime platform
		ExecMode rtold = setExecMode(rt);
		try
		{
			getAndLoadTestConfiguration(testname);
			fullDMLScriptName = getScript();

			// Create the dataset by repeating and shuffling the distinct tokens
			String[][] columns = fromFile ? readTwoColumnStringCSV(DATASET_DIR + DATASET) : new String[][]{{"This is the " +
					"first document","This document is the second document",
					"And this is the third one","Is this the first document"}, {"A", "B", "A", "C"}};
			String[] sentenceColumn = columns[0];
			String[] recodeColumn = recode ? columns[1] : null;
			if(!fromFile)
				writeStringsToCsvFile(sentenceColumn, recodeColumn, baseDirectory + INPUT_DIR + "data", dup);

			int mode = error ? 1 : (dc ? 2 : (pt ? 3 : 0));
			programArgs = new String[]{"-explain", "recompile_runtime", "-stats","-args", fromFile ? DATASET_DIR + DATASET : input("data"),
					output("result"), output("dict"), String.valueOf(recode), String.valueOf(dup),
					String.valueOf(fromFile), String.valueOf(mode)};
			if(error)
				runTest(true, EXCEPTION_EXPECTED, DMLRuntimeException.class, -1);
			else{
				runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
				if(testname == TEST_NAME2){
					double errorValue = readDMLScalarFromOutputDir("result").values()
							.stream().findFirst().orElse(1000.0);
					System.out.println(errorValue);
					assert errorValue <= 10;
				} else {
					FrameBlock dict_frame = readDMLFrameFromHDFS( "dict", Types.FileFormat.CSV);
					int cols = recode? dict_frame.getNumRows() + 1 : dict_frame.getNumRows();
					if(dup)
						cols *= 2;
					if(mode == 0){
						HashMap<MatrixValue.CellIndex, Double> res_actual = readDMLMatrixFromOutputDir("result");
						double[][] result = TestUtils.convertHashMapToDoubleArray(res_actual, Math.min(sentenceColumn.length, 100),
								cols);
						checkResults(sentenceColumn, result, recodeColumn, dict_frame, dup ? 2 : 1);
					}
				}
			}


		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
		finally {
			resetExecMode(rtold);
		}
	}

	private String[][] readTwoColumnStringCSV(String s) {
        try {
            FrameBlock in = readDMLFrameFromHDFS(s, Types.FileFormat.CSV, false);
			String[][] out = new String[2][in.getNumRows()];
			for (int i = 0; i < in.getNumRows(); i++) {
				out[0][i] = in.getString(i, 0);
				out[1][i] = in.getString(i, 1);
			}
			return out;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
	}

	@SuppressWarnings("unchecked")
	public static void checkResults(String[] sentences, double[][] result,
		String[] recodeColumn, FrameBlock dict, int duplicates)
	{
		HashMap<String, Integer>[] indices = new HashMap[duplicates];
		HashMap<String, Integer>[] rcdMaps = new HashMap[duplicates];
		String errors = "";
		int num_errors = 0;
		int max_errors = 100;
		int frameCol = 0;
		// even when the set of tokens is the same for duplicates, the order in which the tokens dicts are merged
		// is not always the same for all columns in multithreaded mode
		for (int i = 0; i < duplicates; i++) {
			indices[i] = new HashMap<>();
			rcdMaps[i] = new HashMap<>();
			for (int j = 0; j < dict.getNumRows(); j++) {
				String[] tuple = dict.getString(j, frameCol).split("\u00b7");
				indices[i].put(tuple[0], Integer.parseInt(tuple[1]) - 1);
			}
			frameCol++;
			if(recodeColumn != null){
				for (int j = 0; j < dict.getNumRows(); j++) {
					String current = dict.getString(j, frameCol);
					if(current == null)
						break;
					String[] tuple = current.split("\u00b7");
					rcdMaps[i].put(tuple[0], Integer.parseInt(tuple[1]));
				}
				frameCol++;
			}
		}

		// only check the first 100 rows
		for (int row = 0; row < Math.min(sentences.length, 100); row++) {
			// build token count dictionary once
			String sentence = sentences[row];
			HashMap<String, Integer> count = new HashMap<>();
			String[] words = tokenize(sentence, false,  "\\s+");
			for (String word : words) {
				if (!word.isEmpty()) {
					word = word.toLowerCase();
					Integer old = count.getOrDefault(word, 0);
					count.put(word, old + 1);
				}
			}
			// iterate through the results of the column encoders
			int offset = 0;
			for (int j = 0; j < duplicates; j++) {
				// init the zeroIndices with all columns
				List<Integer> zeroIndices = new ArrayList<>();
				for (int i = 0; i < indices[j].size(); i++) {
					zeroIndices.add(i);
				}
				// subtract the nnz columns
				for (String word : words) {
					if (!word.isEmpty()) {
						zeroIndices.remove(indices[j].get(word));
					}
				}

				// compare results: bag of words
				for(Map.Entry<String, Integer> entry : count.entrySet()){
					String word = entry.getKey();
					int count_expected = entry.getValue();
					Integer index = indices[j].get(word);
					if(index == null){
						throw new AssertionError("row [" + row + "]: not found word: " + word);
					}
					if(result[row][index + offset] != count_expected){
						String error_message = "bow result[" + row + "," + (index + offset) + "]=" +
								result[row][index + offset] + " does not match the expected: " + count_expected;
						if(num_errors < max_errors)
							errors += error_message + '\n';
						else
							throw new AssertionError(errors + error_message);
						num_errors++;
					}
				}
				for(int zeroIndex : zeroIndices){
					if(result[row][offset + zeroIndex] != 0){
						String error_message = "bow result[" + row + "," + (offset + zeroIndex) + "]=" +
								result[row][offset + zeroIndex] + " does not match the expected: 0";
						if(num_errors < max_errors)
							errors += error_message + '\n';
						else
							throw new AssertionError(errors + error_message);
						num_errors++;
					}
				}
				offset += indices[j].size();
				// compare results: recode
				if(recodeColumn != null){
					if(result[row][offset] != rcdMaps[j].get(recodeColumn[row])){
						String error_message = "recode result[" + row + "," + offset + "]=" +
								result[row][offset]+ " does not match the expected: " + rcdMaps[j].get(recodeColumn[row]);
						if(num_errors < max_errors)
							errors += error_message + '\n';
						else
							throw new AssertionError(errors + error_message);
						num_errors++;
					}
					offset++;
				}
			}
		}
		if (num_errors > 0)
			throw new AssertionError(errors);
	}

	public static void writeStringsToCsvFile(String[] sentences, String[] recodeTokens, String fileName, boolean duplicate) throws IOException {
		Path path = Paths.get(fileName);
		Files.createDirectories(path.getParent());
		try (BufferedWriter bw = Files.newBufferedWriter(path)) {
			for (int i = 0; i < sentences.length; i++) {
				String out = sentences[i] +  (recodeTokens != null ? "," + recodeTokens[i] : "");
				if(duplicate)
					out = out  + ","  + out;
				bw.write(out);
				bw.newLine();
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
