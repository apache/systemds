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

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.io.FileFormatPropertiesCSV;
import org.apache.sysds.runtime.io.FrameReaderFactory;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.transform.tokenize.Tokenizer;
import org.apache.sysds.runtime.transform.tokenize.TokenizerFactory;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

import javax.json.Json;
import javax.json.JsonObject;
import javax.json.JsonObjectBuilder;
import java.io.IOException;


public class TokenizeMultithreadedTest extends AutomatedTestBase  {
    private static final String TEST_DIR = "functions/transform/";
    private static final String TEST_CLASS_DIR = TEST_DIR + TokenizeMultithreadedTest.class.getSimpleName() + "/";

    //dataset and transform tasks without missing values
    private final static String DATASET 	= "20news/20news_subset_untokenized.csv";


    private final static JsonObject ngram_algo_params0 = Json.createObjectBuilder()
            .add("min_gram", 2)
            .add("max_gram", 3)
            .add("regex", "\\W+")
            .build();

    private final static JsonObject count_out_params0 = Json.createObjectBuilder().add("sort_alpha", false).build();
    private final static JsonObject count_out_params1 = Json.createObjectBuilder().add("sort_alpha", true).build();

    private final static JsonObject hash_out_params0 = Json.createObjectBuilder().add("num_features", 128).build();

    public enum TokenizerBuilder {
        WHITESPACE_SPLIT,
        NGRAM,
    }

    public enum TokenizerApplier {
        COUNT,
        HASH,
        POSITION,
    }

    @Override
    public void setUp()  {
        TestUtils.clearAssertionInformation();
        addTestConfiguration(this.getClass().getSimpleName(),
            new TestConfiguration(TEST_CLASS_DIR, this.getClass().getSimpleName(), new String[] { "R" }) );
    }

    @Test
    public void testTokenizeSplitCountLong() {
        runTokenizeTest(ExecMode.SINGLE_NODE, TokenizerBuilder.WHITESPACE_SPLIT,TokenizerApplier.COUNT,
            2000, false, null, count_out_params0);
    }

    @Test
    public void testTokenizeNgramCountLong() {
        runTokenizeTest(ExecMode.SINGLE_NODE, TokenizerBuilder.NGRAM, TokenizerApplier.COUNT,
            2000, false, ngram_algo_params0, count_out_params0);
    }

    @Test
    public void testTokenizeSplitPositionLong() {
        runTokenizeTest(ExecMode.SINGLE_NODE, TokenizerBuilder.WHITESPACE_SPLIT, TokenizerApplier.POSITION,
            2000, false, null, null);
    }

    @Test
    public void testTokenizeNgramPositionLong() {
        runTokenizeTest(ExecMode.SINGLE_NODE, TokenizerBuilder.NGRAM, TokenizerApplier.POSITION,
            2000, false, ngram_algo_params0, null);
    }

    @Test
    public void testTokenizeSplitHashLong() {
        runTokenizeTest(ExecMode.SINGLE_NODE, TokenizerBuilder.WHITESPACE_SPLIT, TokenizerApplier.HASH,
            2000, false, null, hash_out_params0);
    }

    @Test
    public void testTokenizeNgramHashLong() {
        runTokenizeTest(ExecMode.SINGLE_NODE, TokenizerBuilder.NGRAM, TokenizerApplier.HASH,
            2000, false, ngram_algo_params0, hash_out_params0);
    }
    @Test
    public void testTokenizeSplitCountWide() {
        runTokenizeTest(ExecMode.SINGLE_NODE, TokenizerBuilder.WHITESPACE_SPLIT,TokenizerApplier.POSITION,
            2000, true, null, count_out_params0);
    }

    @Test
    public void testTokenizeNgramCountWide() {
        runTokenizeTest(ExecMode.SINGLE_NODE, TokenizerBuilder.NGRAM, TokenizerApplier.POSITION,
            2000, true, ngram_algo_params0, count_out_params0);
    }

    @Test
    public void testTokenizeSplitHashWide() {
        runTokenizeTest(ExecMode.SINGLE_NODE, TokenizerBuilder.WHITESPACE_SPLIT, TokenizerApplier.HASH,
            2000, true, null, hash_out_params0);
    }

    @Test
    public void testTokenizeNgramHashWide() {
        runTokenizeTest(ExecMode.SINGLE_NODE, TokenizerBuilder.NGRAM, TokenizerApplier.HASH,
            2000, true, ngram_algo_params0, hash_out_params0);
    }

    private void runTokenizeTest(ExecMode rt, TokenizerBuilder builder, TokenizerApplier applier,
        int max_tokens, boolean format_wide, JsonObject algo_params, JsonObject out_params) {
        try{
            getAndLoadTestConfiguration(this.getClass().getSimpleName());
            FileFormatPropertiesCSV props = new FileFormatPropertiesCSV();
            props.setHeader(false);
            FrameBlock input = FrameReaderFactory.createFrameReader(Types.FileFormat.CSV, props)
                .readFrameFromHDFS(DATASET_DIR+DATASET, -1L, -1L);
            String spec = createTokenizerSpec(builder, applier, format_wide, algo_params, out_params);
            Tokenizer tokenizer = TokenizerFactory.createTokenizer(spec, max_tokens);
            FrameBlock outS = tokenizer.tokenize(input, 1);
            FrameBlock outM = tokenizer.tokenize(input, 12);
            Assert.assertEquals(outS.getNumRows(), outM.getNumRows());
            Assert.assertEquals(outS.getNumColumns(), outM.getNumColumns());
            TestUtils.compareFrames(DataConverter.convertToStringFrame(outS),
                DataConverter.convertToStringFrame(outM), outS.getNumRows(), outS.getNumColumns());

        } catch (Exception ex){
            throw new RuntimeException(ex);
        }

    }

    private String createTokenizerSpec(TokenizerBuilder builder, TokenizerApplier applier, boolean format_wide, JsonObject algo_params, JsonObject out_params) {
        JsonObjectBuilder spec = Json.createObjectBuilder();
        switch (builder){
            case WHITESPACE_SPLIT:
                spec.add("algo", "split");
                break;
            case NGRAM:
                spec.add("algo", "ngram");
                break;
        }
        switch (applier){
            case COUNT:
                spec.add("out", "count");
                break;
            case POSITION:
                spec.add("out", "position");
                break;
            case HASH:
                spec.add("out", "hash");
                break;
        }
        if(out_params != null)
            spec.add("out_params", out_params);
        if(algo_params != null)
            spec.add("algo_params", algo_params);
        spec.add("format_wide", format_wide);
        spec.add("id_cols",Json.createArrayBuilder().add(2).add(3));
        spec.add("tokenize_col", 4);
        return spec.build().toString();
    }
}
