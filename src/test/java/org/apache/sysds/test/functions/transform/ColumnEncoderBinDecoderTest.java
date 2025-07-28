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

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.decode.*;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

/**
 * Unit test for validating binning column decoding in SystemDS.
 *
 * This test ensures that numeric columns transformed with binning
 * (discretization into intervals or bins) can be correctly decoded
 * back to approximate numeric values using {@link ColumnDecoderBin}.
 *
 * It compares results from the legacy {@link Decoder} API and the
 * newer {@link ColumnDecoder} API to verify decoding consistency.
 */
public class ColumnEncoderBinDecoderTest extends AutomatedTestBase {

    /**
     * Clears assertion state before each test execution.
     */
    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
    }

    /**
     * Test binning-based encoding and decoding workflow.
     *
     * Steps:
     * 1. Generate a sequential numeric column.
     * 2. Apply equi-width binning transformation using {@link MultiColumnEncoder}.
     * 3. Decode the encoded matrix using:
     *    - Legacy {@link Decoder} API (baseline expected result).
     *    - New {@link ColumnDecoder} API.
     * 4. Compare decoded results to ensure both methods match.
     */
    @Test
    public void testColumnEncoderDecoderBin() {
        try {
            // Step 1: Generate numeric data (1 to 20)
            int rows = 20;
            MatrixBlock mb = MatrixBlock.seqOperations(1, rows, 1);
            FrameBlock data = DataConverter.convertToFrameBlock(mb);
            // Step 2: Define binning transformation spec
            String spec = "{ids:true, bin:[{id:1, method:equi-width, numbins:4}]}";

            // Step 3: Encode data using MultiColumnEncoder
            MultiColumnEncoder enc = EncoderFactory.createEncoder(spec, data.getColumnNames(), 1, null);
            MatrixBlock encoded = enc.encode(data);
            FrameBlock meta = enc.getMetaData(new FrameBlock(1, ValueType.STRING));

            // Step 4a: Decode using legacy Decoder API
            Decoder dec = DecoderFactory.createDecoder(spec, data.getColumnNames(), data.getSchema(), meta, encoded.getNumColumns());
            FrameBlock expected = new FrameBlock(data.getSchema());
            dec.decode(encoded, expected);

            // Step 4b: Decode using new ColumnDecoder API
            ColumnDecoder cdec = ColumnDecoderFactory.createDecoder(spec, data.getColumnNames(), data.getSchema(), meta);
            FrameBlock actual = new FrameBlock(data.getSchema());
            cdec.columnDecode(encoded, actual);

            // Step 5: Validate results
            TestUtils.compareFrames(expected, actual, false);
        }
        catch(Exception ex) {
            throw new RuntimeException(ex);
        }
    }
}