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
import org.apache.sysds.runtime.transform.decode.ColumnDecoder;
import org.apache.sysds.runtime.transform.decode.ColumnDecoderFactory;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

/**
 * Test class for validating mixed-method column decoding in SystemDS.
 *
 * This test constructs synthetic data, applies a transform specification
 * involving multiple encoding methods (binning, recoding, dummy coding,
 * and pass-through), encodes it using the transformation framework,
 * and then decodes it back using {@link ColumnDecoder}.
 */
public class ColumnDecoderMixedMethodsTest extends AutomatedTestBase {

    /**
     * Setup method executed before tests.
     * Clears previous assertion information to ensure a clean test state.
     */
    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
    }

    /**
     * Tests decoding of columns transformed by mixed methods:
     * - Column 1: Binning
     * - Column 2: Pass-through
     * - Column 3: Recode
     * - Column 4: Dummy coding
     * - Column 5: Recode
     * - Column 6: Dummy coding
     *
     * The test performs the following steps:
     * 1. Creates synthetic data and converts it into a FrameBlock.
     * 2. Defines a transformation spec combining multiple methods.
     * 3. Encodes the FrameBlock into a MatrixBlock using MultiColumnEncoder.
     * 4. Extracts metadata needed for decoding.
     * 5. Creates a ColumnDecoder using ColumnDecoderFactory.
     * 6. Decodes the encoded matrix and prints the decoded result.
     */
    @Test
    public void testColumnDecoderMixedMethods() {
        try {
            // Step 1: Generate synthetic data
            int rows = 50;
            double[][] arr = new double[rows][6];
            for (int i = 0; i < rows; i++) {
                arr[i][0] = 2*i + 1;        // bin column
                arr[i][1] = 101 + i;        // pass through column
                arr[i][2] = 5*(i % 4) + 2;  // recode column
                arr[i][3] = (i % 4) + 6;    // dummy column
                arr[i][4] = 100 + (i % 3);  // recode column
                arr[i][5] = (i % 2) + 1;    // dummy column
            }

            // Convert data into SystemDS matrix and frame representations
            MatrixBlock mb = DataConverter.convertToMatrixBlock(arr);
            FrameBlock data = DataConverter.convertToFrameBlock(mb);

            // Step 2: Define transformation specification
            String spec ="{ids:true, bin:[{id:1, method:equi-width, numbins:4}], recode:[3,5], dummycode:[4,6]}";

            // Step 3: Encode data using MultiColumnEncoder
            MultiColumnEncoder enc = EncoderFactory.createEncoder(spec, data.getColumnNames(), data.getNumColumns(), null);
            MatrixBlock encoded = enc.encode(data);
            FrameBlock meta = enc.getMetaData(new FrameBlock(data.getNumColumns(), ValueType.STRING));

            //TODO: Error with legacy code
            //Decoder dec = DecoderFactory.createDecoder(spec, data.getColumnNames(), data.getSchema(), meta, encoded.getNumColumns());
            //FrameBlock expected = new FrameBlock(data.getSchema());
            //dec.decode(encoded, expected, 5);


            // Step 4: Decode using ColumnDecoder
            ColumnDecoder cdec = ColumnDecoderFactory.createDecoder(spec, data.getColumnNames(), data.getSchema(), meta);
            FrameBlock actual = new FrameBlock(data.getSchema());
            cdec.columnDecode(encoded, actual);

            System.out.println(actual);
        }
        catch(Exception ex) {
            throw new RuntimeException(ex);
        }
    }
}
