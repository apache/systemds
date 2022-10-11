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

package org.apache.sysds.test.functions.homomorphicEncryption;

import org.apache.sysds.api.DMLException;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.paramserv.NativeHEHelper;
import org.apache.sysds.runtime.controlprogram.paramserv.homomorphicEncryption.PublicKey;
import org.apache.sysds.runtime.controlprogram.paramserv.homomorphicEncryption.SEALClient;
import org.apache.sysds.runtime.controlprogram.paramserv.homomorphicEncryption.SEALServer;
import org.apache.sysds.runtime.instructions.cp.CiphertextMatrix;
import org.apache.sysds.runtime.instructions.cp.PlaintextMatrix;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.meta.MetaDataFormat;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class InOutTest extends AutomatedTestBase {
    private final static String TEST_NAME = "InOutTest";
    private final static String TEST_DIR = "functions/data/";
    private final static String TEST_CLASS_DIR = TEST_DIR + InOutTest.class.getSimpleName() + "/";

    private final int num_clients = 3;

    private final int rows = 100;
    private final int cols = 200;
    private final long seed = 42;

    @Override
    public void setUp() {
        NativeHEHelper.initialize();
        TestUtils.clearAssertionInformation();
        addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "C" }) );
    }

    @Test
    public void endToEndTest() {
        SEALServer server = new SEALServer();

        SEALClient[] clients = new SEALClient[num_clients];
        PublicKey[] partial_pub_keys = new PublicKey[num_clients];
        for (int i = 0; i < num_clients; i++) {
            clients[i] = new SEALClient(server.generateA());
            partial_pub_keys[i] = clients[i].generatePartialPublicKey();
        }

        PublicKey public_key = server.aggregatePartialPublicKeys(partial_pub_keys);

        MatrixObject[] plaintexts = new MatrixObject[num_clients];
        CiphertextMatrix[] ciphertexts = new CiphertextMatrix[num_clients];
        for (int i = 0; i < num_clients; i++) {
            MatrixBlock mb = TestUtils.generateTestMatrixBlock(rows, cols, -100, 100, 1.0, seed+i);
            MatrixObject mo = new MatrixObject(Types.ValueType.FP64, null);
            mo.setMetaData(new MetaDataFormat(new MatrixCharacteristics(rows, cols), Types.FileFormat.BINARY));
            mo.acquireModify(mb);
            mo.release();
            plaintexts[i] = mo;

            clients[i].setPublicKey(public_key);
            ciphertexts[i] = clients[i].encrypt(plaintexts[i]);
        }

        CiphertextMatrix encrypted_sum = server.accumulateCiphertexts(ciphertexts);

        PlaintextMatrix[] partial_decryptions = new PlaintextMatrix[num_clients];
        for (int i = 0; i < num_clients; i++) {
            partial_decryptions[i] = clients[i].partiallyDecrypt(encrypted_sum);
        }

        MatrixObject result = server.average(encrypted_sum, partial_decryptions);

        double[] expected_raw_result = new double[rows*cols];
        double[][] plaintexts_raw = new double[num_clients][];
        for (int i = 0; i < num_clients; i++) {
            plaintexts_raw[i] = plaintexts[i].acquireReadAndRelease().getDenseBlockValues();
        }
        for (int x = 0; x < rows * cols; x++) {
            double sum = 0.0;
            for (int i = 0; i < num_clients; i++) {
                sum += plaintexts_raw[i][x];
            }
            expected_raw_result[x] = sum / num_clients;
        }

        double[] raw_result = result.acquireReadAndRelease().getDenseBlockValues();
        assert result.getNumRows() == rows;
        assert result.getNumColumns() == cols;
        assert raw_result.length == rows*cols;
        TestUtils.compareMatrices(raw_result, expected_raw_result, 5e-8);
    }
}
