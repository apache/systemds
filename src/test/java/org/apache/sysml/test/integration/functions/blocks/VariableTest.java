/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.test.integration.functions.blocks;

import org.junit.Test;

import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;



/**
 * <p><b>Positive tests:</b></p>
 * <ul>
 *  <li>variable passing 1</li>
 *  <li>variable passing 2</li>
 *  <li>variable analysis 1</li>
 * </ul>
 * <p><b>Negative tests:</b></p>
 * 
 * 
 */
public class VariableTest extends AutomatedTestBase
{
	
    private final static String TEST_DIR = "functions/blocks/";
    private final static String TEST_VARIABLE_PASSING_1 = "VariablePassing1";
    private final static String TEST_VARIABLE_PASSING_2 = "VariablePassing2";
    private final static String TEST_VARIABLE_ANALYSIS_1 = "VariableAnalysis1";


    @Override
    public void setUp()
    {
        baseDirectory = SCRIPT_DIR + "functions/blocks/";

        // positive tests
        addTestConfiguration(TEST_VARIABLE_PASSING_1, new TestConfiguration(TEST_DIR, "VariablePassing1Test",
                new String[] { "c", "e", "f" }));
        addTestConfiguration(TEST_VARIABLE_PASSING_2, new TestConfiguration(TEST_DIR, "VariablePassing2Test",
                new String[] { "d" }));
        addTestConfiguration(TEST_VARIABLE_ANALYSIS_1, new TestConfiguration(TEST_DIR, "VariableAnalysis1Test",
                new String[] { "b" }));

        // negative tests
    }

    @Test
    public void testVariablePassing1()
    {
        int rows = 10;
        int cols = 10;

        TestConfiguration config = getTestConfiguration(TEST_VARIABLE_PASSING_1);
        config.addVariable("rows", rows);
        config.addVariable("cols", cols);

        loadTestConfiguration(config);

        double[][] a = getRandomMatrix(rows, cols, -1, 1, 1, -1);
        writeInputMatrix("a", a);
        double[][] b = getRandomMatrix(rows, cols, -1, 1, 1, -1);
        writeInputMatrix("b", b);

        double x = 0;
        double[][] c = new double[rows][cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                x += a[i][j];
            }
        }

        double[][] d = new double[rows][cols];
        double[][] f = new double[rows][cols];
        double[][] e = new double[rows][cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                d[i][j] = x * a[i][j];
                c[i][j] = b[i][j] * d[i][j];
                b[i][j] = b[i][j] * d[i][j];
                f[i][j] = 3 * b[i][j];
                e[i][j] = 3 * d[i][j];
            }
        }

        writeExpectedMatrix("c", c);
        writeExpectedMatrix("f", f);
        writeExpectedMatrix("e", e);

        runTest();

        compareResults(1e-12);
    }
    
    @Test
    public void testVariablePassing2() {
        int rows = 10;
        int cols = 10;
        int factor = 5;
        
        TestConfiguration config = getTestConfiguration(TEST_VARIABLE_PASSING_2);
        config.addVariable("rows", rows);
        config.addVariable("cols", cols);
        config.addVariable("factor", factor);
        
        loadTestConfiguration(config);
        
        double[][] a = getRandomMatrix(rows, cols, -1, 1, 0.5, -1);
        writeInputMatrix("a", a);
        
        double[][] b = new double[cols][rows];
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                b[j][i] = factor * a[i][j];
            }
        }
        
        double[][] c = TestUtils.performMatrixMultiplication(a, b);
        
        double[][] d = TestUtils.performMatrixMultiplication(b, c);
        writeExpectedMatrix("d", d);
        
        runTest();

        compareResults(1e-12);
    }

    @Test
    public void testVariableAnalysis1()
    {
        int rows = 10;
        int cols = 10;

        TestConfiguration config = getTestConfiguration(TEST_VARIABLE_ANALYSIS_1);
        config.addVariable("rows", rows);
        config.addVariable("cols", cols);

        loadTestConfiguration(config);

        double[][] a = getRandomMatrix(rows, cols, -1, 1, 0.5, -1);
        writeInputMatrix("a", a);
        writeExpectedMatrix("b", a);

        runTest();

        compareResults(1e-12);
    }

}
