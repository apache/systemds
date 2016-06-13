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

package org.apache.sysml.test.integration.applications;

import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.runners.Parameterized.Parameters;

import java.util.*;

public abstract class KNNTest extends AutomatedTestBase {

    protected final static String TEST_DIR = "applications/knn/";
    protected final static String TEST_NAME = "KNN";
    protected String TEST_CLASS_DIR =
            TEST_DIR + KNNTest.class.getSimpleName() + "/";

    protected int numRecords, numFeatures, kSelectType, featureSelectType, importanceSelectType;
    protected double logFeatureVarianceDisbalance;

    public KNNTest( int numRecords_,
            int numFeatures_,
            int kSelectType_,
            int featureSelectType_,
            int importanceSelectType_ ) {
        this.numRecords = numRecords_;
        this.numFeatures = numFeatures_;
        this.kSelectType = kSelectType_;
        this.featureSelectType = featureSelectType_;
        this.importanceSelectType = importanceSelectType_;

    }

    @Parameters public static Collection<Object[]> data() {
        //one case for the knn: default settings
        //rows, columns, k-select, feature-select, feature-importance
        Object[][] data = new Object[][] { { 150, 5, 0, 0, 0 } };
        return Arrays.asList( data );
    }

    @Override public void setUp() {
        addTestConfiguration( TEST_CLASS_DIR, TEST_NAME );
    }

    protected void testKNN( ScriptType scriptType ) {
        System.out.println( "------------ BEGIN " + TEST_NAME + " " + scriptType
                + " TEST WITH {" +
                numRecords + ", " +
                numFeatures + ", " +
                kSelectType + ", " +
                featureSelectType + ", " +
                importanceSelectType +
                "} ------------" );
        this.scriptType = scriptType;

        int rows = numRecords;                // # of rows in the training data
        int cols = numFeatures;               // # of features in the training data

        getAndLoadTestConfiguration( TEST_NAME );

        // prepare training data set
        long nnz_in_X = 0;
        long nnz_in_y = 0;

        Random r = new Random( 314159265 );
        double[][] X = getRandomMatrix( rows, cols, 1, 300, 10.0, 34567 );
        double[][] y = getRandomMatrix( rows, 1, 3, 6, 1.0, 31415 );

        double[][] y_t = new double[][] { { 2 } };

        System.out.println("knn input train data values:");

        for (int i = 0; i < rows; i++) {
            for (int column = 0; column < cols; column++)
                System.out.print(X[i][column] + " ");
            System.out.println();
        }
        System.out.println();

        // make the target columns of y variance is inteter
        System.out.println("knn input target values:");
        for (int i = 0; i < rows; i++) {
            y[i][0] = (double) Math.round(y[i][0]);
            System.out.print(y[i][0] + " ");
        }
        System.out.println();
        writeInputMatrixWithMTD( "X", X, true );

        writeInputMatrixWithMTD( "Y", y, true );

        writeInputMatrixWithMTD( "Y_T", y_t, true );

        List<String> proArgs = new ArrayList<String>();

        proArgs.add( "-nvargs" );

        proArgs.add( "select_k=" + String.format( "%d", kSelectType ) );
        proArgs.add(
                "select_feature=" + String.format( "%d", featureSelectType ) );
        proArgs.add( "feature_importance=" + String.format( "%d",
                importanceSelectType ) );
        //proArgs.add("fmt=csv");

        proArgs.add( "X=" + input( "X" ) );
        proArgs.add( "T=" + input( "X" ) );
        proArgs.add( "Y=" + input( "Y" ) );
        proArgs.add( "Y_T=" + input( "Y_T" ) );
        proArgs.add( "PR=" + output( "pr_SYSTEMML" ) );
        proArgs.add( "NNR=" + output( "nnr_SYSTEMML" ) );
        programArgs = proArgs.toArray( new String[ proArgs.size() ] );

        fullDMLScriptName = getScript();

        //used for running R script
        rCmd = getRCmd( input( "X.mtx" ),
                input( "Y.mtx" ),
                expected( "betas_R" ),
                Integer.toString(rows));

        int expectedNumberOfJobs = -1;

        //run the dml script
        runTest( true, EXCEPTION_NOT_EXPECTED, null, expectedNumberOfJobs );

        //get result and compare
        double max_abs_beta = 0.0;

        HashMap<CellIndex, Double> wSYSTEMML_raw = readDMLMatrixFromHDFS(
                "pr_SYSTEMML" );

        HashMap<CellIndex, Double> wSYSTEMML = new HashMap<CellIndex, Double>();
        for ( CellIndex key : wSYSTEMML_raw.keySet() )
            if ( key.column == 1 )
                wSYSTEMML.put( key, wSYSTEMML_raw.get( key ) );

        System.out.println("knn predict values:");
        for ( CellIndex key : wSYSTEMML.keySet() )
            System.out.print(wSYSTEMML_raw.get( key ) + " ");
        System.out.println();
        //run the R script
        runRScript( true );

        HashMap<CellIndex, Double> wR = readRMatrixFromFS( "betas_R" );

        TestUtils.compareMatrices( wR,
                wSYSTEMML,
                max_abs_beta,
                "wR",
                "wSYSTEMML" );
    }

}
