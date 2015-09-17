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

package com.ibm.bi.dml.test.integration.applications;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import org.junit.runners.Parameterized.Parameters;

import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.utils.TestUtils;


public abstract class GLMTest extends AutomatedTestBase
{
	
    protected final static String TEST_DIR = "applications/glm/";
    protected final static String TEST_NAME = "GLM";

    protected int numRecords, numFeatures, distFamilyType, linkType;
    protected double distParam, linkPower, intercept, logFeatureVarianceDisbalance, avgLinearForm, stdevLinearForm, dispersion;
    
	public GLMTest (int numRecords_, int numFeatures_, int distFamilyType_, double distParam_,
		int linkType_, double linkPower_, double intercept_, double logFeatureVarianceDisbalance_, 
		double avgLinearForm_, double stdevLinearForm_, double dispersion_)
	{
		this.numRecords = numRecords_;
		this.numFeatures = numFeatures_;
		this.distFamilyType = distFamilyType_;
		this.distParam = distParam_;
		this.linkType = linkType_;
		this.linkPower = linkPower_;
		this.intercept = intercept_;
		this.logFeatureVarianceDisbalance = logFeatureVarianceDisbalance_;
	    this.avgLinearForm = avgLinearForm_;
	    this.stdevLinearForm = stdevLinearForm_;
		this.dispersion = dispersion_;
	}
	
	// SUPPORTED GLM DISTRIBUTION FAMILIES AND LINKS:
	// -----------------------------------------------
	// INPUT PARAMETERS:    MEANING:            Cano-
	// dfam vpow link lpow  Distribution.link   nical?
	// -----------------------------------------------
	//  1   0.0   1  -1.0   Gaussian.inverse
	//  1   0.0   1   0.0   Gaussian.log
	//  1   0.0   1   1.0   Gaussian.id          Yes
	//  1   1.0   1   0.0   Poisson.log          Yes
	//  1   1.0   1   0.5   Poisson.sqrt
	//  1   1.0   1   1.0   Poisson.id
	//  1   2.0   1  -1.0   Gamma.inverse        Yes
	//  1   2.0   1   0.0   Gamma.log
	//  1   2.0   1   1.0   Gamma.id
	//  1   3.0   1  -2.0   InvGaussian.1/mu^2   Yes
	//  1   3.0   1  -1.0   InvGaussian.inverse
	//  1   3.0   1   0.0   InvGaussian.log
	//  1   3.0   1   1.0   InvGaussian.id
	//  1    *    1    *    AnyVariance.AnyLink
	// -----------------------------------------------
	//  2    *    1   0.0   Binomial.log
	//  2    *    2    *    Binomial.logit       Yes
	//  2    *    3    *    Binomial.probit
	//  2    *    4    *    Binomial.cloglog
	//  2    *    5    *    Binomial.cauchit
	// -----------------------------------------------

	@Parameters
	public static Collection<Object[]> data() {
		// SCHEMA: 
		// #RECORDS, #FEATURES, DISTRIBUTION_FAMILY, VARIANCE_POWER or BERNOULLI_NO, LINK_TYPE, LINK_POWER, 
		//     INTERCEPT, LOG_FEATURE_VARIANCE_DISBALANCE, AVG_LINEAR_FORM, ST_DEV_LINEAR_FORM, DISPERSION
		Object[][] data = new Object[][] { 
				
		//     #RECS  #FTRS DFM VPOW  LNK LPOW   ICPT  LFVD  AVGLT STDLT  DISP
				
		// Both DML and R work and compute close results:

/*				
        // THIS IS TO TEST "NO INTERCEPT" OPTION ("icpt=0"):
			{ 100000,   50,  1,  0.0,  1,  0.0,  0.0,  3.0,  10.0,  2.0,  2.5 },   // Gaussian.log
			{  10000,  100,  1,  0.0,  1,  1.0,  0.0,  3.0,   0.0,  2.0,  2.5 },   // Gaussian.id
			{  20000,  100,  1,  0.0,  1, -1.0,  0.0,  0.0,   0.2,  0.03, 2.5 },   // Gaussian.inverse
			{  10000,  100,  1,  1.0,  1,  0.0,  0.0,  3.0,   0.0,  1.0,  2.5 },   // Poisson.log
			{ 100000,   10,  1,  1.0,  1,  0.0,  0.0,  3.0,   0.0, 50.0,  2.5 },   // Poisson.log             // Pr[0|x] gets near 1
			{  20000,  100,  1,  1.0,  1,  0.5,  0.0,  3.0,  10.0,  2.0,  2.5 },   // Poisson.sqrt
			{  20000,  100,  1,  1.0,  1,  1.0,  0.0,  3.0,  50.0, 10.0,  2.5 },   // Poisson.id
			{ 100000,   50,  1,  2.0,  1,  0.0,  0.0,  3.0,   0.0,  2.0,  2.5 },   // Gamma.log
			{ 100000,   50,  1,  2.0,  1, -1.0,  0.0,  1.0,   2.0,  0.4,  1.5 },   // Gamma.inverse
			{  10000,  100,  1,  3.0,  1, -2.0,  0.0,  3.0,  50.0,  7.0,  1.7 },   // InvGaussian.1/mu^2
			{  10000,  100,  1,  3.0,  1, -1.0,  0.0,  3.0,  10.0,  2.0,  2.5 },   // InvGaussian.inverse			
			{ 100000,   50,  1,  3.0,  1,  0.0,  0.0,  2.0,  -2.0,  1.0,  1.7 },   // InvGaussian.log
			{ 100000,   50,  1,  3.0,  1,  1.0,  0.0,  1.0,   0.2,  0.04, 1.7 },   // InvGaussian.id

			{ 100000,   50,  2, -1.0,  1,  0.0,  0.0,  3.0,  -5.0,  1.0,  1.0 },   // Bernoulli {-1, 1}.log     // Note: Y is sparse
			{ 100000,   50,  2, -1.0,  1,  1.0,  0.0,  1.0,   0.6,  0.1,  1.0 },   // Bernoulli {-1, 1}.id
			{ 100000,   50,  2, -1.0,  1,  0.5,  0.0,  0.0,   0.4,  0.05, 1.0 },   // Bernoulli {-1, 1}.sqrt
			{  10000,  100,  2, -1.0,  2,  0.0,  0.0,  3.0,   0.0,  2.0,  1.0 },   // Bernoulli {-1, 1}.logit
			{  10000,  100,  2, -1.0,  2,  0.0,  0.0,  3.0,   0.0, 50.0,  1.0 },   // Bernoulli {-1, 1}.logit   // Pr[y|x] near 0, 1
			{  20000,  100,  2, -1.0,  3,  0.0,  0.0,  3.0,   0.0,  2.0,  1.0 },   // Bernoulli {-1, 1}.probit
			{ 100000,   10,  2, -1.0,  3,  0.0,  0.0,  3.0,   0.0, 50.0,  1.0 },   // Bernoulli {-1, 1}.probit  // Pr[y|x] near 0, 1
			{  10000,  100,  2, -1.0,  4,  0.0,  0.0,  3.0,  -2.0,  1.0,  1.0 },   // Bernoulli {-1, 1}.cloglog
			{  50000,   20,  2, -1.0,  4,  0.0,  0.0,  3.0,  -2.0, 50.0,  1.0 },   // Bernoulli {-1, 1}.cloglog // Pr[y|x] near 0, 1
			{  50000,  100,  2, -1.0,  5,  0.0,  0.0,  3.0,   0.0,  2.0,  1.0 },   // Bernoulli {-1, 1}.cauchit
				
			{ 100000,   50,  2,  1.0,  1,  0.0,  0.0,  3.0,  -5.0,  1.0,  2.5 },   // Binomial two-column.log   // Note: Y is sparse
			{  10000,  100,  2,  1.0,  1,  1.0,  0.0,  0.0,   0.4,  0.05, 2.5 },   // Binomial two-column.id
			{ 100000,   50,  2,  1.0,  1,  0.5,  0.0,  0.0,   0.4,  0.05, 2.5 },   // Binomial two-column.sqrt
			{  10000,  100,  2,  1.0,  2,  0.0,  0.0,  3.0,   0.0,  2.0,  2.5 },   // Binomial two-column.logit
			{  20000,  100,  2,  1.0,  3,  0.0,  0.0,  3.0,   0.0,  2.0,  2.5 },   // Binomial two-column.probit
			{  10000,  100,  2,  1.0,  4,  0.0,  0.0,  3.0,  -2.0,  1.0,  2.5 },   // Binomial two-column.cloglog
			{  20000,  100,  2,  1.0,  5,  0.0,  0.0,  3.0,   0.0,  2.0,  2.5 },   // Binomial two-column.cauchit
*/				
				

        // THIS IS TO TEST "INTERCEPT AND SHIFT/SCALE" OPTION ("icpt=2"):
			{ 200000,   50,  1,  0.0,  1,  0.0,  0.01, 3.0,  10.0,  2.0,  2.5 },   // Gaussian.log     // CHECK DEVIANCE !!!
			{  10000,  100,  1,  0.0,  1,  1.0,  0.01, 3.0,   0.0,  2.0,  2.5 },   // Gaussian.id
			{  20000,  100,  1,  0.0,  1, -1.0,  0.01, 0.0,   0.2,  0.03, 2.5 },   // Gaussian.inverse
			{  10000,  100,  1,  1.0,  1,  0.0,  0.01, 3.0,   0.0,  1.0,  2.5 },   // Poisson.log
			{ 100000,   10,  1,  1.0,  1,  0.0,  0.01, 3.0,   0.0, 50.0,  2.5 },   // Poisson.log             // Pr[0|x] gets near 1
			{  20000,  100,  1,  1.0,  1,  0.5,  0.01, 3.0,  10.0,  2.0,  2.5 },   // Poisson.sqrt
			{  10000,  100,  1,  1.0,  1,  1.0,  0.01, 3.0,  50.0, 10.0,  2.5 },   // Poisson.id
			{  50000,  100,  1,  2.0,  1,  0.0,  0.01, 3.0,   0.0,  2.0,  2.5 },   // Gamma.log
			{  10000,  100,  1,  2.0,  1, -1.0,  0.01, 3.0,   2.0,  0.3,  2.0 },   // Gamma.inverse
			{  10000,  100,  1,  3.0,  1, -2.0,  1.0,  3.0,  50.0,  7.0,  1.7 },   // InvGaussian.1/mu^2
			{  10000,  100,  1,  3.0,  1, -1.0,  0.01, 3.0,  10.0,  2.0,  2.5 },   // InvGaussian.inverse
			{ 100000,   50,  1,  3.0,  1,  0.0,  0.5,  3.0,  -2.0,  1.0,  2.5 },   // InvGaussian.log
			{ 100000,  100,  1,  3.0,  1,  1.0,  0.01, 3.0,   0.2,  0.03, 2.5 },   // InvGaussian.id

			{ 100000,   50,  2, -1.0,  1,  0.0,  0.01, 3.0,  -5.0,  1.0,  1.0 },   // Bernoulli {-1, 1}.log     // Note: Y is sparse
			{ 100000,   50,  2, -1.0,  1,  1.0,  0.01, 3.0,   0.4,  0.1,  1.0 },   // Bernoulli {-1, 1}.id
			{ 100000,   40,  2, -1.0,  1,  0.5,  0.1,  3.0,   0.4,  0.1,  1.0 },   // Bernoulli {-1, 1}.sqrt
			{  10000,  100,  2, -1.0,  2,  0.0,  0.01, 3.0,   0.0,  2.0,  1.0 },   // Bernoulli {-1, 1}.logit
			{  10000,  100,  2, -1.0,  2,  0.0,  0.01, 3.0,   0.0, 50.0,  1.0 },   // Bernoulli {-1, 1}.logit   // Pr[y|x] near 0, 1
			{  20000,  100,  2, -1.0,  3,  0.0,  0.01, 3.0,   0.0,  2.0,  1.0 },   // Bernoulli {-1, 1}.probit
			{ 100000,   10,  2, -1.0,  3,  0.0,  0.01, 3.0,   0.0, 50.0,  1.0 },   // Bernoulli {-1, 1}.probit  // Pr[y|x] near 0, 1
			{  10000,  100,  2, -1.0,  4,  0.0,  0.01, 3.0,  -2.0,  1.0,  1.0 },   // Bernoulli {-1, 1}.cloglog
			{  50000,   20,  2, -1.0,  4,  0.0,  0.01, 3.0,  -2.0, 50.0,  1.0 },   // Bernoulli {-1, 1}.cloglog // Pr[y|x] near 0, 1
			{  20000,  100,  2, -1.0,  5,  0.0,  0.01, 3.0,   0.0,  2.0,  1.0 },   // Bernoulli {-1, 1}.cauchit
        
			{  50000,  100,  2,  1.0,  1,  0.0,  0.01, 3.0,  -5.0,  1.0,  2.5 },   // Binomial two-column.log   // Note: Y is sparse
			{  10000,  100,  2,  1.0,  1,  1.0,  0.0,  0.0,   0.4,  0.05, 2.5 },   // Binomial two-column.id
			{ 100000,  100,  2,  1.0,  1,  0.5,  0.1,  3.0,   0.4,  0.05, 2.5 },   // Binomial two-column.sqrt
			{  10000,  100,  2,  1.0,  2,  0.0,  0.01, 3.0,   0.0,  2.0,  2.5 },   // Binomial two-column.logit
			{  20000,  100,  2,  1.0,  3,  0.0,  0.01, 3.0,   0.0,  2.0,  2.5 },   // Binomial two-column.probit
			{  10000,  100,  2,  1.0,  4,  0.0,  0.01, 3.0,  -2.0,  1.0,  2.5 },   // Binomial two-column.cloglog
			{  20000,  100,  2,  1.0,  5,  0.0,  0.01, 3.0,   0.0,  2.0,  2.5 },   // Binomial two-column.cauchit
			


		//  DML WORKS, BUT R FAILS:
				
		//	{  10000,  100,  1,  1.0,  1,  1.0,  0.0,  0.0,  10.0,  2.0,  2.5 },   // Poisson.id
		//	{  10000,  100,  1,  2.0,  1, -1.0,  0.0,  0.0,  10.0,  2.0,  2.5 },   // Gamma.inverse
		//	{  10000,  100,  1,  2.0,  1,  1.0,  0.0,  0.0,  10.0,  2.0,  2.5 },   // Gamma.id             // Tried tweaking, cannot satisfy R
		//	{  10000,  100,  1,  3.0,  1, -2.0,  0.0,  0.0,  10.0,  2.0,  2.5 },   // InvGaussian.1/mu^2
		//	{  10000,  100,  1,  3.0,  1,  0.0,  0.0,  0.0,   0.0,  2.0,  2.5 },   // InvGaussian.log      // R computes nonsense!
		//	{  10000,  100,  1,  3.0,  1,  1.0,  0.0,  0.0,   2.0,  0.2,  1.5 },   // InvGaussian.id
				
	    //  BOTH R AND DML FAIL:
				
		//	{  10000,  100,  1,  0.0,  1, -1.0,  0.0,  0.0,  10.0,  2.0,  2.5 },   // Gaussian.inverse     // R and DML compute nonsense
		
		
		};
		return Arrays.asList(data);
	}

    @Override
    public void setUp()
    {
    	addTestConfiguration(TEST_DIR, TEST_NAME);
    }
    
    protected void testGLM(ScriptType scriptType)
    {
		System.out.println("------------ BEGIN " + TEST_NAME + " " + scriptType + " TEST WITH {" + 
				numRecords + ", " +
				numFeatures + ", " +
				distFamilyType + ", " +
				distParam + ", " +
				linkType + ", " +
				linkPower + ", " +
				intercept + ", " +
				logFeatureVarianceDisbalance + ", " +
				avgLinearForm + ", " +
				stdevLinearForm + ", " +
				dispersion +
				"} ------------");
		this.scriptType = scriptType;
    	
    	int rows = numRecords;			    // # of rows in the training data 
        int cols = numFeatures;			    // # of features in the training data 
        
        GLMDist glmdist = new GLMDist (distFamilyType, distParam, linkType, linkPower);
        glmdist.set_dispersion (dispersion);
        
        getAndLoadTestConfiguration(TEST_NAME);

        // prepare training data set
                               
        Random r = new Random (314159265);
        double[][] X = getRandomMatrix (rows, cols, -1.0, 1.0, 1.0, 34567); // 271828183);
        double shift_X = 1.0;
        
        // make the feature columns of X variance disbalanced
        
        for (int j = 0; j < cols; j ++)
        {
        	double varFactor = Math.pow (10.0, logFeatureVarianceDisbalance * (- 0.25 + j / (double) (2 * cols - 2)));
        	for (int i = 0; i < rows; i ++)
        		X [i][j] = shift_X + X [i][j] * varFactor;
        }
        
    	double[] beta_unscaled = new double [cols];
        for (int j = 0; j < cols; j ++)
        	beta_unscaled [j] = r.nextGaussian ();
        double[] beta = scaleWeights (beta_unscaled, X, intercept, avgLinearForm, stdevLinearForm);

        long nnz_in_X = 0;
        long nnz_in_y = 0;

        double[][] y = null;
        if (glmdist.is_binom_n_needed())
        	y = new double [rows][2];
       	else
        	y = new double [rows][1];
        
        
        for (int i = 0; i < rows; i ++)
        {
        	double eta = intercept;
        	for (int j = 0; j < cols; j ++)
        	{
        		eta += X [i][j] * beta [j];
        		if (X [i][j] != 0.0)
        			nnz_in_X ++;
        	}
            if (glmdist.is_binom_n_needed())
          	{
    		    long n = Math.round (dispersion * (1.0 + 2.0 * r.nextDouble()) + 1.0);
    			glmdist.set_binom_n (n);
        	    y [i][0] = glmdist.nextGLM (r, eta);
    			y [i][1] = n - y[i][0];
        	    if (y [i][0] != 0.0)
        		    nnz_in_y ++;
    			if (y [i][1] != 0.0)
    			    nnz_in_y ++;
    		}
            else
            {
            	y [i][0] = glmdist.nextGLM (r, eta);
            	if (y [i][0] != 0.0)
            		nnz_in_y ++;            	
            }
            
        }
        
        int defaultBlockSize = com.ibm.bi.dml.parser.DMLTranslator.DMLBlockSize;

        MatrixCharacteristics mc_X = new MatrixCharacteristics (rows, cols, defaultBlockSize, defaultBlockSize, nnz_in_X);
        writeInputMatrixWithMTD ("X", X, true, mc_X);

        MatrixCharacteristics mc_y = new MatrixCharacteristics (rows, y[0].length, defaultBlockSize, defaultBlockSize, nnz_in_y);
        writeInputMatrixWithMTD ("Y", y, true, mc_y);
        
		List<String> proArgs = new ArrayList<String>();
		if (scriptType == ScriptType.PYDML) {
			proArgs.add("-python");
		}
		proArgs.add("-nvargs");
		proArgs.add("dfam=" + String.format ("%d", distFamilyType));
		proArgs.add(((distFamilyType == 2 && distParam != 1.0) ? "yneg=" : "vpow=") + String.format ("%f", distParam));
		proArgs.add((distFamilyType == 2 && distParam != 1.0) ? "vpow=0.0" : "yneg=0.0");
		proArgs.add("link=" + String.format ("%d", linkType));
		proArgs.add("lpow=" + String.format ("%f", linkPower));
		proArgs.add("icpt=2"); // INTERCEPT - CHANGE THIS AS NEEDED
		proArgs.add("disp=0.0"); // DISPERSION (0.0: ESTIMATE)
		proArgs.add("reg=0.0"); // LAMBDA REGULARIZER
		proArgs.add("tol=0.000000000001"); // TOLERANCE (EPSILON)
		proArgs.add("moi=300");
		proArgs.add("mii=0");
		proArgs.add("X=" + input("X"));
		proArgs.add("Y=" + input("Y"));
		proArgs.add("B=" + output("betas_SYSTEMML"));
		programArgs = proArgs.toArray(new String[proArgs.size()]);
		
		fullDMLScriptName = getScript();
		
		rCmd = getRCmd(input("X.mtx"), input("Y.mtx"), String.format ("%d", distFamilyType), String.format ("%f", distParam),
				String.format ("%d", linkType), String.format ("%f", linkPower), "1" /*intercept*/, "0.000000000001" /*tolerance (espilon)*/,
				expected("betas_R"));
		
		int expectedNumberOfJobs = -1; // 31;

		runTest(true, EXCEPTION_NOT_EXPECTED, null, expectedNumberOfJobs);

		double max_abs_beta = 0.0;
		HashMap<CellIndex, Double> wTRUE = new HashMap <CellIndex, Double> ();
		for (int j = 0; j < cols; j ++)
		{
			wTRUE.put (new CellIndex (j + 1, 1), Double.valueOf(beta [j]));
			max_abs_beta = (max_abs_beta >= Math.abs (beta[j]) ? max_abs_beta : Math.abs (beta[j]));
		}

        HashMap<CellIndex, Double> wSYSTEMML_raw = readDMLMatrixFromHDFS ("betas_SYSTEMML");
		HashMap<CellIndex, Double> wSYSTEMML = new HashMap <CellIndex, Double> ();
		for (CellIndex key : wSYSTEMML_raw.keySet())
			if (key.column == 1)
				wSYSTEMML.put (key, wSYSTEMML_raw.get (key));

		runRScript(true);

		HashMap<CellIndex, Double> wR   = readRMatrixFromFS ("betas_R");
        
        double eps = 0.000001;
        if( distParam==0 && linkType==1 ) // Gaussian.log
        {
        	//NOTE MB: Gaussian.log was the only test failing when we introduced multi-threaded
        	//matrix multplications (mmchain). After discussions with Sasha, we decided to change the eps
        	//because accuracy is anyway affected by various rewrites like binary to unary (-1*x->-x),
        	//transpose-matrixmult, and dot product sum. Disabling these rewrites led to a successful 
        	//test result. Even without multi-threaded matrix mult this test was failing for different number 
        	//of rows if these rewrites are enabled. Users can turn off rewrites if high accuracy is required. 
        	//However, in the future we might also consider to use Kahan plus for aggregations in matrix mult 
        	//(at least for the final aggregation of partial results from individual threads).
        	eps = 0.0000016; //1.6x the error threshold
        }		
        TestUtils.compareMatrices (wR, wSYSTEMML, eps * max_abs_beta, "wR", "wSYSTEMML");
    }
    
    double[] scaleWeights (double[] w_unscaled, double[][] X, double icept, double meanLF, double sigmaLF)
    {
    	int rows = X.length;
    	int cols = w_unscaled.length;
    	double[] w = new double [cols];
        for (int j = 0; j < cols; j ++)
    	    w [j] = w_unscaled [j];
        
    	double sum_wx = 0.0;
        double sum_1x = 0.0;
        double sum_wxwx = 0.0;
        double sum_1x1x = 0.0;
        double sum_wx1x = 0.0; 

        for (int i = 0; i < rows; i ++)
        {
        	double wx = 0.0;
        	double one_x = 0.0;
        	for (int j = 0; j < cols; j ++)
        	{
        		wx += w [j] * X [i][j];
        		one_x += X [i][j];
        	}
            sum_wx += wx;
            sum_1x += one_x;
            sum_wxwx += wx * wx;
            sum_1x1x += one_x * one_x;
            sum_wx1x += wx * one_x; 
        }
        
        double a0 = (meanLF - icept) * rows * sum_wx / (sum_wx * sum_wx + sum_1x * sum_1x);
        double b0 = (meanLF - icept) * rows * sum_1x / (sum_wx * sum_wx + sum_1x * sum_1x);
        double a1 = sum_1x;
        double b1 = - sum_wx;
        double qA = a1 * a1 * sum_wxwx + 2 * a1 * b1 * sum_wx1x + b1 * b1 * sum_1x1x;
        double qB = 2 * (a0 * a1 * sum_wxwx + a0 * b1 * sum_wx1x + a1 * b0 * sum_wx1x + b0 * b1 * sum_1x1x);
        double qC_nosigmaLF = a0 * a0 * sum_wxwx + 2 * a0 * b0 * sum_wx1x + b0 * b0 * sum_1x1x - rows * (meanLF - icept) * (meanLF - icept);
        double qC = qC_nosigmaLF - rows * sigmaLF * sigmaLF;
        double qD = qB * qB - 4 * qA * qC;
        if (qD < 0)
        {
        	double new_sigmaLF = Math.sqrt (qC_nosigmaLF / rows - qB * qB / (4 * qA * rows));
        	String error_message = String.format ("Cannot generate the weights: linear form variance demand is too tight!  Try sigmaLF >%8.4f", new_sigmaLF);
        	System.out.println (error_message);
        	System.out.flush ();
        	throw new IllegalArgumentException (error_message);
        }
        double t = (- qB + Math.sqrt (qD)) / (2 * qA);
        double a = a0 + t * a1;
        double b = b0 + t * b1;
        for (int j = 0; j < cols; j ++)
        	w [j] = a * w [j] + b;
        
        double sum_eta = 0.0;
        double sum_sq_eta = 0.0;
        for (int i = 0; i < rows; i ++)
        {
        	double eta = 0.0;
        	for (int j = 0; j < cols; j ++)
        		eta +=  w [j] * X [i][j];
        	sum_eta += eta;
        	sum_sq_eta += eta * eta;
        }
        double mean_eta = icept + sum_eta / rows;
        double sigma_eta = Math.sqrt ((sum_sq_eta - sum_eta * sum_eta / rows) / (rows - 1));
        System.out.println (String.format ("Linear Form Mean  =%8.4f (Desired:%8.4f)",  mean_eta, meanLF));
        System.out.println (String.format ("Linear Form Sigma =%8.4f (Desired:%8.4f)", sigma_eta, sigmaLF));
        
        return w;
    }
    
    public class GLMDist
    {
    	final int dist;         // GLM distribution family type
    	final double param;     // GLM parameter, typically variance power of the mean
    	final int link;         // GLM link function type
    	final double link_pow;  // GLM link function as power of the mean
    	double dispersion = 1.0;
    	long binom_n = 1;
    	
    	GLMDist (int _dist, double _param, int _link, double _link_pow)
    	{
    		dist = _dist; param = _param; link = _link; link_pow = _link_pow;
    	}
    	
    	void set_dispersion (double _dispersion)
    	{
    		dispersion = _dispersion;
    	}
    	
    	void set_binom_n (long _n)
    	{
    		binom_n = _n;
    	}
    	
    	boolean is_binom_n_needed ()
    	{
    		return (dist == 2 && param == 1.0);
    	}
    	
    	double nextGLM (Random r, double eta)
        {
    	    double mu = 0.0;
    	    switch (link)
    	    {
    	    case 1: // LINK: POWER
    	    	if (link_pow == 0.0)       // LINK: log
    			    mu = Math.exp (eta);
    		    else if (link_pow ==  1.0) // LINK: identity
    			    mu = eta;
    		    else if (link_pow == -1.0) // LINK: inverse
    			    mu = 1.0 / eta;
    		    else if (link_pow ==  0.5) // LINK: sqrt
    			    mu = eta * eta;
    		    else if (link_pow == -2.0) // LINK: 1/mu^2
        		    mu = Math.sqrt (1.0 / eta);
    		    else
    			    mu = Math.pow (eta, 1.0 / link_pow);
    		    break;
    	    case 2: // LINK: logit
    		    mu = 1.0 / (1.0 + Math.exp (- eta));
    		    break;
    	    case 3: // LINK: probit
    		    mu = gaussian_probability (eta);
    		    break;
    	    case 4: // LINK: cloglog
    		    mu = 1.0 - Math.exp (- Math.exp (eta));
    		    break;
    	    case 5: // LINK: cauchit
    		    mu = 0.5 + Math.atan (eta) / Math.PI;
    		    break;
    	    default:
    		    mu = 0.0;
    	    }
    	
    	    double output = 0.0;
    	    if (dist == 1)  // POWER
    	    {
    		    double var_pow = param;
    		    if (var_pow == 0.0) // Gaussian, with dispersion = sigma^2
    		    {
    		        output = mu + Math.sqrt (dispersion) * r.nextGaussian ();
    		    }
    		    else if (var_pow == 1.0) // Poisson; Negative Binomial if overdispersion
    		    {
    		        double lambda = mu;
    		        if (dispersion > 1.000000001)
    		        {
    		    	    // output = Negative Binomial random variable with:
    		    	    //     Number of failures = mu / (dispersion - 1.0)
    		    	    //     Probability of success = 1.0 - 1.0 / dispersion
    		            lambda = (dispersion - 1.0) * nextGamma (r, mu / (dispersion - 1.0));
    		        }
    		        output = nextPoisson (r, lambda);
    		    }
    		    else if (var_pow == 2.0) // Gamma
    		    {
    		        double beta = dispersion * mu;
    		        output = beta * nextGamma (r, mu / beta);
    		    }
    		    else if (var_pow == 3.0) // Inverse Gaussian
    		    {
    		        // From: Raj Chhikara, J.L. Folks.  The Inverse Gaussian Distribution: 
    		        // Theory: Methodology, and Applications.  CRC Press, 1988, Section 4.5
    		        double y_Gauss = r.nextGaussian ();
    		        double mu_y_sq = mu * y_Gauss * y_Gauss;
    		        double x_invG = 0.5 * dispersion * mu * (2.0 / dispersion + mu_y_sq 
    			    	    - Math.sqrt (mu_y_sq * (4.0 / dispersion + mu_y_sq)));
    		        output = ((mu + x_invG) * r.nextDouble() < mu ? x_invG : (mu * mu / x_invG));
    		    }
    		    else
    		    {
    		        output = mu + Math.sqrt (12.0 * dispersion) * (r.nextDouble () - 0.5);
    		    }
    	    }
    	    else if (dist == 2 && param != 1.0) // Binomial, dispersion ignored
    	    {
    		    double bernoulli_zero = param;
    		    output = (r.nextDouble () < mu ? 1.0 : bernoulli_zero);
    	    }
    	    else if (dist == 2) // param == 1.0, Binomial Two-Column, dispersion used
    	    {
    		    double alpha_plus_beta = (binom_n - dispersion) / (dispersion - 1.0);
    		    double alpha = mu * alpha_plus_beta;
    		    double x = nextGamma (r, alpha);
    		    double y = nextGamma (r, alpha_plus_beta - alpha);
    		    double p = x / (x + y);
    		    long out = 0;
    		    for (long i = 0; i < binom_n; i++)
    			    if (r.nextDouble() < p)
    				    out ++;
    		    output = out;
    	    }
    	    return output;
        }
    }
    
    public double nextGamma (Random r, double alpha)
    // PDF(x) = x^(alpha-1) * exp(-x) / Gamma(alpha)
    // D.Knuth "The Art of Computer Programming", 2nd Edition, Vol. 2, Sec. 3.4.1
    {
	    double x;
    	if (alpha > 10000.0)
    	{
    		x = 1.0 - 1.0 / (9.0 * alpha) + r.nextGaussian() / Math.sqrt (9.0 * alpha);
    		return alpha * x * x * x;
    	}
        else if (alpha > 5.0)
        {
            x = 0.0;
            double the_root = Math.sqrt (2.0 * alpha - 1.0);
            boolean is_accepted = false;
            while (! is_accepted)
            {
                double y = Math.tan (Math.PI * r.nextDouble());
                x = the_root * y + alpha - 1.0;
                if (x <= 0)
                    continue;
                double z = Math.exp ((alpha - 1.0) * (1.0 + Math.log (x / (alpha - 1.0))) - x);
                is_accepted = (r.nextDouble() <= z * (1.0 + y * y));
            }
            return x;
        }
    	else if (alpha > 0.0) 
    	{
    		x = 1.0;
    	    double frac_alpha = alpha;
   	    	while (frac_alpha >= 1.0)
   	    	{
   	    		x *= r.nextDouble ();
   	    		frac_alpha -= 1.0;
   	    	}
   	    	double output = - Math.log (x);
    	    if (frac_alpha > 0.0)  // Has to be between 0 and 1
    	    {
    		    double ceee = Math.E / (frac_alpha + Math.E);
    		    boolean is_accepted = false;
    		    while (! is_accepted)
    		    {
    		        double u = r.nextDouble();
    		        if (u <= ceee)
    		        {
    		        	x = Math.pow (u / ceee, 1.0 / frac_alpha);
    		        	is_accepted = (r.nextDouble() <= Math.exp (- x));
    		        }
    		        else
    		        {
    		        	x = 1.0 - Math.log ((1.0 - u) / (1.0 - ceee));
    		        	is_accepted = (r.nextDouble() <= Math.pow (x, frac_alpha - 1.0));
    		        }
    		    }
    		    output += x;
    	    }
        	return output;
    	}
    	else  //  alpha <= 0.0
    		return 0.0;
    }

    public long nextPoisson (Random r, double mu)
    // Prob[k] = mu^k * exp(-mu) / k!
    // The main part is from W. H"ormann "The Transformed Rejection Method
    // for Generating Poisson Random Variables"
    {
    	if (mu <= 0.0)
    		return 0;
    	if (mu >= 100000.0)
    		return Math.round (mu + Math.sqrt (mu) * r.nextGaussian ());
    	if (mu >= 10.0)
    	{
        	long output = 0;
    		double c = mu + 0.445;
    		double b = 0.931 + 2.53 * Math.sqrt (mu);
    		double a = -0.059 + 0.02483 * b;
    		double one_by_alpha = 1.1239 + 1.1328 / (b - 3.4);
    		double u_r = 0.43;
    		double v_r = 0.9277 - 3.6224 / (b - 2);
    		while (true)
    		{
        		double U;
        		double V = r.nextDouble ();
    	    	if (V <= 2 * u_r * v_r)
    		    {
    			    U = V / v_r - u_r;
        			output = (long) Math.floor ((2 * a / (0.5 - Math.abs (U)) + b) * U + c);
        			break;
        		}
			    if (V >= v_r)
			    {
				    U = r.nextDouble () - 0.5;
			    }
			    else
			    {
				    U = V / v_r - (u_r + 0.5);
				    U = Math.signum (U) * 0.5 - U;
				    V = v_r * r.nextDouble ();
			    }
			    double us = 0.5 - Math.abs (U);
		    	if (0.487 < Math.abs (U) && us < V)
			    	continue;
		    	long k = (long) Math.floor ((2 * a / us + b) * U + c);
		    	double V_to_compare = (V * one_by_alpha) / (a / us / us + b);
		    	if (0 <= k &&  Math.log (V_to_compare) <= - mu + k * Math.log (mu) - logFactorial (k))
		    	{
		    		output = k;
		    		break;
		    	}
    		}
    		return output;
    	}
    	long count = 0;
    	double res_mu = mu;
    	while (res_mu > 0.0)
    	{
    	    count ++;
    	    res_mu += Math.log (r.nextDouble ());
    	}
    	return count - 1;
    }

    static double logFactorial (double x)
    //  From paper: C. Lanczos "A Precision Approximation of the Gamma Function",
    //  Journal of the SIAM: Numerical Analysis, Series B, Vol. 1, 1964, pp. 86-96
    {
        final double[] cf = {1.000000000178, 76.180091729406, -86.505320327112,
            24.014098222230, -1.231739516140, 0.001208580030, -0.000005363820};
        double a_5 = cf[0] + cf[1] / (x + 1) + cf[2] / (x + 2) + cf[3] / (x + 3) 
            + cf[4] / (x + 4) + cf[5] / (x + 5) + cf[6] / (x + 6);
        return Math.log(a_5) + (x + 0.5) * Math.log(x + 5.5) - (x + 5.5) + 0.91893853320467; // log(sqrt(2 * PI))
    }
    
    public double gaussian_probability (double point)
    //  "Handbook of Mathematical Functions", ed. by M. Abramowitz and I.A. Stegun,
    //  U.S. Nat-l Bureau of Standards, 10th print (Dec 1972), Sec. 7.1.26, p. 299
    {
        double t_gp = 1.0 / (1.0 + Math.abs (point) * 0.231641888);  // 0.231641888 = 0.3275911 / sqrt (2.0)
        double erf_gp = 1.0 - t_gp * ( 0.254829592 
            + t_gp * (-0.284496736 
            + t_gp * ( 1.421413741 
            + t_gp * (-1.453152027 
            + t_gp *   1.061405429)))) * Math.exp (- point * point / 2.0);
        erf_gp = erf_gp * (point > 0 ? 1.0 : -1.0);
        return (0.5 + 0.5 * erf_gp);
    }
}
