/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.scalability;

import org.junit.Test;

import com.ibm.bi.dml.test.integration.AutomatedScalabilityTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;



public class PageRankTest extends AutomatedScalabilityTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
    @Override
    public void setUp() {
        baseDirectory = "test/scripts/scalability/page_rank/";
        availableTestConfigurations.put("PageRankTest", new TestConfiguration("PageRankTest", new String[] { "p" }));
        matrixSizes = new int[][] {
                { 9914 }
        };
    }
    
    @Test
    public void testPageRank() {
    	loadTestConfiguration("PageRankTest");
        
        addInputMatrix("g", -1, -1, 1, 1, 0.000374962, -1).setRowsIndexInMatrixSizes(0).setColsIndexInMatrixSizes(0);
        addInputMatrix("p", -1, 1, 1, 1, 1, -1).setRowsIndexInMatrixSizes(0);
        addInputMatrix("e", -1, 1, 1, 1, 1, -1).setRowsIndexInMatrixSizes(0);
        addInputMatrix("u", 1, -1, 1, 1, 1, -1).setColsIndexInMatrixSizes(0);
        
        runTest();
    }

}
