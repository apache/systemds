/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.gdfo;


import org.junit.Assert;
import org.junit.Test;

import com.ibm.bi.dml.hops.globalopt.InterestingProperties;
import com.ibm.bi.dml.hops.globalopt.InterestingProperties.Format;
import com.ibm.bi.dml.hops.globalopt.InterestingProperties.Location;
import com.ibm.bi.dml.hops.globalopt.InterestingProperties.Partitioning;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.utils.TestUtils;

public class HashInterestingPropertiesTest extends AutomatedTestBase
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
	}

	@Test
	public void testCopyEquivalence()
	{
		InterestingProperties ips1 = new InterestingProperties(1000, Format.BINARY_BLOCK, Location.MEM, Partitioning.NONE, 1, true);
		InterestingProperties ips2 = new InterestingProperties(ips1);
		InterestingProperties ips3 = new InterestingProperties(ips2);
		
		//test hash functions
		Assert.assertEquals(ips1.hashCode(), ips2.hashCode());
		Assert.assertEquals(ips2.hashCode(), ips3.hashCode());
		
		//test equals (used inside hash maps)
		Assert.assertTrue(ips1.equals(ips2));
		Assert.assertTrue(ips2.equals(ips3));
	}
	
	@Test
	public void testValueEquivalence()
	{
		InterestingProperties ips1 = new InterestingProperties(1000, Format.BINARY_BLOCK, Location.MEM, Partitioning.NONE, 1, true);
		InterestingProperties ips2 = new InterestingProperties(1000, Format.BINARY_BLOCK, Location.MEM, Partitioning.NONE, 1, true);
		InterestingProperties ips3 = new InterestingProperties(2000, Format.BINARY_BLOCK, Location.MEM, Partitioning.NONE, 1, true);
		InterestingProperties ips4 = new InterestingProperties(1000, Format.TEXT_CELL, Location.MEM, Partitioning.NONE, 1, true);
		InterestingProperties ips5 = new InterestingProperties(1000, Format.BINARY_BLOCK, Location.HDFS, Partitioning.NONE, 1, true);
		InterestingProperties ips6 = new InterestingProperties(1000, Format.BINARY_BLOCK, Location.MEM, Partitioning.COL_WISE, 1, true);
		InterestingProperties ips7 = new InterestingProperties(1000, Format.BINARY_BLOCK, Location.MEM, Partitioning.NONE, 1, false);
		
		//test hash functions
		Assert.assertEquals(ips1.hashCode(), ips2.hashCode());
		
		//test equals (used inside hash maps)
		Assert.assertTrue(ips1.equals(ips2));
		Assert.assertFalse(ips2.equals(ips3));
		Assert.assertFalse(ips2.equals(ips4));
		Assert.assertFalse(ips2.equals(ips5));
		Assert.assertFalse(ips2.equals(ips6));
		Assert.assertFalse(ips2.equals(ips7));
	}
	
	@Test
	public void testValueEquivalenceWithNull()
	{
		InterestingProperties ips1 = new InterestingProperties(1000, Format.BINARY_BLOCK, Location.MEM, Partitioning.NONE, 1, true);
		InterestingProperties ips2 = new InterestingProperties(1000, null, Location.MEM, Partitioning.NONE, 1, true);
		InterestingProperties ips3 = new InterestingProperties(1000, Format.BINARY_BLOCK, null, Partitioning.NONE, 1, true);
		InterestingProperties ips4 = new InterestingProperties(1000, Format.BINARY_BLOCK, Location.MEM, null, 1, true);
		InterestingProperties ips5 = new InterestingProperties(1000, Format.ANY, Location.MEM, Partitioning.NONE, 1, true);
		
		//test equals (used inside hash maps)
		Assert.assertFalse(ips1.equals(ips2));
		Assert.assertFalse(ips1.equals(ips3));
		Assert.assertFalse(ips1.equals(ips4));
		Assert.assertFalse(ips1.equals(ips5));
	}
}
