/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.enumerate;

import java.util.HashSet;
import java.util.Set;

import com.ibm.bi.dml.hops.cost.CostEstimationWrapper;

/**
 * Configuration object for enumeration based optimization. Includes parameters like 
 * pruning techniques etc. 
 * 
 * TODO MB cleanup unnecessary information.
 */
public class OptimizerConfig 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private CostEstimationWrapper costEstimator;
	
	public enum CostingStrategy { 
		validOnly, insertDummyops, partialCosting
	};
	
	private CostingStrategy strategy; 
	/**
	 * Define the class of possible interesting properties.
	 * Instances of these classes are derived from different configurations (instances) 
	 * on specific plans.
	 */
	private Set<InterestingProperty> interestingProperties = new HashSet<InterestingProperty>();
	
	/**
	 * Defines the class of possible configuration parameters. Instances of these classes are 
	 * created at each node of a plan instance.
	 */
	private Set<ConfigParam> configurationParameters = new HashSet<ConfigParam>();
	
	public CostEstimationWrapper getCostEstimator() {
		return this.costEstimator;
	}


	public void setCostEstimator(CostEstimationWrapper costEstimator) {
		this.costEstimator = costEstimator;
	}

	public Set<InterestingProperty> getInterestingProperties() {
		return interestingProperties;
	}


	public void setInterestingProperties(
			Set<InterestingProperty> interestingProperties) {
		this.interestingProperties = interestingProperties;
	}

	public void addInterestingProperty(InterestingProperty toAdd) {
		this.interestingProperties.add(toAdd);
	}


	public Set<ConfigParam> getConfigurationParameters() {
		return configurationParameters;
	}


	public void setConfigurationParameters(Set<ConfigParam> configurationParameters) {
		this.configurationParameters = configurationParameters;
	}
	
	public void addConfigParam(ConfigParam toAdd) {
		this.configurationParameters.add(toAdd);
	}


	public CostingStrategy getStrategy() {
		return strategy;
	}


	public void setStrategy(CostingStrategy strategy) {
		this.strategy = strategy;
	}
}
