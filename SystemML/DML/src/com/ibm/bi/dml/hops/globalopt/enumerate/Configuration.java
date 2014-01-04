/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.enumerate;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import com.ibm.bi.dml.hops.Hop;


public class Configuration 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private Map<String, ConfigParam> parameters = new HashMap<String, ConfigParam>();
	
	public Map<String, ConfigParam> getParameters() {
		return parameters;
	}

	public void setParameters(Set<ConfigParam> parameters) {
		for(ConfigParam p : parameters) {
			this.parameters.put(p.getName(), p);
		}
	}
	
	public void setParameters(Map<String, ConfigParam> parameters) {
		this.parameters = parameters;
	}
	
	public void addParam(ConfigParam toAdd) {
		this.parameters.put(toAdd.getName(), toAdd);
	}
	
	public ConfigParam getParamByName(String name) {
		return this.parameters.get(name);
	}
	
	/**
	 * Create a new configuration by adding an additional parameter to the set.
	 * @param toAdd
	 * @return
	 */
	public Configuration generateConfig(ConfigParam toAdd) {
		Configuration copy = new Configuration();
		Map<String, ConfigParam> paramCopy = new HashMap<String, ConfigParam>(this.parameters);
		copy.setParameters(paramCopy);
		copy.addParam(toAdd);
		return copy;
	}

	/**
	 * TODO: consolidate logic
	 * @param hop
	 * @param plan
	 */
	public void generateRewrites(Hop hop, OptimizedPlan plan) {
		ConfigParam blockSize = this.getParamByName(BlockSizeParam.NAME);
		Rewrite reblockRewrite = plan.getRewrite(BlockSizeParam.NAME);
		
		ConfigParam locationParam = this.getParamByName(LocationParam.NAME);
		LocationRewrite locationRewrite = new LocationRewrite();
		locationRewrite.setExecLocation(locationParam.getValue());
		plan.addRewrite(LocationParam.NAME, locationRewrite);
		
		ConfigParam formatParam = this.getParamByName(FormatParam.NAME);
		
		//in this case no reblock is required but just setting the current block size
		if(reblockRewrite == null) {
			BlockSizeRewrite rewrite = new BlockSizeRewrite();
			rewrite.setToBlockSize(blockSize.getValue());
			plan.addRewrite("bs", rewrite);
			
			ReblockRewrite formatRewrite = new ReblockRewrite();
			formatRewrite.setFormat((FormatParam) formatParam);
			formatRewrite.setToBlockSize(blockSize.getValue());
			plan.addRewrite(FormatParam.NAME, formatRewrite);
			
		}else {
			((ReblockRewrite)reblockRewrite).setFormat((FormatParam) formatParam);
		}
		
	}
	
	public void applyToHop(Hop operator) {
		for(ConfigParam p : parameters.values()) {
			p.applyToHop(operator);
		}
	}
	
	@Override
	public String toString() {
		StringBuilder buffer = new StringBuilder();
		buffer.append("[");
		for(ConfigParam p : parameters.values())
			buffer.append(p.getValueString());
		buffer.append("]");
		return buffer.toString();
	}

	public boolean isValidForOperator(Hop operator) {
		for(ConfigParam p : parameters.values()) {
			if(!p.isValidForOperator(operator))
				return false;
		}
		
		return true;
	}
	
}
