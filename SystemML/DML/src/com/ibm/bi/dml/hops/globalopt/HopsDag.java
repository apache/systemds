/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;

import com.ibm.bi.dml.hops.DataGenOp;
import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.FunctionOp;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.Hop.DataOpTypes;
import com.ibm.bi.dml.hops.globalopt.transform.HopsMetaData;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;

/**
 * Visitor that captures all the inputs and outputs of a dag of connected {@link Hops}.
 * Is needed to create a maximal global graph.
 * The implementation of the plan represntation of a global data flow
 */
public class HopsDag implements HopsVisitor 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private Map<String, Hop> dagInputs = new HashMap<String, Hop>();
	private Map<String, Hop> dagOutputs = new HashMap<String, Hop>();
	private Set<String> stitched = new HashSet<String>();
	private Map<String, Set<Hop>> hopsDirectory = new HashMap<String, Set<Hop>>();
	private Map<Hop, HopsMetaData> metaDataDirectory = new HashMap<Hop, HopsMetaData>();
	
	private List<Hop> originalRootHops = new ArrayList<Hop>();
	
	private Map<Hop, ProgramBlock> hopsToBlocks = new HashMap<Hop, ProgramBlock>();
	
	@Override
	public Flag preVisit(Hop hops) {
		if(!this.originalRootHops.contains(hops)) {
			this.originalRootHops.add(hops);
		}
		
		if(this.hopsDirectory.get(hops.getClass().getCanonicalName()) == null) {
			Set<Hop> hopsSet = new HashSet<Hop>();
			this.hopsDirectory.put(hops.getClass().getCanonicalName(), hopsSet);
		}
		this.hopsDirectory.get(hops.getClass().getCanonicalName()).add(hops);
		
		if(hops instanceof DataOp)
		{
			DataOp dataOp = (DataOp)hops;
			DataOpTypes dataType = dataOp.get_dataop();
			if(dataType == DataOpTypes.TRANSIENTREAD || dataType == DataOpTypes.PERSISTENTREAD) {
				this.dagInputs.put(dataOp.get_name(), dataOp);
			}
			
			if(dataType == DataOpTypes.TRANSIENTWRITE || dataType == DataOpTypes.PERSISTENTWRITE) {
				this.dagOutputs.put(dataOp.get_name(), dataOp);
			}
		}
		if(hops instanceof DataGenOp) {
			this.dagInputs.put(hops.get_name(), hops);
		}
		
		if(hops instanceof FunctionOp)
		{
			String[] funcOutputs = ((FunctionOp)hops).getOutputVariableNames(); //MB
			//String[] funcOutputs = ((FunctionOp)hops).getOutputs().toArray(new String[0]);
			if(funcOutputs != null){
				for(String fo : funcOutputs){
					this.dagOutputs.put(fo, hops);
				}
			}
		}
		return Flag.GO_ON;
	}
	
	@Override
	public Flag postVisit(Hop hops) {
		return Flag.GO_ON;
	}
	
	@Override
	public Flag visit(Hop hops) {
		return Flag.GO_ON;
	}
	

	public Map<String, Hop> getDagInputs() {
		return dagInputs;
	}

	public Map<String, Hop> getDagOutputs() {
		return dagOutputs;
	}
	
	public boolean isStitched(String name) {
		return this.stitched.contains(name);
	}
	
	public void addStitched(String stitchedVar) {
		this.stitched.add(stitchedVar);
	}
	
	public void clearStitched() {
		this.stitched.clear();
	}

	public Map<String, Set<Hop>> getHopsDirectory() {
		return hopsDirectory;
	}
	
	public void copyDirectory(Map<String, Set<Hop>> copyFrom) {
		for(Entry<String, Set<Hop>> e: copyFrom.entrySet()) {
			if(this.hopsDirectory.get(e.getKey()) == null){
				this.hopsDirectory.put(e.getKey(), e.getValue());
			}else {
				this.hopsDirectory.get(e.getKey()).addAll(e.getValue());
			}
		}
	}

	@Override
	public boolean matchesPattern(Hop hops) {
		if(hops instanceof CrossBlockOp) 
		{
			return false;
		}
		return true;
	}

	public Map<Hop, HopsMetaData> getMetaDataDirectory() {
		return metaDataDirectory;
	}

	public void setMetaDataDirectory(Map<Hop, HopsMetaData> metaDataDirectory) {
		this.metaDataDirectory = metaDataDirectory;
	}

	@Override
	public boolean traverseBackwards() {
		return false;
	}

	public void addOriginalRootHops(Hop toAdd) {
		this.originalRootHops.add(toAdd);
	}
	
	public void addOriginalRootHops(List<Hop> toAdd) {
		this.originalRootHops.addAll(toAdd);
	}
	
	public List<Hop> getOriginalRootHops() {
		return originalRootHops;
	}

	public void setOriginalRootHops(List<Hop> originalRootHops) {
		this.originalRootHops = originalRootHops;
	}

	public void putHopsToBlock(Hop hop, ProgramBlock block) {
		this.hopsToBlocks.put(hop, block);
	}
	
	public ProgramBlock getBlockForHop(Hop hop) {
		return this.hopsToBlocks.get(hop);
	}
	
	public boolean containsBlockForHop(Hop hop) {
		return this.hopsToBlocks.containsKey(hop);
	}

	public Map<Hop, ProgramBlock> getHopsToBlocks() {
		return hopsToBlocks;
	}

	public void copyHopsToBlocks(Map<Hop, ProgramBlock> hopsToBlocks2) {
		this.hopsToBlocks.putAll(hopsToBlocks2);
	}

	public void linkOutputsWithProgramBlock(ProgramBlock block) {
		for(Hop hops : this.getDagOutputs().values()) {
			this.hopsToBlocks.put(hops, block);
		}
	}
	
}
