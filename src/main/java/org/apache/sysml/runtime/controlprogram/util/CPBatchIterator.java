package org.apache.sysml.runtime.controlprogram.util;

import java.util.Iterator;

import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.cp.Data;

// TODO: Before I go ahead an implement prefetching with this approach
// I want to ensure if this functionality is OK with everyone
public class CPBatchIterator implements Iterator<Data>, Iterable<Data> {
	
	public CPBatchIterator(ExecutionContext ec, String[] iterablePredicateVars) {
		
	}

	@Override
	public Iterator<Data> iterator() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public boolean hasNext() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public MatrixObject next() {
		// TODO Auto-generated method stub
		return null;
	}

}
