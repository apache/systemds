package org.apache.sysml.runtime.controlprogram.util;

import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;

//TODO:
public class SPBatchIterator extends CPBatchIterator {
	public SPBatchIterator(ExecutionContext ec, String[] iterablePredicateVars) {
		super(ec, iterablePredicateVars);
	}
}
