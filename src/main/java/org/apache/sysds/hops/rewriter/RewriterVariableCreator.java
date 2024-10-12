package org.apache.sysds.hops.rewriter;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class RewriterVariableCreator {
	private Map<String, List<String>> defs = new HashMap<>();
	private Map<String, Integer> defCtrs = new HashMap<>();

	public void newVar(String type, String name) {
		Integer ctr = defCtrs.get(name);
	}
}
