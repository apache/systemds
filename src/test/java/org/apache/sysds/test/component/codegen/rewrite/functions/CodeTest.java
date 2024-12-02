package org.apache.sysds.test.component.codegen.rewrite.functions;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.hops.rewriter.dml.DMLExecutor;
import org.junit.Test;

public class CodeTest {
	@Test
	public void test() {
		String str = "X = rand(rows=5000, cols=5000, sparsity=0.1)\n" +
				"Y = rand(rows=5000, cols=5000, sparsity=0.1)\n" +
				"R = X*Y\n" +
				"print(lineage(R))";
		DMLScript.APPLY_GENERATED_REWRITES = true;
		DMLExecutor.executeCode(str, false, "-applyGeneratedRewrites");
	}
}
