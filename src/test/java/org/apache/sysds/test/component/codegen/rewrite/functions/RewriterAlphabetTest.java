package org.apache.sysds.test.component.codegen.rewrite.functions;

import org.apache.sysds.hops.rewriter.RewriterAlphabetEncoder;
import org.junit.Test;

public class RewriterAlphabetTest {

	@Test
	public void testDecode1() {
		int l = 27;
		int n = 5;
		int[] digits = RewriterAlphabetEncoder.fromBaseNNumber(l, n);
		assert digits.length == 3 && digits[0] == 1 && digits[1] == 0 && digits[2] == 2;
	}

	@Test
	public void testEncode1() {
		int[] digits = new int[] { 1, 0, 2 };
		int n = 5;
		int l = RewriterAlphabetEncoder.toBaseNNumber(digits, n);
		assert l == 27;
	}

}
