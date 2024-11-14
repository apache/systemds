package org.apache.sysds.hops.rewriter;

public class RewriterAlphabetEncoder {

	public static int[] fromBaseNNumber(int l, int n) {
		if (l == 0)
			return new int[] { 0 };

		int numDigits = (int)(Math.log(l) / Math.log(n)) + 1;
		int[] digits = new int[numDigits];

		for (int i = numDigits - 1; i >= 0; i--) {
			digits[i] = l % n;
			l = l / n;
		}

		return digits;
	}

	public static int toBaseNNumber(int[] digits, int n) {
		if (digits.length == 0)
			throw new IllegalArgumentException();

		int multiplicator = 1;
		int out = 0;

		for (int i = digits.length - 1; i >= 0; i--) {
			out += multiplicator * digits[i];
			multiplicator *= n;
		}

		return out;
	}
}
