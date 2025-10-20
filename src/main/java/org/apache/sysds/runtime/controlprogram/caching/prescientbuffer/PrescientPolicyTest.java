package org.apache.sysds.runtime.controlprogram.caching.prescientbuffer;

import org.junit.Test;

import java.util.HashSet;
import java.util.Set;

import static org.junit.Assert.*;

public class PrescientPolicyTest {

	@Test
	public void testBasicEviction() {
		PrescientPolicy policy = new PrescientPolicy();

		policy.setAccessTime("block1", 10);
		policy.setAccessTime("block2", 40);
		policy.setAccessTime("block3", 25);

		Set<String> candidates = new HashSet<>();
		assertNull(policy.selectBlockForEviction(candidates));

		candidates.add("block1");
		candidates.add("block2");
		candidates.add("block3");
		assertEquals("block2", policy.selectBlockForEviction(candidates));

	}
}
