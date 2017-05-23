package org.apache.sysml.test.integration.gpu;

import org.apache.sysml.test.gpu.*;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

@RunWith(Suite.class) @SuiteClasses({
		ElementWiseOpTests.class,
		ReorgOpTests.class,
		AggregateUnaryOpTests.class,
		UnaryOpTests.class,
		MatrixMultiplicationOpTest.class,
})
public class ZPackageSuite {

}
