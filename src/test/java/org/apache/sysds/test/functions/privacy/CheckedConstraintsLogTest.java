package org.apache.sysds.test.functions.privacy;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;

import org.apache.sysds.runtime.privacy.CheckedConstraintsLog;
import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;
import org.apache.sysds.test.AutomatedTestBase;
import org.junit.Test;

@net.jcip.annotations.NotThreadSafe
public class CheckedConstraintsLogTest extends AutomatedTestBase {

	@Override
	public void setUp() {
		CheckedConstraintsLog.getCheckedConstraints().clear();
	}

	@Test
	public void addCheckedConstraintsNull(){
		CheckedConstraintsLog.addCheckedConstraints(null);
		assert(CheckedConstraintsLog.getCheckedConstraints() != null && CheckedConstraintsLog.getCheckedConstraints().isEmpty());
	}
	
	@Test
	public void addCheckedConstraintsEmpty(){
		ConcurrentHashMap<PrivacyLevel,LongAdder> checked = new ConcurrentHashMap<>();
		CheckedConstraintsLog.addCheckedConstraints(checked);
		assert(CheckedConstraintsLog.getCheckedConstraints() != null && CheckedConstraintsLog.getCheckedConstraints().isEmpty());
	}

	@Test
	public void addCheckedConstraintsSingleValue(){
		ConcurrentHashMap<PrivacyLevel,LongAdder> checked = getMap(PrivacyLevel.Private, 300);
		CheckedConstraintsLog.addCheckedConstraints(checked);
		assert(CheckedConstraintsLog.getCheckedConstraints().get(PrivacyLevel.Private).longValue() == 300);
	}

	@Test
	public void addCheckedConstraintsTwoValues(){
		ConcurrentHashMap<PrivacyLevel,LongAdder> checked = getMap(PrivacyLevel.Private, 300);
		CheckedConstraintsLog.addCheckedConstraints(checked);
		ConcurrentHashMap<PrivacyLevel,LongAdder> checked2 = getMap(PrivacyLevel.Private, 150);
		CheckedConstraintsLog.addCheckedConstraints(checked2);
		assert(CheckedConstraintsLog.getCheckedConstraints().get(PrivacyLevel.Private).longValue() == 450);
	}

	@Test
	public void addCheckedConstraintsMultipleValues(){
		ConcurrentHashMap<PrivacyLevel,LongAdder> checked = getMap(PrivacyLevel.Private, 300);
		CheckedConstraintsLog.addCheckedConstraints(checked);
		ConcurrentHashMap<PrivacyLevel,LongAdder> checked2 = getMap(PrivacyLevel.Private, 150);
		CheckedConstraintsLog.addCheckedConstraints(checked2);
		ConcurrentHashMap<PrivacyLevel,LongAdder> checked3 = getMap(PrivacyLevel.PrivateAggregation, 150);
		CheckedConstraintsLog.addCheckedConstraints(checked3);
		assert(CheckedConstraintsLog.getCheckedConstraints().get(PrivacyLevel.Private).longValue() == 450 
		    && CheckedConstraintsLog.getCheckedConstraints().get(PrivacyLevel.PrivateAggregation).longValue() == 150);
	}

	private ConcurrentHashMap<PrivacyLevel,LongAdder> getMap(PrivacyLevel level, long value){
		ConcurrentHashMap<PrivacyLevel,LongAdder> checked = new ConcurrentHashMap<>();
		LongAdder valueAdder = new LongAdder();
		valueAdder.add(value);
		checked.put(level, valueAdder);
		return checked;
	}
}