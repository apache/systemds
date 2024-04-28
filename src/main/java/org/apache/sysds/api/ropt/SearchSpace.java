package org.apache.sysds.api.ropt;

import java.util.EnumSet;
import java.util.HashMap;
import java.util.Set;

public class SearchSpace {

    public enum InstanceType {
        M5, M5a, M6i, M6a, M6g, M7i, M7a, M7g, // general purpose - vCores:mem~=1:4
        C5, C5a, C6i, C6a, C6g, C7i, C7a, C7g, // compute optimized - vCores:mem~=1:2
        R5, R5a, R6i, R6a, R6g, R7i, R7a, R7g; // memory optimized - vCores:mem~=1:8
    }

    public enum InstanceSize {
        _XLARGE, _2XLARGE, _4XLARGE, _8XLARGE, _12XLARGE, _16XLARGE, _24XLARGE, _32XLARGE, _48XLARGE
    }

    private static final String EC2_REGEX = "^([a-z]+)([0-9])(a|g|i?)\\.([a-z0-9]*)$";
    public static final int MIN_EXECUTORS = 0; // allow single node configuration
    public static final int MAX_EXECUTORS = 10; // TODO: think of reasonable max number

    public static class SearchPoint {
        private final InstanceType instanceTypeExecutor;
        private final InstanceSize instanceSizeExecutor;
        private final InstanceType instanceTypeDriver;
        private final InstanceSize instanceSizeDriver;
        private final int numberExecutors;

        public SearchPoint(InstanceType instanceTypeExecutor, InstanceSize instanceSizeExecutor,
                           InstanceType instanceTypeDriver, InstanceSize instanceSizeDriver, int numberExecutors) {
            this.instanceTypeExecutor = instanceTypeExecutor;
            this.instanceSizeExecutor = instanceSizeExecutor;
            this.instanceTypeDriver = instanceTypeDriver;
            this.instanceSizeDriver = instanceSizeDriver;
            this.numberExecutors = numberExecutors;
        }

        public InstanceType getInstanceTypeExecutor() {
            return instanceTypeExecutor;
        }

        public InstanceSize getInstanceSizeExecutor() {
            return instanceSizeExecutor;
        }

        public InstanceType getInstanceTypeDriver() {
            return instanceTypeDriver;
        }

        public InstanceSize getInstanceSizeDriver() {
            return instanceSizeDriver;
        }

        public int getNumberExecutors() {
            return numberExecutors;
        }

        public String getInstanceNameExecutor() {
            if (numberExecutors == 0 || instanceTypeExecutor == null || instanceSizeExecutor == null) {
                return "";
            }
            return SearchSpace.getInstanceName(instanceTypeExecutor, instanceSizeExecutor);
        }

        public String getInstanceNameDriver() {
            return SearchSpace.getInstanceName(instanceTypeDriver, instanceSizeDriver);
        }

        @Override
        public String toString() {
            return "Driver: " + getInstanceNameDriver() + "; Executors: "
                    + numberExecutors + " " + getInstanceNameExecutor();
        }
    }

    // NOTE(1): Keeping separate domain sets for executors and driver instances
    //  as defining different range limits, despite the generally available
    //  instance types, is possible (and maybe desired: e.g. GPU types for driver)

    // NOTE(2): Using 'EnumSet' and having the enums declared in an ascending oder
    //  ensures that an iteration over the set always follows the ascending order

    private Set<InstanceType> _instanceTypeDomainExecutors;
    private HashMap<InstanceType, Set<InstanceSize>> _instanceSizeDomainExecutors;
    private Set<InstanceType> _instanceTypeDomainDriver;
    private HashMap<InstanceType, Set<InstanceSize>> _instanceSizeDomainDriver;
    //private final int[] _executorRange; NOTE: Probably obsolete
    public SearchSpace(HashMap<String, CloudInstance> availableInstances) {
        // init data structures
        _instanceTypeDomainExecutors = EnumSet.noneOf(InstanceType.class);
        _instanceSizeDomainExecutors = new HashMap<>();
        _instanceTypeDomainDriver = EnumSet.noneOf(InstanceType.class);
        _instanceSizeDomainDriver = new HashMap<>();
        //_executorRange = new int[]{MIN_EXECUTORS, MAX_EXECUTORS};

        for (CloudInstance ci: availableInstances.values()) {
            InstanceType type = getTypeEnum(ci.getInstanceType());
            // ignore all instances which type is not defined in 'InstanceType'
            if (type == null)
                continue;
            InstanceSize size = getSizeEnum(ci.getInstanceSize());
            // ignore all instances which size is not defined in 'InstanceType'
            if (size == null)
                continue;
            // add to the domain of the executor instances
            // TODO: add a condition
            _instanceTypeDomainExecutors.add(type);
            if (!_instanceSizeDomainExecutors.containsKey(type)) {
                _instanceSizeDomainExecutors.put(type, EnumSet.of(size));
            } else {
                _instanceSizeDomainExecutors.get(type).add(size);
            }
            // add to the domain of the driver instance
            // TODO: add a condition, e.g ic.getGPUs() != 0
            _instanceTypeDomainDriver.add(type);
            if (!_instanceSizeDomainDriver.containsKey(type)) {
                _instanceSizeDomainDriver.put(type, EnumSet.of(size));
            } else {
                _instanceSizeDomainDriver.get(type).add(size);
            }
        }
    }

    public Set<InstanceType> getInstanceTypeDomainExecutors() {
        return _instanceTypeDomainExecutors;
    }

    public HashMap<InstanceType, Set<InstanceSize>> getInstanceSizeDomainExecutors() {
        return _instanceSizeDomainExecutors;
    }

    public Set<InstanceType> getInstanceTypeDomainDriver() {
        return _instanceTypeDomainDriver;
    }

    public HashMap<InstanceType, Set<InstanceSize>> getInstanceSizeDomainDriver() {
        return _instanceSizeDomainDriver;
    }

    public static InstanceType getTypeEnum(String s) {
        try {
            return InstanceType.valueOf(s.substring(0,1).toUpperCase()+s.substring(1));
        } catch (IllegalArgumentException e) {
            return null;
        }
    }

    public static InstanceSize getSizeEnum(String s) {
        try {
            return InstanceSize.valueOf("_" + s.toUpperCase());
        } catch (IllegalArgumentException e) {
            return null;
        }
    }

    public static String getInstanceName(InstanceType type, InstanceSize size) {
        return (type.toString()+"."+size.toString().substring(1)).toLowerCase();
    }
}
