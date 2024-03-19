package org.apache.sysds.api.ropt;

import java.util.EnumSet;
import java.util.Set;

public class EMRUtils {
    // NOTE: does NOT match additional capabilities: '-flex' and 'q'
    public static final String EC2_REGEX = "^([a-z]+)([0-9])(a|g|i?)([bdnez]*)\\.([a-z0-9]*)$";

    public enum InstanceFamilyId {
        GENERAL_CURRENT_GEN,
        COMPUTE_CURRENT_GEN,
        GPU_CURRENT_GEN,
        HI_MEM_CURRENT_GEN,
        STORAGE_CURRENT_GEN,
        GENERAL_PREVIOUS_GEN,
        COMPUTE_PREVIOUS_GEN,
        HI_MEM_PREVIOUS_GEN,
        STORAGE_PREVIOUS_GEN
    }

    public static Set<InstanceFamilyId> setOfCurrentGenEC2() {
        return EnumSet.of(
                InstanceFamilyId.GENERAL_CURRENT_GEN,
                InstanceFamilyId.COMPUTE_CURRENT_GEN,
                InstanceFamilyId.GPU_CURRENT_GEN,
                InstanceFamilyId.HI_MEM_CURRENT_GEN,
                InstanceFamilyId.STORAGE_CURRENT_GEN
        );
    }
}
