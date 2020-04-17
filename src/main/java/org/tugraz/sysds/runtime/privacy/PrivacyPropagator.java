package org.tugraz.sysds.runtime.privacy;

public class PrivacyPropagator {

    public static PrivacyConstraint MergeBinary(PrivacyConstraint privacyConstraint1, PrivacyConstraint privacyConstraint2) {
        if ( privacyConstraint1 != null && privacyConstraint2 != null) {
            if ( privacyConstraint1.getPrivacy() || privacyConstraint2.getPrivacy()){
                return new PrivacyConstraint(true);
            } else return new PrivacyConstraint(false);
        } 
        else if (privacyConstraint1 != null) return privacyConstraint1;
        else if (privacyConstraint2 != null) return privacyConstraint2; 
        else return null;
    }

}