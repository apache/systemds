This document give a breaf instruction for developing SystemML project using an IDE. 


# Import Systemml Project to Eclipse
Eclipse IDE include:
* Scala IDE [http://scala-ide.org/]
* Eclipse Juno with scala plug-in

 File -> Import -> Maven -> Existing Maven Projects

There are serval tips to resolve below compiler error:
* `invalid cross-compiled libraries` error
Since Scala IDE bundles the latest versions (2.10.5 and 2.11.6 at this point), you need do add one  in Eclipse Preferences -> Scala -> Installations by pointing to the lib/ directory of your Scala 2.10.4 distribution. Once this is done, select all Spark projects and right-click, choose Scala -> Set Scala Installation and point to the 2.10.4 installation. This should clear all errors about invalid cross-compiled libraries. A clean build should succeed now.

* `incompatation scala version` error
Change IDE scala version `project->propertiest->scala compiler -> scala installation` to `Fixed scala Installation: 2.10.5`

* `Not found type * ` error.
Run command `mvn package`, and do `project -> refresh`

* `maketplace not found` error for Eclipse Luna
Except scala IDE pulgin install, please make sure get update from "http://alchim31.free.fr/m2e-scala/update-site" to update maven connector for scala.

# Import SystemML project to IntelliJ

 1. Download IntelliJ and install the Scala plug-in for IntelliJ.
 2. Go to "File -> Import Project", locate the spark source directory, and select "Maven Project".
 3. In the Import wizard, it's fine to leave settings at their default. However it is usually useful to enable "Import Maven projects automatically", since changes to the project structure will automatically update the IntelliJ project.