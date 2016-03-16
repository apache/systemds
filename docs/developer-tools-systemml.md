---
layout: global
displayTitle: SystemML Developer Tools
title: SystemML Developer Tools
description: SystemML Developer Tools
---
<!--
{% comment %}
Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements.  See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to you under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
{% endcomment %}
-->

Useful Tools for Developing SystemML:

* This will become a table of contents (this text will be scraped).
{:toc}


## Eclipse

Eclipse [Luna SR2](https://eclipse.org/downloads/packages/release/luna/sr2) can be used for an integrated development development environment with SystemML code.  Maven integration is required which is included in the [Eclipse IDE for Java Developers](https://eclipse.org/downloads/packages/eclipse-ide-java-developers/lunasr2) package.

To get started in Eclipse, import SystemML's pom.xml file as an existing Maven project.  After import is completed, the resulting Eclipse installation should include two maven connectors.

![About Eclipse](img/developer-tools/about-eclipse.png "About Eclipse")

![Eclipse Details](img/developer-tools/eclipse-details.png "Eclipse Details")


## Eclipse with Scala

An additional Maven connector is required for working with Scala code in Eclipse.  The [Maven Integration for Scala IDE](http://scala-ide.org/docs/tutorials/m2eclipse/) plugin can be installed into Eclipse from [this](http://alchim31.free.fr/m2e-scala/update-site/) update site.  

![Maven Scala Integration](img/developer-tools/maven-scala.png "Maven Scala Integration")

![Installed Maven Connectors](img/developer-tools/maven-connectors.png "Installed Maven Connectors")
<br></br>
The [Scala IDE 4.0.0 plugin for Eclipse](http://scala-ide.org/download/prev-stable.html) is known to work for mixed Java and Scala development in [Eclipse IDE for Java Developers Luna SR2](https://eclipse.org/downloads/packages/eclipse-ide-java-developers/lunasr2).  <b>Release 4.0.0</b> of the Scala IDE for Eclipse plugin can be downloaded [here](http://download.scala-ide.org/sdk/lithium/e44/scala211/stable/site_assembly-20150305-1905.zip) and then installed through Eclipse's 'Install New Software' menu item.  Only the Scala IDE and ScalaTest components are needed.

![Scala IDE Components](img/developer-tools/scala-components.png "Scala IDE Components")

![Eclipse Scala Details](img/developer-tools/eclipse-scala.png "Eclipse Scala Details")

Note the corresponding Eclipse project needs to include the Scala nature.  Typically this occurs automatically during project import but also can be invoked directly by right clicking on the systemml project node in the Eclipse Package Explorer view.

![Add Scala Nature](img/developer-tools/scala-nature.png "Add Scala Nature")


## Eclipse Java Only (How to skip Scala)

Since the core SystemML code is written in Java, developers may prefer not to use Eclipse in a mixed Java/Scala environment.  To configure Eclipse to skip the Scala code of SystemML and avoid installing any Scala-related components, Maven lifecycle mappings can be created.  The simplest way to create these mappings is to use Eclipse's quick fix option to resolve the following pom.xml errors which occur if Maven Integration for Scala is not present.

![Scala pom errors](img/developer-tools/pom-scala-errors.png "Scala pom errors")

![Scala pom ignore](img/developer-tools/pom-scala-ignore.png "Scala pom Quick Fix")

The lifecycle mappings are stored in a workspace metadata file as specified in Eclipse's Maven Lifecycle Mappings Preferences page.  The pom.xml file itself is unchanged which allows the Scala portion to be built outside of Eclipse using mvn command line.


## IntelliJ

IntelliJ can also be used since it provides good support for mixed Java and Scala projects as described [here](https://cwiki.apache.org/confluence/display/SPARK/Useful+Developer+Tools#UsefulDeveloperTools-IntelliJ).
