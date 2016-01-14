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

# Apache SystemML Documentation

The primary SystemML documentation is written in markdown (*.md) and can be converted to HTML using
[Jekyll](http://jekyllrb.com).

Jekyll (and optionally Pygments) can be installed on the Mac OS in the following manner.

    $ brew install ruby
    $ gem install jekyll
    $ gem install jekyll-redirect-from
    $ brew install python
    $ pip install Pygments
    $ gem install pygments.rb

For installation on other platforms, please consult the Jekyll documentation.

To generate SystemML documentation in HTML, navigate to the ```docs``` folder, the root directory of the
documentation. From there, you can have Jekyll convert the markdown files to HTML. If you run in server mode,
Jekyll will serve up the generated documentation by default at http://127.0.0.1:4000. Modifications
to *.md files will be converted to HTML and can be viewed in a web browser.

    $ jekyll serve -w