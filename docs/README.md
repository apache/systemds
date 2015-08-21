# SystemML Documentation

The primary SystemML documentation is written in markdown (*.md) and can be converted to HTML using
[Jekyll](http://jekyllrb.com).

Jekyll (and optionally Pygments) can be installed on the Mac OS in the following manner.

    $ brew install ruby
    $ gem install jekyll
    $ gem install jekyll-redirect-from
    $ brew install python
    $ pip install Pygments

For installation on other platforms, please consult the Jekyll documentation.

To generate SystemML documentation in HTML, navigate to the ```docs``` folder, the root directory of the
documentation. From there, you can have Jekyll convert the markdown files to HTML. If you run in server mode,
Jekyll will serve up the generated documentation by default at http://127.0.0.1:4000. Modifications
to *.md files will be converted to HTML and can be viewed in a web browser.

    $ jekyll serve -w