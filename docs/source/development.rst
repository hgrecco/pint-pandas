.. development:

Development
===========

Releasing
---------

The steps to release a new version of pint-pandas are shown below.
Please do all the steps below and all the steps for both release platforms.

First step
~~~~~~~~~~

#. Test installation with dependencies ``make test-install``
#. Update ``CHANGELOG.rst``:

    - add a header for the new version between ``master`` and the latest bullet point
    - this should leave the section underneath the master header empty

#. ``git add .``
#. ``git commit -m "Prepare for release of vX.Y.Z"``
#. ``git tag vX.Y.Z``
#. Test version updated as intended with ``make test-install``

PyPI
~~~~

If uploading to PyPI, do the following (otherwise skip these steps)

#. ``make publish-on-testpypi``
#. Go to `test PyPI <https://test.pypi.org/project/pint-pandas/>`_ and check that the new release is as intended. If it isn't, stop and debug.

Assuming test PyPI worked, now upload to the main repository

#. ``make publish-on-pypi``
#. Go to `pint-pandas's PyPI`_ and check that the new release is as intended.
#. Test the install with ``make test-pypi-install``.

Push to repository
~~~~~~~~~~~~~~~~~~

Finally, push the tags and the repository

#. ``git push``
#. ``git push --tags``

Conda
~~~~~

#. If you haven't already, fork the `pint-pandas conda feedstock`_. In your fork, add the feedstock upstream with ``git remote add upstream https://github.com/conda-forge/pint-pandas-feedstock`` (``upstream`` should now appear in the output of ``git remote -v``)
#. Update your fork's master to the upstream master with:

    #. ``git checkout master``
    #. ``git fetch upstream``
    #. ``git reset --hard upstream/master``

#. Create a new branch in the feedstock for the version you want to bump to.
#. Edit ``recipe/meta.yaml`` and update:

    - version number in line 1 (don't include the 'v' in the version tag)
    - the build number to zero (you should only be here if releasing a new version)
    - update ``sha256`` in line 9 (you can get the sha from `pint-pandas's PyPI`_ by clicking on 'Download files' on the left and then clicking on 'SHA256' of the ``.tar.gz`` file to copy it to the clipboard)

#. ``git add .``
#. ``git commit -m "Update to vX.Y.Z"``
#. ``git push``
#. Make a PR into the `pint-pandas conda feedstock`_
#. If the PR passes (give it at least 10 minutes to run all the CI), merge
#. Check https://anaconda.org/conda-forge/pint-pandas to double check that the version has increased (this can take a few minutes to update)

.. _`pint-pandas's PyPI`: https://pypi.org/project/pint-pandas/
.. _`pint-pandas conda feedstock`: https://github.com/conda-forge/pint-pandas-feedstock


Why is there a ``Makefile`` in a pure Python repository?
--------------------------------------------------------

Whilst it may not be standard practice, a ``Makefile`` is a simple way to automate general setup (environment setup in particular).
Hence we have one here which basically acts as a notes file for how to do all those little jobs which we often forget e.g. setting up environments, running tests (and making sure we're in the right environment), building docs, setting up auxillary bits and pieces.
