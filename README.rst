PyAutoFit Workspace Test
=========================

Welcome to the **PyAutoFit** test workspace.

This workspace mirrors the `autolens_workspace <https://github.com/Jammy2211/autolens_workspace>`_ but runs the example
scripts and pipelines fast, by skipping the non-linear search. It is used by **PyAutoFit** developers to perform
automated integration tests of example scripts.

To run the pipelines in this project you must add the autolens_workspace_test directory to your PYTHONPATH:

.. code-block:: bash

    export PYTHONPATH=$PYTHONPATH:/mnt/c/Users/Jammy/Code/PyAutoJAX/autolens_workspace_test

You can run an integration test as follows:

.. code-block:: bash

    python slam/imaging/no_lens_light/source_lp/mass_total/no_hyper.py


Workspace Version
=================

This version of the workspace are built and tested for using **PyAutoFit v2025.5.10.1**.