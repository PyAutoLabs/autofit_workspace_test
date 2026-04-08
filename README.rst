PyAutoFit Workspace Test
=========================

Welcome to the **PyAutoFit** test workspace.

This workspace mirrors the `autolens_workspace <https://github.com/Jammy2211/autolens_workspace>`_ but runs the example
scripts and pipelines fast, by skipping the non-linear search. It is used by **PyAutoFit** developers to perform
automated integration tests of example scripts.

To run the pipelines in this project you must add the autolens_workspace_test directory to your PYTHONPATH:

.. code-block:: bash

    export PYTHONPATH=$PYTHONPATH:/mnt/c/Users/Jammy/Code/PyAutoLabs/autolens_workspace_test

You can run an integration test as follows:

.. code-block:: bash

    python slam/imaging/no_lens_light/source_lp/mass_total/no_hyper.py


Workspace Version
=================

This version of the workspace are built and tested for using **PyAutoFit v2026.4.5.3**.

Build Configuration
===================

The ``config/`` directory contains two files used by the automated build and test system
(CI, smoke tests, and pre-release checks). These are not relevant to normal workspace usage.

- ``config/no_run.yaml`` — scripts to skip during automated runs. Each entry is a filename stem
  or path pattern with an inline comment explaining why it is skipped.
- ``config/env_vars.yaml`` — environment variables applied to each script during automated runs.
  Defines default values (e.g. test mode, small datasets) and per-script overrides for scripts
  that need different settings.