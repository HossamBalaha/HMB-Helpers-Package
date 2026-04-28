Examples
========

This page lists the example scripts included with the HMB package and gives
basic instructions to run them. Examples are grouped by platform where
applicable and are intended as demonstrations. They may require additional
dependencies or input data; consult the header comments of each example script
for usage details.

Available example scripts
-------------------------


The repository contains the following example scripts (located under ``HMB/Examples``):

- ``PyTorch_UNet_Segmentation.py``
- ``TF_UNet_Training.py``
- ``TF_UNet_EvalPredict.py``
- ``Timm_FineTune_Classification.py``
- ``Timm_Statistics_Analysis_Ablations.py``
- ``Train TF Pretrained Attention Model from DataFrame.py``
- ``Explain TF Pretrained Attention Model from DataFrame.py``
- ``Machine_Learning_Pipeline.py``

Platform helper scripts
-----------------------

To simplify running the examples, the repository provides per-example wrapper
scripts grouped by platform under the ``HMB/Examples/BAT Files`` and
``HMB/Examples/SH Files`` directories. Each wrapper has the same base name as
its corresponding Python example and invokes or configures that example. There
is no single top-level "run_examples" aggregator in the project; run the
wrapper for the example you want to execute.

How to run
----------

From the repository root on Windows (cmd.exe) you can run an example wrapper
directly. For example, to run the Timm statistics example use the corresponding
batch file:

.. code-block:: bat

   call "HMB\Examples\BAT Files\Timm_Statistics_Analysis_Ablations.bat"

On POSIX (bash) run the matching shell wrapper, for example:

.. code-block:: bash

   bash "HMB/Examples/SH Files/Timm_Statistics_Analysis_Ablations.sh"

If you prefer to run a single example directly, invoke the corresponding
Python file. Quoting paths is recommended when file or folder names contain
spaces. For example, on Windows:

.. code-block:: bat

   python "HMB\Examples\Timm_Statistics_Analysis_Ablations.py"

or on POSIX:

.. code-block:: bash

   python3 "HMB/Examples/Timm_Statistics_Analysis_Ablations.py"

Important
---------

- Examples can require large models, GPU access, or specific dataset files and
  therefore may not run out-of-the-box.
- Review each example header for required environment variables, expected data
  paths and recommended dependency versions.

See also
--------

- :doc:`index` (main documentation)
