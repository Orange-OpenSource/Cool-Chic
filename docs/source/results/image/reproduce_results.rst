Reproduce our best results
==========================

We provide already encoded files as bitstreams in
``results/image/<dataset_name>/`` where ``<dataset_name>`` can be ``kodak``,
``clic20-pro-valid``, ``jvet``.


These bitstreams allow to reproduce the results stated above qnd reflect the
current best rate-distortion results achievable with Cool-chic.

For each dataset, a script decodes all the bitstreams.

.. code:: bash

    (venv) python results/decode_one_dataset.py image <dataset_name> # Can take a few minutes


The file ``results/image/<dataset_name>/results.tsv`` provides the
results that should be obtained.
