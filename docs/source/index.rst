Cool-chic
=========

.. raw:: html

    <style> .gray {color:gray} </style>

.. role:: gray


**Cool-chic** (pronounced :gray:`/kul ʃik/` as in French 🥖🧀🍷) is a
low-complexity neural image codec based on overfitting. It offers image coding
performance competitive with H.266/VVC for 2000 multiplications per decoded
pixel.


Cool-chic 3.2 improvements
""""""""""""""""""""""""""

* **Fast CPU-only decoder** as proposed in `Overfitted image coding at reduced
  complexity, Blard et al. <https://arxiv.org/abs/2403.11651>`_

  * Decode a 512x768 image in **100 ms**
  * C API for **binary arithmetic coding**

* Encoding (*i.e.* training) time **reduced by 35%**

* **Rate reduction of 5%** compared to Cool-chic 3.1.


Check-out the `release history
<https://github.com/Orange-OpenSource/Cool-Chic/releases>`_ to see previous
versions of Cool-chic.

.. attention::

   🛑 Cool-chic 3.2 temporarily disables video coding. If you really want to
   compress videos you can

   * Go back to 3.1: ``git clone --branch v3.1
     https://github.com/Orange-OpenSource/Cool-Chic.git``

   * Wait for Cool-chic 4.0 for better and faster video coding 😉.

Thanks
""""""

Special thanks go to:

* **Hyunjik Kim, Matthias Bauer, Lucas Theis, Jonathan Richard Schwarz and Emilien
  Dupont** for their great work enhancing Cool-chic: `C3: High-performance and
  low-complexity neural compression from a single image or video, Kim et al.
  <https://arxiv.org/abs/2312.02753>`_


.. image:: assets/logo_orange.png
  :align: center
  :height: 150
  :alt: Logo orange



.. toctree::
   :maxdepth: 1
   :caption: Getting started
   :hidden:

   Quickstart <getting_started/quickstart>
   Example <getting_started/example>

.. toctree::
   :maxdepth: 1
   :caption: Decoding
   :hidden:

   Decoding a bitstream <decoding/decoding_images>


.. toctree::
   :maxdepth: 1
   :caption: Encoding your own files
   :hidden:

   Overview <encoding/overview>
   Decoder configuration <encoding/architecture>
   Encoder configuration <encoding/preset>

.. Video <encoding/video>

.. toctree::
   :maxdepth: 1
   :caption: Compression performance
   :hidden:

   Results <getting_started/results>
   Literature <getting_started/literature>


.. toctree::
   :caption: Code Documentation
   :hidden:

   At a glance <code_documentation/overview>
   Encoder <code_documentation/encoder/index>

.. Decoder <code_documentation/decoder/index>


.. .. toctree::
..    :maxdepth: 1
..    :caption: Code Documentation

..    encoding_management <encoding_management>
..    models <models>
..    utils <utils>
..    visu <visu>


