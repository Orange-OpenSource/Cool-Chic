What's new in Cool-chic 3.4?
============================


Cool-chic 3.4 introduces an enhanced upsampling with more expressive convolutive
filters. More information are available in the :doc:`upsampling documentation <./../../code_documentation/encoder/component/core/upsampling>`

New 3.4 version offers the following improvements:

* New and improved latent **upsampling module**

  * Leverage symmetric and separable convolution kernels to reduce complexity & parameters count

  * Learn two filters per upsampling step instead of one for all upsampling steps

* 1% to 5% **rate reduction** for the same image quality

* **30% complexity reduction** using a smaller Auto-Regressive Module

  * From 2000 MAC / decoded pixel to 1300 MAC / decoded pixel

  * **10% faster** decoding speed

Check-out the `release history
<https://github.com/Orange-OpenSource/Cool-Chic/releases>`_ to see previous
versions of Cool-chic.

.. attention::

   🛑 Cool-chic 3.4 temporarily disables video coding. If you really want to
   compress videos you can

   * Go back to 3.1: ``git clone --branch v3.1
     https://github.com/Orange-OpenSource/Cool-Chic.git``

   * Wait for Cool-chic 4.0 for better and faster video coding 😉.
