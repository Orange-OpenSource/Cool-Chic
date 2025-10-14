What's new in Cool-chic 4.2?
============================

Cool-chic 4.2 focuses on perceptually-oriented image coding. This release draws
heavily on the following paper: `Good, Cheap, and Fast: Overfitted Image
Compression with Wasserstein Distortion, Ballé et al
<https://arxiv.org/abs/2412.00505>`_.

* **Wasserstein Distance** as a distortion metric: ``--tune=wasserstein``

* Decoder-side **common randomness** for more high-frequency in the decoded image

* Improved image coding performance: around **-50% rate** versus Cool-chic 4.1 for the same visual quality

* Low decoding complexity **1728 MAC / pixel**


A pair of bitstreams in ``samples/bitstreams/`` illustrates the benefits of the
``--tune=wasserstein`` options. See the :ref:`decoding example
<decoding_example>` to decode them and see the pictures.

Check-out the `release history
<https://github.com/Orange-OpenSource/Cool-Chic/releases>`_ to see previous
versions of Cool-chic.
