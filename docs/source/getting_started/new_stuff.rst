What's new in Cool-chic 4.1?
============================

Cool-chic 4.1 focuses on video coding, improving particularly the temporal
prediction through better and lighter sub-pixel motion compensation. This
release is linked to the following paper: `Efficient Sub-pixel Motion
Compensation in Learned Video Codecs, Ladune et al
<https://arxiv.org/pdf/2507.21926>`_.

* Replace 2-tap bilinear filtering with **sinc-based 8-tap filters**

* **Downsampled motion fields** for lighter decoding

* Improved video compression performance: **-23.6% rate** versus Cool-chic 4.0

* Decrease motion-related complexity by 30%, **from 391 to 214 MAC / decoded pixel**


Check-out the `release history
<https://github.com/Orange-OpenSource/Cool-Chic/releases>`_ to see previous
versions of Cool-chic.
