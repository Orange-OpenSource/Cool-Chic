:layout: simple

Decoding complexity
===================

As described in the :doc:`decoder configuration <../../encoding/architecture>`
section, Cool-chic decoder can be made as simple (or complex) as desired. We
present here the performance-complexity continuum obtained through the :ref:`4
provided configurations <decoder_cfg_files>` (*VLOP*, *LOP*, *MOP*, *HOP*). We
also add a *MIX* configuration, where we pick the best configuration out of the
4 available ones for each image. We used the ``slow_100k`` :ref:`encoding preset
<encoder_cfg_files>`.

Decoding complexity is measured both in multiplication per decoded pixel and
decoding time, obtained on a single core of an **AMD EPYC 7282 16-Core
Processor**.

Kodak
*****

.. image:: ../../assets/kodak/all_complexity_dec.png
  :alt: Kodak rd results

CLIC20 Pro Valid
****************

.. image:: ../../assets/clic20-pro-valid/all_complexity_dec.png
  :alt: CLIC20 rd results


JVET Class B
************

.. image:: ../../assets/jvet/all_complexity_dec_classB.png
  :alt: JVET class B rd results

JVET Class C
************

.. image:: ../../assets/jvet/all_complexity_dec_classC.png
  :alt: JVET class C rd results

JVET Class D
************

.. image:: ../../assets/jvet/all_complexity_dec_classD.png
  :alt: JVET class D rd results

JVET Class E
************

.. image:: ../../assets/jvet/all_complexity_dec_classE.png
  :alt: JVET class E rd results

JVET Class F
************

.. image:: ../../assets/jvet/all_complexity_dec_classF.png
  :alt: JVET class F rd results

JVET All Classes
****************

.. image:: ../../assets/jvet/all_complexity_dec_classBCDEF.png
  :alt: JVET class BCDEF rd results


