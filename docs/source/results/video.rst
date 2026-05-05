Video results
=============

.. raw:: html

    <style> .red {color:red} </style>
    <style> .green {color:green} </style>


.. role:: red

.. role:: green

Video compression performance are presented on the entire JVET test sequences.
PSNR in the YUV420 domain are obtained as follows:

.. math::

    \mathrm{PSNR_{YUV420}} = -10 \log_{10} \left(\frac{4\mathrm{MSE}_Y + \mathrm{MSE}_U + \mathrm{MSE}_V}{6}\right).



Random-access compression
*************************

The following table sums up the compression performance obtained by Cool-chic
4.0 against several conventional codecs in a random access configuration
settings. An intra period of 32 frames is used. Results are measure through
`BD-rates <https://github.com/Anserw/Bjontegaard_metric>`_ which represents the
relative rate required for Cool-chic to obtain the same quality than some other
codec.

.. raw:: html

    <style type="text/css">
    .tg .tg-m5nv{border-color:#656565;text-align:center;vertical-align:top}
    .tg .tg-dfl2{border-color:#656565;font-family:inherit;text-align:center;vertical-align:top}
    .tg .tg-uflc{border-color:#656565;color:#009901;text-align:center;vertical-align:top}
    .tg .tg-86ol{border-color:#656565;font-family:inherit;font-weight:bold;text-align:center;vertical-align:center}
    .tg .tg-9mze{border-color:#656565;font-family:inherit;font-style:italic;text-align:center;vertical-align:top}
    .tg .tg-u3ui{border-color:#656565;font-family:inherit;font-weight:bold;text-align:center;vertical-align:top}
    .tg .tg-qch7{border-color:#656565;color:#009901;font-family:inherit;text-align:center;vertical-align:top}
    .tg .tg-arzi{border-color:#656565;color:#cb0000;text-align:center;vertical-align:top}
    .tg .tg-xd3r{border-color:#656565;color:#cb0000;font-family:inherit;text-align:center;vertical-align:top}
    .tg .tg-aaaa{border-color:#656565;color:#cb0000;font-weight:bold;font-family:inherit;text-align:center;vertical-align:top}
    .tg .tg-5niz{border-color:#656565;color:#9b9b9b;font-family:inherit;text-align:center;vertical-align:top}
    .tg .tg-x9uu{border-color:#656565;font-weight:bold;text-align:center;vertical-align:top}
    .tg .tg-qch8{border-color:#656565;color:#009901;font-weight:bold;font-family:inherit;text-align:center;vertical-align:top}
    .tg .tg-1keu{border-color:#656565;color:#9b9b9b;text-align:center;vertical-align:top}
    </style>
  <table class="tg"><thead>
  <tr>
      <th class="tg-86ol" rowspan="2">Sequence</th>
      <th class="tg-86ol" colspan="4">BD-rate of Cool-chic 4.1.0 against [%]</th>
      <th class="tg-86ol" colspan="1">Decoding complexity</th>
  </tr>
  <tr>
      <th class="tg-86ol">HEVC (HM 16)</th>
      <th class="tg-86ol">HEVC (x265-medium)</th>
      <th class="tg-86ol">AVC (x264-medium)</th>
      <th class="tg-86ol">Cool-chic 4.0</th>
      <th class="tg-86ol">MAC / pixel</th>
  </tr></thead>
  <tbody>
  <tr>
      <td class="tg-9mze">C-BasketballDrill</td>
      <td class="tg-xd3r">+30.8</td>
      <td class="tg-qch7">-5.2</td>
      <td class="tg-qch7">-29.5</td>
      <td class="tg-qch7">-17.5</td>
      <td class="tg-dfl2">639</td>
  </tr>
  <tr>
      <td class="tg-9mze">C-BQMall</td>
      <td class="tg-xd3r">+77.1</td>
      <td class="tg-xd3r">+16.6</td>
      <td class="tg-qch7">-6.0</td>
      <td class="tg-qch7">-21.1</td>
      <td class="tg-dfl2">639</td>
  </tr>
  <tr>
      <td class="tg-9mze">C-PartyScene</td>
      <td class="tg-xd3r">+28.3</td>
      <td class="tg-qch7">-15.1</td>
      <td class="tg-qch7">-39.2</td>
      <td class="tg-qch7">-39.1</td>
      <td class="tg-dfl2">639</td>
  </tr>
  <tr>
      <td class="tg-9mze">C-RaceHorses</td>
      <td class="tg-xd3r">+79.6</td>
      <td class="tg-xd3r">+17.4</td>
      <td class="tg-xd3r">+8.6</td>
      <td class="tg-qch7">-16.8</td>
      <td class="tg-dfl2">639</td>
  </tr>
  <tr>
      <td class="tg-u3ui">Average</td>
      <td class="tg-aaaa">+54.0</td>
      <td class="tg-aaaa">+3.4</td>
      <td class="tg-qch8">-14.0</td>
      <td class="tg-qch8">-23.6</td>
      <td class="tg-u3ui">639</td>
  </tr>
  </tbody></table>
