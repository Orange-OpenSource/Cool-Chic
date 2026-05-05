Quickstart
==========

Installing necessary packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. attention::

    The setup requires ``python >= 3.10``. It is presented here with Python 3.10 but
    should work with more recent versions.

Here are the instructions to install Cool-chic

.. code:: bash

    # Get the repository
    ~$ git clone https://github.com/Orange-OpenSource/Cool-Chic.git && cd Cool-Chic

    # You should create a virtual environment when installing Cool-chic
    ~/Cool-Chic$ sudo apt update && sudo apt install -y pip
    ~/Cool-Chic$ python3 -m pip install virtualenv
    ~/Cool-Chic$ python3 -m virtualenv venv && source venv/bin/activate

    # Cool-chic needs some external packages (e.g., PyTorch) to work.
    (venv) ~/Cool-Chic/$ pip install -r requirements.txt


Sanity check
~~~~~~~~~~~~

A simple sanity check script is provided. It performs a very fast encoding of
different image and video formats, write bitstreams and decode them.

.. code:: bash

    (venv) ~/Cool-Chic$ python -m test.sanity_check


You're good to go!
******************
