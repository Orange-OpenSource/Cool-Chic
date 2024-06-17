Quickstart
==========

Installing necessary packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. attention::

    The setup requires ``python >= 3.10``. It is presented here with Python 3.10 but
    should work with more recent versions.

The first step is to install some necessary packages and to clone Cool-chic. We
need to install ``python3.10-dev`` to compile and bind the Cool-chic C API.

.. code:: bash

    # We need to get g++, python3.10-dev and pip to compile the Cool-chic
    # C API and bind it to python.
    ~$ sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt update
    ~$ sudo apt install -y build-essential python3.10-dev pip
    ~$ git clone https://github.com/Orange-OpenSource/Cool-Chic.git && cd Cool-Chic


You should create a virtual environment when installing Cool-chic

.. code:: bash

    ~/Cool-Chic$ python3.10 -m pip install virtualenv                          # Install virtual env if needed
    ~/Cool-Chic$ python3.10 -m virtualenv venv && source venv/bin/activate     # Create and activate a virtual env named "venv"

Cool-chic can then be installed through pip, which retrieves the required
package (torch etc.) and compiles the Cool-chic C API.

.. code:: bash

    (venv) ~/Cool-Chic/$ pip install -e .


Sanity check
~~~~~~~~~~~~

A simple sanity check script is provided. It performs a very fast encoding of an
image, write a bitstream and decode it.

.. code:: bash

    (venv) ~/Cool-Chic$ python -m test.sanity_check


You're good to go!
******************
