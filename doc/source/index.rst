.. tc_gan documentation master file, created by
   sphinx-quickstart on Sun Oct  1 00:13:13 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TC-GAN documentation!
================================

Quick links:

- Entry points:

  `.run.bptt_wgan`
    WGAN with BPTT-based gradient calculation.

  `.run.bptt_moments`
    Moment matching with BPTT-based gradient calculation.

  `.run.gan`
    GAN with gradient calculated exactly at the fixed-point.

  `.run.cgan`
    cGAN with gradient calculated exactly at the fixed-point.

  `.run.moments`
    Moment matching with gradient calculated exactly at the fixed-point.

- Submodules:

  `.networks`

      Implementations GANs and their components including:

      `.EulerSSNModel`
          Implementation of SSN in Theano and Lasagne.

      `.TuningCurveGenerator`
          Tuning curve generator based on `.EulerSSNModel`.

      `.BPTTWassersteinGAN`
          WGAN based on `.TuningCurveGenerator`.

  `.ssnode`
    Fixed-point solver for SSN.

  `.gradient_expressions`
    Implementation of analytical generator gradient at fixed-point.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   ssn.rst
   api/modules.rst
   glossary


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
