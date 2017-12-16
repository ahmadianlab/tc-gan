===========================================================
 Stabilized Supralinear Network (SSN) as a GAN's Generator
===========================================================

.. note:: Work in progress!

.. _general-construction:

General construction
====================

We regard an SSN as (possibly conditional) generator in the GAN
framework by constructing the generator function :math:`G` by:

.. math::

   G(z, c) = P \left( \Big(F \big(z, I(s, c) \big) \Big)_{s \in S}, c \right)

where

- :math:`S` is the (finite) points in the :term:`tuning curve
  domain` sampled during the experiment.
- :math:`I: S \times C \to X` is the input-constructing function
  (:term:`stimulator`) where :math:`C` is the set of
  :term:`cGAN-conditions` and :math:`X` (= :math:`\mathbb R^{2N}`) is
  the input space of the SSN.
- :math:`F: Z \times X \to Y` is the :term:`fixed-point "solver"
  <fixed-point solver>` of the SSN given a noise variable :math:`z \in
  Z` and an input in :math:`X`.  It returns a fixed point [#]_ in the
  state space :math:`Y` (= :math:`\mathbb R^{2N}`) of the SSN.
- :math:`P: Y^S \times C \to \mathrm{dom}(D)` is the :term:`probe
  function` which converts entire SSN states across different tuning
  curve conditions :math:`s \in S` to a (lower dimensional) tuning
  curve which in turn fed into the discriminator.  This is analogous
  to neural recordings in actual experiments.
  (cf., `.get_reduced`)

Note that both input-constructing function :math:`I` and probe
function :math:`P` can "see" the condition :math:`c \in C`.  This
allows us to represent :term:`cGAN-conditions` as sliding probes or
sliding stimuli.  Since the first argument to :math:`P` is a tuple of
:math:`|S|` fixed points, :math:`P` may include some normalization of
the tuning curve.

.. [#] We assume such fixed-point is uniquely determined (say, the SSN
   is monostable and :math:`F` returns only stable one).

To fit a parametrized model :math:`F` using GAN, one need to
appropriately construct :math:`(S, C, I, P)` to define the generator.
Such construction has to be rich enough for the GAN to determine the
parameter but also simple enough for experimental data collection can
be done.


SSN dynamics
============

For a network with recurrent connectivity matrix :math:`\mathbf{W}`,
the vector of firing rates :math:`\mathbf{r}` is governed by the
differential equation

.. math::

   \mathbf{T} \frac{d\mathbf{r}}{dt}
   = - \mathbf{r} + f\left(\mathbf{W} \mathbf{r} + \mathbf{I}\right),
   \qquad
   {f}(\mathbf{u}) \equiv \left( f(u_{i}) \right)_{i=1}^{2N}
   = \left( k [u_{i}]_+^n \right)_{i=1}^{2N}

where the diagonal matrix :math:`\mathbf{T} = \text{Diag}\!\left(
(\tau_i)_{i=1}^{2N} \right)` denotes the neural relaxation time
constants, and :math:`\mathbf{I}` denotes the external or stimulus
input (which is set by the input-constructing function :math:`I`).
Below, for an arbitrary neuron :math:`i`, we denote its type by
:math:`\alpha(i)\in \{E,I\}` and its topographic location by
:math:`x_i`.

The fixed-point solver :math:`F: Z \times X \to Y` produces a fixed
point :math:`\hat{\mathbf{r}} \in Y` given the noise :math:`z \in Z`
(which in turn sets :math:`\mathbf{W}`) and the input
:math:`\mathbf{I} \in X`.


Stimulus to SSN
===============

We let the stimulus input to neuron :math:`i` be

.. math::

   I_i(s) = A\,
   \sigma\!\left( \frac{{s}/{2} + x_i}{l} \right)\,
   \sigma\!\left( \frac{{s}/{2} - x_i}{l} \right)

where :math:`\sigma(u) = (1+\exp(-u))^{-1}` is the logistic function,
:math:`A` denotes the stimulus intensity (`contrast`), and :math:`s
\in S` (= `bandwidths`) are the stimulus size.

.. seealso::

   `.run.gan.learn`, `.stimuli.input`
