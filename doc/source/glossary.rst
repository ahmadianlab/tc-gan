==========
 Glossary
==========

.. glossary::

   tuning curve domain

     Stimuli "sweep" given to each identical neuron; in our main case,
     it's bandwidths.  In principle, we can vary, say, both bandwidths
     and offsets for the same neuron.  In this case, tuning curve
     domain is ``bandwidths x offsets``.

     See: :ref:`general-construction`

   tuning curve codomain

     Neural output, e.g., firing rate.
     (OK. Nobody would use this. Just for completeness.)

   cGAN-conditions
   tuning curve modifier

     This is the :term:`stimulus parameter` for *each tuning curve*.
     However, every neuron need *not* to exhaust the full set of this
     parameter.

     See: :ref:`general-construction`

   experiment parameter
   stimulus parameter

     Any parameter varied during the experiment.  *Both* :term:`tuning
     curve domain` *and* :term:`cGAN-conditions` (:term:`tuning curve
     modifier`) are such parameters.

     See: :ref:`general-construction`

   probe function
   probe
   subsample

     See: :ref:`general-construction`
