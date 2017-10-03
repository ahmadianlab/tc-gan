.. glossary::

   tuning curve domain

     Stimuli "sweep" given to each identical neuron; in our main case,
     it's bandwidths.  In principle, we can vary, say, both bandwidths
     and offsets for the same neuron.  In this case, tuning curve
     domain is ``bandwidths x offsets``.

   tuning curve codomain

     "Neural output".
     (OK. Nobody would use this. Just for completeness.)

   cGAN-conditions
   tuning curve modifier

     This is the stimulus parameter for *each tuning curve*.

   experiment parameter
   stimulus parameter

     Any parameter varied during the experiment.  *Both* :term:`tuning
     curve domain` *and* :temr:`cGAN-conditions` (:term:`tuning curve
     modifier`) are such parameters.

   probe function
   probe
   subsample
