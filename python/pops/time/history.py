"""pops.time history policy descriptors (Spec 5 sec.5.3.1).

A history policy describes how the ring buffer behind ``T.prev(lag)`` is seeded on the
first macro-step (step 0), when no genuine ``U^{n-1}`` exists yet. These are inert
authoring descriptors: they carry NO runtime data and emit NO IR on their own. The
``keep_history`` lowering records the chosen policy on the :class:`TimeState` so a later
runtime / codegen phase can honor it; the historical cold start (the runtime fills every
slot on the first store) is the default when no policy is given.
"""


class CopyCurrent:
    """Cold-start policy: seed every history slot with the current state ``U^n``.

    This mirrors the runtime's historical behavior (a multistep scheme degenerates to a
    one-step scheme on step 0, e.g. Adams-Bashforth 2 takes a Forward-Euler first step).
    It is the conventional default and carries no parameters.
    """

    kind = "copy_current"

    def __repr__(self):
        return "CopyCurrent()"
