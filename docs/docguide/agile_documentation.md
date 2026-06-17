<!--
Original adaptation, in our own words, of Scott W. Ambler's "Core Practices for
Lean/Agile Documentation":
https://agilemodeling.com/essays/agileDocumentationBestPractices.htm
The source essay is (c) Ambysoft Inc. and carries no open license, so it is NOT
vendored here the way the Google CC-BY-3.0 files are. This page is our own summary,
tailored to adc_cpp, and may be edited freely. It is the "how much, what, and when"
companion to the Google guides' "how to write".
-->

# Agile Documentation

"Just barely good enough" beats comprehensive. A document earns its place by
serving a clear, present need; the rest is cost and rot.

The Google [best practices](best_practices.md) say *how* to write a document well.
This page says *whether*, *what*, and *when* to write one at all. It is adapted from
Scott Ambler's
[Core Practices for Lean/Agile Documentation](https://agilemodeling.com/essays/agileDocumentationBestPractices.htm),
which the Google guide itself points to.

Contents:

1.  [Writing](#writing)
1.  [Simplification](#simplification)
1.  [What to document](#what-to-document)
1.  [When to document](#when-to-document)
1.  [In general](#in-general)

## Writing

*   **Prefer executable specifications over static prose.** A test pins down a
    requirement and proves it at the same time. In adc_cpp the ctest suite and the
    golden/replay cases are the real specification of behavior; reach for a static
    document only for what a test cannot say (intent, trade-offs, the shape of the
    whole).
*   **Document stable concepts, not speculative ones.** Write late, once a thing has
    stopped moving. Keep rough notes as you go so nothing is lost, but do not polish
    a design the next iteration will rewrite. (This is why a release's CHANGELOG
    section is cut at release time, not sketched up front.)
*   **Generate system documentation.** Let the tools read the code: Doxygen for the
    C++ reference, Sphinx autodoc for the Python API, doxysphinx to fuse them. A
    generated page cannot drift from what it describes.

## Simplification

*   **Just simple enough, not too simple.** A trustworthy ten-page overview beats a
    five-hundred-page reference nobody believes. Prefer a roadmap that points into
    the source over a transcription of it, a sketch over an elaborate diagram, a
    reference over a copy.
*   **Fewest documents, least overlap.** Say each thing once, in one place, and link
    to it. Build large documents out of small single-topic ones instead of repeating
    yourself; every duplicated fact is a second place to forget to update.
*   **Put information where its reader will look for it.** A design decision usually
    belongs in a code comment, an API contract in the docstring the tool renders, a
    cross-cutting idea on its own page. Record it once, where it does the most good
    (single source of truth, as with the version in `CMakeLists.txt` or page
    ownership in `docmap.toml`).
*   **Make it public.** A document people can find and watch change carries
    information across the team; one buried in a drive does not. Mark its status
    (draft or released) so readers know how far to lean on it.

## What to document

*   **Document with a purpose.** Create a document only when it serves a clear,
    important, present goal. No single template fits every project, so decide each
    time rather than filling in a form.
*   **Serve the actual reader.** Find out who will read it and what they will do with
    it, then write the smallest thing that lets them do it. The best-written page is
    worthless if its reader never needs it or never finds it.
*   **The reader judges sufficiency.** You make the document meaningful; the reader
    decides whether it is enough. Treat their acceptance as the quality gate.

## When to document

*   **Iterate.** Write a little, show it, take the feedback, adjust. Good documents
    are grown, not poured in one cast.
*   **Prefer a better channel when one exists.** Documentation carries memory across
    distance and time; for understanding, a conversation, a worked example, or
    pairing usually beats paper. Reach for a document when the people who need the
    knowledge are not in the room.
*   **Build on the models you actually keep current.** If a diagram is worth keeping
    up to date, it is worth basing a document on. If it was left to rot, it was not
    pulling its weight to begin with.
*   **Update only when it hurts.** A slightly stale document is often fine. Spend the
    edit once the gap between the page and reality costs more than the fix would. The
    `docmap.toml` freshness gate is this rule made mechanical: it speaks up only when
    a documented dependency has actually moved.

## In general

*   **Treat documentation as a requirement.** Estimate it, prioritize it, put it on
    the same stack as features. Time spent writing a doc is time a feature did not
    get, so make that a deliberate, visible choice.
*   **Make people justify a documentation request.** Ask what they will use it for
    and how. Often the honest answer is a security blanket, and the real worry has a
    cheaper cure than a page nobody reads.
*   **Some documentation is necessary.** Working software is the goal; letting the
    next person maintain, operate, and extend it is the close second. Capture the
    high-level and the durable (overviews, decisions, anything a regulation or a
    successor will need) and skip the minutiae the code already states.
*   **Lean on people who write.** Pair on a document the way you pair on code, share
    ownership so more than one person tends it, and read it back aloud to catch the
    rough passages.
