# Privacy notice

This notice is published under the Nigeria Data Protection Act 2023 ("the Act"). It explains what
personal data the RiskRank project processes about software developers, why, and what rights those
developers have. It is published here instead of contacting each developer, because contacting every
developer whose commits appear in the histories used would involve disproportionate effort
(section 27(2)(b)).

## Who is responsible

The data controller is the author of the project, the owner of this repository
([github.com/wisdomerhabor](https://github.com/wisdomerhabor)). RiskRank is an MIT professional
master's project at Miva Open University. The Act applies because the author, who controls the
processing, is resident in Nigeria (section 2(2)(a)).

## Whose data, and what data

The project uses publicly available datasets and open-source repositories, in line with their licences.
The personal data it draws from them is limited to:

- author and committer identifiers in commit histories (names or email addresses), including those in
  commit trailers;
- the developer metrics derived from them: the experience measures and the count of developers who
  previously changed the modified files;
- any names in commit messages, which are read for fix keywords;
- linked issue records, of which only the number, title, labels, creation date and pull-request flag
  are kept (no usernames).

The experience measures and developer counts published in the ApacheJIT dataset contain no author
identifier, but they can be linked to public commits, so the project treats them as personal data too.

No sensitive personal data, such as health or religious information, is processed.

## Purpose

The purpose is software quality research: predicting which code modules and changes are most likely to
contain defects, so that limited testing effort can be allocated across them. Developer experience has
been used as a change measure in just-in-time defect prediction since Kamei et al. (2013).

The system is designed and presented strictly as a mechanism for allocating testing effort across code
modules and changes. The project does not assess individual developers and does not report results by
developer. No dashboard view or other system output shows results by developer.

## Legal basis

The legal basis is the legitimate interest in software quality research (section 25(1)(b)(v)). Using
published commit histories for this research is compatible with the purpose for which they were
published: section 24(4)(b) treats further processing for scientific research as compatible with the
original purpose.

Section 25(2) allows this basis only when three tests are met:

1. The processing does not override developers' fundamental rights, freedoms and interests. Results are
   not reported by developer and no developer is assessed.
2. The interest is not incompatible with the other legal bases in section 25(1)(b)(i) to (iv). The author
   has no contract, legal duty or public task concerning these developers, and no vital interest is at
   stake.
3. Developers can reasonably expect this processing. They published their commits in public open-source
   projects, and datasets such as ApacheJIT have already been built from such histories.

The processing is unlikely to create a high risk for developers, because it uses published commit
histories and reports no results by developer, so no data privacy impact assessment is required
(section 28(1)).

## Recipients

- **GitHub** acts as the author's data processor for the project's workflows (section 29).
- The Flask fork (`wisdomerhabor/flask`) receives upstream commits directly from the Flask repository on
  GitHub. Its CI job calculates module metrics only and processes no developer metrics.
- In this repository's own CI runs, author identifiers stay on the runner, whose virtual machine is
  discarded when the job ends. Only derived metrics are sent to the scoring service inside the job, and
  they are not stored or written to a plan file, build annotation or log.
- Google Colab receives no personal data.

## What is published, and what is withheld

Published: the source code, mining and labelling scripts, settings files, run records, the batch boundaries for each deployment push, the list of bug-fixing
commits used for labelling, the mined module-level metrics with their labels, the trained model files
and the aggregate results. The model files and their background data hold fitted parameters, cluster
centres and summary statistics, not individual records.

In the published bounded-check results, upstream commit identifiers are replaced with case numbers.

Withheld: the mined commit-level records, the developer metrics, the rebuilt ApacheJIT partitions and
the walkthrough records. The project withholds records that link developers to defects. Commit-level
results remain attributable to their authors through the public history.

The source repositories are public, so anyone can inspect the original commit histories directly. The
project's safeguards therefore protect its own outputs, not the public histories.

## Transfers outside Nigeria

Section 41(1) of the Act restricts transfers of personal data from Nigeria to other countries. The
project makes no such transfer. The Flask fork receives upstream commits directly from GitHub, and
workflows that run on GitHub do so with GitHub as the author's processor.

## How long data is kept

| Data | Kept until |
|---|---|
| Clones of the mined repositories (Flask and ZooKeeper) | Deleted with the local data, within 30 days after the university confirms the final project result. They can be cloned again from the recorded commits. |
| Developer metrics and mined commit-level records on the author's workstation | Deleted within 30 days after the university confirms the final project result. |
| The downloaded ApacheJIT file and rebuilt partitions | Deleted with the local data. |
| Author identifiers in a CI run | Until the job ends. |
| Plan files, build annotations and logs on GitHub | Seven days, the retention period set on both repositories. |
| The Flask fork on GitHub | Deleted within 30 days after the university confirms the final project result. Its history remains public in the Flask repository. |
| Published scripts, models and aggregate results | Remain public. They contain no personal data. |

The project log records the deletion date.

## Your rights

Under the Act you have rights over personal data about you, including the right to be informed, to
access it, to have it corrected or erased, to restrict its processing, and to object.

**Objection (section 36).** You may object to this processing. If you object, the project removes your
commits from the mined commit-level data and recalculates the affected results. Models already
published are not retrained.

**No automated decisions.** The project makes no automated decisions about developers.

**Complaints.** You may complain to the Nigeria Data Protection Commission (section 46(1)).

## Contact

To object, ask a question or exercise any other right, open an issue on this repository
([github.com/wisdomerhabor/riskrank-ci/issues](https://github.com/wisdomerhabor/riskrank-ci/issues)),
or contact the repository owner through their GitHub profile
([github.com/wisdomerhabor](https://github.com/wisdomerhabor)). An issue is public: to object, it is
enough to name the repository and the commit identifiers or the author identifier concerned.

## If you rebuild the withheld datasets

An examiner or anyone else who rebuilds the withheld datasets from the published scripts handles
personal data and should follow the same rules: no results by developer, the same retention limits,
and deletion when the work is done.
