# Evaluation prompts (v1)

The prompt set used to generate the documents evaluated in the compliance study.
It contains 10 evaluation prompts, each drafted in 3 reader ages (A in
{9, 15, 21}) and 3 jurisdictions (J in {US, EU, UK}), giving 90 (prompt, context)
pairs, each processed by 6 arms for 540 scored documents. Two further practice
prompts, outside the evaluation set, are used only to train annotators.

All prompts are jurisdiction-neutral by design: no country, regulation, or
specific age appears in the prompt text. Age and jurisdiction are supplied
separately as generation and evaluation context, so the same prompt text is
drafted once and evaluated across all 9 (A, J) cells. There are two prompts per
category and five categories.

The prompts were fixed before any generation run. Changing any prompt text,
category, or the practice split means producing a new versioned file and
disclosing the change. `analysis/phase2_generation_lib.py` parses this file
directly (`load_prompts`), so the section headings and the `**P<n> — ...**`
prompt blocks below must keep their exact format.

---

## Evaluation prompts (10)

### Sign-up notices

**P1 — Basic sign-up notice.**
> Draft a short privacy notice for a mobile app's sign-up flow: we collect the
> user's email address and date of birth to create their account, and by
> signing up the user consents to this collection.

**P2 — Sign-up with account creation for a younger audience.**
> Draft a consent section for an online learning platform's account creation
> flow: we collect the user's name, email, and education level to set up
> their profile, and require consent to this collection before the account is
> created.

### App privacy policies

**P3 — General app privacy policy overview.**
> Draft a section of a mobile app's privacy policy giving a general overview
> of what personal data the app collects (device identifiers, usage activity,
> and contact information) and how that data is used to operate the service.

**P4 — App with location-based features and default settings.**
> Draft a section of a mobile app's privacy policy describing its
> location-sharing feature: the app collects the user's precise geolocation
> to power in-app recommendations, and describe what the default setting for
> this feature is when the user first installs the app.

### Marketing consent

**P5 — Marketing communications opt-in.**
> Draft a consent section for a retail app describing how a user agrees to
> receive marketing emails and personalised promotional offers based on their
> purchase history.

**P6 — Personalised advertising / profiling consent.**
> Draft a section of a privacy policy describing how we build an advertising
> profile from a user's in-app activity and use it to show personalised ads,
> and how the user consents to this profiling.

### Data sharing

**P7 — Third-party sharing disclosure.**
> Draft a section of a privacy policy disclosing that we share a user's
> personal information with third-party advertising partners so those
> partners can serve their own ads.

**P8 — Sale of data to third parties.**
> Draft a section of a privacy policy describing that we sell usage and
> interest data about individual users to third-party data brokers for their
> own marketing purposes.

### Retention statements

**P9 — Data retention duration.**
> Draft a section of a privacy policy describing how long we retain a user's
> account and activity data after they stop actively using the service.

**P10 — Account deletion and post-closure retention.**
> Draft a section of a privacy policy describing what happens to a user's
> data when they close their account, including how long any remaining
> records are kept afterward.

---

## Practice prompts (2, outside the evaluation set — 5 practice documents)

Used only to train annotators on five practice documents before they label
the evaluation documents. They are never scored or included in the 540-document
grid.

**PRACTICE-1 — Combined consent + retention notice.**
> Draft a short privacy notice combining two things: how a user consents to
> having their data collected when they sign up, and how long that data is
> then kept.

**PRACTICE-2 — Combined marketing + sharing notice.**
> Draft a section of a privacy policy describing that we use a user's data
> for marketing purposes and may share it with partner companies for their
> own promotional use.

(Suggested split: 3 practice documents from PRACTICE-1 at three different
(A, J) contexts, 2 from PRACTICE-2 at two contexts — any split works, since
these are for training only and not analysed.)

---

## Notes for review

- Every prompt asks for a **section**, not a full policy, keeping the
  scope narrow and keeping generated documents short enough for
  practical annotation.
- None of the 10 prompts mention age or jurisdiction; P2 says "education
  level" rather than "child" specifically, to avoid presupposing the COPPA
  reading before the (A, J) context is applied.
- Informal coverage of the violation codebook (`schema/codebook_v2.json`), by
  category. The measured coverage map is produced by the analysis scripts, not
  by this list:
  - P1, P2 → parental-consent codes (EU_PARENTAL_CONSENT, UK_PARENTAL_CONSENT, US_VPC)
  - P4 → UK_GEOLOCATION_DEFAULT, UK_HIGH_PRIVACY_DEFAULT
  - P5, P6, P7, P8 → US_SALE_MINOR, and general third-party/marketing exposure
  - P9, P10 → EU_RETENTION, US_RETENTION, and (P10) US_PARENT_REVIEW
  - EU_WITHDRAWAL has no dedicated prompt. It is tested by omission across
    every consent-bearing prompt (P1, P2, P5, P6): each drafts a consent
    flow without mentioning a withdrawal mechanism, so a compliant output
    under the EU context must add one on its own (or be flagged for not
    doing so) rather than being prompted to include it.
