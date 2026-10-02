"""Reader-facing notice sentences for the 31 certified rules.

Supports: every CP-GTR-family arm of the compliance evaluation. As extracted,
a rule's template is a drafter-facing instruction ("Add a requirement for
verifiable parental consent...") rather than a sentence that can appear in a
privacy notice, so a repair that appended it would be marked non-compliant by
any annotator. CORRECTIONS holds a manually written reader-facing replacement
for each of the 31 unique rules (keyed by content hash), and the script writes
them to the JSON file that analysis/phase2_harness.py applies at load time.

Constraints on every rewrite: state only what the rule's right-hand side adds
(or removes, for a subtractive rule) relative to its left-hand side; never add
an obligation the rule does not structurally encode; and never name an age or
jurisdiction that the rule's own context predicate does not carry. Where the
old template reflects an extraction-fidelity problem (treaty reporting
language, an enumeration fragment, an unverified number), the note records it
and the new sentence states only what the rule actually encodes.

Content hashes cover rule structure, name and source but not the template, so
the corrections do not change the frozen library hashes. The raw extraction
cache is left untouched.

Run:
    python analysis/phase2_taskI_generate_corrections.py
Order: before phase2_taskN_build_recheck.py and phase2_generation_run*.py. The
SHA-256 of the JSON output is asserted by phase2_generation_lib.py and
phase2_build_fidelity_recheck_xlsx.py, so a regenerated file must match it.

Inputs : none (the corrections are defined in this file).
Outputs: analysis/realization_templates_corrections.json,
         analysis/realization_templates.md (the old and new text per rule).
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# (content_hash, name, source, old_template, new_template, note)
CORRECTIONS = [
    ("54499e4f23ed10b402583689b5f63e9b037f1d28f8e061c55e61b36427913eec",
     "AddParentalConsentForChildConsentProcess", "OHCHR.txt",
     "When a child (DataSubject) is linked to a ConsentProcess, ensure the ConsentProcess requires ParentalConsent, reflecting the child's inherent right to life and protection.",
     "A parent or legal guardian must give consent before we collect your information.",
     "Rule name implies a child-only scope but Phi is unconditional (no age gate) -- 'child' dropped so the sentence doesn't claim a restriction Phi doesn't carry."),

    ("23e999bce96b7346ae1359eec1850737bc9fb8cfe30edcbcc35ae50d34b91a30",
     "AddParentalConsentRequirementForChildAdoption", "OHCHR.txt",
     "Add a requirement that the adoption consent process must obtain informed parental consent from the child's parents, relatives, or legal guardians, as determined by the competent authority.",
     "A parent, relative, or legal guardian must give informed consent before this process may proceed.",
     "'Adoption' dropped -- the rule's L pattern matches ANY ConsentProcess node, not specifically an adoption process; keeping 'adoption' would misdescribe the rule when it fires on an unrelated consent process."),

    ("74ebc3fd78ee1e57b4fd086dd4f440fbfc66a2afddaef1ad41bdba0fb39cb487",
     "AddWithdrawalPermissionToConsentProcess", "OHCHR.txt",
     "Add a withdrawal permission to the consent process so the data subject can fully respect their privacy at every stage of the proceedings.",
     "You may withdraw your consent at any time.",
     None),

    ("3188e5d1696575695290a59966563be6e9830493254a18f7ee61ddf46986a393",
     "DeleteInfoWhenNoParentalConsent", "coppa_16cfr312.txt",
     "If the operator has not obtained parental consent for the collection of a parent or child's name or online contact information, replace any indefinite retention period with a bounded retention period (i.e., delete the information after a reasonable time).",
     "If parental consent has not been obtained, your information will be deleted within a reasonable time rather than retained indefinitely.",
     None),

    ("7799508da7e362554f4578a36e4a9b7335ae848e5ef6fe9ad67281dcf8c2891e",
     "InterpreterAssistanceRequirement", "OHCHR.txt",
     "If a child (DataSubject) cannot understand or speak the language used, add a DataProcessing node representing interpreter assistance and a grants edge from the associated ConsentProcess to this DataProcessing, thereby ensuring the child receives free interpreter assistance.",
     "If you cannot understand or speak the language used, free interpreter assistance will be provided.",
     "'Child' dropped -- Phi is unconditional (no age gate)."),

    ("3e62a311cc627fa1a3ed097e14fffdef25bd758e6d2b70645b524f4949122af6",
     "OperatorCollectsPersistentIdentifierInternalUse_NoNotice", "coppa_16cfr312.txt",
     "When an operator collects only a persistent identifier for the sole purpose of supporting internal website or online‑service operations, delete any existing sellsTo relationship to third parties, thereby removing the requirement to provide notice to data subjects.",
     "We do not sell or share your persistent identifier with third parties when it is used only to support our internal website or service operations.",
     None),

    ("48d0e8fdba86c14b460cc0654b65b85e4c417c7f1c98fc7308ea395c1639e81a",
     "Prohibit selling child data (UN)", "OHCHR.txt",
     "Operator must not sell personal data to ThirdParty to protect children from exploitation.",
     "We do not sell your personal data to third parties.",
     "'To protect children from exploitation'/child framing dropped -- Phi is unconditional (no age gate)."),

    ("bf53fedb863eefffcb05318ccf56599bdcab71ef421b56857a5fc04e655fab30",
     "ProhibitDataSellingWithoutConsent", "uk_aadc2.txt",
     "Remove any sellsTo relationship between the operator and third parties unless proper consent is obtained.",
     "We do not sell your data to third parties without your consent.",
     None),

    ("8913dbe8fe57dce11d41f1826d89196621b52cdc7986a701f9be0678424eb633",
     "ProhibitPersistentIdentifierSale", "coppa_16cfr312.txt",
     "Do not allow operators to sell persistent identifiers to third parties without appropriate consent.",
     "We do not sell your persistent identifier to third parties without appropriate consent.",
     None),

    ("9ef661e94b9947d5da7be1d4a0499795543e33570dedf49143063b250dbbabbf",
     "Provision 2", "OHCHR.txt",
     "Prohibit operators from selling data to third parties when the reservation is incompatible with the convention's purpose.",
     "We do not sell your data to third parties.",
     "Old template is treaty-reservation language, not a user-facing statement -- extraction-fidelity mismatch (same category as the earlier fidelity-audit finding), simplified to what R actually encodes (sellsTo removed)."),

    ("b1ccfb74badf69583b95870fecb3b3373f59a8736eeab9e7c1b14e8972bab7bf",
     "Provision0", "coppa_16cfr312.txt",
     "Operator must limit retention of child data to a bounded duration and delete it using reasonable security measures.",
     "We will retain your information only for a limited time and delete it using reasonable security measures.",
     "'Child' dropped -- Phi carries jurisdiction (US) only, no age gate."),

    ("59aece82386be85988c2e75fd9b26a22681bddf22ad0234a5c9c1e3a4b18d58f",
     "Provision0_OptOutIdentifierRestriction", "ccpa_civ_1798.txt",
     "When a consumer has opted out (Withdrawal), the business must not use or share their identifier (remove DataProcessing grant).",
     "Once you opt out, we will stop using or sharing your identifier.",
     None),

    ("c3bc3e4948b355647580e25568b4889dbd87384f7e66822ef6af9767f8d4a036",
     "Provision1", "ccpa_civ_1798.txt",
     "Retention must be limited to a bounded duration rather than indefinite.",
     "We will retain your information only for a limited time, not indefinitely.",
     None),

    ("6b907952ebe4b4f225932dee0334b9618da9ac394c6e9899855ea9d971c32abb",
     "Provision1", "uk_aadc2.txt",
     "Default settings must not allow unlimited sharing; remove unspecified sellsTo edges.",
     "By default, we do not share your data with third parties.",
     None),

    ("33b714c404f5c4a90ffb7828fe7e448da7c4891de3eb3d26b72d2d3e93c42ccc",
     "Provision1", "ccpa_civ_1798.txt",
     "Prohibit operators from selling consumer report data to third parties.",
     "We do not sell your consumer report data to third parties.",
     None),

    ("ddd030d02f418324db87eab951301a08b72d601557d88423bfec5c8c03a655ec",
     "Provision2", "coppa_16cfr312.txt",
     "(iv) Referral to the Commission of operators who engage in a pattern or practice of violating the self-regulatory guidelines; or",
     "We do not sell your data to third parties.",
     "Old template is a fragment of statutory enumeration, not a sentence at all -- extraction-fidelity mismatch, simplified to what R actually encodes (sellsTo removed)."),

    ("017fa099566e439eda89c109940e110ea49fe728787e1e432330cadadf0b95e2",
     "Provision2_ContractorThirdPartySharingProhibition", "ccpa_civ_1798.txt",
     "Prohibit operator from selling to third parties outside the direct business relationship.",
     "We do not sell your data to third parties outside our direct business relationship with you.",
     None),

    ("e477c6fddc4b07408eb10410afc84324d517c0e099c6a34910a53edf644d116d",
     "Provision3", "uk_aadc2.txt",
     "Replace indefinite retention with a bounded duration to avoid unlimited visibility of personal data.",
     "We will retain your information only for a limited time, not indefinitely.",
     None),

    ("1d274efd4422cb09cc59728d6bc870e170651c8a12f5afd25836a38d963bf3d4",
     "Provision4", "ccpa_civ_1798.txt",
     "Prohibit operators from selling personal data to third parties unless authorized by regulation.",
     "We do not sell your personal data to third parties unless authorized by regulation.",
     None),

    ("1a2490f719af18c502c5c8628292897d082710a143cd6192a1311adacf349ceb",
     "Provision4", "coppa_16cfr312.txt",
     "Do not treat a link to a child‑directed site as making the website child‑directed; remove the sellsTo edge.",
     "We do not sell your data to third parties.",
     "Old template is interpretive guidance to operators, not a user-facing statement -- extraction-fidelity mismatch, simplified to what R actually encodes (sellsTo removed)."),

    ("bcdfac426f8fb808082c72afc0cb7952608fde3fde54e1ccaeb7e1df8e0e746b",
     "RecruitmentAgeBoundRule", "OHCHR.txt",
     "Replace an indefinite duration clause with a bounded duration specifying that recruitment may only occur for persons aged at least fifteen years and less than eighteen years, thereby giving priority to the oldest within that range.",
     "We will retain your information only for a limited time, not indefinitely.",
     "Old template names ages 15/18 that this rule's Phi does NOT carry (age_op/age both null) -- a pre-existing extraction-fidelity defect (matches the earlier fidelity-audit finding for this exact rule: source text states explicit ages, extractor left age_op/age unset). All age language removed."),

    ("17d6a892dc40549d745643cf24ff6cb04af002e00ae9c7792916d1f616f21049",
     "RemoveParentalConsentWhenNotTargetedToChildren", "coppa_16cfr312.txt",
     "If a consent process does not target children as its primary audience, delete the ParentalConsent requirement.",
     "Because this service is not primarily directed to children, parental consent is not required for this collection.",
     "Subtractive rule (removes the ParentalConsent requirement) -- new template states the now-absent fact rather than an added obligation."),

    ("33eb9ee1dd4c8d15e871fec30b9edaad6d7ee76746dece3f1bb30d80d7d8cec9",
     "Replace indefinite retention period with a bounded duration", "OHCHR.txt",
     "Replace an indefinite data‑retention period with a specific bounded duration so that reports provide concrete information about how long data may be kept, satisfying the requirement for sufficient detail on implementation.",
     "We will retain your information only for a limited time, not indefinitely.",
     "Old template is CRC treaty state-reporting language, not a user-facing statement -- same extraction-fidelity category the earlier fidelity audit already flagged for OHCHR-sourced rules."),

    ("e3fcf2ccba772f3744d94670193af7ce926657d31529d9b1496ed44a274e1770",
     "Replace indefinite retention with a 30‑day bounded period after instrument deposit", "OHCHR.txt",
     "When a State deposits its instrument of ratification or accession, replace any indefinite retention period (DurationIndefinite) with a fixed 30‑day period (DurationBounded) to ensure the Convention enters into force thirty days after the deposit.",
     "We will retain your information only for a limited time, not indefinitely.",
     "Old template names '30 days', but R only encodes a generic DurationBounded node -- no specific day count is structurally present. '30 days' removed as an unverified over-claim. Old template is also treaty entry-into-force language, unrelated to data retention -- same extraction-fidelity category as the earlier fidelity-audit finding for this exact rule."),

    ("75193b645534babe0b18cd68698513c3fdc2fba7c29256f968dded80814f0d74",
     "Replace indefinite retention with bounded retention to comply with limitation on restrictions", "OHCHR.txt",
     "Replace any indefinite data retention period with a specific bounded duration, ensuring that restrictions on the exercise of data subject rights are limited to what is necessary, lawful, and proportionate.",
     "We will retain your information only for a limited time, not indefinitely.",
     "Old template is CRC limitation-on-restrictions language, not a user-facing statement -- same extraction-fidelity category as the earlier fidelity-audit finding."),

    ("47443c1b6dee27afb3933639efbea88acce5cf32fc4d8e5a6d288bc058d9151f",
     "ReplaceIndefiniteRetentionWithBoundedDuration", "OHCHR.txt",
     "Replace any indefinite data‑retention period with a specific bounded duration, ensuring that any restriction on the exercise of the data‑subject's right is limited to what is required by law and necessary.",
     "We will retain your information only for a limited time, not indefinitely.",
     "Old template is CRC rights-restriction language, not a user-facing statement -- same extraction-fidelity category as the earlier fidelity-audit finding."),

    ("35bab113cb9fc920f27b6f60a577e661d95385df274f720a98c10218f799eaf3",
     "RequireParentalConsentForChildDataProcessing", "OHCHR.txt",
     "Add a ParentalConsent node and a 'requires' edge from the ConsentProcess to ensure that any consent process involving a child includes a parental consent requirement, thereby protecting the child from discrimination based on parental status, activities, opinions, or beliefs.",
     "A parent or legal guardian must give consent before we collect your information.",
     "'Child'/discrimination framing dropped -- Phi is unconditional (no age gate). Old template also describes a graph-edit operation to a drafter ('Add a ParentalConsent node and a requires edge'), not a sentence addressed to a reader -- the exact defect Task I targets."),

    ("955b5cb74d805d744270c9a1e2809e10c443f6f201442a41ce659126b176eee7",
     "RequireParentalConsentForChildProtection", "OHCHR.txt",
     "Add a requirement that the consent process must obtain parental consent to protect the child's well‑being, reflecting the obligation to consider the rights and duties of parents or legal guardians.",
     "A parent or legal guardian must give consent before we collect your information.",
     "Drafter-instruction phrasing ('Add a requirement...') replaced with a reader-facing statement; Phi is unconditional (no age gate), so no age framing added."),

    ("6d234a1d4bb79145a1807b23de8153ba3034561d50384fd1ca20a567a81a2db8",
     "RequireParentalConsentForChildren", "coppa_16cfr312.txt",
     "Add a requirement for verifiable parental consent to the consent process for children under 13.",
     "If you are under 13, a parent or guardian must give verifiable consent before we collect your information.",
     "THE rule from the P1 pilot's actual bad output. Phi carries age_op='<', age=13, jurisdiction='US' -- the 'under 13' condition IS in Phi, so stating it is accurate (unlike the unconditional rules above)."),

    ("9db8f19ab63934788a6bace3665dae54ed661e28a6cc9d6a5639ae271776985e",
     "StatuteOfLimitations_AdminAction", "ccpa_civ_1798.txt",
     "Administrative actions must be commenced within five years of the violation; fraudulent concealment tolls the period.",
     "We will retain your information only for a limited time, not indefinitely.",
     "Old template is an admin-enforcement statute-of-limitations clause, completely unrelated to what R actually encodes (indefinite retention -> bounded) -- a severe extraction-fidelity mismatch, simplified to the structural delta."),

    ("84a9763dac0ecd8eb22ebcc6c5d17700cd88fc42fc8920d5053bd7e71dff71da",
     "Under13ParentalConsentRequirement", "coppa_16cfr312.txt",
     "Add a parental consent requirement to the consent process for data subjects identified as under age 13 before any collection, use, or disclosure of personal information is permitted.",
     "If you are under 13, a parent or guardian must give consent before we collect, use, or share your personal information.",
     "Phi carries age_op='<', age=13, jurisdiction='US' -- stating the age condition is accurate here."),
]

assert len(CORRECTIONS) == 31, len(CORRECTIONS)
assert len({c[0] for c in CORRECTIONS}) == 31, "duplicate content_hash"

corrections_map = {c[0]: c[4] for c in CORRECTIONS}
(REPO_ROOT / "analysis" / "realization_templates_corrections.json").write_text(
    json.dumps(corrections_map, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {len(corrections_map)} corrections to analysis/realization_templates_corrections.json")

# ---- build the markdown record ----
lines = [
    "# Realization template corrections",
    "",
    "**Correction made after a pilot run and before any evaluation data was generated.** "
    "The pilot's CP-GTR output for context (A=9, J=US) appended the sentence "
    "\"Add a requirement for verifiable parental consent to the consent process for children under 13.\" -- "
    "an instruction to a drafter, not a sentence addressed to a reader; any annotator would mark it non-compliant. "
    "This defect was not unique to that one rule -- every rule's `template` field comes straight from Stage-1 extraction "
    "(`cpgtr/extract.py`, `cache/canonical_extraction_cache.jsonl`), which was never asked to produce reader-facing prose, "
    "only a structured L/K/R/Phi spec. All 31 unique rules across LIB-FULL (and its 13-rule LIB-FAITHFUL-P subset) are "
    "corrected below.",
    "",
    "**Constraints applied to every rewrite:** state only what the rule's `R` adds (or removes, for a subtractive/"
    "substitutive rule) relative to `L`; never add an obligation the rule does not structurally encode; never name an "
    "age or jurisdiction the rule's own `Phi` (its `age_op`/`age`/`jurisdiction` fields) does not carry. Two rules "
    "(`RequireParentalConsentForChildren`, `Under13ParentalConsentRequirement`) genuinely have `age_op='<', age=13, "
    "jurisdiction='US'` in `Phi`, so stating \"if you are under 13\" for those two is accurate, not an over-claim -- "
    "every other rule's `Phi` is either fully unconditional or gated on jurisdiction only, so no age language appears "
    "in their new templates.",
    "",
    "**A number of old templates turned out to be extraction-fidelity mismatches independent of the drafter/reader "
    "framing issue** -- source text about treaty reporting obligations, entry-into-force clauses, or admin-enforcement "
    "statutes of limitations, extracted with a `template` field that doesn't describe a user-facing consequence at all "
    "(the same category of defect the earlier fidelity audit already flagged for several OHCHR-sourced "
    "rules). Flagged per rule below; not a new finding for the two rules the earlier audit already named "
    "(`RecruitmentAgeBoundRule`'s unverified ages, the CRC entry-into-force clause), but several more instances turned "
    "up while doing this correction.",
    "",
    "Implementation: `analysis/realization_templates_corrections.json` (content_hash -> new template), applied by "
    "`analysis/phase2_harness.py: apply_template_corrections()` immediately after `build_rules_for_library()` "
    "reconstructs Rule objects -- `Rule.content_hash()` is computed from L/K/R structure + name + source, NOT the "
    "template text (verified: `cpgtr/rules.py: Rule.content_hash()`), so this does not change any rule's content_hash "
    "and does not invalidate `phase2_library.json`'s frozen SHA-256 hashes (which cover rule membership, not template "
    "text). `cache/canonical_extraction_cache.jsonl` (the raw extraction record) is left untouched -- the correction is "
    "applied at load time, not by editing the extraction provenance.",
    "",
    "| Rule | Source | Old template | New template | Note |",
    "|---|---|---|---|---|",
]
for ch, name, source, old, new, note in CORRECTIONS:
    note_s = note.replace("|", "\\|") if note else ""
    old_s = old.replace("|", "\\|")
    new_s = new.replace("|", "\\|")
    lines.append(f"| `{name}` | `{source}` | {old_s} | **{new_s}** | {note_s} |")

(REPO_ROOT / "analysis" / "realization_templates.md").write_text("\n".join(lines), encoding="utf-8")
print("Wrote analysis/realization_templates.md")
