# Unsupervised → Validate → Supervised: Applied to SAP MM Data Migration

## Context

When migrating data into SAP Materials Management (MM), organisations extract raw legacy data from old ERPs, spreadsheets, or home-grown systems. This data arrives **without clean SAP field mappings, with duplicates, and with inconsistent quality**.

The unsupervised → validate → supervised pipeline fits perfectly here:

- **Unsupervised** — let the raw legacy data reveal its own natural clusters, duplicates, and anomalies before anyone manually reviews it
- **Validate** — SAP MM consultants and business owners (material master team, procurement, finance) confirm what each discovered group actually means in SAP terms
- **Supervised** — train a model that applies those validated decisions at scale across millions of records, and continues working for post-go-live incoming data

---

## Key SAP MM Tables Referenced

| Table | Description |
|---|---|
| `MARA` | General Material Data (material number, type, base UoM, material group) |
| `MARC` | Plant-specific Material Data (MRP type, procurement type, lot size) |
| `MARD` | Storage Location Stock Data |
| `MAKT` | Material Descriptions (multilingual) |
| `MBEW` | Material Valuation (valuation class, price control, moving average price) |
| `EKKO` | Purchasing Document Header (PO, contract) |
| `EKPO` | Purchasing Document Item (material, qty, net price, plant) |
| `LFA1` | Vendor Master — General Data (name, country, tax ID) |
| `LFB1` | Vendor Master — Company Code Data (payment terms, reconciliation account) |
| `EINA` | Purchasing Info Record — General Data |
| `EINE` | Purchasing Info Record — Purchasing Org Data (price, conditions) |

---

## Use Case 1 — Material Master Deduplication (MARA / MAKT)

### Problem
Legacy system has 150,000 material records. No consistent naming convention — the same physical item exists under multiple descriptions:
- `"Steel Bolt M8 x 25"`, `"BOLT M8X25 SS"`, `"Bolt, stainless, M8, 25mm"`

SAP requires one clean `MATNR` per unique material.

### Unsupervised Step
Apply **TF-IDF vectorisation + cosine similarity clustering** (or fuzzy string matching + DBSCAN) across `MAKT-MAKTX` (material description) combined with numeric attributes from `MARA` (base unit, weight, material group `MATKL`).

Clusters emerge grouping near-identical items regardless of how they were named in the legacy system.

### Validation
The Material Master team reviews each cluster:
- Cluster of 3 → confirmed duplicates → keep one, map others to it
- Cluster of 2 → different grades (304 vs 316 stainless) → keep both, add distinguishing attribute

They produce a **merge map**: `legacy_id → canonical_MATNR`.

### Supervised Step
Train a **binary classifier (XGBoost or Sentence-BERT fine-tuned)** on the validated merge map:
- Input: pair of material descriptions + attributes
- Output: `duplicate` / `distinct`

This model handles the remaining 80% of records automatically, and continues deduplicating materials created post-go-live.

### SAP Migration Impact
Eliminates split stock, redundant info records (`EINA`/`EINE`), and inflated material counts in `MARA` before the cutover load.

---

## Use Case 2 — Material Type Classification (MARA-MTART)

### Problem
Legacy system has a single `item_category` field with 40 inconsistent values mapped by different teams over 20 years. SAP MM requires every material to have a valid `MTART` from a defined set:

| SAP MTART | Meaning |
|---|---|
| `ROH` | Raw material |
| `HALB` | Semi-finished product |
| `FERT` | Finished product |
| `HAWA` | Trading goods |
| `DIEN` | Service |
| `NLAG` | Non-stock material |
| `VERP` | Packaging material |

### Unsupervised Step
**K-Means / Hierarchical Clustering** on `MARA` attributes:
- `MATKL` (material group), `MEINS` (base UoM), `NTGEW` (net weight), `BRGEW` (gross weight), `MTPOS_MARA` (general item category group), procurement data from `MARC`

Seven natural clusters emerge — closely matching the seven common `MTART` values.

### Validation
SAP MM consultants review 50–100 samples from each cluster and assign the correct `MTART`. Edge cases (e.g. a packaging item used as a traded good) are escalated to the business owner.

### Supervised Step
A **multi-class Random Forest** is trained on the consultant-labeled records:
- Input: legacy attributes
- Output: predicted `MTART`

Achieves ~94% accuracy on held-out test set. Remaining 6% flagged for manual review.

### SAP Migration Impact
`MTART` controls which SAP screens and fields are active for a material. Wrong assignment = wrong configuration = failed transactions at go-live. Getting this right in migration prevents costly post-go-live master data corrections.

---

## Use Case 3 — Vendor Master Deduplication & Consolidation (LFA1 / LFB1)

### Problem
Three legacy systems being merged into one SAP instance. Each had its own vendor master. The same supplier (e.g. a global logistics company) appears as:
- `"DHL Supply Chain Ltd"` in System A
- `"DHL Logistics"` in System B
- `"Deutsche Post DHL"` in System C

SAP requires one `LIFNR` (vendor account) per legal entity per company code.

### Unsupervised Step
**Record linkage clustering** using:
- Fuzzy name matching (Jaro-Winkler similarity on `LFA1-NAME1`)
- Exact match on tax ID (`LFA1-STCD1`/`STCD2`) and bank account (`LFBK`)
- Address similarity (`LFA1-STRAS`, `ORT01`, `PSTLZ`, `LAND1`)

DBSCAN identifies clusters of vendor records that are likely the same legal entity across the three systems.

### Validation
The Accounts Payable / Procurement team reviews each cluster:
- Confirms true duplicates (same legal entity, same payment terms)
- Separates legitimate distinct entities (same name, different country subsidiaries)
- Chooses the "golden record" to become the SAP `LIFNR`, merging `LFB1` payment terms and `LFM1` purchasing data

### Supervised Step
A **gradient-boosted classifier** trained on the validated pairs:
- Input: name similarity score, tax ID match flag, address similarity, bank account match
- Output: `same_vendor` / `different_vendor`

Applied to all remaining unresolved pairs and to new vendor creation requests post-go-live.

### SAP Migration Impact
Duplicate vendors cause duplicate payments (AP risk), split purchasing history, and inflated vendor count. Consolidation before migration reduces `LFA1` records by 20–40% in typical multi-system merges.

---

## Use Case 4 — Valuation Class Assignment (MBEW-BKLAS)

### Problem
SAP's `BKLAS` (valuation class in `MBEW`) links materials to G/L accounts in Finance. Every material needs the correct valuation class or goods movements post-go-live post to the wrong accounts. Legacy systems have no equivalent field.

Common valuation classes:

| BKLAS | Typical Use |
|---|---|
| `3000` | Raw materials |
| `7900` | Trading goods |
| `7920` | Finished products |
| `3100` | Spare parts |

### Unsupervised Step
**Hierarchical clustering** on `MARA` and `MARC` attributes:
- `MATKL` (material group), `MTART`, `MTPOS_MARA`, `BESKZ` (procurement type: in-house vs external), `SOBSL` (special procurement key)
- Historical purchase price range from `EINA-NETPR`

Clusters naturally align with Finance's valuation class boundaries.

### Validation
The Finance / Controlling team reviews cluster-to-`BKLAS` proposals:
- Confirms accounting logic (does this cluster belong on the balance sheet as raw material inventory or finished goods inventory?)
- Adjusts any clusters that cross G/L account boundaries for regulatory reasons

### Supervised Step
A **Decision Tree classifier** (chosen for its explainability — Finance needs to audit it) is trained on the validated assignments.

The tree's rules are exported and reviewed by the Finance Controller before deployment, providing a human-readable audit trail alongside the ML model.

### SAP Migration Impact
Incorrect `BKLAS` causes financial misstatement from day one of go-live. Automating this with a validated, explainable model replaces weeks of manual spreadsheet mapping.

---

## Use Case 5 — Purchase Order Anomaly Detection & Data Quality Scoring (EKKO / EKPO)

### Problem
Historical PO data being migrated to SAP contains years of entry errors:
- Zero-price line items (`EKPO-NETPR = 0`)
- Quantities ordered in wrong UoM (1000x or 0.001x expected)
- Delivery dates in the past or 50 years in the future
- Plant/storage location combinations that do not exist in the new SAP system

No error labels exist — nobody logged "this PO line was a data entry mistake" at the time.

### Unsupervised Step
**Isolation Forest + statistical outlier detection** on `EKPO` features:
- Price deviation from `EINA` info record price (if available)
- Quantity vs. historical average for that material/vendor pair
- Date range sanity checks
- `WERKS` (plant) and `LGORT` (storage location) combination validity

Anomaly scores assigned to every PO line. High-score records clustered by anomaly type.

### Validation
The Purchasing / Data Migration team reviews high-scoring records by anomaly cluster:
- Confirms which are genuine data entry errors → labelled `bad_data`
- Confirms which are legitimate exceptions (e.g. free-of-charge sample delivery at zero price) → labelled `valid_exception`
- Defines a **data quality threshold** score for go/no-go migration cutoff

### Supervised Step
An **XGBoost classifier** trained on the labelled anomaly records:
- Scores every remaining PO line as `migrate_as-is` / `needs_review` / `exclude`
- Integrated into the migration ETL pipeline as an automated data quality gate

Post-go-live, the same model validates new PO lines created via EDI or batch interfaces before they reach `EKKO`/`EKPO`.

### SAP Migration Impact
Migrating bad PO data creates phantom commitments in Controlling, incorrect open item lists, and wrong MRP inputs. The anomaly model acts as an automated data quality gate before cutover.

---

## End-to-End Migration Pipeline Summary

```
Legacy System Extract (raw tables)
        │
        ▼
 Unsupervised Learning
 (clustering, anomaly detection, record linkage)
        │
        ▼
 Domain Validation
 (MM consultants, Finance, Procurement, Data owners)
        │
        ▼
 Labelled Dataset Created
        │
        ▼
 Supervised Model Trained
        │
   ┌────┴─────────────────────┐
   ▼                          ▼
Bulk auto-label          Flag edge cases
remaining records        for manual review
   │
   ▼
SAP Migration Load
(LSMW / BAPI / IDoc / S/4HANA Migration Cockpit)
        │
        ▼
 Post Go-Live:
 Same model validates ongoing master data creation
```

---

## Summary Table

| Use Case | SAP Tables | Unsupervised Method | Validated By | Supervised Method | SAP Field / Risk Prevented |
|---|---|---|---|---|---|
| Material Deduplication | MARA, MAKT | TF-IDF + DBSCAN | Material Master team | Sentence-BERT classifier | MATNR duplicates |
| Material Type Classification | MARA, MARC | K-Means | MM Consultants | Random Forest (multi-class) | MTART wrong config |
| Vendor Consolidation | LFA1, LFB1 | Record linkage + DBSCAN | AP / Procurement | GBT binary classifier | LIFNR duplicates, duplicate payments |
| Valuation Class Assignment | MBEW, MARA | Hierarchical Clustering | Finance / Controlling | Decision Tree (explainable) | BKLAS financial misstatement |
| PO Data Quality | EKKO, EKPO | Isolation Forest | Purchasing / Migration team | XGBoost quality gate | Bad data entering SAP |

---

## Key Takeaways for SAP Projects

1. **SAP field values are the labels.** The migration project's job is to discover what SAP value each legacy record should receive — unsupervised learning surfaces candidate groupings; domain experts assign the SAP value.

2. **Explainability matters in SAP Finance contexts.** Use Decision Trees or logistic models where Finance Controllers need to audit the classification logic, not just trust a black-box score.

3. **The model becomes a migration ETL component.** Once validated, the supervised model plugs into LSMW, BAPI wrapper scripts, or the S/4HANA Migration Cockpit as an automated pre-load quality filter.

4. **Post-go-live value.** The same trained model validates master data created after go-live — preventing the same data quality issues from re-entering SAP through manual entry or inbound interfaces.
