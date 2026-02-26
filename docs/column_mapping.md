# Kowope Mart Dataset — Column Mapping

> Source: Zindi DSN 2020 AI Bootcamp Qualification Hackathon variable descriptions.

## Dataset Size
- **Train**: 56,000 rows
- **Test**: 24,000 rows
- **Total**: 80,000 rows
- ⚠️ This is the FULL dataset — there is no 2M-row version.

## Column Descriptions

### ID & Target
| Column | Description |
|--------|-------------|
| `Applicant_ID` | Unique Customer Application Identification number |
| `default_status` | Target variable (yes/no — whether customer defaulted) |

### Risk Scoring (2 features)
| Column | Description |
|--------|-------------|
| `form_field1` | Customer Creditworthiness score based on historical data |
| `form_field2` | Score measuring the number and riskiness of credit inquiries |

### Default Severity (3 features)
| Column | Description |
|--------|-------------|
| `form_field3` | Severity of default on any loan(s) |
| `form_field4` | Severity of default on auto loan(s) |
| `form_field5` | Severity of default on education loan(s) |

### Credit Amounts — in NGN (8 features)
| Column | Description |
|--------|-------------|
| `form_field6` | Min credit available on all revolving credit cards |
| `form_field7` | Max credit available on active credit lines |
| `form_field8` | Max credit available on all revolving credit cards |
| `form_field9` | Sum of available credit on cards with 1 missed payment |
| `form_field10` | Total credit available on accepted credit lines |
| `form_field11` | Dues collected post-default (due > 500 NGN) |
| `form_field12` | Sum of amount due on active credit cards |
| `form_field13` | Annual amount paid towards all credit cards |

### Credit Utilization (4 features)
| Column | Description |
|--------|-------------|
| `form_field19` | # active credit cards with ≥75% utilization |
| `form_field20` | # active credit lines with ≥75% utilization |
| `form_field21` | Avg utilization of active revolving credit cards (%) |
| `form_field22` | Avg utilization of credit lines activated in last 2 years (%) |

### Behavioral / Risk Indicators
| Column | Description |
|--------|-------------|
| `form_field42` | Financial stress index of the borrower |
| `form_field43` | # credit lines with no missed payments in 2yr, but flagged high-risk by market prediction |
| `form_field44` | Ratio: max amount due on active lines / sum of amounts due |
| `form_field45` | # mortgage loans with 2 missed payments |
| `form_field46` | # auto loans with 2 missed payments |

### Categorical
| Column | Description |
|--------|-------------|
| `form_field47` | Type of product the applicant applied for (only categorical feature) |

### Other
| Column | Description |
|--------|-------------|
| `form_field14` | (int64, no nulls — description not found, likely a count or flag) |
| `form_field48` | Undefined Variable (per Zindi) |
| `form_field15–18, 23–41, 49–50` | Descriptions not available in public sources — likely additional credit bureau metrics |

## Key Takeaway
These are **credit bureau variables** — NOT raw application fields like Loan_Amount, Total_Income, or LGA.
The roadmap's preprocessing assumptions (LGA correction, currency parsing, geo-zoning) do not apply.
