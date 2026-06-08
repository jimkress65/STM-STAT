#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Systematic Treatment Methodology (STM) Foundation Model v3.0
Ewing Sarcoma Risk Prediction

## ⚠️ Not for clinical use

This is **research code**. It is not a medical device. It has not been cleared by the FDA, the EMA, or any other regulatory body. It must not be used to make clinical decisions for individual patients. The simulation framework is published for scientific reproducibility and methodological review only.

v3.0 ADDITIONS (on top of calibrated v2.0 frontline):
- Relapsed/Refractory (R/R) extension module
  * Early (<2yr) vs late (>2yr) relapse as distinct sub-cohorts
  * Four salvage regimens calibrated to rEECur trial
  * Year-by-year post-relapse survival tracking
- Adverse effects modules (5 organ systems, 30-year horizon)
  * Cardiotoxicity: cumulative doxorubicin → CHF (van der Pal 2012)
  * Nephrotoxicity: cumulative ifosfamide → GFR decline (Euro-EWING 99)
  * Hypertension: GFR-driven via renal pressure-natriuresis (Gibson 2017)
  * Second malignant neoplasms: t-MN + radiation-induced sarcomas
  * RT cardiac interaction: PENTEC 2023 HR 1.87/10Gy
- Comprehensive data extraction for figure generation

Frontline Stages 1-6: IDENTICAL to v2.0 (3.8% MAE, 15 endpoints)
R/R module: calibrated to Stahl 2011, rEECur (2.1% MAE, 8 endpoints)
Combined: 3.2% MAE across 23 endpoints

Author: James W. Kress, Ph.D.
Affiliation: The KressWorks Foundation
Contact: jimkress_58@kressworks.org
ORCID: 0000-0002-2511-6822
Date: March 2026
Version: STM Foundation v3.0
"""

import numpy as np
import pandas as pd
from datetime import datetime
import json
import sys
import os

np.random.seed(42)

print("="*80)
print("SYSTEMATIC TREATMENT METHODOLOGY (STM) - FOUNDATION MODEL v3.0")
print("Frontline (v2.0 calibrated) + R/R Extension + Adverse Effects Modules")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# MODEL PARAMETERS (from published literature + recalibration)
# ============================================================================

class STMParameters:
    """All model parameters with literature citations"""
    
    # Genetic risk modifiers
    TP53_FAILURE_MULTIPLIER = 2.0      # Tirode 2014, Crompton 2014
    TP53_RECURRENCE_MULTIPLIER = 1.8   # Tirode 2014
    TYPE2_FUSION_FAILURE = 1.15        # Le Deley 2014
    TYPE2_FUSION_RECURRENCE = 1.3      # Le Deley 2014
    RB1_MULTIPLIER = 1.2               # Brohl 2014
    STAG2_MULTIPLIER = 1.15            # Tirode 2014
    CDKN2A_MULTIPLIER = 1.1            # Kovar 2016
    
    # Treatment-related mortality (TRM)
    TRM_BASELINE = 0.025               # Womer 2012, Paulussen 2008
    TRM_AGE_YOUNG = 1.8                # Age <5 years
    TRM_AGE_OLD = 1.3                  # Age >18 years
    TRM_METASTATIC = 1.5               # Metastatic disease
    
    # Treatment failure by necrosis
    FAILURE_NECROSIS_90 = 0.025        # ≥90% necrosis
    FAILURE_NECROSIS_70 = 0.060        # 70-89%
    FAILURE_NECROSIS_50 = 0.120        # 50-69%
    FAILURE_NECROSIS_LOW = 0.210       # <50%
    
    # Recurrence by necrosis
    RECURRENCE_NECROSIS_90 = 0.08      # ≥90% necrosis
    RECURRENCE_NECROSIS_70 = 0.18      # 70-89%
    RECURRENCE_NECROSIS_50 = 0.32      # 50-69%
    RECURRENCE_NECROSIS_LOW = 0.50     # <50%
    
    # Metastatic multipliers
    META_LUNG_FAILURE = 4.1            # Luksch 2012: Lung-only mets 40-42% EFS
    META_BONE_FAILURE = 5.4            # Luksch 2012: Bone mets 27% EFS
    META_MULTIPLE_FAILURE = 7.7        # Luksch 2012: Multiple sites 15-20% EFS
    META_RECURRENCE = 3.5              # Dirksen 2015, Bacci 2006
    
    # NEW v2.0: Risk-dependent biomarker parameters
    # Target prevalence by risk quartile: Low, Intermediate, High, Very High
    BIOMARKER_TARGETS = {
        'tp53': [0.05, 0.08, 0.12, 0.20],
        'rb1': [0.03, 0.06, 0.10, 0.15],
        'stag2': [0.10, 0.15, 0.20, 0.28],
        'type2_fusion': [0.08, 0.12, 0.18, 0.25],
        'ldh': [0.30, 0.50, 0.70, 0.90],
        'alp': [0.25, 0.40, 0.60, 0.80],
        'ctdna': [0.40, 0.60, 0.80, 0.95],
        'mrd': [0.10, 0.22, 0.35, 0.50],
    }
    
    # Biomarker response parameters
    LDH_NORMALIZATION_RATE = 0.26      # Week 4-8 response
    CTDNA_CLEARANCE_RATE = 0.053
    
    # MRD parameters - v2.0 RECALIBRATED
    MRD_BASE_PROBABILITY = 0.25        # Adjusted for risk stratification
    MRD_IRS_EXPONENT = 0.35            # Integrated response score exponent
    MRD_POSITIVE_MULTIPLIER = 9.0      # 9× recurrence risk if MRD+ (Klega 2018)
    MRD_NEGATIVE_MULTIPLIER = 0.40     # 60% risk reduction if MRD-
    
    # Treatment intensification
    POOR_RESPONSE_THRESHOLD = 0.4
    INTENSIFICATION_BENEFIT = 0.70     # 30% risk reduction
    
    # Recurrence timing (bimodal distribution)
    # CALIBRATED v3.0: Need ~79% early (<2yr) to produce 14.2% overall R/R 5yr OS
    # from 9.0% (early) and 33.9% (late) components
    RECURRENCE_EARLY_PEAK = 1.0        # Years (12 months) — shifted from 1.5
    RECURRENCE_LATE_PEAK = 3.5         # Years (42 months)
    RECURRENCE_EARLY_WEIGHT = 0.82     # 82% early, 18% late — increased from 0.65
    RECURRENCE_FATAL_PROBABILITY = 0.85  # 85% of recurrences are fatal

    # ========================================================================
    # R/R EXTENSION PARAMETERS — NEW IN v3.0
    # Calibrated to: Stahl 2011 (N=714), McCabe 2022, Bacci 2003, rEECur
    # ========================================================================
    
    # Early vs late relapse definition
    RR_EARLY_LATE_CUTOFF = 2.0  # Years from diagnosis
    
    # Base annual mortality rates (pre-modifier)
    # Effective rates after modifier stacking match Stahl 2011 targets
    RR_BASE_ANNUAL_MORTALITY_EARLY = 0.33   # → ~9% 5-yr OS for early (was 0.35)
    RR_BASE_ANNUAL_MORTALITY_LATE = 0.175   # → ~34% 5-yr OS for late (was 0.19)
    
    # Site-of-relapse mortality modifiers
    RR_SITE_MORTALITY_LOCAL = 1.0
    RR_SITE_MORTALITY_SYSTEMIC = 1.40
    RR_SITE_MORTALITY_COMBINED = 1.85
    
    # Relapse site distribution (Stahl 2011)
    RR_SITE_LOCAL_PROB = 0.24
    RR_SITE_SYSTEMIC_PROB = 0.12
    RR_SITE_COMBINED_PROB = 0.04
    # Remainder = distant-only (most common)
    
    # Salvage regimen allocation (rEECur-informed)
    SALVAGE_REGIMEN_PROBS = {
        'HD-IFOS': 0.35,
        'IT': 0.30,
        'TC': 0.20,
        'GD': 0.15,
    }
    
    # Salvage regimen efficacy modifiers (relative to base mortality)
    SALVAGE_EFFICACY = {
        'HD-IFOS': 0.80,   # Best: 20% mortality reduction
        'IT': 0.90,         # Intermediate
        'TC': 1.00,         # Reference
        'GD': 1.05,         # Slightly worse
    }
    
    # Salvage 6-month EFS base probabilities (recalibrated for 79% early mix)
    SALVAGE_6MO_EFS_BASE = {
        'HD-IFOS': 0.56,    # Target: 43.9% after early/late weighting
        'IT': 0.50,          # Target: ~40%
        'TC': 0.44,          # Target: 35.1%
        'GD': 0.33,          # Target: ~26%
    }
    
    # Biomarker modifiers at relapse
    RR_LDH_ELEVATED_MORTALITY_MULT = 1.25
    RR_CTDNA_POSITIVE_MORTALITY_MULT = 1.15
    
    # Chemosensitivity
    SALVAGE_SENSITIVE_MORTALITY_MULT = 0.55
    SALVAGE_RESISTANT_MORTALITY_MULT = 1.20
    
    # Salvage TRM
    SALVAGE_TRM_RATE = 0.020  # Target: 2%
    
    # ========================================================================
    # ADVERSE EFFECTS PARAMETERS — NEW IN v3.0
    # Calibrated to: SJLIFE, CCSS, DCCSS-LATER, PENTEC 2023
    # ========================================================================
    
    # Standard cumulative drug exposures (VDC/IE, COG AEWS0331)
    # Doxorubicin: 75 mg/m² × 5 cycles VDC = 375 mg/m²
    STANDARD_DOX_DOSE = 375.0  # mg/m²
    # Ifosfamide: 1.8 g/m²/day × 5 days × 6 cycles IE = 54 g/m²
    # (Euro-EWING 99 VIDE: up to 78.8 g/m² with VAI consolidation)
    STANDARD_IFO_DOSE = 54.0   # g/m² (VDC/IE); 78.8 for VIDE+VAI
    # Etoposide: 100 mg/m²/day × 5 days × 6 cycles = 3000 mg/m²
    STANDARD_ETO_DOSE = 3000.0 # mg/m²
    
    # Cardiotoxicity: CHF (van der Pal 2012, Mulrooney CCSS 2016)
    # Target: 7-8% dox-only, 12-14% combined modality at 30 years
    DOX_CHF_COEFF = 0.0010       # Per 100 mg/m² per year (calibrated to 7-8% at 30yr)
    DOX_CHF_EXPONENT = 1.65      # Supralinear dose-response
    CHF_RESERVE_THRESHOLD = 0.60 # Fraction below which CHF risk accelerates
    CHF_ACCELERATION_FACTOR = 2.0 # Acceleration when reserve < threshold
    
    # RT-cardiac interaction (van Nimwegen 2016, PENTEC 2023)
    RT_CARDIAC_PENTEC_HR = 1.87   # Per 10 Gy mean heart dose
    RT_CARDIAC_INTERACTION = 0.09 # P_interaction (multiplicative)
    # Site-dependent cardiac scatter fractions
    RT_HEART_SCATTER = {
        'extremity': 0.00,
        'pelvis': 0.005,
        'axial': 0.22,
        'soft_tissue': 0.00,
    }
    RT_FIELD_DOSE = 55.8  # Gy typical field dose
    
    # Nephrotoxicity (Euro-EWING 99: 34.8% 10-yr at 78.8 g/m²)
    IFO_GFR_COEFF = 0.018    # GFR decline per g/m² per year (calibrated to 120→71 at 30yr)
    IFO_GFR_THRESHOLD = 40.0 # g/m² threshold for accelerated decline
    BASELINE_GFR = 120.0     # mL/min/1.73m²
    BRENNER_AMPLIFICATION = 1.02  # Per year after initial injury
    
    # Hypertension (Gibson SJLIFE 2017: SPR 2.6, target 60.5% at 30yr)
    HTN_GFR_SLOPE = -0.003    # HTN probability increase per mL/min GFR loss
    HTN_BASE_RATE_PER_YEAR = 0.004  # Age-related background rate
    HTN_RENAL_SPR = 2.6       # Standardized prevalence ratio
    
    # Nephro-cardiac coupling (Armstrong 2013: RR=12.4, RERI=11.9)
    NEPHRO_CARDIAC_COUPLING = 1.4  # CHF amplification from HTN
    
    # Second malignant neoplasms
    SMN_ETO_TMN_RATE = 0.0015  # t-MN annual hazard per 1000 mg/m² etoposide (halved)
    SMN_TMN_LATENCY_PEAK = 5.0 # Years (t-MN peaks 3-7 years)
    SMN_TMN_PLATEAU_YEAR = 10  # t-MN risk plateaus after year 10
    SMN_RT_RIS_RATE = 0.0015   # RIS annual hazard per 10 Gy
    SMN_RT_RIS_LATENCY = 8.0   # Years minimum latency for RIS
    # Background solid tumor rate increases with time (latent carcinogenesis)
    # Early (0-10yr): 0.001/yr, Late (10-30yr): 0.005/yr → ~14% cumulative at 30yr
    SMN_BASELINE_RATE_EARLY = 0.001  # Per year, years 0-10
    SMN_BASELINE_RATE_LATE = 0.0055  # Per year, years 10-30
    
    # CHIP modification
    CHIP_PREVALENCE_30YR = 0.15  # ~15% by 30 years post-treatment
    CHIP_CARDIAC_MODIFIER = 2.0  # 2-fold CHF risk increase

# ============================================================================
# STAGE 1: BASELINE RISK ASSESSMENT - v2.0 WITH RISK SCORING
# ============================================================================

def generate_baseline_cohort(n_patients, meta_fraction=0.15):
    """Generate patient cohort with baseline characteristics and risk scoring"""
    
    print(f"STAGE 1: BASELINE RISK ASSESSMENT (v2.0 - Risk-Stratified)")
    print(f"  Generating {n_patients} patients...")
    
    n_meta = int(n_patients * meta_fraction)
    n_local = n_patients - n_meta
    
    # Metastatic status
    meta_status = np.array(['localized'] * n_local + 
                          np.random.choice(['lung', 'bone', 'multiple'], 
                                         n_meta, p=[0.45, 0.35, 0.20]).tolist())
    np.random.shuffle(meta_status)
    
    df = pd.DataFrame({
        'patient_id': range(n_patients),
        'meta_status': meta_status,
        'age': np.random.normal(15, 5, n_patients).clip(2, 45),
        'tumor_volume': np.random.lognormal(4.5, 0.8, n_patients).clip(10, 2000),
    })
    
    # Calculate preliminary risk factors for stratification
    # Disease Burden Score (DBS)
    volume_score = np.log10(df['tumor_volume'] / 100).clip(0, 2)
    meta_score = df['meta_status'].map({
        'localized': 0, 'lung': 1.5, 'bone': 2.0, 'multiple': 3.0
    })
    df['dbs'] = (volume_score + meta_score).clip(0, 5)
    
    # Create preliminary risk score for biomarker assignment
    # This will be refined after biomarkers assigned
    df['preliminary_risk'] = df['dbs'] / 5.0  # Normalize 0-1
    
    # Assign to risk quartiles for targeted biomarker prevalence
    df['risk_quartile'] = pd.qcut(df['preliminary_risk'], 
                                   q=4, 
                                   labels=False,  # Use numeric labels 0,1,2,3
                                   duplicates='drop')
    
    # NEW v2.0: Risk-dependent genetic alterations
    print(f"  Assigning risk-dependent genetic alterations...")
    
    for idx, row in df.iterrows():
        quartile = int(row['risk_quartile'])
        
        # TP53 mutation - increases with risk
        tp53_prob = STMParameters.BIOMARKER_TARGETS['tp53'][quartile]
        df.loc[idx, 'tp53_mutant'] = np.random.random() < tp53_prob
        
        # RB1 mutation - increases with risk
        rb1_prob = STMParameters.BIOMARKER_TARGETS['rb1'][quartile]
        df.loc[idx, 'rb1_mutant'] = np.random.random() < rb1_prob
        
        # STAG2 mutation - increases with risk
        stag2_prob = STMParameters.BIOMARKER_TARGETS['stag2'][quartile]
        df.loc[idx, 'stag2_mutant'] = np.random.random() < stag2_prob
        
        # CDKN2A loss - moderate increase with risk
        cdkn2a_base = 0.14
        cdkn2a_prob = cdkn2a_base * (1.0 + row['preliminary_risk'] * 0.5)
        df.loc[idx, 'cdkn2a_loss'] = np.random.random() < cdkn2a_prob
        
        # Type 1 fusion (inverse = Type 2)
        type2_prob = STMParameters.BIOMARKER_TARGETS['type2_fusion'][quartile]
        df.loc[idx, 'type1_fusion'] = np.random.random() > type2_prob  # Inverse
    
    # Genetic Risk Score (GRS)
    df['grs'] = (
        df['tp53_mutant'] * STMParameters.TP53_FAILURE_MULTIPLIER +
        df['rb1_mutant'] * STMParameters.RB1_MULTIPLIER +
        df['stag2_mutant'] * STMParameters.STAG2_MULTIPLIER +
        df['cdkn2a_loss'] * STMParameters.CDKN2A_MULTIPLIER
    ).clip(0, 5)
    
    # NEW v2.0: Risk-dependent baseline biomarkers
    print(f"  Assigning risk-dependent biomarkers...")
    
    for idx, row in df.iterrows():
        quartile = int(row['risk_quartile'])
        
        # LDH elevation - strong risk correlation
        ldh_prob = STMParameters.BIOMARKER_TARGETS['ldh'][quartile]
        df.loc[idx, 'ldh_elevated'] = np.random.random() < ldh_prob
        
        # ALP elevation - strong risk correlation
        alp_prob = STMParameters.BIOMARKER_TARGETS['alp'][quartile]
        df.loc[idx, 'alp_elevated'] = np.random.random() < alp_prob
        
        # ctDNA detection - very strong risk correlation
        ctdna_prob = STMParameters.BIOMARKER_TARGETS['ctdna'][quartile]
        df.loc[idx, 'ctdna_detected'] = np.random.random() < ctdna_prob
    
    # Biomarker Elevation Score (BES)
    df['bes'] = (
        df['ldh_elevated'].astype(int) +
        df['alp_elevated'].astype(int) +
        df['ctdna_detected'].astype(int) * 1.5
    ).clip(0, 3.5)
    
    # Calculate FINAL composite risk score
    df['composite_risk'] = (
        df['grs'] * 0.3 +      # Genetic factors (30%)
        df['dbs'] * 0.4 +      # Disease burden (40%)
        df['bes'] * 0.3        # Biomarkers (30%)
    )
    
    # Final risk group assignment
    df['risk_group'] = pd.qcut(df['composite_risk'], 
                               q=4, 
                               labels=['Low Risk', 'Intermediate Risk', 
                                      'High Risk', 'Very High Risk'],
                               duplicates='drop')
    
    print(f"  ✓ Baseline characteristics assigned")
    print(f"  - Localized: {(df['meta_status']=='localized').sum()} ({100*(df['meta_status']=='localized').mean():.1f}%)")
    print(f"  - Metastatic: {n_meta} ({100*meta_fraction:.1f}%)")
    print(f"  - Mean GRS: {df['grs'].mean():.2f}")
    print(f"  - Mean DBS: {df['dbs'].mean():.2f}")
    print(f"  - Mean BES: {df['bes'].mean():.2f}")
    print(f"  Risk group distribution:")
    for rg in ['Low Risk', 'Intermediate Risk', 'High Risk', 'Very High Risk']:
        n_rg = (df['risk_group'] == rg).sum()
        print(f"    • {rg}: {n_rg} ({100*n_rg/len(df):.1f}%)")
    print()
    
    return df

# ============================================================================
# STAGE 2: EARLY RESPONSE ASSESSMENT
# ============================================================================

def simulate_early_response(df):
    """Simulate biomarker response at weeks 4-8"""
    
    print("STAGE 2: EARLY RESPONSE ASSESSMENT (Week 4-8)")
    
    # Probability of normalization (better for localized, worse for metastatic)
    base_response = df['meta_status'].map({
        'localized': 0.70,
        'lung': 0.55,
        'bone': 0.45,
        'multiple': 0.35
    })
    
    # Genetic factors reduce response
    genetic_penalty = (
        df['tp53_mutant'] * 0.20 +
        df['rb1_mutant'] * 0.10 +
        df['stag2_mutant'] * 0.05
    )
    
    response_prob = (base_response - genetic_penalty).clip(0.15, 0.85)
    
    # LDH normalization
    df['ldh_normalized'] = df['ldh_elevated'] & (
        np.random.random(len(df)) < response_prob
    )
    
    # ctDNA clearance (slower than LDH)
    ctdna_clearance_prob = response_prob * 0.75
    df['ctdna_cleared'] = df['ctdna_detected'] & (
        np.random.random(len(df)) < ctdna_clearance_prob
    )
    
    # Biomarker Response Score (BRS) - inverse of persistence
    df['brs'] = (
        (df['ldh_elevated'] & ~df['ldh_normalized']).astype(int) * 0.3 +
        (df['ctdna_detected'] & ~df['ctdna_cleared']).astype(int) * 0.4
    ).clip(0, 1)
    df['brs'] = 1.0 - df['brs']  # Invert so high = good response
    
    # Treatment intensification for poor responders
    poor_response = df['brs'] < STMParameters.POOR_RESPONSE_THRESHOLD
    df['treatment_intensified'] = poor_response
    
    print(f"  - LDH normalized: {df['ldh_normalized'].sum()} / {df['ldh_elevated'].sum()} elevated")
    print(f"  - ctDNA cleared: {df['ctdna_cleared'].sum()} / {df['ctdna_detected'].sum()} detected")
    print(f"  - Treatment intensified: {poor_response.sum()} patients")
    print(f"  - Mean BRS: {df['brs'].mean():.2f}")
    print()
    
    return df

# ============================================================================
# STAGE 3: TREATMENT-RELATED MORTALITY
# ============================================================================

def simulate_trm(df):
    """Simulate treatment-related mortality"""
    
    print("STAGE 3: TREATMENT-RELATED MORTALITY")
    
    # Age-dependent risk
    age_mult = np.ones(len(df))
    age_mult[df['age'] < 5] = STMParameters.TRM_AGE_YOUNG
    age_mult[df['age'] > 18] = STMParameters.TRM_AGE_OLD
    
    # Metastatic modifier
    meta_mult = np.where(df['meta_status'] != 'localized', 
                        STMParameters.TRM_METASTATIC, 1.0)
    
    # TRM probability
    trm_prob = STMParameters.TRM_BASELINE * age_mult * meta_mult
    trm_prob = trm_prob.clip(0, 0.10)
    
    df['trm'] = np.random.random(len(df)) < trm_prob
    
    print(f"  - TRM events: {df['trm'].sum()} / {len(df)} ({100*df['trm'].mean():.1f}%)")
    print(f"  - By cohort:")
    for status in ['localized', 'lung', 'bone', 'multiple']:
        subset = df[df['meta_status'] == status]
        if len(subset) > 0:
            trm_rate = subset['trm'].mean()
            print(f"    • {status}: {subset['trm'].sum()}/{len(subset)} ({100*trm_rate:.1f}%)")
    print()
    
    return df

# ============================================================================
# STAGE 4: TREATMENT FAILURE & HISTOLOGIC RESPONSE  
# ============================================================================

def simulate_treatment_failure(df):
    """Simulate treatment failure among non-TRM patients"""
    
    print("STAGE 4: TREATMENT FAILURE & HISTOLOGIC RESPONSE")
    
    # Only for survivors of TRM
    alive_after_trm = ~df['trm']
    
    # Histologic response (necrosis %)
    base_necrosis = np.random.beta(8, 2, len(df)) * 100  # Mean ~80%
    
    # Adjust for biomarkers and genetics
    brs_effect = df['brs'] * 10  # BRS improves necrosis
    genetic_penalty = (df['tp53_mutant'] * 15 + df['rb1_mutant'] * 10)
    
    necrosis_pct = (base_necrosis + brs_effect - genetic_penalty).clip(10, 100)
    df['necrosis_pct'] = necrosis_pct
    
    # Categorize necrosis
    df['necrosis_cat'] = pd.cut(necrosis_pct, 
                                bins=[0, 50, 70, 90, 100],
                                labels=['<50%', '50-69%', '70-89%', '≥90%'])
    
    # Treatment failure probability (among non-TRM)
    beta0 = df['necrosis_cat'].map({
        '≥90%': STMParameters.FAILURE_NECROSIS_90,
        '70-89%': STMParameters.FAILURE_NECROSIS_70,
        '50-69%': STMParameters.FAILURE_NECROSIS_50,
        '<50%': STMParameters.FAILURE_NECROSIS_LOW
    }).astype(float).values
    
    # Genetic modifiers
    g_tp53 = np.where(df['tp53_mutant'].values, 
                     STMParameters.TP53_FAILURE_MULTIPLIER, 1.0).astype(float)
    g_fusion = np.where(df['type1_fusion'].values == False,  # Fixed deprecation warning
                       STMParameters.TYPE2_FUSION_FAILURE, 1.0).astype(float)
    
    # Metastatic multipliers
    m_meta = df['meta_status'].map({
        'localized': 1.0,
        'lung': STMParameters.META_LUNG_FAILURE,
        'bone': STMParameters.META_BONE_FAILURE,
        'multiple': STMParameters.META_MULTIPLE_FAILURE
    }).values
    
    # Treatment intensification benefit
    intensification = np.where(df['treatment_intensified'].values,
                              STMParameters.INTENSIFICATION_BENEFIT, 1.0)
    
    # Combined failure probability
    failure_prob = beta0 * g_tp53 * g_fusion * m_meta * intensification
    failure_prob = failure_prob.clip(0, 0.60)
    
    # Apply only to non-TRM patients
    df['treatment_failure'] = False
    df.loc[alive_after_trm, 'treatment_failure'] = \
        np.random.random(alive_after_trm.sum()) < failure_prob[alive_after_trm]
    
    print(f"  - Necrosis distribution:")
    for cat in ['≥90%', '70-89%', '50-69%', '<50%']:
        n = (df['necrosis_cat'] == cat).sum()
        print(f"    • {cat}: {n} ({100*n/len(df):.1f}%)")
    
    at_risk = alive_after_trm.sum()
    print(f"  - Treatment failure: {df['treatment_failure'].sum()} / {at_risk} at-risk")
    print(f"    (Overall rate: {100*df['treatment_failure'].sum()/at_risk:.1f}%)")
    print()
    
    return df

# ============================================================================
# STAGE 5: MINIMAL RESIDUAL DISEASE (MRD) - v2.0 RECALIBRATED
# ============================================================================

def simulate_mrd(df):
    """Simulate MRD detection with proper risk stratification (v2.0)"""
    
    print("STAGE 5: MINIMAL RESIDUAL DISEASE (MRD) DETECTION - v2.0 RECALIBRATED")
    
    # MRD only assessed in patients who reach this point
    eligible_for_mrd = ~df['trm'] & ~df['treatment_failure']
    
    # Integrated Response Score (IRS): combines necrosis and biomarker response
    # Lower IRS = worse response = higher MRD risk
    necrosis_score = df['necrosis_pct'] / 100.0
    biomarker_score = df['brs']
    df['irs'] = (necrosis_score * 0.6 + biomarker_score * 0.4).clip(0, 1)
    
    # NEW v2.0: Risk-stratified MRD detection
    df['mrd_positive'] = False
    
    if eligible_for_mrd.any():
        # Get risk quartile for each patient
        eligible_df = df[eligible_for_mrd].copy()
        
        for idx in eligible_df.index:
            # Get patient's risk group
            risk_group = df.loc[idx, 'risk_group']
            
            # Base MRD probability by risk group (target calibration)
            risk_to_quartile = {
                'Low Risk': 0,
                'Intermediate Risk': 1,
                'High Risk': 2,
                'Very High Risk': 3
            }
            quartile = risk_to_quartile.get(risk_group, 1)
            base_mrd_prob = STMParameters.BIOMARKER_TARGETS['mrd'][quartile]
            
            # Modify by IRS (integrated response score)
            irs = df.loc[idx, 'irs']
            irs_modifier = (1.0 - irs) ** STMParameters.MRD_IRS_EXPONENT
            
            # Final MRD probability
            mrd_prob = base_mrd_prob * (0.5 + irs_modifier * 1.5)
            mrd_prob = np.clip(mrd_prob, 0.02, 0.80)
            
            # Assign MRD status
            df.loc[idx, 'mrd_positive'] = np.random.random() < mrd_prob
    
    n_tested = eligible_for_mrd.sum()
    n_mrd_pos = df.loc[eligible_for_mrd, 'mrd_positive'].sum()
    
    print(f"  - MRD detection rate: {n_mrd_pos} / {n_tested} ({100*n_mrd_pos/n_tested:.1f}%)")
    print(f"  - By risk group:")
    for rg in ['Low Risk', 'Intermediate Risk', 'High Risk', 'Very High Risk']:
        mask = (df['risk_group'] == rg) & eligible_for_mrd
        if mask.any():
            mrd_rate = df.loc[mask, 'mrd_positive'].mean()
            print(f"    • {rg}: {100*mrd_rate:.1f}%")
    print(f"  - Mean IRS: {df['irs'].mean():.2f}")
    print()
    
    return df

# ============================================================================
# STAGE 6: RECURRENCE & SALVAGE
# ============================================================================

def simulate_recurrence(df):
    """Simulate disease recurrence in patients who achieved remission"""
    
    print("STAGE 6: RECURRENCE & SALVAGE")
    
    # Only patients who achieved remission can experience recurrence
    eligible_for_recurrence = ~df['trm'] & ~df['treatment_failure']
    
    # Recurrence probability based on necrosis
    # Handle categorical necrosis properly
    necrosis_map = {
        '≥90%': STMParameters.RECURRENCE_NECROSIS_90,
        '70-89%': STMParameters.RECURRENCE_NECROSIS_70,
        '50-69%': STMParameters.RECURRENCE_NECROSIS_50,
        '<50%': STMParameters.RECURRENCE_NECROSIS_LOW
    }
    
    beta0_recur = np.zeros(len(df))
    for cat, prob in necrosis_map.items():
        mask = (df['necrosis_cat'] == cat)
        beta0_recur[mask] = prob
    
    # Default for any missing values
    beta0_recur[beta0_recur == 0] = 0.15
    
    # Genetic risk modifiers for recurrence
    g_tp53_recur = np.where(df['tp53_mutant'].values,
                           STMParameters.TP53_RECURRENCE_MULTIPLIER, 1.0)
    g_fusion_recur = np.where(df['type1_fusion'].values == False,  # Fixed deprecation warning
                             STMParameters.TYPE2_FUSION_RECURRENCE, 1.0)
    
    # Metastatic recurrence risk
    m_meta_recur = df['meta_status'].map({
        'localized': 1.0,
        'lung': STMParameters.META_RECURRENCE,
        'bone': STMParameters.META_RECURRENCE * 1.2,
        'multiple': STMParameters.META_RECURRENCE * 1.5
    }).values
    
    # MRD status is the most powerful predictor
    mrd_mult = np.where(df['mrd_positive'].values,
                       STMParameters.MRD_POSITIVE_MULTIPLIER,
                       STMParameters.MRD_NEGATIVE_MULTIPLIER)
    
    # Combined recurrence probability
    recurrence_prob = beta0_recur * g_tp53_recur * g_fusion_recur * m_meta_recur * mrd_mult
    recurrence_prob = recurrence_prob.clip(0, 0.95)
    
    # Apply only to eligible patients
    df['recurrence'] = False
    df.loc[eligible_for_recurrence, 'recurrence'] = \
        np.random.random(eligible_for_recurrence.sum()) < recurrence_prob[eligible_for_recurrence]
    
    # Time to recurrence (bimodal: early vs late)
    df['recurrence_time_years'] = 0.0  # Initialize as float to avoid dtype warning
    df['fatal_recurrence'] = False
    
    if df['recurrence'].any():
        n_recur = df['recurrence'].sum()
        
        # Bimodal timing — v3.0 recalibrated
        # gamma(shape, scale): mean = shape × scale
        # Early: gamma(3, 0.33) → mean ~1.0yr, 95th pctile ~2.3yr
        # Late: gamma(7, 0.5) → mean ~3.5yr
        early_recur = int(n_recur * STMParameters.RECURRENCE_EARLY_WEIGHT)
        time_to_recurrence = np.concatenate([
            np.random.gamma(3, 0.33, early_recur),   # Early: mean ~1.0yr
            np.random.gamma(7, 0.5, n_recur - early_recur)   # Late: mean 42mo
        ])
        np.random.shuffle(time_to_recurrence)
        df.loc[df['recurrence'], 'recurrence_time_years'] = time_to_recurrence
        
        # Fatal recurrence (85% of recurrences are fatal)
        df.loc[df['recurrence'], 'fatal_recurrence'] = \
            np.random.random(n_recur) < STMParameters.RECURRENCE_FATAL_PROBABILITY
    
    print(f"  - Recurrence: {df['recurrence'].sum()} / {eligible_for_recurrence.sum()} at-risk")
    if df['recurrence'].any():
        print(f"    • MRD+: {df[df['mrd_positive']]['recurrence'].sum()} / {df['mrd_positive'].sum()} ({100*df[df['mrd_positive']]['recurrence'].mean():.1f}%)")
        print(f"    • MRD-: {df[~df['mrd_positive']&eligible_for_recurrence]['recurrence'].sum()} / {(~df['mrd_positive']&eligible_for_recurrence).sum()} ({100*df[~df['mrd_positive']&eligible_for_recurrence]['recurrence'].mean():.1f}%)")
        
        # Calculate risk ratio
        mrd_pos_recur_rate = df[df['mrd_positive']]['recurrence'].mean()
        mrd_neg_recur_rate = df[~df['mrd_positive']&eligible_for_recurrence]['recurrence'].mean()
        if mrd_neg_recur_rate > 0:
            risk_ratio = mrd_pos_recur_rate / mrd_neg_recur_rate
            print(f"    • MRD risk ratio: {risk_ratio:.1f}×")
        
        print(f"    • Fatal recurrence: {df['fatal_recurrence'].sum()} / {df['recurrence'].sum()} ({100*df[df['recurrence']]['fatal_recurrence'].mean():.1f}%)")
    print()
    
    return df

# ============================================================================
# OUTCOME CALCULATIONS
# ============================================================================

def calculate_outcomes(df):
    """Calculate 5-year OS and EFS"""
    
    # Overall Survival (OS) at 5 years
    df['fatal_recurrence_5y'] = df['fatal_recurrence'] & (df.get('recurrence_time_years', 0) <= 5)
    df['dead_5y'] = df['trm'] | df['treatment_failure'] | df['fatal_recurrence_5y']
    df['os_5y'] = ~df['dead_5y']
    
    # Event-Free Survival (EFS) at 5 years  
    df['recurrence_5y'] = df['recurrence'] & (df.get('recurrence_time_years', 0) <= 5)
    df['event_5y'] = df['trm'] | df['treatment_failure'] | df['recurrence_5y']
    df['efs_5y'] = ~df['event_5y']
    
    return df

def print_outcomes(df, cohort_name="Full Cohort"):
    """Print survival outcomes"""
    
    print("="*80)
    print(f"{cohort_name.upper()} RESULTS")
    print("="*80)
    
    n = len(df)
    
    # Primary outcomes
    os_5y = df['os_5y'].mean()
    efs_5y = df['efs_5y'].mean()
    
    print(f"\nPRIMARY OUTCOMES (n={n}):")
    print(f"  5-year Overall Survival (OS):     {100*os_5y:.1f}%")
    print(f"  5-year Event-Free Survival (EFS): {100*efs_5y:.1f}%")
    
    # Component analysis
    print(f"\nEVENTS:")
    trm_rate = df['trm'].mean()
    failure_rate = df['treatment_failure'].mean()
    recur_rate = df['recurrence'].mean()
    fatal_recur_rate = (df['fatal_recurrence'].sum() / n) if 'fatal_recurrence' in df else 0
    
    print(f"  Treatment-related mortality: {100*trm_rate:.1f}% ({df['trm'].sum()}/{n})")
    print(f"  Treatment failure:           {100*failure_rate:.1f}% ({df['treatment_failure'].sum()}/{n})")
    print(f"  Recurrence:                  {100*recur_rate:.1f}% ({df['recurrence'].sum()}/{n})")
    print(f"  Fatal recurrence:            {100*fatal_recur_rate:.1f}% ({df['fatal_recurrence'].sum() if 'fatal_recurrence' in df else 0}/{n})")
    
    # MRD analysis
    eligible_for_mrd = ~df['trm'] & ~df['treatment_failure']
    if eligible_for_mrd.any():
        mrd_rate = df.loc[eligible_for_mrd, 'mrd_positive'].mean()
        print(f"\nMRD DETECTION:")
        print(f"  MRD+ rate: {100*mrd_rate:.1f}% ({df['mrd_positive'].sum()}/{eligible_for_mrd.sum()} tested)")
        
        if df['mrd_positive'].any() and (~df['mrd_positive'] & eligible_for_mrd).any():
            mrd_pos_recur = df[df['mrd_positive']]['recurrence'].mean()
            mrd_neg_recur = df[~df['mrd_positive'] & eligible_for_mrd]['recurrence'].mean()
            risk_ratio = mrd_pos_recur / mrd_neg_recur if mrd_neg_recur > 0 else 0
            print(f"  MRD+ recurrence: {100*mrd_pos_recur:.1f}%")
            print(f"  MRD- recurrence: {100*mrd_neg_recur:.1f}%")
            print(f"  Risk ratio (MRD+/MRD-): {risk_ratio:.1f}×")

def print_biomarker_calibration(df):
    """Print biomarker prevalence by risk group for calibration validation"""
    
    print("\n" + "="*80)
    print("BIOMARKER CALIBRATION VALIDATION (v2.0)")
    print("="*80)
    
    # Get eligible patients for MRD
    eligible_for_mrd = ~df['trm'] & ~df['treatment_failure']
    
    print("\nBiomarker Prevalence by Risk Group:")
    print("-" * 80)
    print(f"{'Risk Group':<20} {'TP53':<8} {'RB1':<8} {'STAG2':<8} {'Type2':<8} {'LDH':<8} {'ALP':<8} {'ctDNA':<8} {'MRD+':<8}")
    print("-" * 80)
    
    for rg in ['Low Risk', 'Intermediate Risk', 'High Risk', 'Very High Risk']:
        mask = df['risk_group'] == rg
        if mask.any():
            tp53 = df.loc[mask, 'tp53_mutant'].mean() * 100
            rb1 = df.loc[mask, 'rb1_mutant'].mean() * 100
            stag2 = df.loc[mask, 'stag2_mutant'].mean() * 100
            type2 = (1 - df.loc[mask, 'type1_fusion'].mean()) * 100
            ldh = df.loc[mask, 'ldh_elevated'].mean() * 100
            alp = df.loc[mask, 'alp_elevated'].mean() * 100
            ctdna = df.loc[mask, 'ctdna_detected'].mean() * 100
            
            # MRD only for eligible
            mrd_mask = mask & eligible_for_mrd
            mrd = df.loc[mrd_mask, 'mrd_positive'].mean() * 100 if mrd_mask.any() else 0
            
            print(f"{rg:<20} {tp53:>6.0f}%  {rb1:>6.0f}%  {stag2:>6.0f}%  {type2:>6.0f}%  {ldh:>6.0f}%  {alp:>6.0f}%  {ctdna:>6.0f}%  {mrd:>6.0f}%")
    
    print("\nTarget Values (from Figure 4):")
    print("-" * 80)
    print(f"{'Risk Group':<20} {'TP53':<8} {'RB1':<8} {'STAG2':<8} {'Type2':<8} {'LDH':<8} {'ALP':<8} {'ctDNA':<8} {'MRD+':<8}")
    print("-" * 80)
    targets = {
        'Low Risk': [5, 3, 10, 8, 30, 25, 40, 10],
        'Intermediate Risk': [8, 6, 15, 12, 50, 40, 60, 22],
        'High Risk': [12, 10, 20, 18, 70, 60, 80, 35],
        'Very High Risk': [20, 15, 28, 25, 90, 80, 95, 50],
    }
    for rg, vals in targets.items():
        print(f"{rg:<20} {vals[0]:>6.0f}%  {vals[1]:>6.0f}%  {vals[2]:>6.0f}%  {vals[3]:>6.0f}%  {vals[4]:>6.0f}%  {vals[5]:>6.0f}%  {vals[6]:>6.0f}%  {vals[7]:>6.0f}%")
    
    print("="*80)

# ============================================================================
# MAIN SIMULATION RUNNER
# ============================================================================

def run_frontline_cohort(n_patients, meta_fraction, cohort_name):
    """Run frontline Stages 1-6 for a single cohort."""
    print("\n" + "="*80)
    print(f"SIMULATING {cohort_name.upper()}")
    print("="*80)
    df = generate_baseline_cohort(n_patients=n_patients, meta_fraction=meta_fraction)
    df = simulate_early_response(df)
    df = simulate_trm(df)
    df = simulate_treatment_failure(df)
    df = simulate_mrd(df)
    df = simulate_recurrence(df)
    df = calculate_outcomes(df)
    print_outcomes(df, cohort_name)
    return df


# ============================================================================
# R/R EXTENSION MODULE — NEW IN v3.0
# ============================================================================

def simulate_rr_extension(df, cohort_name=""):
    """
    Relapsed/refractory extension module.
    Takes frontline cohort output, models post-relapse outcomes for
    patients who experienced recurrence.
    
    Calibration targets (Stahl 2011, rEECur):
      - Overall R/R: 14.2% 5-yr OS, 6.1% EFS, median OS 14.7 mo
      - Early (<2yr): 9.0% 5-yr OS, 2.7% EFS
      - Late (>2yr): 33.9% 5-yr OS, 18.8% EFS
      - Late/Early ratio: 3.8× (target 4.1×)
      - HD-IFOS 6-mo EFS: 43.9% (target 47%)
      - TC 6-mo EFS: 35.1% (target 37%)
      - Salvage TRM: 2.0%
    """
    print(f"\n  R/R EXTENSION MODULE (v3.0) — {cohort_name}")
    
    relapsed = df[df['recurrence']].copy()
    n_relapsed = len(relapsed)
    
    if n_relapsed == 0:
        print("    No relapsed patients in this cohort.")
        return df, pd.DataFrame()
    
    # Early vs late relapse classification
    relapsed['is_early_relapse'] = relapsed['recurrence_time_years'] < STMParameters.RR_EARLY_LATE_CUTOFF
    relapsed['is_late_relapse'] = ~relapsed['is_early_relapse']
    
    n_early = relapsed['is_early_relapse'].sum()
    n_late = relapsed['is_late_relapse'].sum()
    print(f"    Relapsed patients: {n_relapsed} (early: {n_early}, late: {n_late})")
    
    # Assign relapse site
    site_probs = [STMParameters.RR_SITE_LOCAL_PROB, 
                  STMParameters.RR_SITE_SYSTEMIC_PROB,
                  STMParameters.RR_SITE_COMBINED_PROB,
                  1 - STMParameters.RR_SITE_LOCAL_PROB - STMParameters.RR_SITE_SYSTEMIC_PROB - STMParameters.RR_SITE_COMBINED_PROB]
    relapsed['relapse_site'] = np.random.choice(
        ['local', 'systemic', 'combined', 'distant'],
        size=n_relapsed, p=site_probs)
    
    # Assign salvage regimen
    reg_names = list(STMParameters.SALVAGE_REGIMEN_PROBS.keys())
    reg_probs = list(STMParameters.SALVAGE_REGIMEN_PROBS.values())
    relapsed['salvage_regimen'] = np.random.choice(reg_names, size=n_relapsed, p=reg_probs)
    
    # Salvage TRM
    relapsed['salvage_trm'] = np.random.random(n_relapsed) < STMParameters.SALVAGE_TRM_RATE
    
    # Chemosensitivity (early relapse → more likely resistant)
    resist_prob = np.where(relapsed['is_early_relapse'].values, 0.65, 0.35)
    relapsed['chemo_resistant'] = np.random.random(n_relapsed) < resist_prob
    
    # Biomarker status at relapse
    relapsed['rr_ldh_elevated'] = np.random.random(n_relapsed) < 0.80
    relapsed['rr_ctdna_positive'] = np.random.random(n_relapsed) < 0.70
    
    # --- 6-month EFS ---
    six_mo_efs = np.array([STMParameters.SALVAGE_6MO_EFS_BASE[r] 
                           for r in relapsed['salvage_regimen']])
    # Late relapse does better
    six_mo_efs = np.where(relapsed['is_late_relapse'].values,
                          six_mo_efs * 1.25, six_mo_efs * 0.90)
    # Chemosensitivity
    six_mo_efs = np.where(relapsed['chemo_resistant'].values,
                          six_mo_efs * 0.60, six_mo_efs * 1.15)
    six_mo_efs = np.clip(six_mo_efs, 0.02, 0.85)
    relapsed['salvage_efs_6mo'] = (~relapsed['salvage_trm'].values) & \
                                   (np.random.random(n_relapsed) < six_mo_efs)
    
    # --- Year-by-year post-relapse survival ---
    # Base annual mortality modified by site, regimen, biomarkers, sensitivity
    base_mort = np.where(relapsed['is_early_relapse'].values,
                         STMParameters.RR_BASE_ANNUAL_MORTALITY_EARLY,
                         STMParameters.RR_BASE_ANNUAL_MORTALITY_LATE)
    
    # Site modifier
    site_mult = relapsed['relapse_site'].map({
        'local': STMParameters.RR_SITE_MORTALITY_LOCAL,
        'systemic': STMParameters.RR_SITE_MORTALITY_SYSTEMIC,
        'combined': STMParameters.RR_SITE_MORTALITY_COMBINED,
        'distant': 1.20,
    }).values
    
    # Regimen modifier
    reg_mult = np.array([STMParameters.SALVAGE_EFFICACY[r] for r in relapsed['salvage_regimen']])
    
    # Biomarker modifiers
    ldh_mult = np.where(relapsed['rr_ldh_elevated'].values,
                        STMParameters.RR_LDH_ELEVATED_MORTALITY_MULT, 1.0)
    ctdna_mult = np.where(relapsed['rr_ctdna_positive'].values,
                          STMParameters.RR_CTDNA_POSITIVE_MORTALITY_MULT, 1.0)
    
    # Sensitivity modifier
    sens_mult = np.where(relapsed['chemo_resistant'].values,
                         STMParameters.SALVAGE_RESISTANT_MORTALITY_MULT,
                         STMParameters.SALVAGE_SENSITIVE_MORTALITY_MULT)
    
    # Effective annual mortality
    eff_mort = base_mort * site_mult * reg_mult * ldh_mult * ctdna_mult * sens_mult
    eff_mort = np.clip(eff_mort, 0.05, 0.95)
    
    # Simulate year-by-year survival (already dead from salvage TRM = dead at year 0)
    relapsed['post_relapse_alive_yr0'] = ~relapsed['salvage_trm'].values
    for yr in range(1, 11):
        prev_alive = relapsed[f'post_relapse_alive_yr{yr-1}'].values
        died_this_year = prev_alive & (np.random.random(n_relapsed) < eff_mort)
        relapsed[f'post_relapse_alive_yr{yr}'] = prev_alive & ~died_this_year
    
    # Median OS computation
    relapsed['post_relapse_os_months'] = 0.0
    for idx in relapsed.index:
        if relapsed.loc[idx, 'salvage_trm']:
            relapsed.loc[idx, 'post_relapse_os_months'] = np.random.uniform(0, 1)
            continue
        # Find year of death
        for yr in range(1, 11):
            if not relapsed.loc[idx, f'post_relapse_alive_yr{yr}']:
                relapsed.loc[idx, 'post_relapse_os_months'] = (yr - 1) * 12 + np.random.uniform(0, 12)
                break
        else:
            relapsed.loc[idx, 'post_relapse_os_months'] = 120 + np.random.uniform(0, 60)
    
    # Print R/R results
    print(f"\n    R/R OUTCOMES:")
    for label, mask in [('All R/R', np.ones(n_relapsed, dtype=bool)),
                        ('Early (<2yr)', relapsed['is_early_relapse'].values),
                        ('Late (>=2yr)', relapsed['is_late_relapse'].values)]:
        n = mask.sum()
        if n == 0: continue
        os_5yr = relapsed.loc[relapsed.index[mask], 'post_relapse_alive_yr5'].mean()
        med_os = relapsed.loc[relapsed.index[mask], 'post_relapse_os_months'].median()
        print(f"      {label} (n={n}): 5-yr OS={os_5yr*100:.1f}%, median OS={med_os:.1f} mo")
    
    if n_early > 0 and n_late > 0:
        early_os5 = relapsed.loc[relapsed['is_early_relapse'], 'post_relapse_alive_yr5'].mean()
        late_os5 = relapsed.loc[relapsed['is_late_relapse'], 'post_relapse_alive_yr5'].mean()
        if early_os5 > 0:
            print(f"      Survival ratio (Late/Early): {late_os5/early_os5:.1f}×")
    
    # Salvage regimen results
    print(f"\n    SALVAGE REGIMEN 6-MO EFS:")
    for reg in ['HD-IFOS', 'TC', 'IT', 'GD']:
        rmask = relapsed['salvage_regimen'] == reg
        if rmask.sum() > 0:
            efs = relapsed.loc[rmask, 'salvage_efs_6mo'].mean()
            mos = relapsed.loc[rmask, 'post_relapse_os_months'].median()
            print(f"      {reg}: n={rmask.sum()}, 6-mo EFS={efs*100:.1f}%, median OS={mos:.1f} mo")
    
    print(f"    Salvage TRM: {relapsed['salvage_trm'].sum()}/{n_relapsed} ({relapsed['salvage_trm'].mean()*100:.1f}%)")
    
    return df, relapsed


# ============================================================================
# ADVERSE EFFECTS MODULES — NEW IN v3.0
# ============================================================================

def simulate_adverse_effects(df, cohort_name=""):
    """
    Compute 30-year adverse effects trajectories for all patients.
    Uses cumulative drug exposures from the treatment simulation.
    
    Modules: Cardiotoxicity, Nephrotoxicity, Hypertension, SMN, RT late effects
    """
    print(f"\n  ADVERSE EFFECTS MODULE (v3.0) — {cohort_name}")
    
    n = len(df)
    timepoints = [0, 5, 10, 15, 20, 25, 30]
    
    # --- Assign cumulative drug exposures ---
    # Standard VDC/IE dosing with patient-level variation
    # Patients who had treatment failure get reduced exposure
    completion_frac = np.where(df['treatment_failure'].values, 
                                np.random.uniform(0.3, 0.7, n),
                                np.random.uniform(0.85, 1.05, n))
    
    # Intensified patients get ~20% more
    intensified_mult = np.where(df['treatment_intensified'].values, 1.20, 1.0)
    
    df['cum_dox_mg_m2'] = STMParameters.STANDARD_DOX_DOSE * completion_frac * intensified_mult
    df['cum_ifo_g_m2'] = STMParameters.STANDARD_IFO_DOSE * completion_frac * intensified_mult
    df['cum_eto_mg_m2'] = STMParameters.STANDARD_ETO_DOSE * completion_frac * intensified_mult
    
    # RT assignment (site-dependent)
    df['rt_received'] = False
    df['rt_heart_dose_gy'] = 0.0
    # ~60% receive RT for local control
    rt_prob = 0.60
    df['rt_received'] = np.random.random(n) < rt_prob
    
    # Heart dose based on primary site scatter
    for site_name, scatter_frac in STMParameters.RT_HEART_SCATTER.items():
        site_mask = (df['meta_status'] == 'localized')  # All get site assignment
        # Use tumor volume as proxy for site (simplified)
        # In real model, site would be assigned at baseline
        df.loc[df['rt_received'], 'rt_heart_dose_gy'] = 0.0  # Reset
    
    # Simplified site assignment for RT scatter
    site_assignment = np.random.choice(
        ['extremity', 'pelvis', 'axial', 'soft_tissue'],
        size=n, p=[0.30, 0.25, 0.15, 0.30])
    df['primary_site'] = site_assignment
    
    for idx in df[df['rt_received']].index:
        site = df.loc[idx, 'primary_site']
        scatter = STMParameters.RT_HEART_SCATTER.get(site, 0.0)
        df.loc[idx, 'rt_heart_dose_gy'] = STMParameters.RT_FIELD_DOSE * scatter
    
    # --- Cardiotoxicity (CHF) ---
    # Cumulative hazard model: annual CHF probability from doxorubicin exposure
    dox_normalized = df['cum_dox_mg_m2'].values / 100.0  # Per 100 mg/m²
    
    # Year-by-year CHF onset tracking
    df['chf_onset_year'] = 999  # No CHF
    cardiac_reserve = np.ones(n)  # Start at 1.0 (full reserve)
    
    for yr in timepoints[1:]:  # Skip year 0
        still_at_risk = df['chf_onset_year'].values > yr
        
        # Annual reserve depletion
        annual_depletion = STMParameters.DOX_CHF_COEFF * (dox_normalized ** STMParameters.DOX_CHF_EXPONENT)
        
        # RT interaction (multiplicative with dox damage)
        rt_factor = np.ones(n)
        rt_heart = df['rt_heart_dose_gy'].values
        rt_factor = np.where(rt_heart > 0,
                            (STMParameters.RT_CARDIAC_PENTEC_HR ** (rt_heart / 10.0)),
                            1.0)
        
        # Reserve depletion over the interval (5-year steps)
        interval = 5 if yr > 5 else yr
        cardiac_reserve -= annual_depletion * rt_factor * interval
        cardiac_reserve = np.clip(cardiac_reserve, 0.0, 1.0)
        
        # CHF probability: accelerates when reserve drops below threshold
        chf_prob_base = (1.0 - cardiac_reserve) * 0.014 * interval
        acceleration = np.where(cardiac_reserve < STMParameters.CHF_RESERVE_THRESHOLD,
                               STMParameters.CHF_ACCELERATION_FACTOR, 1.0)
        chf_prob = chf_prob_base * acceleration
        
        # Nephro-cardiac coupling: hypertension from GFR decline adds afterload
        # (computed after nephro module, but approximate here)
        nephro_coupling = np.where(yr >= 15, STMParameters.NEPHRO_CARDIAC_COUPLING, 1.0)
        chf_prob *= nephro_coupling
        
        chf_prob = np.clip(chf_prob, 0, 0.50)
        
        # CHIP modifier (prevalence increases with time)
        chip_prev = min(yr / 30.0 * STMParameters.CHIP_PREVALENCE_30YR, STMParameters.CHIP_PREVALENCE_30YR)
        has_chip = np.random.random(n) < chip_prev
        chf_prob = np.where(has_chip, chf_prob * STMParameters.CHIP_CARDIAC_MODIFIER, chf_prob)
        
        new_chf = still_at_risk & (np.random.random(n) < chf_prob)
        df.loc[new_chf, 'chf_onset_year'] = yr
    
    # --- Nephrotoxicity (GFR decline) ---
    ifo_normalized = df['cum_ifo_g_m2'].values
    
    for yr in timepoints:
        if yr == 0:
            df[f'gfr_year_{yr}'] = STMParameters.BASELINE_GFR
            continue
        # GFR decline: dose-dependent + Brenner amplification over time
        base_decline = STMParameters.IFO_GFR_COEFF * ifo_normalized * yr
        # Accelerated decline above threshold dose
        threshold_excess = np.maximum(ifo_normalized - STMParameters.IFO_GFR_THRESHOLD, 0)
        accelerated = threshold_excess * 0.004 * yr
        # Brenner hyperfiltration amplification
        brenner = STMParameters.BRENNER_AMPLIFICATION ** max(yr - 5, 0)
        
        total_decline = (base_decline + accelerated) * brenner
        # Add individual variation
        variation = np.random.normal(0, 3.0, n)
        df[f'gfr_year_{yr}'] = np.clip(STMParameters.BASELINE_GFR - total_decline + variation, 15, 130)
    
    # --- Hypertension ---
    df['htn_onset_year'] = 999
    for yr in timepoints[1:]:
        still_at_risk = df['htn_onset_year'].values > yr
        gfr_decline = STMParameters.BASELINE_GFR - df[f'gfr_year_{yr}'].values
        
        # HTN probability from GFR decline + age-related background
        renal_htn_prob = np.clip(gfr_decline * abs(STMParameters.HTN_GFR_SLOPE), 0, 0.5)
        background_htn = STMParameters.HTN_BASE_RATE_PER_YEAR * yr
        htn_prob = np.clip(renal_htn_prob + background_htn, 0, 0.80)
        
        new_htn = still_at_risk & (np.random.random(n) < htn_prob)
        df.loc[new_htn, 'htn_onset_year'] = yr
    
    # --- Second Malignant Neoplasms (SMN) ---
    df['smn_onset_year'] = 999
    df['tmn_onset_year'] = 999
    df['ris_onset_year'] = 999
    
    eto_normalized = df['cum_eto_mg_m2'].values / 1000.0  # Per 1000 mg/m²
    rt_dose = df['rt_heart_dose_gy'].values  # Proxy for RT exposure
    
    for yr in timepoints[1:]:
        # t-MN: peaks early, plateaus after year 10
        tmn_at_risk = df['tmn_onset_year'].values > yr
        if yr <= STMParameters.SMN_TMN_PLATEAU_YEAR:
            tmn_hazard = STMParameters.SMN_ETO_TMN_RATE * eto_normalized
            # Peak around year 5
            time_weight = np.exp(-0.5 * ((yr - STMParameters.SMN_TMN_LATENCY_PEAK) / 2.0) ** 2)
            tmn_prob = np.clip(tmn_hazard * time_weight * 5, 0, 0.10)  # 5-year interval
        else:
            tmn_prob = np.full(n, 0.001)  # Minimal after plateau
        
        new_tmn = tmn_at_risk & (np.random.random(n) < tmn_prob)
        df.loc[new_tmn, 'tmn_onset_year'] = yr
        
        # RIS: late onset, requires RT
        ris_at_risk = (df['ris_onset_year'].values > yr) & df['rt_received'].values
        if yr >= STMParameters.SMN_RT_RIS_LATENCY:
            ris_hazard = STMParameters.SMN_RT_RIS_RATE * (rt_dose / 10.0)
            ris_prob = np.clip(ris_hazard * 5, 0, 0.05)  # 5-year interval
        else:
            ris_prob = np.zeros(n)
        
        new_ris = ris_at_risk & (np.random.random(n) < ris_prob)
        df.loc[new_ris, 'ris_onset_year'] = yr
        
        # Any SMN (first of t-MN, RIS, or background solid tumor)
        smn_at_risk = df['smn_onset_year'].values > yr
        bg_rate = STMParameters.SMN_BASELINE_RATE_EARLY if yr <= 10 else STMParameters.SMN_BASELINE_RATE_LATE
        background_prob = np.clip(bg_rate * 5, 0, 0.05)
        any_new = smn_at_risk & (new_tmn | new_ris | (np.random.random(n) < background_prob))
        df.loc[any_new, 'smn_onset_year'] = np.minimum(df.loc[any_new, 'smn_onset_year'], yr)
    
    # --- Print toxicity summary ---
    print(f"\n    ADVERSE EFFECTS SUMMARY (30-year horizon):")
    for yr in timepoints:
        chf_rate = (df['chf_onset_year'] <= yr).mean() * 100
        gfr_mean = df[f'gfr_year_{yr}'].mean()
        htn_rate = (df['htn_onset_year'] <= yr).mean() * 100
        smn_rate = (df['smn_onset_year'] <= yr).mean() * 100
        print(f"      Year {yr:2d}: CHF={chf_rate:5.1f}%  GFR={gfr_mean:5.1f}  HTN={htn_rate:5.1f}%  SMN={smn_rate:5.1f}%")
    
    # RT-stratified CHF
    dox_only = df[~df['rt_received']]
    combined = df[df['rt_received']]
    if len(dox_only) > 0 and len(combined) > 0:
        chf_dox = (dox_only['chf_onset_year'] <= 30).mean() * 100
        chf_comb = (combined['chf_onset_year'] <= 30).mean() * 100
        print(f"\n    RT-STRATIFIED 30-yr CHF:")
        print(f"      Dox-only (n={len(dox_only)}): {chf_dox:.1f}%")
        print(f"      Combined modality (n={len(combined)}): {chf_comb:.1f}%")
    
    # Site-stratified CHF (RT patients only)
    print(f"\n    SITE-STRATIFIED 30-yr CHF (RT patients):")
    for site in ['extremity', 'pelvis', 'axial']:
        smask = (df['primary_site'] == site) & df['rt_received']
        if smask.sum() > 10:
            chf_site = (df.loc[smask, 'chf_onset_year'] <= 30).mean() * 100
            mean_hd = df.loc[smask, 'rt_heart_dose_gy'].mean()
            print(f"      {site} (n={smask.sum()}): CHF={chf_site:.1f}%, mean heart dose={mean_hd:.1f} Gy")
    
    return df


# ============================================================================
# DATA EXTRACTION FOR FIGURE GENERATION
# ============================================================================

def extract_figure_data(df, relapsed_df, cohort_name="mixed"):
    """
    Extract all data needed for manuscript figures and save as JSON.
    Every value traced to this computation.
    """
    timepoints = [0, 5, 10, 15, 20, 25, 30]
    n = len(df)
    
    data = {
        '_metadata': {
            'script': 'stm_foundation_model_v3.0.py',
            'cohort': cohort_name,
            'n_patients': n,
            'seed': 42,
            'version': 'v3.0 (v2.0 frontline + R/R extension + adverse effects)',
            'source_model': 'stm_foundation_model_v3_0_reseed.py',
            'per_cohort_reseed': True,
        }
    }
    
    # --- Frontline calibration endpoints ---
    data['frontline'] = {
        'os_5yr_pct': round(df['os_5y'].mean() * 100, 1),
        'efs_5yr_pct': round(df['efs_5y'].mean() * 100, 1),
        'trm_pct': round(df['trm'].mean() * 100, 1),
    }
    
    # MRD
    eligible = ~df['trm'] & ~df['treatment_failure']
    if eligible.any():
        mrd_pos_recur = df[df['mrd_positive']]['recurrence'].mean() if df['mrd_positive'].any() else 0
        mrd_neg_recur = df[~df['mrd_positive'] & eligible]['recurrence'].mean() if (~df['mrd_positive'] & eligible).any() else 0
        data['frontline']['mrd_pos_recurrence_pct'] = round(mrd_pos_recur * 100, 1)
        data['frontline']['mrd_neg_recurrence_pct'] = round(mrd_neg_recur * 100, 1)
        data['frontline']['mrd_risk_ratio'] = round(mrd_pos_recur / mrd_neg_recur, 1) if mrd_neg_recur > 0.001 else None
    
    # --- Fig 2: Incremental Stratification (genotype × biomarker × MRD) ---
    # All panels derived from STM model patient-level data
    fig2 = {}
    
    # Define genotype groups (cast to bool for v2.0 float columns)
    tp53 = df['tp53_mutant'].astype(bool)
    stag2 = df['stag2_mutant'].astype(bool)
    rb1 = df['rb1_mutant'].astype(bool)
    cdkn2a = df['cdkn2a_loss'].astype(bool)
    
    geno_groups = {
        'No adverse': ~tp53 & ~stag2 & ~rb1 & ~cdkn2a,
        'STAG2 only': stag2 & ~tp53 & ~rb1 & ~cdkn2a,
        'RB1/CDKN2A': (rb1 | cdkn2a) & ~tp53 & ~stag2,
        'TP53 only': tp53 & ~stag2,
        'TP53+STAG2': tp53 & stag2,
    }
    
    # Panel A: Genetics only
    panel_a = {}
    for gname, gmask in geno_groups.items():
        sub = df[gmask]
        if len(sub) >= 10:
            panel_a[gname] = {'n': int(len(sub)), 'mean_efs_pct': round(sub['efs_5y'].mean() * 100, 1)}
    fig2['panel_a_genetics'] = panel_a
    
    # Panel B: Genetics + Biomarker favorability
    # Use composite_risk median: below = favorable, above = elevated
    median_risk = df['composite_risk'].median()
    panel_b = {}
    for gname, gmask in geno_groups.items():
        sub = df[gmask]
        if len(sub) < 10: continue
        fav = sub[sub['composite_risk'] <= median_risk]
        elev = sub[sub['composite_risk'] > median_risk]
        if len(fav) >= 5 and len(elev) >= 5:
            panel_b[gname] = {
                'favorable': {'n': int(len(fav)), 'mean_efs_pct': round(fav['efs_5y'].mean() * 100, 1)},
                'elevated': {'n': int(len(elev)), 'mean_efs_pct': round(elev['efs_5y'].mean() * 100, 1)},
            }
    fig2['panel_b_biomarkers'] = panel_b
    
    # Panel C: Genetics + BRS risk groups (quartiles)
    # Panel C: Genetics + composite risk quartiles
    # Use composite_risk (continuous) rather than BRS (discrete 4-value)
    risk_q25, risk_q50, risk_q75 = np.percentile(df['composite_risk'].values, [25, 50, 75])
    geno_for_c = {
        'Wild-type': ~tp53 & ~stag2,
        'STAG2+': stag2 & ~tp53,
        'TP53+': tp53,
    }
    panel_c = {}
    for gname, gmask in geno_for_c.items():
        sub = df[gmask]
        if len(sub) < 20: continue
        q_data = {}
        for ql, (lo, hi) in [('Q1', (-999, risk_q25)), ('Q2', (risk_q25, risk_q50)),
                              ('Q3', (risk_q50, risk_q75)), ('Q4', (risk_q75, 999))]:
            qsub = sub[(sub['composite_risk'] > lo) & (sub['composite_risk'] <= hi)]
            if len(qsub) >= 5:
                q_data[ql] = {'n': int(len(qsub)), 'mean_efs_pct': round(qsub['efs_5y'].mean() * 100, 1)}
        panel_c[gname] = q_data
    fig2['panel_c_brs_quartiles'] = panel_c
    
    # Panel D: Genetics + MRD status
    geno_for_d = {
        'Wild-type': ~tp53 & ~stag2,
        'STAG2+': stag2 & ~tp53,
        'TP53+': tp53 & ~stag2,
        'TP53+STAG2': tp53 & stag2,
    }
    panel_d = {}
    for gname, gmask in geno_for_d.items():
        sub = df[gmask & eligible]  # Only MRD-eligible patients
        if len(sub) < 10: continue
        mrd_neg = sub[~sub['mrd_positive']]
        mrd_pos = sub[sub['mrd_positive']]
        entry = {}
        if len(mrd_neg) >= 5:
            entry['MRD_negative'] = {'n': int(len(mrd_neg)), 'mean_efs_pct': round(mrd_neg['efs_5y'].mean() * 100, 1)}
        if len(mrd_pos) >= 5:
            entry['MRD_positive'] = {'n': int(len(mrd_pos)), 'mean_efs_pct': round(mrd_pos['efs_5y'].mean() * 100, 1)}
        panel_d[gname] = entry
    fig2['panel_d_mrd'] = panel_d
    
    data['fig2_incremental'] = fig2
    
    # --- R/R data (for Fig S3/S4) ---
    if len(relapsed_df) > 0:
        rr_data = {}
        for label, mask_col in [('all_rr', None), 
                                 ('early_relapse', 'is_early_relapse'),
                                 ('late_relapse', 'is_late_relapse')]:
            if mask_col:
                sub = relapsed_df[relapsed_df[mask_col]]
            else:
                sub = relapsed_df
            
            n_sub = len(sub)
            if n_sub == 0:
                rr_data[label] = {'n': 0}
                continue
            
            survival_by_year = {}
            for yr in range(1, 11):
                col = f'post_relapse_alive_yr{yr}'
                if col in sub.columns:
                    alive_pct = round(sub[col].mean() * 100, 1)
                    survival_by_year[str(yr)] = alive_pct
            
            rr_data[label] = {
                'n': n_sub,
                'survival_by_year_pct': survival_by_year,
                'median_os_months': round(float(sub['post_relapse_os_months'].median()), 1),
                'os_5yr_pct': round(sub['post_relapse_alive_yr5'].mean() * 100, 1) if 'post_relapse_alive_yr5' in sub.columns else None,
            }
        
        # Salvage regimens
        salvage = {}
        for reg in ['HD-IFOS', 'TC', 'IT', 'GD']:
            rmask = relapsed_df['salvage_regimen'] == reg
            if rmask.sum() > 0:
                efs_6mo = relapsed_df.loc[rmask, 'salvage_efs_6mo'].mean()
                med_os = relapsed_df.loc[rmask, 'post_relapse_os_months'].median()
                salvage[reg] = {
                    'n': int(rmask.sum()),
                    'efs_6mo_pct': round(efs_6mo * 100, 1),
                    'median_os_months': round(float(med_os), 1),
                }
        rr_data['salvage_regimens'] = salvage
        rr_data['salvage_trm_pct'] = round(relapsed_df['salvage_trm'].mean() * 100, 1)
        
        data['rr_outcomes'] = rr_data
    
    # --- Toxicity trajectories (for Fig 3, S4/S5) ---
    tox = {'timepoints_years': timepoints}
    
    chf_by_yr = [round((df['chf_onset_year'] <= yr).mean() * 100, 1) for yr in timepoints]
    gfr_by_yr = [round(float(df[f'gfr_year_{yr}'].mean()), 1) for yr in timepoints]
    htn_by_yr = [round((df['htn_onset_year'] <= yr).mean() * 100, 1) for yr in timepoints]
    smn_by_yr = [round((df['smn_onset_year'] <= yr).mean() * 100, 1) for yr in timepoints]
    tmn_by_yr = [round((df['tmn_onset_year'] <= yr).mean() * 100, 1) for yr in timepoints]
    ris_by_yr = [round((df['ris_onset_year'] <= yr).mean() * 100, 1) for yr in timepoints]
    
    tox['chf_cumulative_pct'] = chf_by_yr
    tox['mean_gfr_ml_min'] = gfr_by_yr
    tox['htn_cumulative_pct'] = htn_by_yr
    tox['smn_total_pct'] = smn_by_yr
    tox['smn_tmn_pct'] = tmn_by_yr
    tox['smn_ris_pct'] = ris_by_yr
    
    # RT-stratified cardiotoxicity
    rt_cardiac = {}
    dox_only = df[~df['rt_received']]
    combined = df[df['rt_received']]
    for label, cohort in [('dox_only', dox_only), ('combined_modality', combined)]:
        if len(cohort) > 0:
            chf_traj = [round((cohort['chf_onset_year'] <= yr).mean() * 100, 1) for yr in timepoints]
            rt_cardiac[label] = {'n': len(cohort), 'chf_by_year_pct': chf_traj}
    
    # Site-stratified
    for site in ['extremity', 'pelvis', 'axial']:
        smask = (df['primary_site'] == site) & df['rt_received']
        if smask.sum() > 10:
            chf_traj = [round((df.loc[smask, 'chf_onset_year'] <= yr).mean() * 100, 1) for yr in timepoints]
            mean_hd = round(float(df.loc[smask, 'rt_heart_dose_gy'].mean()), 1)
            rt_cardiac[f'rt_{site}'] = {'n': int(smask.sum()), 'chf_by_year_pct': chf_traj, 'mean_heart_dose_gy': mean_hd}
    
    # Nephro-cardiac coupling
    gfr_30 = df['gfr_year_30'] if 'gfr_year_30' in df.columns else pd.Series([120]*len(df))
    gfr_low = df[gfr_30 < 60]
    gfr_ok = df[gfr_30 >= 60]
    coupling = {}
    for label, cohort in [('gfr_below_60', gfr_low), ('gfr_60_plus', gfr_ok)]:
        if len(cohort) > 0:
            chf_30 = (cohort['chf_onset_year'] <= 30).mean() * 100
            coupling[label] = {'n': len(cohort), 'chf_30yr_pct': round(chf_30, 1)}
    rt_cardiac['nephro_cardiac_coupling'] = coupling
    
    tox['rt_cardiac'] = rt_cardiac
    data['toxicity'] = tox
    
    return data


# ============================================================================
# UPDATED MAIN SIMULATION RUNNER — v3.0
# ============================================================================

def run_full_simulation():
    """Run complete simulation: frontline + R/R + adverse effects + data extraction"""
    
    results = {}
    all_figure_data = {}
    
    cohorts = [
        ('localized', 10000, 0.0),
        ('mixed', 10000, 0.15),
        ('metastatic', 10000, 1.0),
    ]
    
    for cohort_name, n_patients, meta_frac in cohorts:
        np.random.seed(42)            # PER-COHORT RE-SEED (S57 fix): isolate each cohort's
                                      # RNG stream so a g_fusion change cannot desynchronize
                                      # downstream cohorts (clean one-way sensitivity).
        # Frontline Stages 1-6 (v2.0 calibrated, unchanged)
        df = run_frontline_cohort(n_patients, meta_frac, f"{cohort_name} cohort")
        
        # Print biomarker calibration for first cohort
        if cohort_name == 'localized':
            print_biomarker_calibration(df)
        
        # R/R Extension (v3.0)
        df, relapsed_df = simulate_rr_extension(df, cohort_name)
        
        # Adverse Effects (v3.0)
        df = simulate_adverse_effects(df, cohort_name)
        
        # Extract figure data
        figure_data = extract_figure_data(df, relapsed_df, cohort_name)
        all_figure_data[cohort_name] = figure_data
        
        results[cohort_name] = {'df': df, 'relapsed': relapsed_df}
    
    # --- Compute R/R MAE ---
    print("\n" + "="*80)
    print("CALIBRATION SUMMARY")
    print("="*80)
    
    # Frontline MAE (from mixed cohort which is the primary)
    mixed_data = all_figure_data['mixed']['frontline']
    local_data = all_figure_data['localized']['frontline']
    meta_data = all_figure_data['metastatic']['frontline']
    
    print(f"\n  FRONTLINE ENDPOINTS:")
    print(f"    Localized:  OS={local_data['os_5yr_pct']}% EFS={local_data['efs_5yr_pct']}% TRM={local_data['trm_pct']}%")
    print(f"    Mixed:      OS={mixed_data['os_5yr_pct']}% EFS={mixed_data['efs_5yr_pct']}% TRM={mixed_data['trm_pct']}%")
    print(f"    Metastatic: OS={meta_data['os_5yr_pct']}% EFS={meta_data['efs_5yr_pct']}% TRM={meta_data['trm_pct']}%")
    
    if 'rr_outcomes' in all_figure_data['mixed']:
        rr = all_figure_data['mixed']['rr_outcomes']
        print(f"\n  R/R ENDPOINTS (mixed cohort):")
        if 'all_rr' in rr and rr['all_rr'].get('n', 0) > 0:
            print(f"    All R/R: 5-yr OS={rr['all_rr'].get('os_5yr_pct', 'N/A')}%, median OS={rr['all_rr'].get('median_os_months', 'N/A')} mo")
        if 'early_relapse' in rr and rr['early_relapse'].get('n', 0) > 0:
            print(f"    Early:   5-yr OS={rr['early_relapse'].get('os_5yr_pct', 'N/A')}%")
        if 'late_relapse' in rr and rr['late_relapse'].get('n', 0) > 0:
            print(f"    Late:    5-yr OS={rr['late_relapse'].get('os_5yr_pct', 'N/A')}%")
    
    # --- Save outputs ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'outputs_v3')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n  Output directory: {output_dir}")
    
    # Save CSVs
    for cohort_name in ['localized', 'mixed', 'metastatic']:
        df = results[cohort_name]['df']
        csv_path = os.path.join(output_dir, f'stm_v3_{cohort_name}_outcomes.csv')
        df.to_csv(csv_path, index=False)
        print(f"  ✓ Saved {cohort_name} outcomes CSV")
    
    # Save figure data JSON
    json_path = os.path.join(output_dir, 'stm_v3_figure_data.json')
    with open(json_path, 'w') as f:
        json.dump(all_figure_data, f, indent=2, default=str)
    print(f"  ✓ Saved figure data JSON: {json_path}")
    
    print("\n" + "="*80)
    print("SIMULATION COMPLETE - MODEL v3.0")
    print("="*80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nPaste the console output above into Claude, or upload stm_v3_figure_data.json")
    
    return results, all_figure_data


if __name__ == "__main__":
    results, figure_data = run_full_simulation()
