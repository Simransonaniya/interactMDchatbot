"""
Clinical Cases Ground Truth Store for InteractMD.
Strict source-of-truth definitions for all virtual patient scenarios.
"""

CLINICAL_CASES = {
    "case-acs-1": {
        "id": "case-acs-1",
        "title": "Acute Crushing Retrosternal Chest Pain",
        "shortDescription": "58-year-old male with sudden onset substernal chest heaviness, diaphoresis, and radiation to the left jaw.",
        "specialty": "Cardiology",
        "difficulty": "Intermediate",
        "estimatedMinutes": 15,
        "patient": {
            "id": "pt-robert-chen",
            "name": "Robert Chen",
            "age": 58,
            "gender": "Male",
            "occupation": "Architectural Project Manager",
            "presentationComplaint": "Severe pressure and heaviness in my chest that started less than an hour ago.",
            "initialStatement": "Doctor, please... It feels like an elephant is sitting right in the middle of my chest. I started feeling dizzy and breaking out in a cold sweat on my way into the office.",
            "mood": "Anxious, pale, clutching center of chest with a closed fist (Levine sign)",
            "appearance": "Diaphoretic, breathing shallowly, speech interrupted by discomfort."
        },
        "initialVitals": {
            "heartRate": 98,
            "bloodPressure": "154/94",
            "respiratoryRate": 20,
            "oxygenSaturation": 97,
            "temperature": 37.1,
            "painScore": 8
        },
        "facts": {
            "onset": "started approximately 45 minutes ago while walking up two flights of stairs to his office desk.",
            "provocationPalliative": "nothing makes it better, even resting in a chair; taking deep breaths doesn't change it.",
            "quality": "a deep, heavy crushing pressure, like someone is squeezing my heart in a vice.",
            "radiation": "radiates up into the left side of his jaw, lower teeth, and down the inner aspect of his left arm.",
            "severity": "rates it an 8 out of 10 right now, at its peak when climbing stairs it was a 9/10.",
            "timing": "continuous and unremitting since it began 45 minutes ago; has not come in waves.",
            "associatedSymptoms": [
                "Profuse cold sweats (diaphoresis)",
                "Mild lightheadedness and feeling faint",
                "Nausea without vomiting",
                "Shortness of breath"
            ],
            "pertinentNegatives": [
                "No sharp pleuritic pain with breathing",
                "No sudden tearing pain between shoulder blades",
                "No fever or chills",
                "No calf swelling or recent long-distance travel"
            ],
            "pastMedicalHistory": [
                "Essential Hypertension diagnosed 6 years ago",
                "Hyperlipidemia (elevated LDL)",
                "No prior heart attack or stroke"
            ],
            "medications": [
                "Amlodipine 5 mg daily",
                "Atorvastatin 20 mg daily (admits to missing doses frequently)"
            ],
            "allergies": ["No known drug allergies (NKDA)"],
            "familyHistory": ["Father had a fatal myocardial infarction at age 52; Mother has type 2 diabetes."],
            "socialHistory": [
                "Smokes 0.5 packs per day for 25 years (12.5 pack-years)",
                "Drinks 1-2 glasses of wine on weekends",
                "Denies illicit drug use, including cocaine or amphetamines."
            ]
        },
        "physicalFindings": [
            {
                "id": "exam-cv",
                "system": "Cardiovascular",
                "name": "Precordial & Heart Auscultation",
                "findingDescription": "Tachycardic regular rhythm. S1 and S2 present. Soft S4 gallop audible at apex. No pericardial friction rub. JVP estimated at 3 cm above sternal angle at 45 degrees. Peripheral pulses equal and palpable bilaterally.",
                "isAbnormal": True,
                "clinicalSignificance": "S4 gallop reflects decreased left ventricular compliance secondary to acute myocardial ischemia."
            },
            {
                "id": "exam-pulm",
                "system": "Respiratory",
                "name": "Lung Auscultation & Chest Wall Palpation",
                "findingDescription": "Clear to auscultation bilaterally. No wheezing, rhonchi, or basilar crackles. Chest wall tenderness is absent; pain is NOT reproducible with manual palpation of costochondral junctions.",
                "isAbnormal": False,
                "clinicalSignificance": "Absence of chest wall tenderness rules against costochondritis; clear lung fields indicate no acute cardiogenic pulmonary edema at present."
            },
            {
                "id": "exam-general",
                "system": "General",
                "name": "General Appearance & Diaphoresis",
                "findingDescription": "Cool, clammy extremities with marked forehead and palmar diaphoresis. Capillary refill approximately 2.5 seconds. Pupils equal and reactive. Mucous membranes moist.",
                "isAbnormal": True,
                "clinicalSignificance": "Significant sympathetic autonomic activation typical in acute myocardial ischemia."
            },
            {
                "id": "exam-abdomen",
                "system": "Abdominal",
                "name": "Abdominal Palpation & Epigastric Exam",
                "findingDescription": "Soft, non-tender, non-distended. No guarding, rebound, or organomegaly. No pulsatile abdominal mass detected.",
                "isAbnormal": False,
                "clinicalSignificance": "Non-tender abdomen reduces likelihood of acute perforated peptic ulcer or cholecystitis masquerading as lower chest pain."
            }
        ],
        "investigations": [
            {
                "id": "inv-ecg",
                "name": "12-Lead Electrocardiogram (STAT)",
                "category": "Cardiology / Point-of-Care",
                "turnaroundMinutes": 2,
                "value": "ST Elevation in II, III, aVF",
                "interpretation": "Sinus rhythm at 96 bpm. 2.5mm ST-segment elevation in leads II, III, and aVF with reciprocal ST depression in leads I and aVL. Hyperacute T-waves in inferior leads.",
                "isAbnormal": True,
                "findingsDetail": [
                    "Acute ST-segment Elevation Myocardial Infarction (Inferior STEMI - Right Coronary Artery territory).",
                    "Reciprocal changes in high lateral leads (I, aVL)."
                ]
            },
            {
                "id": "inv-troponin",
                "name": "High-Sensitivity Cardiac Troponin I (hs-cTnI)",
                "category": "Laboratory",
                "turnaroundMinutes": 20,
                "value": "185 ng/L (Markedly Elevated)",
                "normalRange": "< 14 ng/L",
                "interpretation": "Positive for acute myocardial necrosis. Baseline initial rise observed 50 mins post-symptom onset.",
                "isAbnormal": True
            },
            {
                "id": "inv-cxr",
                "name": "Portable Chest Radiograph (CXR)",
                "category": "Imaging",
                "turnaroundMinutes": 15,
                "value": "Normal cardiothoracic ratio, clear fields",
                "interpretation": "Normal cardiothoracic ratio. No widening of the superior mediastinum. Clear lung parenchymal fields.",
                "isAbnormal": False
            }
        ],
        "diagnosisOptions": [
            {
                "id": "dx-stemi",
                "name": "ST-Elevation Myocardial Infarction (Inferior STEMI)",
                "category": "Cardiovascular",
                "isCorrectPrimary": True,
                "isHighDifferential": True
            },
            {
                "id": "dx-dissection",
                "name": "Acute Aortic Dissection (Type A)",
                "category": "Vascular",
                "isCorrectPrimary": False,
                "isHighDifferential": True
            },
            {
                "id": "dx-pe",
                "name": "Pulmonary Embolism",
                "category": "Pulmonary",
                "isCorrectPrimary": False,
                "isHighDifferential": False
            },
            {
                "id": "dx-gerd",
                "name": "Gastroesophageal Reflux Disease (GERD) / Esophageal Spasm",
                "category": "Gastroenterology",
                "isCorrectPrimary": False,
                "isHighDifferential": False
            }
        ],
        "managementProtocols": [
            {
                "id": "mgmt-aspirin",
                "label": "Administer Chewable Aspirin 324 mg immediately",
                "isCorrect": True,
                "feedback": "Essential immediate antiplatelet therapy in suspected ACS to reduce mortality."
            },
            {
                "id": "mgmt-cath",
                "label": "Activate Cardiac Catheterization Lab for Emergent PCI (<90 min door-to-balloon)",
                "isCorrect": True,
                "feedback": "Definitive primary reperfusion strategy for acute inferior STEMI."
            },
            {
                "id": "mgmt-heparin",
                "label": "IV Anticoagulation (Unfractionated Heparin bolus + infusion)",
                "isCorrect": True,
                "feedback": "Prevents thrombus propagation before and during percutaneous intervention."
            },
            {
                "id": "mgmt-nsaids",
                "label": "High-dose Ibuprofen or Ketorolac IV for pain control",
                "isCorrect": False,
                "feedback": "NSAIDs (except aspirin) are contraindicated in acute MI due to increased risk of mortality, reinfarction, and cardiac rupture."
            }
        ],
        "scoringRubric": {
            "redFlagsToScreen": [
                "Radiation to jaw or left arm",
                "Autonomic symptoms (diaphoresis, nausea, presyncope)",
                "Absence of pleuritic chest wall tenderness",
                "Aortic dissection exclusion (pulse differentials, tearing back pain)"
            ]
        }
    },
    "case-dyspnea-2": {
        "id": "case-dyspnea-2",
        "title": "Acute Severe Dyspnea & Expiratory Wheezing",
        "shortDescription": "32-year-old female presents with sudden onset breathlessness, audible wheezing, and inability to speak in full sentences.",
        "specialty": "Pulmonology",
        "difficulty": "Novice",
        "estimatedMinutes": 12,
        "patient": {
            "id": "pt-elena-rostova",
            "name": "Elena Rostova",
            "age": 32,
            "gender": "Female",
            "occupation": "Elementary School Teacher",
            "presentationComplaint": "I can barely catch my breath... My inhaler isn't working.",
            "initialStatement": "Doctor... I can't... breathe... Used my blue inhaler... four times... still tight...",
            "mood": "Frightened, tachypneic, tripod positioning",
            "appearance": "Accessory muscle use (intercostal retractions), speaking in 2-3 word phrases."
        },
        "initialVitals": {
            "heartRate": 112,
            "bloodPressure": "128/82",
            "respiratoryRate": 28,
            "oxygenSaturation": 91,
            "temperature": 36.8,
            "painScore": 4
        },
        "facts": {
            "onset": "started 3 hours ago after visiting a friend who has two cats; worsened progressively.",
            "provocationPalliative": "albuterol MDI provided only 10 minutes of partial relief before tightness returned.",
            "quality": "feels like breathing through a thin straw; chest feels constricted like a tight band.",
            "radiation": "diffuse chest tightness, no radiation to neck, back, or arms.",
            "severity": "describes breathlessness as severe (8/10 difficulty breathing).",
            "timing": "worsened steadily over the past 3 hours.",
            "associatedSymptoms": [
                "Audible expiratory wheezing",
                "Non-productive dry hacking cough",
                "Chest tightness",
                "Throat itching"
            ],
            "pertinentNegatives": [
                "No fever, chills, or purulent sputum",
                "No unilateral pleuritic chest pain",
                "No leg swelling or prior deep vein thrombosis",
                "No facial angioedema or urticarial skin rash"
            ],
            "pastMedicalHistory": [
                "Moderate persistent asthma diagnosed at age 11",
                "Allergic rhinitis and cat dander allergy",
                "One prior ICU admission for asthma at age 19"
            ],
            "medications": [
                "Albuterol 90 mcg HFA inhaler as needed",
                "Fluticasone/Salmeterol 100/50 mcg DPI daily (admits poor compliance)"
            ],
            "allergies": ["Cat dander, grass pollen, Aspirin causes wheezing"],
            "familyHistory": ["Mother has asthma; Brother has eczema."],
            "socialHistory": ["Never smoked cigarettes; denies vaping or illicit substances."]
        },
        "physicalFindings": [
            {
                "id": "exam-pulm-wheeze",
                "system": "Respiratory",
                "name": "Chest Auscultation & Respiratory Mechanics",
                "findingDescription": "Markedly prolonged expiratory phase. High-pitched polyphonic expiratory wheezes throughout all lung fields. Supraclavicular and intercostal retractions present.",
                "isAbnormal": True,
                "clinicalSignificance": "Severe diffuse bronchospasm with significant airflow limitation."
            }
        ],
        "investigations": [
            {
                "id": "inv-peak-flow",
                "name": "Peak Expiratory Flow (PEF)",
                "category": "Point-of-Care",
                "turnaroundMinutes": 1,
                "value": "190 L/min (42% of personal best 450 L/min)",
                "interpretation": "Severe exacerbation (Red Zone < 50% predicted).",
                "isAbnormal": True
            }
        ],
        "diagnosisOptions": [
            {
                "id": "dx-asthma",
                "name": "Severe Acute Asthma Exacerbation",
                "category": "Pulmonary",
                "isCorrectPrimary": True,
                "isHighDifferential": True
            },
            {
                "id": "dx-anaphylaxis",
                "name": "Anaphylaxis with Respiratory Manifestations",
                "category": "Allergy / Immunology",
                "isCorrectPrimary": False,
                "isHighDifferential": True
            }
        ],
        "managementProtocols": [
            {
                "id": "mgmt-neb-albuterol",
                "label": "Inhaled Albuterol + Ipratropium nebulization back-to-back x 3",
                "isCorrect": True,
                "feedback": "First-line dual bronchodilator therapy for acute severe bronchospasm."
            },
            {
                "id": "mgmt-systemic-steroids",
                "label": "Oral Prednisone 60 mg or IV Methylprednisolone 60 mg STAT",
                "isCorrect": True,
                "feedback": "Crucial systemic corticosteroid to resolve airway mucosal inflammation."
            }
        ],
        "scoringRubric": {
            "redFlagsToScreen": [
                "Silent chest or exhaustion signs",
                "Accessory muscle use",
                "Previous intubation or ICU history"
            ]
        }
    },
    "case-abdomen-3": {
        "id": "case-abdomen-3",
        "title": "Migratory Right Lower Quadrant Abdominal Pain",
        "shortDescription": "24-year-old male with dull periumbilical pain that migrated to the right iliac fossa over 14 hours, with anorexia and low-grade fever.",
        "specialty": "Gastroenterology / Surgery",
        "difficulty": "Novice",
        "estimatedMinutes": 12,
        "patient": {
            "id": "pt-marcus-vance",
            "name": "Marcus Vance",
            "age": 24,
            "gender": "Male",
            "occupation": "Software Engineer",
            "presentationComplaint": "My stomach started hurting around my belly button yesterday, but now it has moved down to the right side and hurts when I walk.",
            "initialStatement": "Doc, every bump in the road on the way here was pure agony. It started near my navel but now it's focused low down on the right side.",
            "mood": "Guarded, grimacing when moving or coughing",
            "appearance": "Lying supine with right hip slightly flexed to relieve abdominal tension."
        },
        "initialVitals": {
            "heartRate": 88,
            "bloodPressure": "122/76",
            "respiratoryRate": 16,
            "oxygenSaturation": 99,
            "temperature": 37.9,
            "painScore": 7
        },
        "facts": {
            "onset": "started yesterday evening around 7 PM around the navel, migrated to right lower quadrant 6 hours ago.",
            "provocationPalliative": "coughing, walking, and jumping worsen the pain significantly; lying still brings partial relief.",
            "quality": "initially dull, cramping periumbilical ache; now sharp, localized, constant right lower quadrant ache.",
            "radiation": "migrated from periumbilical region to McBurney's point in the right lower quadrant.",
            "severity": "rates it 7 out of 10 currently.",
            "timing": "constant and progressively worsening over the past 16 hours.",
            "associatedSymptoms": [
                "Complete loss of appetite (anorexia - couldn't eat his favorite pizza)",
                "Nausea with one episode of non-bloody vomiting 4 hours ago",
                "Low-grade fever and mild chills",
                "Constipation since yesterday"
            ],
            "pertinentNegatives": [
                "No dysuria, hematuria, or flank pain",
                "No diarrhea or hematochezia",
                "No history of similar pain or prior abdominal surgery"
            ],
            "pastMedicalHistory": ["No chronic medical conditions."],
            "medications": ["None. Took 500 mg acetaminophen 3 hours ago with minimal relief."],
            "allergies": ["No known drug allergies (NKDA)"],
            "familyHistory": ["Non-contributory."],
            "socialHistory": ["Non-smoker, drinks socially on weekends, no illicit drugs."]
        },
        "physicalFindings": [
            {
                "id": "exam-mcburney",
                "system": "Abdominal",
                "name": "Focused Right Lower Quadrant Palpation",
                "findingDescription": "Marked focal tenderness at McBurney's point with involuntary guarding. Positive Rovsing's sign (pain in RLQ on palpation of LLQ). Positive Psoas sign on passive right hip extension.",
                "isAbnormal": True,
                "clinicalSignificance": "Classic peritoneal signs indicating localized inflammation of the vermiform appendix."
            }
        ],
        "investigations": [
            {
                "id": "inv-us-appendix",
                "name": "Abdominal Ultrasound (Right Lower Quadrant)",
                "category": "Imaging",
                "turnaroundMinutes": 30,
                "value": "Non-compressible, blind-ending tubular structure measuring 8.2 mm in diameter",
                "interpretation": "Acutely inflamed appendix with periappendiceal fat stranding. Positive sonographic McBurney's sign.",
                "isAbnormal": True
            }
        ],
        "diagnosisOptions": [
            {
                "id": "dx-appendicitis",
                "name": "Acute Appendicitis",
                "category": "Surgery / Gastrointestinal",
                "isCorrectPrimary": True,
                "isHighDifferential": True
            },
            {
                "id": "dx-mesenteric-adenitis",
                "name": "Mesenteric Adenitis",
                "category": "Gastroenterology",
                "isCorrectPrimary": False,
                "isHighDifferential": True
            }
        ],
        "managementProtocols": [
            {
                "id": "mgmt-npo-surgery",
                "label": "Keep NPO (Nil Per Os) & request Urgent General Surgery Consultation",
                "isCorrect": True,
                "feedback": "Prepare for appendectomy and surgical evaluation."
            },
            {
                "id": "mgmt-iv-antibiotics",
                "label": "IV Fluid resuscitation and broad-spectrum IV Antibiotics (Ceftriaxone + Metronidazole)",
                "isCorrect": True,
                "feedback": "Appropriate perioperative antimicrobial coverage."
            }
        ],
        "scoringRubric": {
            "redFlagsToScreen": [
                "Signs of appendiceal perforation (diffuse peritonitis, high fever, hypotension)",
                "Migration pattern from periumbilical to RLQ",
                "Anorexia presence"
            ]
        }
    },
    "case-cap-4": {
        "id": "case-cap-4",
        "title": "Fever, Productive Cough & Altered Mental Status",
        "shortDescription": "74-year-old female nursing home resident brought to ED with 3 days of fever, purulent rust-colored sputum, confusion, and tachypnea.",
        "specialty": "Infectious Disease / Geriatrics",
        "difficulty": "Advanced",
        "estimatedMinutes": 18,
        "patient": {
            "id": "pt-dorothy-gable",
            "name": "Dorothy Gable",
            "age": 74,
            "gender": "Female",
            "occupation": "Retired Librarian",
            "presentationComplaint": "Patient brought by EMS from assisted living due to worsening cough, high fever, and acute confusion.",
            "initialStatement": "The windows... they're shaking... Where is Harold? It's so cold in here...",
            "mood": "Confused, lethargic, shivering",
            "appearance": "Flushed cheeks, accessory respiratory muscle use, incoherent responses at times."
        },
        "initialVitals": {
            "heartRate": 118,
            "bloodPressure": "88/56",
            "respiratoryRate": 30,
            "oxygenSaturation": 88,
            "temperature": 39.2,
            "painScore": 5
        },
        "facts": {
            "onset": "cough and fatigue started 3 days ago; acute confusion and severe lethargy developed this morning.",
            "provocationPalliative": "coughing hurts her right side; taking deep breaths causes sharp right-sided pain.",
            "quality": "sharp right lower thoracic pain on inspiration (pleuritic); persistent deep chest congestion.",
            "radiation": "localized to the right lower chest wall, does not radiate to the neck or back.",
            "severity": "patient unable to provide accurate numeric pain score due to delirium.",
            "timing": "progressively deteriorating over 72 hours.",
            "associatedSymptoms": [
                "Purulent, rust-colored sputum production",
                "Severe shaking chills (rigors)",
                "New-onset cognitive disorientation to time and place",
                "Anorexia and generalized profound weakness"
            ],
            "pertinentNegatives": [
                "No neck stiffness or photophobia (Kernig/Brudzinski negative)",
                "No focal neurological deficits (no limb paresis or facial droop)",
                "No rash or petechiae"
            ],
            "pastMedicalHistory": [
                "Moderate COPD (GOLD stage 2)",
                "Type 2 Diabetes Mellitus (HbA1c 7.8%)",
                "Mild vascular cognitive impairment"
            ],
            "medications": [
                "Tiotropium inhaler daily",
                "Metformin 1000 mg twice daily"
            ],
            "allergies": ["Penicillin (develops hives and lip swelling)"],
            "familyHistory": ["Non-contributory for acute illness."],
            "socialHistory": ["Former smoker (30 pack-years, quit 10 years ago). Assisted living resident."]
        },
        "physicalFindings": [
            {
                "id": "exam-lung-consolidation",
                "system": "Respiratory",
                "name": "Chest Auscultation & Percussion",
                "findingDescription": "Dullness to percussion over the right lower lung base. Bronchial breath sounds and coarse inspiratory crackles audible over the right lower lobe. Increased tactile fremitus and positive egophony (E to A change).",
                "isAbnormal": True,
                "clinicalSignificance": "Definitive physical signs of dense right lower lobe lobar consolidation."
            }
        ],
        "investigations": [
            {
                "id": "inv-cxr-cap",
                "name": "Urgent Portable Chest Radiograph (CXR)",
                "category": "Imaging",
                "turnaroundMinutes": 15,
                "value": "Dense dense alveolar consolidation in the right lower lobe with visible air bronchograms",
                "interpretation": "Lobar pneumonia involving the right lower lobe. No pleural effusion or pneumothorax.",
                "isAbnormal": True
            }
        ],
        "diagnosisOptions": [
            {
                "id": "dx-severe-cap",
                "name": "Severe Community-Acquired Pneumonia (CAP) with Sepsis",
                "category": "Infectious Disease",
                "isCorrectPrimary": True,
                "isHighDifferential": True
            }
        ],
        "managementProtocols": [
            {
                "id": "mgmt-sepsis-bundle",
                "label": "Implement Sepsis Bundle: 30 mL/kg IV Crystalloid bolus + Blood Cultures STAT + Supplemental Oxygen",
                "isCorrect": True,
                "feedback": "Immediate hemodynamic resuscitation and oxygenation for septic shock criteria."
            },
            {
                "id": "mgmt-pneumonia-abx",
                "label": "Empiric Broad-Spectrum IV Antibiotics (Respiratory Fluoroquinolone: Levofloxacin due to severe Penicillin allergy)",
                "isCorrect": True,
                "feedback": "Timely (<1 hour) administration of appropriate antibiotic covering Streptococcus pneumoniae."
            }
        ],
        "scoringRubric": {
            "redFlagsToScreen": [
                "Sepsis alert criteria (Hypotension SBP < 90, Tachycardia > 100, Tachypnea > 22)",
                "CURB-65 criteria (Confusion, Urea, Respiratory rate >= 30, BP, Age >= 65 = High Risk)",
                "Penicillin allergy awareness"
            ]
        }
    }
}
