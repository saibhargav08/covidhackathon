# covidhackathon

The goals of this project are threefold: (1) to explore development of a deep learning algorithm to distinguish chest X-rays of individuals with respiratory illness testing positive for COVID-19 from other X-rays, (2) to promote discovery of patterns in such X-rays via deep learning interpretability algorithms, and (3) to build more robust and extensible deep learning infrastructure trained on a variety of data types, to aid in the global response to COVID-19.

# Background
The 2019 novel coronavirus (COVID-19) presents several unique features. While the diagnosis is confirmed using polymerase chain reaction (PCR), infected patients with pneumonia may present on chest X-ray and computed tomography (CT) images with a pattern that is only moderately characteristic for the human eye Ng, 2020. COVID-19’s rate of transmission depends on our capacity to reliably identify infected patients with a low rate of false negatives. In addition, a low rate of false positives is required to avoid further increasing the burden on the healthcare system by unnecessarily exposing patients to quarantine if that is not required. Along with proper infection control, it is evident that timely detection of the disease would enable the implementation of all the supportive care required by patients affected by COVID-19.
In late January, a Chinese team published a paper detailing the clinical and paraclinical features of COVID-19. They reported that patients present abnormalities in chest CT images with most having bilateral involvement Huang 2020. Bilateral multiple lobular and subsegmental areas of consolidation constitute the typical findings in chest CT images of intensive care unit (ICU) patients on admission Huang 2020. In comparison, non-ICU patients show bilateral ground-glass opacity and subsegmental areas of consolidation in their chest CT images Huang 2020. In these patients, later chest CT images display bilateral ground-glass opacity with resolved consolidation Huang 2020.
Our goal is to use these images to develop deep learning-based approaches to predict and understand the infection.
The tasks are as follows using chest X-ray or CT (preference for X-ray) as input to predict these tasks:
Healthy vs Pneumonia (prototype already implemented Chester with ~74% AUC, validation study here)
Bacterial vs Viral vs COVID-19 Pneumonia
Survival of patient


# Model Evaluation:
Confusion matrix:


F-1 Score, precision, recall

Training Graph with validations


# Limitations:
1. Supported image extensions are: pbm, pgm, ppm, sr, ras,jpeg, jpg, jpe,jp2,tiff, tif,png
2. Cannot detect other pnuemonial infections(SARS, MERS, viral, bacterial etc)
3. Not a production ready as there is little data know on the actual covid-19 cases


# Resources:
1. https://github.com/ieee8023/covid-chestxray-dataset
2. https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
3. https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia


