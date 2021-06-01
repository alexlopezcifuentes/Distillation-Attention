import matplotlib.pyplot as plt

"""
Using a DCT-driven Loss in Attention-based Knowledge-Distillation for Scene Recognition

AblationAlfaFigure.py
Python file to plot the ablation study regarding Alpha parameter.
Python file used to create the figure from the paper.

Fully developed by Anonymous Code Author.
"""

# Read Precisions from the files itself. Is by hand, make better the code automatic.
Top1 = [43.73, 45.71, 46.19, 46.65, 46.62, 46.38, 47.42, 47.54, 47.46, 47.35, 45.89, 48.26, 48.43, 48.02]
Top5 = [65.61, 67.07, 68.28, 69.22, 69.18, 68.15, 69.85, 70.34, 69.24, 70.40, 67.71, 70.09, 70.75, 69.26]
MCA = [10.77, 11.63, 12.79, 13.35, 13.08, 12.70, 13.19, 12.90, 14.30, 13.11, 12.49, 14.80, 14.56, 13.64]
AlfaValues = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 3, 5]

# Values for Vanilla
Top1_v = [40.97] * len(AlfaValues)
Top5_v = [63.94] * len(AlfaValues)
MCA_v = [10.24] * len(AlfaValues)

color1 = 'tab:red'
color2 = 'tab:blue'


plt.figure()
plt.plot(AlfaValues, Top1, '*-', markersize=5, color=color1, label='Top@1 DCT')
plt.plot(AlfaValues, Top1_v, '*-', markersize=5, color=color2, label='Top@1 Vanilla R18')
plt.plot(AlfaValues, Top5, '*-.', markersize=5, color=color1, label='Top@5 DCT')
plt.plot(AlfaValues, Top5_v, '*-.', markersize=5, color=color2, label='Top@5 Vanilla R18')
plt.plot(AlfaValues, MCA, '*--', markersize=5, color=color1, label='MCA DCT')
plt.plot(AlfaValues, MCA_v, '*--', markersize=5, color=color2, label='MCA Vanilla R18')
plt.xlabel(r'$\alpha$ Parameter')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('Ablation Alfa.pdf', dpi=300)
