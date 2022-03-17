import os
import numpy as np
import matplotlib.pyplot as plt
import yaml

Colors = {'AT':'tab:red',
          'PKT':'tab:purple',
          'VID':'tab:brown',
          'CRD':'tab:gray',
          'Review':'tab:green'}


KDMethods = ["AT", "VID", "PKT", "REVIEW"]

# Results with original alpha values
original_results = {'AT': {'0': 45.43},
                    'PKT': {'0': 44.59},
                    'VID': {'0': 42.09},
                    'CRD': {'0': 45.46},
                    'Review': {'0': 44.56}}


results_path = os.path.join('Results', 'ADE20K', 'SOTA_Alpha_Search')

folders = os.listdir(results_path)

results_dict = dict()
for i, model_str in enumerate(folders):
    method_used = model_str.split(' ')[4]

    model_path = os.path.join(results_path, model_str)

    model_CONFIG = yaml.safe_load(open(os.path.join(model_path, 'config_' + method_used.upper() + '.yaml'), 'r'))

    model_results = yaml.safe_load(open(os.path.join(model_path, 'ResNet18 ADE20K Validation Report.yaml'), 'r'))

    if method_used not in results_dict:
        results_dict[method_used] = dict()

    results_dict[method_used][model_CONFIG['DISTILLATION']['PERCENTAGE_CHANGE_ALPHA']] = float((model_results['VALIDATION']['ACCURACY TOP1']).split()[0])

plt.figure()

for key in results_dict.keys():
    model_dict = results_dict[key]

    x = list()
    y = list()
    for result_key in model_dict.keys():
        if 1 >= float(result_key) >= -1:
            x.append(float(result_key)*100)
            y.append(float(model_dict[result_key]))

    # Add original alpha value result
    x.append(0)
    y.append(original_results[key]['0'])

    x_sorted, y_sorted = zip(*sorted(zip(x, y)))

    plt.plot(x_sorted, y_sorted, '*-', markersize=5, color=Colors[key], label=key)

plt.xlabel('% of alpha variation')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('SOTA_Alpha_Study_Curves.svg')


plt.figure()
results = list()
for key in results_dict.keys():
    model_dict = results_dict[key]

    results_aux = list()
    for result_key in model_dict.keys():
        if 1 >= float(result_key) >= -1:
            results_aux.append(float(model_dict[result_key]))
    results.append(results_aux)


plt.boxplot(results, whis=10)
plt.axhline(y=47.35, color='r', linestyle='-', label='Ours')
plt.plot(np.arange(1, 6), (44.56, 45.46, 42.09, 45.43, 44.59), 'xb', label=r'Originally proposed $\alpha$')
plt.xticks(ticks=np.arange(1, 6), labels=('Review', 'CRD', 'VID', 'AT', 'PKT'))
plt.xlabel('Method')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('SOTA_Alpha_Study Boxplot.svg')

for key in results_dict.keys():
    max_prec = 0
    percetange = 0
    model_dict = results_dict[key]

    for result_key in model_dict.keys():
        if float(model_dict[result_key]) > max_prec:
            max_prec = float(model_dict[result_key])
            percetange = result_key

    print('Best value for {} is a {} obtained with {}'.format(key, max_prec, percetange))