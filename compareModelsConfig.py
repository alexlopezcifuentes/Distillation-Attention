import os
import sys
from Libs.getConfiguration import getValidationConfiguration



Model1Path = os.path.join('CIFAR100', 'ID 2 ResNet20C CIFAR100 DFT')
Model2Path = os.path.join('CIFAR100', 'ResNet20C CIFAR100 AT')

CONFIGModel1 = getValidationConfiguration(Model1Path)
CONFIGModel2 = getValidationConfiguration(Model2Path)

c = 0
for key, conf1 in CONFIGModel1.items():
    conf2 = CONFIGModel2[key]

    for key2, conf1_2 in conf1.items():
        if key2 in conf2.keys():
            conf2_2 = conf2[key2]

            if type(conf1_2) is dict:
                for key3, conf1_3 in conf1_2.items():
                    conf2_3 = conf2_2[key3]

                    if conf1_3 != conf2_3:
                        print('{}-{}-{} is not the same. Value for Model 1: {}. Value for Model 2: {}.'.format(key, key2, key3, conf1_3, conf2_3))
                        c += 1
            else:
                if conf1_2 != conf2_2:
                    print('{}-{} is not the same. Value for Model 1: {}. Value for Model 2: {}.'.format(key, key2, conf1_2, conf2_2))
                    c += 1
        else:
            print('Key {} is in Model 1 config but not in Model 2.'.format(key2))

if c == 0:
    print('All the configuration is the same.')
