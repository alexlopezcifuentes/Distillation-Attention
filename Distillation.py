import torch
from getConfiguration import getValidationConfiguration
import resnet
import PlottingUtils
import numpy as np
import Utils as utils
import os
import matplotlib.pyplot as plt
from scipy import ndimage
import gaussfitter
from scipy.stats import multivariate_normal
import torch.nn.functional as F
import torch.fft


def defineTeacher(teacher_model, multiscale=False):
    CONFIG_teacher = getValidationConfiguration(teacher_model)

    if CONFIG_teacher['MODEL']['ARCH'].lower() == 'resnet18':
        model_teacher = resnet.resnet18(num_classes=CONFIG_teacher['DATASET']['N_CLASSES'],
                                        multiscale=multiscale)
    elif CONFIG_teacher['MODEL']['ARCH'].lower() == 'resnet50':
        model_teacher = resnet.resnet50(num_classes=CONFIG_teacher['DATASET']['N_CLASSES'],
                                        multiscale=multiscale)

    # Load weigths
    completeTeacherPath = os.path.join('Results', teacher_model, 'Models', CONFIG_teacher['MODEL']['ARCH'] + '_' +
                                       CONFIG_teacher['DATASET']['NAME'] + '_best.pth.tar')

    checkpoint_teacher = torch.load(completeTeacherPath)
    model_teacher.load_state_dict(checkpoint_teacher['model_state_dict'])

    print('Teacher weights loaded!')

    return model_teacher


def calculate_frechet_distance(mu1_l, sigma1_l, mu2_l, sigma2_l, eps=1e-6):

    loss_l = list()
    for mu1, sigma1, mu2, sigma2 in zip(mu1_l, sigma1_l, mu2_l, sigma2_l):
        # First term. Mean Differences
        diff = torch.dot(torch.abs(mu1 - mu2), torch.abs(mu1 - mu2))

        # Second term. Covariances
        covmean = 2 * torch.pow(torch.matmul(sigma1, sigma2), 0.5)

        # Product might be almost singular
        if not torch.all(torch.isfinite(covmean)):
            offset = (torch.eye(covmean.shape[0]) * eps).cuda()
            covmean = 2 * torch.pow(torch.matmul(sigma1 + offset, sigma2 + offset), 0.5)

        # Numerical error might give slight imaginary component
        if torch.is_complex(covmean):
            covmean = covmean.real

        # Traces
        tr1 = torch.trace(sigma1)
        tr2 = torch.trace(sigma2)
        tr_covmean = torch.trace(covmean)

        loss_l.append(diff + tr1 + tr2 + tr_covmean)

    return torch.stack(loss_l, dim=0)


def getMean(AM):

    bs, h, w = AM.shape
    x = torch.arange(start=0, end=w, step=1).cuda()
    y = torch.arange(start=0, end=h, step=1).cuda()
    grid_y, grid_x = torch.meshgrid(x, y)

    means = list()
    for b in range(bs):
        map = AM[b, :]

        mean = torch.zeros(2).cuda()
        mean[0] = torch.sum(grid_x.flatten() * map.flatten())
        mean[1] = torch.sum(grid_y.flatten() * map.flatten())

        means.append(mean)

    return means


def getMeanandCov(AM):

    bs, h, w = AM.shape
    x = torch.arange(start=0, end=w, step=1).cuda()
    y = torch.arange(start=0, end=h, step=1).cuda()
    grid_y, grid_x = torch.meshgrid(x, y)

    means = list()
    covs = list()
    for b in range(bs):
        map = AM[b, :]

        mean = torch.zeros(2).cuda()
        # Mean x
        mean[0] = torch.sum(grid_x.flatten() * map.flatten())
        # Mean y
        mean[1] = torch.sum(grid_y.flatten() * map.flatten())

        cov = torch.zeros(2, 2).cuda()
        # STD x
        stdx = torch.sqrt(torch.sum(map.flatten() * torch.pow((grid_x.flatten() - mean[0]), 2)))
        # STD y
        stdy = torch.sqrt(torch.sum(map.flatten() * torch.pow((grid_y.flatten() - mean[1]), 2)))
        # STD yx
        stdyx = torch.sqrt(torch.sum(map.flatten() * torch.abs((grid_x.flatten() - mean[0]) * (grid_y.flatten() - mean[1]))))

        cov[0, 0] = stdx
        cov[0, 1] = stdyx
        cov[1, 0] = stdyx
        cov[1, 1] = stdy

        means.append(mean)
        covs.append(cov)

    return means, covs


def getMassCenterLoss(AMs_student, AMs_teacher):
    # Center of mass
    mean_s = getMean(AMs_student)
    mean_t = getMean(AMs_teacher)

    # Coordinates of center of mass to tensor
    mean_s = torch.stack(mean_s, dim=0)
    mean_t = torch.stack(mean_t, dim=0)

    # Euclidean Distances
    destillation_loss = torch.sqrt(torch.pow(mean_s[:, 0] - mean_t[:, 0], 2) + torch.pow(mean_s[:, 1] - mean_t[:, 1], 2))

    return destillation_loss


def distillationLoss(loss, features_student, features_teacher):

    # Compute Loss
    bs, chns, h, w = features_student[-1].shape

    if loss.lower() == 'l2' or loss.lower() == 'newl2':
        normalization = 'minmax'
    elif loss.lower() == 'dft':
        normalization = 'minmax'
    else:
        normalization = 'softmax'

    n_scales = len(features_teacher)
    scale_loss = torch.zeros(bs).float().cuda()
    for scale in range(n_scales):
        # Compute AMs
        if not loss.lower() == 'l2paper':
            AMs_student = PlottingUtils.returnAM(features_student[scale], resize=False, normalization=normalization)
            AMs_teacher = PlottingUtils.returnAM(features_teacher[scale], resize=False, normalization=normalization)

        if loss.lower() == 'newl2':
            # L2 Norm Loss
            destillation_loss = torch.sqrt(torch.sum(torch.pow(AMs_teacher.view(AMs_teacher.shape[0], -1) - AMs_student.view(AMs_teacher.shape[0], -1), 2), dim=1))

        elif loss.lower() == 'l2':
            destillation_loss = torch.pow(AMs_teacher - AMs_student, 2)
            destillation_loss = torch.mean(destillation_loss.view(destillation_loss.shape[0], -1), dim=1)

        elif loss.lower() == 'l2paper':
            destillation_loss = at_loss(features_student[scale], features_teacher[scale])

        elif loss.lower() == 'inner':
            # Inner product between two matrices
            destillation_loss = torch.sum(AMs_teacher * AMs_student)

        elif loss.lower() == 'masscenter':
            destillation_loss = getMassCenterLoss(AMs_student, AMs_teacher)

        elif loss.lower() == 'gaussian_fid':
            mean_s, cov_s = getMeanandCov(AMs_student)
            mean_t, cov_t = getMeanandCov(AMs_teacher)

            destillation_loss = calculate_frechet_distance(mean_s, cov_s, mean_t, cov_t)

        elif loss.lower() == 'gaussian':
            mean_s, cov_s = getMeanandCov(AMs_student)
            mean_t, cov_t = getMeanandCov(AMs_teacher)

            mean_s = torch.stack(mean_s)
            cov_s = torch.stack(cov_s)
            cov_s = cov_s.view(cov_s.shape[0], -1)

            mean_t = torch.stack(mean_t)
            cov_t = torch.stack(cov_t)
            cov_t = cov_s.view(cov_t.shape[0], -1)

            feat_s = torch.cat((mean_s, cov_s), dim=1)
            feat_t = torch.cat((mean_t, cov_t), dim=1)

            destillation_loss = torch.sqrt(torch.sum(torch.pow(feat_s - feat_t, 2), dim=1))

            # destillation_loss.append(torch.tensor(calculate_frechet_distance(mean_s, cov_s, mean_t, cov_t)))

            # Y, X = np.mgrid[0:student.shape[0]:1, 0:student.shape[1]:1]
            # pos = np.dstack((Y, X))

            # rv = multivariate_normal(mean_t, cov_t)
            # gaussian = rv.pdf(pos)

            # plt.figure()
            # plt.imshow(teacher)
            # plt.title('Teacher Activation Map')
            # plt.savefig('Teacher.png')
            #
            # plt.figure()
            # plt.imshow(gaussian)
            # plt.title('Gaussian fitted to teacher')
            # plt.savefig('Gaussian.png')
            #
            # fig = plt.figure()
            # ax = plt.axes(projection='3d')
            # ax.plot_wireframe(X, Y, teacher, color='black')
            # ax.plot_wireframe(X, Y, rv.pdf(pos), color='red')
            # ax.set_xlabel('x')
            # ax.set_ylabel('y')
            # ax.set_zlabel('z')
            # ax.set_title('Student Activation Map (Black) + Gaussian (Red)')
            # plt.savefig('3D.png')

            # destillation_loss = torch.stack(destillation_loss, dim=0)
        elif loss.lower() == 'dft':
            bs, h, w = AMs_student.shape

            torch.autograd.set_detect_anomaly(True)

            # Substract mean value
            mean_s = torch.repeat_interleave(torch.unsqueeze(torch.mean(AMs_student.view(bs, -1), dim=1),dim=1), h, dim=1)
            mean_s = torch.repeat_interleave(torch.unsqueeze(mean_s, dim=2), w, dim=2)

            mean_t = torch.repeat_interleave(torch.unsqueeze(torch.mean(AMs_teacher.view(bs, -1), dim=1), dim=1), h, dim=1)
            mean_t = torch.repeat_interleave(torch.unsqueeze(mean_t, dim=2), w, dim=2)

            AMs_student = AMs_student - mean_s
            AMs_teacher = AMs_teacher - mean_t

            # DCT STUDENT
            ffts_s = torch.fft.fft(torch.fft.fft(AMs_student, dim=1), dim=2)
            ffts_s = torch.abs(torch.real(ffts_s))

            # Normalize between 0-1
            min_s = torch.repeat_interleave(torch.unsqueeze(torch.min(ffts_s.view(bs, -1), dim=1)[0], dim=1), h, dim=1)
            min_s = torch.repeat_interleave(torch.unsqueeze(min_s, dim=2), w, dim=2)
            max_s = torch.repeat_interleave(torch.unsqueeze(torch.max(ffts_s.view(bs, -1), dim=1)[0], dim=1), h, dim=1)
            max_s = torch.repeat_interleave(torch.unsqueeze(max_s, dim=2), w, dim=2)

            ffts_s = (ffts_s - min_s) / max_s


            # DCT TEACHER
            ffts_t = torch.fft.fft(torch.fft.fft(AMs_teacher, dim=1), dim=2)
            ffts_t = torch.abs(torch.real(ffts_t))

            # Normalize between 0-1
            min_t = torch.repeat_interleave(torch.unsqueeze(torch.min(ffts_t.view(bs, -1), dim=1)[0], dim=1), h, dim=1)
            min_t = torch.repeat_interleave(torch.unsqueeze(min_t, dim=2), w, dim=2)
            max_t = torch.repeat_interleave(torch.unsqueeze(torch.max(ffts_t.view(bs, -1), dim=1)[0], dim=1), h, dim=1)
            max_t = torch.repeat_interleave(torch.unsqueeze(max_t, dim=2), w, dim=2)

            ffts_t = (ffts_t - min_t) / max_t

            # if scale == 0:
            #     for i in range(20):
            #         CAM_s = AMs_student[i, :].cpu()
            #         CAM_t = AMs_teacher[i, :].cpu()
            #         fs = ffts_s[i].cpu()
            #         ft = ffts_t[i].cpu()
            #
            #         plt.figure()
            #         plt.subplot(2, 2, 1)
            #         plt.imshow(CAM_s.cpu().detach())
            #         plt.title('Activation Map Student', fontsize=10)
            #         plt.colorbar()
            #         plt.subplot(2, 2, 2)
            #         plt.imshow(CAM_t.cpu().detach())
            #         plt.title('Activation Map Teacher', fontsize=10)
            #         plt.colorbar()
            #         plt.subplot(2, 2, 3)
            #         plt.imshow(fs.cpu().detach())
            #         plt.title('2DFT Student', fontsize=10)
            #         plt.colorbar()
            #         plt.subplot(2, 2, 4)
            #         plt.imshow(ft.cpu().detach())
            #         plt.title('2DFT Teacher', fontsize=10)
            #         plt.colorbar()
            #         plt.show()

            ffts_s = ffts_s.reshape(ffts_s.shape[0], -1)
            ffts_t = ffts_t.reshape(ffts_t.shape[0], -1)

            destillation_loss = torch.sqrt(torch.sum(torch.pow(ffts_s - ffts_t, 2), dim=1))

        else:
            print('Distillation function indicated in the configuration file is not available.')
            exit()

        scale_loss += destillation_loss

    return scale_loss


def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()
