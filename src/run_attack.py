import argparse
import torch
import numpy as np

from nn.enums import ExplainingMethod
from nn.networks import ExplainableNet
from nn.utils import get_expl, plot_overview, clamp, load_image, make_dir
import json

import torchvision.models as models
import torch.nn.functional as F
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD

def get_beta(i, num_iter):
    """
    Helper method for beta growth
    """
    start_beta, end_beta = 10.0, 100.0
    return start_beta * (end_beta / start_beta) ** (i / num_iter)


def main() -> None:

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--num_iter', type=int, default=1500, help='number of iterations')
    argparser.add_argument('--img_index', type=str, default=1, help='image index to run attack on')
    argparser.add_argument('--img', type=str, default='../data/collie4.jpeg', help='image net file to run attack on')

    argparser.add_argument('--target_img', type=str, default='../data/fiddler_crab.jpg',
                           help='imagenet file used to generate target expl')
    argparser.add_argument('--lr', type=float, default=0.0002, help='lr')
    argparser.add_argument('--cuda', help='enable GPU mode', action='store_true')
    argparser.add_argument('--output_dir', type=str, default='../output/', help='directory to save results to')
    argparser.add_argument('--beta_growth', help='enable beta growth', action='store_true')
    argparser.add_argument('--prefactors', nargs=2, default=[1e11, 1e6], type=float,
                           help='prefactors of losses (diff expls, class loss)')
    argparser.add_argument('--method', help='algorithm for expls',
                           choices=['lrp', 'guided_backprop', 'gradient', 'integrated_grad',
                                    'pattern_attribution', 'grad_times_input'],
                           default='lrp')
    args = argparser.parse_args()

    # options
    device = torch.device("cuda" if args.cuda else "cpu")
    method = getattr(ExplainingMethod, args.method)

    # instantiate a model (could also be a TensorFlow or JAX model)
    # att_model = models.resnet18(pretrained=True).eval().to(device)
    att_model = models.vgg16(pretrained=True).eval().to(device)
    # Explain Code
    data_mean = np.array([0.485, 0.456, 0.406])
    data_std = np.array([0.229, 0.224, 0.225])

    exp_model = ExplainableNet(att_model, data_mean=data_mean, data_std=data_std, beta=1000 if args.beta_growth else None)
    if method == ExplainingMethod.pattern_attribution:
        exp_model.load_state_dict(torch.load('../models/model_vgg16_pattern_small.pth'), strict=False)
    exp_model = exp_model.eval().to(device)
    # Attack Code

    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = PyTorchModel(att_model, bounds=(0, 1), preprocessing=preprocessing)

    # get data and test the model
    # wrapping the tensors with ep.astensors is optional, but it allows
    # us to work with EagerPy tensors in the following
    images, labels = samples(fmodel, dataset="imagenet", batchsize=7)
    clean_acc = accuracy(fmodel, images, labels)
    print(f"clean accuracy:  {clean_acc * 100:.1f} %")

    # apply the attack
    attack = LinfPGD()
    epsilons = [
        0.0,
        0.0002,
        0.0005,
        0.0008,
        0.001,
        0.0015,
        0.002,
        0.003,
        0.01,
        0.1,
        0.3,
        0.5,
        1.0,
    ]
    raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=epsilons)

    print("images size", images.size())
    # Explanation code
    x_image = clipped_advs[len(clipped_advs)-3][args.img_index].unsqueeze(0)
    if len(args.target_img) > 3:
        x_target = load_image(data_mean, data_std, device, args.target_img)
    else:
        x_target = images[args.img_index].unsqueeze(0)

    print("ximage", x_image.size())
    x_adv = x_image.clone().detach().requires_grad_()
    image0_expl, image0_acc, image0_idx = get_expl(exp_model, images[args.img_index].unsqueeze(0), method)
    org_expl, org_acc, org_idx = get_expl(exp_model, x_image, method)
    org_expl = org_expl.detach().cpu()
    target_expl, _, _ = get_expl(exp_model, x_target, method)
    target_expl = target_expl.detach()

    optimizer = torch.optim.Adam([x_adv], lr=args.lr)

    acc, class_idx = exp_model.classify(x_adv)
    print("adv class", class_idx)
    acc, class_idx = exp_model.classify(images[args.img_index].unsqueeze(0))
    print("original class", class_idx)

    for i in range(args.num_iter):
        if args.beta_growth:
            exp_model.change_beta(get_beta(i, args.num_iter))

        optimizer.zero_grad()

        # calculate loss
        adv_expl, adv_acc, class_idx = get_expl(exp_model, x_adv, method, desired_index=org_idx)
        loss_expl = F.mse_loss(adv_expl, target_expl)
        loss_output = F.mse_loss(adv_acc, org_acc.detach())
        total_loss = args.prefactors[0]*loss_expl + args.prefactors[1]*loss_output

        # update adversarial example
        total_loss.backward()
        optimizer.step()

        # clamp adversarial example
        # Note: x_adv.data returns tensor which shares data with x_adv but requires
        #       no gradient. Since we do not want to differentiate the clamping,
        #       this is what we need
        x_adv.data = clamp(x_adv.data, data_mean, data_std)
        if i % 100 == 0:
            print("Iteration {}: Total Loss: {}, Expl Loss: {}, Output Loss: {}".format(i, total_loss.item(), loss_expl.item(), loss_output.item()))

    # test with original model (with relu activations)
    exp_model.change_beta(None)
    adv_expl, adv_acc, class_idx = get_expl(exp_model, x_adv, method)
    acc, class_idx = exp_model.classify(x_adv)
    print("adv class after expl change", class_idx)
    # save results
    output_dir = make_dir(args.output_dir)
    # commented out target image target_expl, replaced with image0_expl
    plot_overview([x_target, x_image, x_adv], [image0_expl, org_expl, adv_expl], data_mean, data_std, \
                    captions=['Target Image', 'Original Image', 'Manipulated Image', 'Original Explanation', 'Adversarial Explanation', 'Manipulated Adv. Explanation'], \
                    filename=f"{output_dir}overview_{args.method}.png")
    torch.save(x_adv, f"{output_dir}x_{args.method}.pth")





    # calculate and report the robust accuracy (the accuracy of the model when
    # it is attacked)
    robust_accuracy = 1 - ep.astensor(success).float32().mean(axis=-1)
    # robust_accuracy = 1 - torch.tensor(success, dtype=torch.float32).mean(axis=-1)

    print("robust accuracy for perturbations with")
    for eps, acc in zip(epsilons, robust_accuracy):
        print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")

    # we can also manually check this
    # we will use the clipped advs instead of the raw advs, otherwise
    # we would need to check if the perturbation sizes are actually
    # within the specified epsilon bound
    print()
    print("we can also manually check this:")
    print()
    print("robust accuracy for perturbations with")
    for eps, advs_ in zip(epsilons, clipped_advs):
        acc2 = accuracy(fmodel, advs_, labels)
        print(f"  Linf norm ≤ {eps:<6}: {acc2 * 100:4.1f} %")
        print("    perturbation sizes:")
        perturbation_sizes = ep.astensor(advs_ - images).norms.linf(axis=(1, 2, 3)).numpy()
        print("    ", str(perturbation_sizes).replace("\n", "\n" + "    "))
        if acc2 == 0:
            break


if __name__ == "__main__":
    main()