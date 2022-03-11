import torch

from python.sgxutils import SGXUtils

def main(args):
    sgxutils = SGXUtils()

    # For whatever reason, Weights is Row-Major and X is Column-Major. This is stupid.

    l = torch.nn.Linear(args.in_features, args.out_features, bias=False).cuda()
    x = torch.randn(args.batch, args.in_features).cuda()
    w_t = torch.transpose(l.weight, 0, 1)

    print("Weights:")
    print(w_t)
    print("X:")
    print(x)


    # given the weight; precompute w * r
    sgxutils.precompute(w_t, args.batch)

    # x_blinded = x + r
    x_blinded = sgxutils.addNoise(x)
    print("x_blinded:")
    print(x_blinded)

    # y_blinded = w * x_blinded
    y_blinded = l(x_blinded)
    print("y_blinded:")
    print(y_blinded)

    # y_recovered = y_blinded - w * r
    y_recovered = sgxutils.removeNoise(y_blinded)
    print("y_recovered:")
    print(y_recovered)
    s = sgxutils.nativeMatMul(w_t, x)
    print("s")
    print(s)

    y_expected = l(x)
    print("y_expected:")
    print(y_expected)

    print("Total diffs:", abs(y_expected - y_recovered).sum())

    return 0
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_features', type=int, default=10, help="Input feature of Linear layer")
    parser.add_argument('--out_features', type=int, default=30,  help="Output feature of Linear layer")
    parser.add_argument('--batch', type=int, default=30, help="Input batch size")

    args = parser.parse_args()
    main(args)