import torch
import time

from python.sgxutils import SGXUtils

def main(args):

    fast_times = []
    slow_times = []
    speedups = []
    for i in range(1, 6):
        sgxutils = SGXUtils()
        print("============================")
        print("\tRun #%d" % i)
        print("============================")
        l = torch.nn.Linear(args.in_features, args.out_features, bias=False).cuda()
        x = torch.randn(args.batch, args.in_features).cuda()
        w = l.weight

        # print("Weight (%d, %d):" % (w.size(0), w.size(1)))
        # print(w)
        # print("X (%d, %d):" % (x.size(0), x.size(1)))
        # print(x)


        # Time tracking for Fast Mult
        start_time = time.time()
        # given the weight; precompute w * r
        sgxutils.precompute(w, args.batch)

        # x_blinded = x + r
        x_blinded = sgxutils.addNoise(x)
        # print("x_blinded:")
        # print(x_blinded)

        # y_blinded = w * x_blinded
        y_blinded = l(x_blinded)
        # print("y_blinded:")
        # print(y_blinded)

        # y_recovered = y_blinded - w * r
        y_recovered = sgxutils.removeNoise(y_blinded)
        # print("y_recovered:")
        # print(y_recovered)
        fast_time = time.time() - start_time

        # Time tracking for Native Mult
        start_time = time.time()
        s = sgxutils.nativeMatMul(w, x)
        # print("s")
        # print(s)
        slow_time = time.time() - start_time

        y_expected = l(x)
        # print("y_expected:")
        # print(y_expected)

        print("Total diffs:", abs(y_expected - y_recovered).sum())

        fast_times.append(fast_time)
        slow_times.append(slow_time)
        speedups.append(slow_time / fast_time)

    print("----------------------")
    print("Fast Times:\t[ %s ]" % (", ".join(str(f) for f in fast_times)))
    print("Native Times:\t[ %s ]" % (", ".join(str(f) for f in slow_times)))
    print("Speedups:\t[ %s ]" % (", ".join(str(f) for f in speedups)))
    print()
    t_sum = 0.0
    for t in fast_times:
        t_sum += t
    fast_avg = t_sum / len(fast_times)
    t_sum = 0.0
    for t in slow_times:
        t_sum += t
    slow_avg = t_sum / len(slow_times)
    print("Average Fast Time:\t%s" % str(fast_avg))
    print("Average Native Time:\t%s" % str(slow_avg))
    print("Average Speedup:\t%s" % str(slow_avg / fast_avg))

    return 0
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_features', type=int, default=10, help="Input feature of Linear layer")
    parser.add_argument('--out_features', type=int, default=30,  help="Output feature of Linear layer")
    parser.add_argument('--batch', type=int, default=30, help="Input batch size")

    args = parser.parse_args()
    main(args)