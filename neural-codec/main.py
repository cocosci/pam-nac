from cmrl import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test and see it argument parser works.')
    parser.add_argument('--learning_rate', type=float, help='learning_rate for training tanh NN')
    parser.add_argument('--learning_rate_followers', type=str,
                        help='learning_rate for training greedy followers NN', default='0.00002 0.000002')
    parser.add_argument('--epoch', type=int, help='epoch to train tanh NN.')
    parser.add_argument('--epoch_followers', type=str, help='epoch to fine tuning NN.', default='5 5 30 30')
    parser.add_argument('--from_where_step', type=int, help='0: from beginning; '
                                                            '1 from the second resnet or the first follower...',
                        default=0)
    parser.add_argument('--batch_size', type=int, help='batch size.', default=128)
    parser.add_argument('--num_resnets', type=int, help='num_resnets.', default=1)
    parser.add_argument('--training_mode', type=str, help='How to train the NN.',default='3')
    parser.add_argument('--base_model_id', type=str, help='which model to re-train?',default='0')
    parser.add_argument('--suffix', type=str, help='save model name suffix..', default='')
    parser.add_argument('--window_size', type=int, help='window_size', default=512)
    parser.add_argument('--bottleneck_kernel_and_dilation', type=str, help='bottleneck_kernel_and_dilation',
                        default='9 9 100 20 1 2')
    parser.add_argument('--is_pam', type=str, help='is_pam', default=0.0)
    parser.add_argument('--the_strides', type=str, help='the_strides', default='2 2')
    parser.add_argument('--save_unique_mark', type=str, help='save_unique_mark', default='')
    parser.add_argument('--loss_coeff', type=str, help='loss_coeff')
    parser.add_argument('--res_scalar', type=float, help='res_scalar', default=5)
    parser.add_argument('--pretrain_step', type=int, help='pretrain_step', default=0)
    parser.add_argument('--target_entropy', type=float, help='target_entropy', default=4)
    parser.add_argument('--num_bins_for_follower', type=str, help='num_bins_for_follower', default='32 32')
    args = parser.parse_args()
    print(args)

    audio_coding_ae = CMRL(args)
    
    if args.training_mode == '1':
        audio_coding_ae.model(training_mode='cascaded', arg=args)
    elif args.training_mode == '2':
        audio_coding_ae.model(training_mode='retrain_from_somewhere', arg=args)
    elif args.training_mode == '3':
        audio_coding_ae.model(training_mode='one_module', arg=args)
    elif args.training_mode == '0':
        audio_coding_ae.model(training_mode='feedforward', arg=args)
    else:
        print('WRONG INPUT...')