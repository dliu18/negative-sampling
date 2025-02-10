import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--loss_func', type=str,default="sg",
                        help="loss function to use between 'sg' or 'sg aug'")
    parser.add_argument('--base_model', type=str,default="n2v",
                        help="base model to use between 'n2v' or 'line'")
    parser.add_argument('--test_set', type=str,default="test",
                        help="Is either 'test' or 'valid' i.e. validation")
    parser.add_argument('--test_set_frac', type=float,default=0.2,
                        help="Size of the test set split, as a fraction of the original dataset")
    parser.add_argument('--lam', type=float,default=0.0,
                        help="dimension regularization hyperparameter")
    parser.add_argument('--batch_size', type=int,default=128,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--alpha', type=float,default=0.0,
                        help="Multinomial exponent for skip gram negative samples")
    parser.add_argument('--K', type=int,default=1,
                        help="The number of negative samples per positive sample")
    parser.add_argument('--dropout', type=int,default=0,
                        help="using the dropout or not")
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='gowalla',
                        help="available datasets: [lastfm, gowalla, yelp2018, amazon-book]")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[20]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int,default=1,
                        help="enable tensorboard")
    parser.add_argument('--board_path', type=str,default="",
                        help="path to save tensorboard logs within runs/")
    parser.add_argument('--n_negative', type=int,default=10,
                        help="Cadence (in epochs) at which the dimension regularization is applied.")
    parser.add_argument('--n2v_p', type=float,default=1.0,
                        help="p hyperparameter for n2v random walks. Small values encourage local exploration.")
    parser.add_argument('--n2v_q', type=float,default=1.0,
                        help="q hyperparameter for n2v random walks. Small values encourage distant (DFS) exploration.")
    parser.add_argument('--comment', type=str,default="")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--bypass_skipgram', type=bool,default=False)
    parser.add_argument('--epochs', type=int,default=1000)
    parser.add_argument('--classifier_epochs', type=int,default=5)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='lgn', help='rec-model, support [mf, lgn]')
    return parser.parse_args()
