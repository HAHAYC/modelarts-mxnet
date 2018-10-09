# Copyright 2018 Deep Learning Service of Huawei Cloud. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Copyright 2018 Deep Learning Service of Huawei Cloud. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# use example
"""
# parameter input example
use_auto_lr =  1
train_file = flower_256_q100_train.rec
val_file = flower_256_q100_test.rec
num_classes = 102
checkpoint_url = s3://yangjie/resnet/resnet-50

# UI input example
MXNet-1.1.0-python2.7
data_url = /obs-lpf/data/flower/
train_url = /yangjie/ckpt/
"""


import os

os.environ['MXNET_CPU_WORKER_NTHREADS'] = '40'
os.environ['USE_AUTO_LR'] = '1'
import argparse
import logging

logging.basicConfig(level=logging.DEBUG)
import mxnet as mx
import moxing.mxnet as mox
import time
from moxing.mxnet.utils import contrib_metrics

def get_fine_tune_model(symbol, arg_params, num_classes, layer_name):
    """
    symbol: the pre-trained network symbol
    arg_params: the argument parameters of the pre-trained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = symbol.get_internals()
    net = all_layers[layer_name + '_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k: arg_params[k] for k in arg_params if 'fc' not in k})
    return (net, new_args)


def get_data():
    train_file = mox.get_hyper_parameter('train_file')
    val_file = mox.get_hyper_parameter('val_file')
    data_url = mox.get_hyper_parameter('data_url')
    train_path = os.path.join(data_url, train_file)
    val_path = os.path.join(data_url, val_file)
    data_shape = tuple([int(l) for l in args.image_shape.split(',')])
    if mox.file.is_directory(train_path) and mox.file.is_directory(val_path):
        data_set = mox.get_data_iter('imageraw',
                                     hyper_train={'data_shape':data_shape, 'batch_size':args.batch_size},
                                     hyper_val={'data_shape':data_shape, 'batch_size':args.batch_size},
                                     num_process=128)
    else:
        data_set = mox.get_data_iter('imagerecord',
                                     hyper_train={'data_shape':data_shape, 'batch_size':args.batch_size},
                                     hyper_val={'data_shape':data_shape, 'batch_size':args.batch_size})
    return data_set


def get_optimizer_params(lr, mom, wd):
    optimizer_params = mox.get_optimizer_params(
        num_examples=args.num_examples,
        lr=lr,
        mom=mom,
        wd=wd,
        lr_scheduler_mode='None_Scheduler',
        batch_size=args.batch_size,
        num_epoch=args.num_epoch)
    return optimizer_params


def get_model(network):
    num_gpus = mox.get_hyper_parameter('num_gpus')
    devs = mx.cpu() if num_gpus is None or num_gpus == 0 else [mx.gpu(int(i))
                                                               for i in
                                                               range(num_gpus)]
    model = mx.mod.Module(
        context=devs,
        symbol=network
    )
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '1'
    return model


def get_parameter(model, data_set, arg_params, aux_params, metrics):
    logging.info("search beigin")
    start_time = time.time()
    hyper_param_seletor = mox.TrainingHyperSelector(data_set, model,
                                                arg_params=arg_params,
                                                aux_params=aux_params,
                                                num_examples=args.num_examples,
                                                batch_size=args.batch_size,
                                                model_size=None,
                                                by_acc=True)

    best_lr, best_mom, best_wd = hyper_param_seletor.search(
        pre_train_epoch=args.pre_train_epoch,
        search_momentum=args.search_momentum,
        init_learning_rate=args.init_learning_rate,
        end_learning_rate=args.end_learning_rate,
        evaluate_every_n_steps=args.pre_train_evaluate_every_n_steps,
        metrics=metrics)

    search_time = time.time()
    logging.info("Best lr %f, Best weight_decay %f, Best momentum %f ",
                 best_lr, best_wd, best_mom)
    logging.info('Time: %f h', ((search_time - start_time) / 3600))

    return best_lr, best_mom, best_wd


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="fine-tune a dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint_url', type=str,
                        help='the pre-trained model')
    parser.add_argument('--train_url', type=str, help='the path model saved')
    parser.add_argument('--layer_before_fullc', type=str, default='flatten0',
                        help='the name of the layer before the last fullc layer')
    parser.add_argument('--num_classes', type=int, help='the number of classes')
    parser.add_argument('--num_examples', type=int, default=7370,
                        help='the number of training examples')
    parser.add_argument('--image_shape', type=str, default='3, 224, 224', help='the shape of input data')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--load_epoch', type=int, default=0)
    parser.add_argument('--num_epoch', type=int, default=10000,
                        help='the number of training epochs')
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--mom', type=float, default=None)
    parser.add_argument('--wd', type=float, default=None)
    parser.add_argument('--pre_train_epoch', type=int, default=3)
    parser.add_argument('--search_momentum', type=bool, default=True)
    parser.add_argument('--init_learning_rate', type=float, default=None)
    parser.add_argument('--end_learning_rate', type=float, default=None)
    parser.add_argument('--pre_train_evaluate_every_n_steps', type=int,
                        default=1)
    parser.add_argument('--evaluate_every_n_steps', type=int, default=None)
    parser.add_argument('--save_frequency', type=int, default=1,
                        help='how many epochs to save model')
    parser.add_argument('--labels_name_url', type=str, default=None, help='labels_name for metric to match')
    parser.add_argument('--export_model', type=int, default=1, help='1: export model for predict job \
                                                                     0: not export model')
    args, _ = parser.parse_known_args()

    mox.set_hyper_parameter('num_samples', args.num_examples)

    best_lr = args.lr
    best_mom = args.mom
    best_wd = args.wd

    # load data
    data_set = get_data()

    # load pretrained model
    sym, arg_params, aux_params = mox.load_model(args.checkpoint_url,
                                                 args.load_epoch)

    # remove the last fullc layer
    (new_sym, new_args) = get_fine_tune_model(
        sym, arg_params, args.num_classes, args.layer_before_fullc)

    # load model
    model = get_model(new_sym)
    params_tuple = (new_args, aux_params)

    # metrics contain recall etc.
    labels_name = []
    label_num = []
    if args.train_url is not None and len(args.train_url):
        for i in range(args.num_classes):
            label_num.append(i)
        if args.labels_name_url is not None:
            with mox.file.File(args.labels_name_url, 'r') as file:
                data = file.readlines()
                for line in data:
                    labels_name.append(line.split()[0])
        else:
            labels_name = label_num
        metrics = contrib_metrics.GetMetricsmulticlass(labels=label_num, total_label=labels_name, train_url=args.train_url)
        mox.set_hyper_parameter('acc_name', 'metrics')
    else:
        metrics = [mx.metric.Accuracy(), mx.metric.CrossEntropy()]

    if args.lr is None or args.mom is None or args.wd is None:
        best_lr, best_mom, best_wd = get_parameter(model, data_set, new_args,
                                                   aux_params, metrics)

    optimizer_params = get_optimizer_params(best_lr, best_mom, best_wd)

    if args.train_url is not None and len(args.train_url):
        worker_id = mox.get_hyper_parameter('worker_id')
        save_path = args.train_url if worker_id == 0 else "%s-%d" % (args.train_url, worker_id)
        epoch_end_callbacks = mx.callback.do_checkpoint(save_path, args.save_frequency)
    else:
        epoch_end_callbacks = None

    # train
    mox.set_hyper_parameter('train_type', 'auto_lr')
    mox.set_hyper_parameter('evaluate_every_n_steps',
                            args.evaluate_every_n_steps)
    mox.run(data_set, model, params_tuple,
            run_mode=mox.ModeKeys.TRAIN,
            optimizer=args.optimizer,
            optimizer_params=optimizer_params,
            batch_size=args.batch_size,
            epoch_end_callbacks=epoch_end_callbacks,
            load_epoch=args.load_epoch,
            num_epoch=args.num_epoch,
            metrics=metrics, force_init=True)

    if args.export_model == 1 and args.train_url is not None and len(args.train_url):
        end_epoch = args.num_epoch // args.save_frequency * args.save_frequency
        params_path = '%s-%04d.params' % (save_path, end_epoch)
        json_path = ('%s-symbol.json' % save_path)
        logging.info(params_path + 'used to predict')
        pred_params_path = os.path.join(args.train_url, 'model', 'pred_model-0000.params')
        pred_json_path = os.path.join(args.train_url, 'model', 'pred_model-symbol.json')
        mox.file.copy(params_path, pred_params_path)
        mox.file.copy(json_path, pred_json_path)
        for i in range(args.save_frequency, args.num_epoch + 1, args.save_frequency):
            mox.file.remove('%s-%04d.params' % (save_path, i))
        mox.file.remove(json_path)
        mox.file.remove(os.path.join(args.train_url, 'metric.json'))
