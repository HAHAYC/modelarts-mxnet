from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import mxnet as mx
import moxing.mxnet as mox
import logging
logging.basicConfig(level=logging.DEBUG)
import os
import sys
sys.path.append('/usr/local/anaconda2/lib/python2.7/site-packages/moxing/mxnet/module/rcnn')
from moxing.mxnet.module.rcnn.tester import Predictor, pred_eval
from moxing.mxnet.data.rcnn_load.dataset.pascal_voc_eval import voc_eval
import moxing.mxnet as mox
import argparse
import fine_tune_config
from moxing.mxnet.data.rcnn_load.loader import AnchorLoader
from moxing.mxnet.utils.safe_pickle import restricted_loads
import xml.etree.ElementTree as ET
import six as _six
if _six.PY2:
    import cPickle
else:
    import pickle as cPickle
import pprint
from moxing.mxnet.data.rcnn_load.loader import TestLoader
import numpy as np
from moxing.mxnet.config.rcnn_config import config
import logging
import mxnet as mx
from moxing.mxnet.data.rcnn_load.dataset.imdb import IMDB
from moxing.framework import file
from moxing.framework.common.metrics import object_detection_metrics

class AllDataEval(IMDB):
    def __init__(self, image_set, root_path, devkit_path, classes, save_url=None):
        """
        fill basic information to initialize imdb
        :param image_set: 2007_trainval, 2007_test, etc
        :param root_path: 'selective_search_data' and 'cache'
        :param devkit_path: data and results
        :param save_url: url to store the metric.json
        :return: imdb object
        """
        super(AllDataEval, self).__init__(image_set, root_path, devkit_path, save_url)  # set self.name
        self.save_url = save_url
        self.root_path = root_path
        self.devkit_path = devkit_path
        self.data_path = os.path.join(devkit_path, 'eval')
        self.classes = classes
        self.num_classes = len(self.classes)
        self.num_images = 0
        self.config = {'comp_id': 'comp4',
                       'use_diff': False,
                       'min_size': 2}
        self.image_set_list = []

    def image_path_from_index(self, index):
        """
        given image index, find out full path
        :param index: index of a specific image
        :return: full path of this image
        """
        image_file = os.path.join(self.data_path, 'Images', index + '.jpg')
        #assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        ## support obs
        assert file.exists(image_file), 'Path does not exist: {}'.format(image_file)
        ## support obs
        return image_file

    def gt_roidb(self):
        """
        return ground truth image regions database
        :return: imdb[image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        # read cache
        cache_path = os.path.join(self.devkit_path, 'cache')
        if not mox.file.exists(cache_path):
            mox.file.make_dirs(cache_path)
        cache_file = os.path.join(cache_path, 'object_detection_gt_roidb_eval.pkl')
        if mox.file.exists(cache_file):
            if _six.PY2:
                data_string = mox.file.read(cache_file)
                roidb = cPickle.loads(data_string)
            else:
                data_string = mox.file.read(cache_file, binary=True)
                roidb = cPickle.loads(data_string, encoding='iso-8859-1')
            self.num_images = len(roidb)
            logging.info('roidb loaded from %s' % (cache_file))
            if mox.file.exists(os.path.join(self.data_path)):
                data_path = os.path.join(self.data_path, 'Images')
                assert mox.file.exists(data_path), "data file path not exists"
                self.image_set_list = mox.file.list_directory(data_path)
                for i in range(len(self.image_set_list)):
                    self.image_set_list[i] = self.image_set_list[i][0:-4]
                self.num_images = len(self.image_set_list)
            return roidb

        gt_roidb = None
        if mox.file.exists(os.path.join(self.data_path)):
            data_path = os.path.join(self.data_path, 'Images')
            assert mox.file.exists(data_path), "data file path not exists"
            self.image_set_list = mox.file.list_directory(data_path)
            for i in range(len(self.image_set_list)):
                self.image_set_list[i] = self.image_set_list[i][0:-4]
            self.num_images = len(self.image_set_list)
            with mox.file.File(os.path.join(self.devkit_path, 'cache', self.image_set + '.txt'), 'w') as file_txt:
                for i in self.image_set_list:
                    file_txt.write(i + '\n')
            gt_roidb = [self.load_pascal_annotation(index) for index in self.image_set_list]
            # with open(cache_file, 'wb') as fid:
            #     cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
            ## support obs
            data_string = cPickle.dumps(gt_roidb, cPickle.HIGHEST_PROTOCOL)
            file.write(cache_file, data_string, binary=True)
            ## support obs
            logging.info('%s wrote gt roidb to %s' % (self.name, cache_file))

        return gt_roidb

    def load_pascal_annotation(self, index):
        """
        for a given index, load image and bounding boxes info from XML file
        :param index: index of a specific image
        :return: record['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        import xml.etree.ElementTree as ET
        roi_rec = dict()
        roi_rec['image'] = self.image_path_from_index(index)
        #size = cv2.imread(roi_rec['image']).shape
        ## support obs
        size = mx.img.imdecode(file.read(roi_rec['image'], binary=True)).shape
        ## support obs
        roi_rec['height'] = size[0]
        roi_rec['width'] = size[1]

        filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
        #tree = ET.parse(filename)
        ## support obs
        tree = ET.ElementTree()
        parser = ET.XMLParser(target=ET.TreeBuilder())
        parser.feed(file.read(filename, binary=True))
        tree._root = parser.close()
        ## support obs
        objs = tree.findall('object')
        if not self.config['use_diff']:
            non_diff_objs = [obj for obj in objs if int(obj.find('difficult').text) == 0]
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        class_to_index = dict(zip(self.classes, range(self.num_classes)))
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = class_to_index[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        roi_rec.update({'boxes': boxes,
                        'gt_classes': gt_classes,
                        'gt_overlaps': overlaps,
                        'max_classes': overlaps.argmax(axis=1),
                        'max_overlaps': overlaps.max(axis=1),
                        'flipped': False})
        return roi_rec

    def evaluate_detections(self, detections):
        """
        top level evaluations
        :param detections: result matrix, [bbox, confidence]
        :return: None
        """
        # make all these folders for results
        result_dir = os.path.join(self.devkit_path, 'cache', 'results')
        # if not os.path.exists(result_dir):
        #     os.mkdir(result_dir)
        ## support obs
        if not file.exists(result_dir):
            file.make_dirs(result_dir)
        ## support obs
        year_folder = os.path.join(self.devkit_path, 'cache', 'results')
        # if not os.path.exists(year_folder):
        #     os.mkdir(year_folder)IMDB
        ## support obs
        if not file.exists(year_folder):
            file.make_dirs(year_folder)
        ## support obs
        res_file_folder = os.path.join(self.devkit_path, 'cache', 'results', 'Main')
        # if not os.path.exists(res_file_folder):
        #     os.mkdir(res_file_folder)
        ## support obs
        if not file.exists(res_file_folder):
            file.make_dirs(res_file_folder)
        ## support obs

        self.write_pascal_results(detections)
        self.do_python_eval()

    def get_result_file_template(self):
        """
        this is a template
        <comp_id>_det_test_aeroplane.txt
        :return: a string template
        """
        res_file_folder = os.path.join(self.devkit_path, 'cache', 'results', 'Main')
        comp_id = self.config['comp_id']
        filename = comp_id + '_det_' + self.image_set + '_{:s}.txt'
        path = os.path.join(res_file_folder, filename)
        return path

    def write_pascal_results(self, all_boxes):
        """
        write results files in pascal devkit path
        :param all_boxes: boxes to be processed [bbox, confidence]
        :return: None
        """
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            logging.info('Writing %s results file' % cls)
            filename = self.get_result_file_template().format(cls)
            # with open(filename, 'wt') as f:
            #     for im_ind, index in enumerate(self.image_set_index):
            #         dets = all_boxes[cls_ind][im_ind]
            #         if len(dets) == 0:
            #             continue
            #         # the VOCdevkit expects 1-based indices
            #         for k in range(dets.shape[0]):
            #             f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
            #                     format(index, dets[k, -1],
            #                            dets[k, 0] + 1, dets[k, 1] + 1, dets[k, 2] + 1, dets[k, 3] + 1))
            ## support obs
            write_buff = ""
            for im_ind, index in enumerate(self.image_set_list):
                dets = all_boxes[cls_ind][im_ind]
                if len(dets) == 0:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    write_buff += '{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(index,
                                  dets[k, -1], dets[k, 0] + 1, dets[k, 1] + 1,
                                  dets[k, 2] + 1, dets[k, 3] + 1)
            file.write(filename, write_buff, binary=True)
            ## support obs

    def do_python_eval(self):
        """
        python evaluation wrapper
        :return: None
        """
        annopath = os.path.join(self.data_path, 'Annotations', '{0!s}.xml')
        imageset_file = os.path.join(self.devkit_path, 'cache', self.image_set + '.txt')
        annocache = os.path.join(self.devkit_path, 'cache', self.name + '_annotations.pkl')
        aps = []
        rec_prec_ap = {}
        class_metrics_map = {}
        rec_avg = 0
        prec_avg = 0
        ap_avgs = 0
        rec_avgs = 0
        prec_avgs = 0

        # The PASCAL VOC metric changed in 2010
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            filename = self.get_result_file_template().format(cls)
            rec, prec, ap = voc_eval(filename, annopath, imageset_file, cls, annocache=annocache,
                                     ovthresh=0.5, use_07_metric=True)
            aps += [ap]
            rec_avg = rec.sum()/rec.size
            rec_avgs += rec_avg
            prec_avg = prec.sum()/prec.size
            prec_avgs += prec_avg
            rec_prec_ap['id'] = cls_ind
            rec_prec_ap['recall'] = rec_avg
            rec_prec_ap['precision'] = prec_avg
            rec_prec_ap['accuracy'] = ap
            ap_avgs += ap
            class_metrics_map[cls] = rec_prec_ap.copy()
            logging.info('AP for {} = {:.4f}'.format(cls, ap))
        logging.info('Mean AP = {:.4f}'.format(np.mean(aps)))
        rec_prec_ap['recall'] = rec_avgs/(self.num_classes - 1)
        rec_prec_ap['precision'] = prec_avgs/(self.num_classes - 1)
        rec_prec_ap['accuracy'] = ap_avgs/(self.num_classes - 1)
        class_metrics_map['total'] = rec_prec_ap
        object_detection_metrics.get_metrics(self.save_url, class_metrics_map)

def e2e_eval(args):
    # default
    assert args.train_url != None, 'checkpoint_url should not be None'
    assert args.load_epoch != None, 'load_epoch should not be None'
    # set environment parameters
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    os.environ['PYTHONUNBUFFERED'] = '1'
    config.TEST.HAS_RPN = True
    pprint.pprint(mox.rcnn_config.config)
    # load classes name
    classes = get_label_names(args)
    # load data
    imdb = AllDataEval('test', 'data', mox.rcnn_config.default.dataset_path, classes, args.train_url)
    roidb = imdb.gt_roidb()
    test_data = TestLoader(roidb, batch_size=1, shuffle=False, has_rpn=True)
    # load symbol
    symbol = mox.get_model('object_detection', 'resnet_rcnn', num_classes=2, is_train=False)
    # load model params
    model_prefix = args.train_url
    load_epoch = args.load_epoch
    arg_params, aux_params = mox.rcnn_load_param(
        model_prefix, load_epoch,
        convert=True, data=test_data, process=True,
        is_train=False, sym=symbol)
    max_data_shape = [('data', (1, 3, max([v[0] for v in mox.rcnn_config.config.SCALES]),
        max([v[1] for v in mox.rcnn_config.config.SCALES])))]
    # create predictor
    devs = [mx.gpu(0)]
    predictor = Predictor(
        symbol,
        data_names=[k[0] for k in test_data.provide_data],
        label_names=None,
        context=devs,
        max_data_shapes=max_data_shape,
        provide_data=test_data.provide_data,
        provide_label=test_data.provide_label,
        arg_params=arg_params,
        aux_params=aux_params)

    # start detection
    pred_eval(predictor, test_data, imdb, vis=False, thresh=0.001)

def add_parameter():
    parser = argparse.ArgumentParser(description='train faster rcnn',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint_url', type=str, help='the pre-trained model')
    parser.add_argument('--train_url', type=str, help='the path model saved')
    parser.add_argument('--num_classes', type=int, help='the number of classes')
    parser.add_argument('--load_epoch', type=int, help='load the model on epoch use checkpoint_url')
    parser.add_argument('--num_epoch', type=int, help='the number of training epochs')
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--mom', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=0.0005)
    parser.add_argument('--network', type=str, default='resnet_rcnn', help='name of network')
    parser.add_argument('--labels_name_url', type=str, default=None, help='the classes txt file name')
    parser.add_argument('--export_model', type=int, default=1, help='1: export model for predict job \
                                                                     0: not export model')
    args, _ = parser.parse_known_args()
    return args

def set_config(args):
    mox.rcnn_config.config = fine_tune_config.config
    mox.rcnn_config.default = fine_tune_config.default
    if args.checkpoint_url is not None:
        mox.rcnn_config.default.pretrained = args.checkpoint_url
    if args.load_epoch is not None:
        mox.rcnn_config.default.pretrained_epoch = args.load_epoch
    if args.num_classes is not None:
        mox.rcnn_config.config.NUM_CLASSES = args.num_classes
    if args.lr is not None:
        mox.rcnn_config.default.base_lr = args.lr
    if args.train_url is not None:
        mox.rcnn_config.default.e2e_prefix = args.train_url
    if args.num_epoch is not None:
        mox.rcnn_config.default.e2e_epoch = args.num_epoch
    mox.rcnn_config.default.dataset_path = mox.get_hyper_parameter('data_url')

def get_label_names(args):
  classes = ['__background__']
  if args.labels_name_url is not None:
      classes_read = []
      path_classese_txt = os.path.join(mox.rcnn_config.default.dataset_path,
                                       args.labels_name_url)
      assert mox.file.exists(path_classese_txt), 'No such file in the path'
      with mox.file.File(path_classese_txt, 'r') as f:
          data = f.readlines()
          for line in data:
              classes_read.append(line.split())
      for item_read in classes_read:
          if len(item_read) == 2:
              if item_read[0] == 'name:':
                  classes.append(eval(item_read[1]))
  else:
      search_path = os.path.join(mox.rcnn_config.default.dataset_path, 'train/Annotations')
      for annotation_file in mox.file.list_directory(search_path):
            filename = os.path.join(search_path, annotation_file)
            tree = ET.ElementTree()
            parser = ET.XMLParser(target=ET.TreeBuilder())
            parser.feed(mox.file.read(filename, binary=True))
            tree._root = parser.close()
            objs = tree.findall('object')
            non_diff_objs = [obj for obj in objs if
                             int(obj.find('difficult').text) == 0]
            objs = non_diff_objs
            for ix, obj in enumerate(objs):
                class_name = obj.find('name').text.lower().strip()
                if class_name not in classes:
                    classes.append(class_name)
  logging.info("classes names: %s", classes)
  return classes

def gt_roidb(image_set_list, args):
    cache_path = os.path.join(mox.rcnn_config.default.dataset_path, 'cache')
    if not mox.file.exists(cache_path):
        mox.file.make_dirs(cache_path)
    cache_file = os.path.join(cache_path, 'object_detection_gt_roidb.pkl')
    if mox.file.exists(cache_file):
        if _six.PY2:
            data_string = mox.file.read(cache_file)
            roidb = cPickle.loads(data_string)
        else:
            data_string = mox.file.read(cache_file, binary=True)
            roidb = cPickle.loads(data_string, encoding='iso-8859-1')
        logging.info('roidb loaded from %s' % (cache_file))
        return roidb

    classes = get_label_names(args)
    gt_roidb = [load_annotation(item, classes) for item in image_set_list]
    data_string = cPickle.dumps(gt_roidb, cPickle.HIGHEST_PROTOCOL)
    if _six.PY2:
        mox.file.write(cache_file, data_string)
    else:
        mox.file.write(cache_file, data_string, binary=True)
    logging.info('write gt roidb to %s' % (cache_file))
    return gt_roidb

def load_annotation(item, classes):
    num_classes = mox.rcnn_config.config.NUM_CLASSES

    roi_rec = dict()
    roi_rec['image'] = os.path.join(mox.rcnn_config.default.dataset_path, 'train', 'Images', item)
    size = mx.img.imdecode(mox.file.read(roi_rec['image'], binary=True)).shape
    roi_rec['height'] = size[0]
    roi_rec['width'] = size[1]

    filename = os.path.join(mox.rcnn_config.default.dataset_path, 'train',
                            'Annotations', item.split('.')[0] + '.xml')
    tree = ET.ElementTree()
    parser = ET.XMLParser(target=ET.TreeBuilder())
    parser.feed(mox.file.read(filename, binary=True))
    tree._root = parser.close()
    objs = tree.findall('object')
    non_diff_objs = [obj for obj in objs if int(obj.find('difficult').text) == 0]
    objs = non_diff_objs
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, num_classes), dtype=np.float32)

    class_to_index = dict(zip(classes, range(num_classes)))
    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1
        cls = class_to_index[obj.find('name').text.lower().strip()]
        boxes[ix, :] = [x1, y1, x2, y2]
        gt_classes[ix] = cls
        overlaps[ix, cls] = 1.0

    roi_rec.update({'boxes': boxes,
                    'gt_classes': gt_classes,
                    'gt_overlaps': overlaps,
                    'max_classes': overlaps.argmax(axis=1),
                    'max_overlaps': overlaps.max(axis=1),
                    'flipped': False})
    return roi_rec

def get_train_data_iter(sym, ctx, args):
    data_path = mox.rcnn_config.default.dataset_path
    train_data = None
    if mox.file.exists(os.path.join(data_path, "train")):
        data_path = os.path.join(data_path, "train/Images")
        assert mox.file.exists(data_path), "train file path not exists"
        image_set_list = mox.file.list_directory(data_path)
        roidb = gt_roidb(image_set_list, args)
        feat_sym = sym.get_internals()['rpn_cls_score_output']
        input_batch_size = mox.rcnn_config.config.TRAIN.BATCH_IMAGES * len(ctx)
        train_data = AnchorLoader(
            feat_sym=feat_sym,
            roidb=roidb,
            batch_size=input_batch_size,
            shuffle=True,
            ctx=ctx,
            work_load_list=None,
            feat_stride=mox.rcnn_config.config.RPN_FEAT_STRIDE,
            anchor_scales=mox.rcnn_config.config.ANCHOR_SCALES,
            anchor_ratios=mox.rcnn_config.config.ANCHOR_RATIOS,
            aspect_grouping=mox.rcnn_config.config.TRAIN.ASPECT_GROUPING)
    return train_data

def get_model(data_set, symbol, ctx):
    input_batch_size = data_set.batch_size
    max_data_shape = [('data', (input_batch_size, 3, max([v[0] for v in mox.rcnn_config.config.SCALES]),
                                max([v[1] for v in mox.rcnn_config.config.SCALES])))]
    max_data_shape, max_label_shape = data_set.infer_shape(max_data_shape)
    max_data_shape.append(('gt_boxes', (input_batch_size, 100, 5)))
    logger = logging.getLogger()
    logger.info('providing maximum shape %s %s' % (max_data_shape, max_label_shape))
    model = mox.MutableModule(
        symbol,
        data_names=[k[0] for k in data_set.provide_data],
        label_names=[k[0] for k in data_set.provide_label],
        logger=logger, context=ctx, work_load_list=None,
        max_data_shapes=max_data_shape, max_label_shapes=max_label_shape,
        fixed_param_prefix=mox.rcnn_config.config.FIXED_PARAMS)
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    os.environ['PYTHONUNBUFFERED'] = '1'
    return model

def check_data_url():
    data_path = mox.get_hyper_parameter('data_url')
    if mox.file.exists(os.path.join(data_path, "train")) and \
       mox.file.exists(os.path.join(data_path, "eval")):
        return
    if mox.file.exists(os.path.join(data_path, "Images")) and \
       mox.file.exists(os.path.join(data_path, "Annotations")):
        img_path = os.path.join(data_path, "Images")
        ann_path = os.path.join(data_path, "Annotations")
        image_set_list = mox.file.list_directory(img_path)
        num_examples = len(image_set_list)
        train_num = int(num_examples * 0.8)
        shuffle_list = range(num_examples)
        import random
        random.shuffle(shuffle_list)
        train_img_list = []
        eval_img_list = []
        for idx, item in enumerate(shuffle_list):
            if idx < train_num:
                train_img_list.append(image_set_list[item])
            else:
                eval_img_list.append(image_set_list[item])
        new_train_img_path = os.path.join(data_path, "tmp/train/Images")
        new_train_ann_path = os.path.join(data_path, "tmp/train/Annotations")
        new_eval_img_path = os.path.join(data_path, "tmp/eval/Images")
        new_eval_ann_path = os.path.join(data_path, "tmp/eval/Annotations")
        mox.file.make_dirs(new_train_img_path)
        mox.file.make_dirs(new_train_ann_path)
        mox.file.make_dirs(new_eval_img_path)
        mox.file.make_dirs(new_eval_ann_path)
        for img_name in train_img_list:
            ann_name = img_name.split('.')[0] + '.xml'
            mox.file.copy(os.path.join(img_path, img_name),
                          os.path.join(new_train_img_path, img_name))
            mox.file.copy(os.path.join(ann_path, ann_name),
                          os.path.join(new_train_ann_path, ann_name))
        for img_name in eval_img_list:
            ann_name = img_name.split('.')[0] + '.xml'
            mox.file.copy(os.path.join(img_path, img_name),
                          os.path.join(new_eval_img_path, img_name))
            mox.file.copy(os.path.join(ann_path, ann_name),
                          os.path.join(new_eval_ann_path, ann_name))
        mox.set_hyper_parameter('data_url', os.path.join(data_path, "tmp"))

def remove_tmp_dirs():
    data_path = mox.get_hyper_parameter('data_url')
    cache_path = os.path.join(data_path, 'cache')
    if mox.file.exists(cache_path):
        mox.file.remove(cache_path, recursive=True)
    if "tmp" in data_path:
        mox.file.remove(data_path, recursive=True)

def train_faster_rcnn():
    args = add_parameter()
    check_data_url()
    mox.file.set_auth(path_style=True)
    set_config(args)
    num_gpus = mox.get_hyper_parameter('num_gpus')
    ctx = [mx.cpu()] if num_gpus is None or num_gpus == 0 else [mx.gpu(int(i)) for i in range(num_gpus)]
    num_classes = mox.rcnn_config.config.NUM_CLASSES
    # training
    symbol = mox.get_model('object_detection', args.network, num_classes=num_classes)

    data_set = get_train_data_iter(sym=symbol, ctx=ctx, args=args)
    num_examples = len(data_set.roidb)
    batch_size = data_set.batch_size
    initializer = mx.init.Xavier(factor_type="in", magnitude=2.34)
    metrics = [mox.contrib_metrics.RPNAccMetric(),
               mox.contrib_metrics.RPNLogLossMetric(),
               mox.contrib_metrics.RPNL1LossMetric(),
               mox.contrib_metrics.RCNNAccMetric(),
               mox.contrib_metrics.RCNNLogLossMetric(),
               mox.contrib_metrics.RCNNL1LossMetric(),
               mx.metric.CompositeEvalMetric()]

    if 'rfcn' in args.network:
        means = np.tile(np.array(mox.rcnn_config.config.TRAIN.BBOX_MEANS), num_classes * 7 * 7)
        stds = np.tile(np.array(mox.rcnn_config.config.TRAIN.BBOX_STDS), num_classes * 7 * 7)
    else:
        means = np.tile(np.array(mox.rcnn_config.config.TRAIN.BBOX_MEANS), num_classes)
        stds = np.tile(np.array(mox.rcnn_config.config.TRAIN.BBOX_STDS), num_classes)

    if args.train_url is not None and len(args.train_url):
        worker_id = mox.get_hyper_parameter('worker_id')
        save_path = args.train_url if worker_id == 0 else "%s-%d" % (args.train_url, worker_id)
        epoch_end_callbacks = mox.rcnn_do_checkpoint(save_path, means, stds)
    else:
        epoch_end_callbacks = None

    params_tuple = mox.rcnn_load_param(
        prefix=mox.rcnn_config.default.pretrained,
        epoch=mox.rcnn_config.default.pretrained_epoch,
        convert=True, data=data_set, sym=symbol, is_train=True)

    lr = mox.rcnn_config.default.base_lr
    lr_factor = 0.1
    lr_iters = [int(epoch * num_examples / batch_size)
                for epoch in [int(i) for i in mox.rcnn_config.default.e2e_lr_step.split(',')]]
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(lr_iters, lr_factor)
    optimizer_params = {'momentum': args.mom,
                        'wd': args.wd,
                        'learning_rate': lr,
                        'lr_scheduler': lr_scheduler,
                        'rescale_grad': (1.0 / batch_size),
                        'clip_gradient': 5}

    mox.run(data_set=(data_set, None),
            optimizer=args.optimizer,
            optimizer_params=optimizer_params,
            run_mode=mox.ModeKeys.TRAIN,
            model=get_model(data_set, symbol, ctx),
            epoch_end_callbacks=epoch_end_callbacks,
            initializer=initializer,
            batch_size=batch_size,
            params_tuple=params_tuple,
            metrics=metrics,
            num_epoch=mox.rcnn_config.default.e2e_epoch)
    if args.train_url is not None and len(args.train_url):
        args.load_epoch = mox.rcnn_config.default.e2e_epoch
        e2e_eval(args=args)
        if args.export_model == 1:
            end_epoch = mox.rcnn_config.default.e2e_epoch
            params_path = '%s-%04d.params' % (save_path, end_epoch)
            json_path = ('%s-symbol.json' % save_path)
            logging.info(params_path + 'used to predict')
            pred_params_path = os.path.join(args.train_url, 'model', 'pred_model-0000.params')
            pred_json_path = os.path.join(args.train_url, 'model', 'pred_model-symbol.json')
            mox.file.copy(params_path, pred_params_path)
            symbol = mox.get_model('object_detection', 'resnet_rcnn',
                                    num_classes=num_classes, is_train=False)
            symbol.save(pred_json_path)
            for i in range(1, end_epoch + 1, 1):
                mox.file.remove('%s-%04d.params' % (save_path, i))
            mox.file.remove(json_path)
            mox.file.remove(os.path.join(args.train_url, 'metric.json'))

    remove_tmp_dirs()

if __name__ == '__main__':
    train_faster_rcnn()
