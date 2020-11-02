from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pandas as pd
import pickle
import os
import tensorflow as tf

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils, plot_model
from keras_frcnn.simple_parser import get_data
from keras_frcnn import resnet as nn
from keras.callbacks import TensorBoard

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="train_path", help="Path to training data.")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois", help="Number of RoIs to process at once.", default=32)
parser.add_option("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=false).", action="store_true", default=True)
parser.add_option("--vf", dest="vertical_flips", help="Augment with vertical flips in training. (Default=false).", action="store_true", default=True)
parser.add_option("--rot", "--rot_90", dest="rot_90", help="Augment with 90 degree rotations in training. (Default=false).", action="store_true", default=True)
parser.add_option("--num_epochs", type="int", dest="num_epochs", help="Number of epochs.", default=40)
parser.add_option("--config_filename", dest="config_filename", help=
				"Location to store all the metadata related to the training (to be used when testing).",
				default="config.pickle")


(options, args) = parser.parse_args()

C = config.Config()

train_path = options.train_path
output_weight_path = C.model_path
C.num_rois = int(options.num_rois)
config_output_filename = options.config_filename
C.use_horizontal_flips = bool(options.horizontal_flips)
C.use_vertical_flips = bool(options.vertical_flips)
C.rot_90 = bool(options.rot_90)
record_path = 'record.csv'

# Path for pre-trained weights
base_net_weights = C.pretrained_weights

train_imgs, classes_count, class_mapping = get_data(train_path)

if 'bg' not in classes_count:
	classes_count['bg'] = 0
	class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping

print('Training images per class:')
pprint.pprint(classes_count)
print('\nNum classes (including bg) = {}'.format(len(classes_count)))
print(class_mapping)

with open(config_output_filename, 'wb') as config_f:
	pickle.dump(C,config_f)
	print(f'\nConfig has been written to {config_output_filename}, and can be loaded when testing to ensure correct results\n')

random.seed(1)
random.shuffle(train_imgs)

print('Num training samples: {}\n'.format(len(train_imgs)))

data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length, mode='train')


input_shape_img = (None, None, 3)
img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)

model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)

# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

tbcallback = TensorBoard(log_dir = 'logs', histogram_freq = 1, write_graph = True, write_images = False)
tbcallback.set_model(model_all)

if not os.path.isfile(output_weight_path):
    try:
        print('loading weights from {}'.format(base_net_weights))
        model_rpn.load_weights(base_net_weights, by_name=True)
        model_classifier.load_weights(base_net_weights, by_name=True)
    except:
        print('Could not load pretrained model weights. Weights can be found in the keras application folder \
        https://github.com/fchollet/deep-learning-models/releases/tag/v0.2')
    
    record_df = pd.DataFrame(columns=['mean_overlapping_bboxes', 'class_acc', 'loss_rpn_cls', 'loss_rpn_regr', 'loss_class_cls', 'loss_class_regr', 'curr_loss', 'elapsed_time'])

else:
    print('loading weights from {}'.format(output_weight_path))
    model_rpn.load_weights(output_weight_path, by_name=True)
    model_classifier.load_weights(output_weight_path, by_name=True)
    
    record_df = pd.read_csv(record_path)
    
    r_mean_overlapping_bboxes = record_df['mean_overlapping_bboxes']
    r_class_acc = record_df['class_acc']
    r_loss_rpn_cls = record_df['loss_rpn_cls']
    r_loss_rpn_regr = record_df['loss_rpn_regr']
    r_loss_class_cls = record_df['loss_class_cls']
    r_loss_class_regr = record_df['loss_class_regr']
    r_curr_loss = record_df['curr_loss']
    r_elapsed_time = record_df['elapsed_time']
    
    print('\nAlready trained {} epochs'.format(len(record_df)))



optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')

# Saving model for visulaization
plot_model(model = model_all, to_file = 'model_all.png', show_shapes = True, show_layer_names = True)

total_epochs = len(record_df)
r_epochs = len(record_df)
epoch_length = len(train_imgs)
num_epochs = int(options.num_epochs)
iter_num = 0

total_epochs += num_epochs

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []


if len(record_df) == 0:
    best_loss = np.Inf
else:
    best_loss = np.min(r_curr_loss)

sel_samp = []

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

train_names = ['train_loss_rpn_cls', 'train_loss_rpn_regr', 'train_loss_class_cls', 'train_loss_class_regr','train_total_loss', 'train_acc']

start_time = time.time()

for epoch_num in range(num_epochs):
    progbar = generic_utils.Progbar(epoch_length)
    print('\nEpoch {}/{}'.format(r_epochs + 1, total_epochs))
    
    r_epochs += 1
    while True:
        try:
            if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
                mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
                rpn_accuracy_rpn_monitor = []
                #print('\nAverage number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
                
                if mean_overlapping_bboxes == 0:
                    print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')
                    
            X, Y, img_data = next(data_gen_train)
            
            loss_rpn = model_rpn.train_on_batch(X, Y)
            
            P_rpn = model_rpn.predict_on_batch(X)
            
            R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, use_regr=True, overlap_thresh=0.7, max_boxes=300)
            # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
            X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)
            
            if X2 is None:
                rpn_accuracy_rpn_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                continue
                
            neg_samples = np.where(Y1[0, :, -1] == 1)
            pos_samples = np.where(Y1[0, :, -1] == 0)
            
            if len(neg_samples) > 0:
                neg_samples = neg_samples[0]
            else:
                neg_samples = []
                
            if len(pos_samples) > 0:
                pos_samples = pos_samples[0]
            else:
                pos_samples = []
                
            rpn_accuracy_rpn_monitor.append(len(pos_samples))
            rpn_accuracy_for_epoch.append((len(pos_samples)))
            
            if C.num_rois > 1:
                if len(pos_samples) < C.num_rois//2:
                    selected_pos_samples = pos_samples.tolist()
                else:
                    selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()
                try:
                    selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
                except:
                    selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()
                
                sel_samples = selected_pos_samples + selected_neg_samples
            else:
                # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                selected_pos_samples = pos_samples.tolist()
                selected_neg_samples = neg_samples.tolist()
                if np.random.randint(0, 2):
                    sel_samples = random.choice(neg_samples)
                else:
                    sel_samples = random.choice(pos_samples)
            
            
            sel_samp.append(len(selected_pos_samples))

            loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])
            
            losses[iter_num, 0] = loss_rpn[1]
            losses[iter_num, 1] = loss_rpn[2]
            
            losses[iter_num, 2] = loss_class[1]
            losses[iter_num, 3] = loss_class[2]
            losses[iter_num, 4] = loss_class[3]
            
            iter_num += 1
            
            progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                                        ('detector_cls', np.mean(losses[:iter_num, 2])), ('detector_regr', np.mean(losses[:iter_num, 3]))])
            
            if iter_num == epoch_length:
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_regr = np.mean(losses[:, 1])
                loss_class_cls = np.mean(losses[:, 2])
                loss_class_regr = np.mean(losses[:, 3])
                class_acc = np.mean(losses[:, 4])
                total_samp = np.mean(sel_samp)
                
                mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                rpn_accuracy_for_epoch = []
                
                if C.verbose:
                    print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
                    print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                    print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                    print('Loss RPN regression: {}'.format(loss_rpn_regr))
                    print('Loss Detector classifier: {}'.format(loss_class_cls))
                    print('Loss Detector regression: {}'.format(loss_class_regr))
                    print('Elapsed time: {}'.format(time.time() - start_time))
                    print('Total samples: {}'.format(total_samp))
                    elapsed_time = (time.time()-start_time)/60
                    
                curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                iter_num = 0
                start_time = time.time()
                write_log(tbcallback, train_names, [loss_rpn_cls, loss_rpn_regr, loss_class_cls, loss_class_regr, curr_loss, class_acc], epoch_num)
                
                if curr_loss < best_loss:
                    if C.verbose:
                        print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
                    best_loss = curr_loss
                    model_all.save_weights(output_weight_path)
                    
                new_row = {'mean_overlapping_bboxes':round(mean_overlapping_bboxes, 3),
                            'class_acc':round(class_acc, 3),
                            'loss_rpn_cls':round(loss_rpn_cls, 3),
                            'loss_rpn_regr':round(loss_rpn_regr, 3),
                            'loss_class_cls':round(loss_class_cls, 3),
                            'loss_class_regr':round(loss_class_regr, 3),
                            'curr_loss':round(curr_loss, 3),
                            'elapsed_time':round(elapsed_time, 3)}
                
                record_df = record_df.append(new_row, ignore_index=True)
                record_df.to_csv('record.csv', index=0)
                
                break
                
        except Exception as e:
            print('Exception: {}'.format(e))
            print('error{}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__,)
            continue    

print('Training complete, exiting.')