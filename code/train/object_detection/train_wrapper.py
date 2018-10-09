import os
import subprocess
import sys
import moxing as mox

script_dir = '/home/work'
cache_dir = os.environ['DLS_LOCAL_CACHE_PATH']
#AWS_ACCESS_KEY_ID = os.environ['AWS_ACCESS_KEY_ID']
#AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY']
USER_AWS_ACCESS_KEY_ID = os.environ['USER_AWS_ACCESS_KEY_ID']
USER_AWS_SECRET_ACCESS_KEY = os.environ['USER_AWS_SECRET_ACCESS_KEY']
USER_S3_ACCESS_KEY_ID = os.environ['USER_S3_ACCESS_KEY_ID']
USER_S3_SECRET_ACCESS_KEY = os.environ['USER_S3_SECRET_ACCESS_KEY']

local_checkpoint_url = os.path.join(cache_dir, 'checkpoint_url')
if not os.path.exists(local_checkpoint_url):
    os.mkdir(local_checkpoint_url)

args = sys.argv[1:]
concat_args = ''
model_name = ''
built_in_tf_cnn_model_names = ['resnet_v1_50', 'vgg_16']
built_in_tf_object_detection_model_names = ['faster_rcnn_resnet_v1_50']
built_in_mx_cnn_model_names = ['resnet_v2_50']
built_in_mx_object_detection_model_names = ['faster_rcnn_resnet_v2_50']
built_in_mx_segmentation_model_names = ['segnet_vgg_bn_16']

ckpt_url = ""
for arg in args:
    if arg.strip().startswith('--checkpoint_url'):
        obs_checkpoint_url = arg[17:]
        # concat_args = concat_args + ' --checkpoint_url=' + local_checkpoint_url
        ckpt = local_checkpoint_url + '/'
    elif arg.strip().startswith('--model_name'):
        model_name = arg[13:]
        assert (model_name in built_in_tf_cnn_model_names or model_name in built_in_tf_object_detection_model_names \
                or model_name in built_in_mx_cnn_model_names or model_name in built_in_mx_object_detection_model_names \
                or model_name in built_in_mx_segmentation_model_names)
        # concat_args = concat_args + ' --model_name=' + model_name
        ckpt =  ckpt + model_name     
    else:
        concat_args = concat_args + ' ' + arg

# Step 1: download the pretrained model files from EiWizard OBS to local file system of the container
'''
returnCode = subprocess.call(
    '`which python` '+ os.path.join(script_dir, "dls-downloader.py") +' -r -s ' + obs_checkpoint_url + ' -d ' + local_checkpoint_url,
    shell=True)
if returnCode != 0:
    os._exit(returnCode)
'''
mox.file.copy_parallel(obs_checkpoint_url, local_checkpoint_url)
# Step 2: change AK/SK to user's AK/SK
os.environ['AWS_ACCESS_KEY_ID'] = USER_AWS_ACCESS_KEY_ID
os.environ['AWS_SECRET_ACCESS_KEY'] = USER_AWS_SECRET_ACCESS_KEY
os.environ['S3_ACCESS_KEY_ID'] = USER_S3_ACCESS_KEY_ID
os.environ['S3_SECRET_ACCESS_KEY'] = USER_S3_SECRET_ACCESS_KEY
# Step 3: start training
current_dir = os.path.dirname(os.path.realpath("__file__"))
cnn_code_dir_name = 'cnn'
object_detection_dir_name = 'object_detection'
segmentation_dir_name = 'image_segmentation'
print('obs_checkpoint_url: %s' % obs_checkpoint_url)
if obs_checkpoint_url.endswith('/'):
    obs_checkpoint_url = obs_checkpoint_url[:-1]
if not model_name:
    model_name = obs_checkpoint_url[obs_checkpoint_url.rfind('/')+1:]
    arg = '--model_name=' + model_name
    ckpt = ckpt + model_name
    #concat_args = concat_args + ' ' + arg
concat_args = concat_args + ' --checkpoint_url=' + ckpt

code_dir_name = ''
print('model_name: %s ' % model_name)
if model_name in built_in_tf_cnn_model_names or model_name in built_in_mx_cnn_model_names:
    code_dir_name = cnn_code_dir_name
elif model_name in built_in_tf_object_detection_model_names or model_name in built_in_mx_object_detection_model_names:
    code_dir_name = object_detection_dir_name
elif model_name in built_in_mx_segmentation_model_names:
    code_dir_name = segmentation_dir_name
else:
    raise ValueError('no such built-in model.')
print('code_dir_name : %s' % code_dir_name)

print('cmd : %s', "`which python` " + os.path.join(current_dir, code_dir_name, 'train.py') + concat_args)

returnCode = subprocess.call("`which python` " + os.path.join(current_dir, code_dir_name, 'train.py') + concat_args, shell=True)
if returnCode != 0:
    os._exit(returnCode)
