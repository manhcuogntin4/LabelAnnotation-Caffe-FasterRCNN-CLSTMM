import os
import cv2
import time
import cPickle
import datetime
import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import pandas as pd
from PIL import Image
import cStringIO as StringIO
import urllib
import exifutil
from tools.axademo import detect_cni

import caffe

#CAFFE_ROOT = '/home/haoming/Documents/caffe'
CAFFE_ROOT ='/home/cuong-nguyen/2016/Workspace/brexia/Septembre/Codesource/caffe-master'
REPO_DIR = os.path.abspath(os.path.dirname(__file__))
MODELE_DIR = os.path.join(REPO_DIR, 'models/googlenet')
DATA_DIR = os.path.join(REPO_DIR, 'data/googlenet')
UPLOAD_FOLDER = '/tmp/caffe_demos_uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])

# Obtain the flask app object
app = flask.Flask(__name__)


@app.route('/')
def index():
    return flask.render_template('index.html', has_result=False)


@app.route('/classify_url', methods=['GET'])
def classify_url():
    imageurl = flask.request.args.get('imageurl', '')
    try:
        string_buffer = StringIO.StringIO(
            urllib.urlopen(imageurl).read())
        image = caffe.io.load_image(string_buffer)

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open image from URL.')
        )

    logging.info('Image: %s', imageurl)
    result = app.clf.classify_image(image)
    return flask.render_template(
        'index.html', has_result=True, result=result, imagesrc=imageurl,
        is_cni=False)


def load_image(filename, color=True):
    img = cv2.imread(filename).astype(np.float32) / 255.
    img = img[...,::-1]
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
        logging.info('Saving to %s.', filename)
        # image = exifutil.open_oriented_im(filename)
        image = load_image(filename)

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )

    result = app.clf.classify_image(image)
    proba_list = result[2]  # array of (cls_name, probability)
    is_cni = True # proba_list[0][0] == 'cni'

    ptime = 0.0
    images = []
    images.append(image) # for classification display
    bboxes, texts = [], []
    if is_cni:
        logging.info('Extracting Region of Interest in CNI...')
        cnis, preproc_time = detect_cni(filename) # img, res
        ptime += preproc_time
        for img, res, pt in cnis:
            ptime += pt
            images.append(img)
            bbox, text_info = [], {}
            for cls in res:
                # print cls, res[cls]
                bbox.append(res[cls][0])
                text_info[cls] = (res[cls][1], '%.3f' % (res[cls][2]))   # (content, prob)
            bboxes.append(bbox)
            texts.append(text_info)
    else:
        images.append(image) ## for future use, just to avoid bug now

    return flask.render_template(
        'index.html', has_result=True, result=result, doc_info=texts, proc_time='%.3f' % (ptime),
        imagesrc=embed_image_html(images, bboxes), is_cni=is_cni
    )

def embed_image_html(images, bboxes):
    """Creates an image embedded in HTML base64 format."""
    embed_images = []
    for im_ind, image in enumerate(images):        
        if im_ind == 0:
            image_pil = Image.fromarray((255 * image).astype('uint8'))
            image_pil = image_pil.resize((256, 256))
            string_buf = StringIO.StringIO()
            image_pil.save(string_buf, format='png')
            data = string_buf.getvalue().encode('base64').replace('\n', '')
            embed_images.append('data:image/png;base64,' + data)
        else:
            for bbox in bboxes[im_ind - 1]:
                img = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                image_pil = Image.fromarray(img.astype('uint8'))
                string_buf = StringIO.StringIO()
                image_pil.save(string_buf, format='png')
                data = string_buf.getvalue().encode('base64').replace('\n', '')
                embed_images.append('data:image/png;base64,' + data)
        
    return embed_images


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
    )


class ImagenetClassifier(object):
    default_args = {
        'model_def_file': (
            '{}/deploy.prototxt'.format(MODELE_DIR)),
        'pretrained_model_file': (
            '{}/train_val.caffemodel'.format(DATA_DIR)),
        'mean_file': (
            '{}/python/caffe/imagenet/ilsvrc_2012_mean.npy'.format(CAFFE_ROOT)),
        'class_labels_file': (
            '{}/synset_words.txt'.format(MODELE_DIR)),
        'bet_file': (
            '{}/data/ilsvrc12/imagenet.bet.pickle'.format(CAFFE_ROOT)),
    }
    for key, val in default_args.iteritems():
        if not os.path.exists(val):
            raise Exception(
                "File for {} is missing. Should be at: {}".format(key, val))
    default_args['image_dim'] = 256
    default_args['raw_scale'] = 255.

    def __init__(self, model_def_file, pretrained_model_file, mean_file,
                 raw_scale, class_labels_file, bet_file, image_dim, gpu_mode):
        logging.info('Loading net and associated files...')
        if gpu_mode:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        net = caffe.Net(model_def_file, pretrained_model_file, caffe.TEST)
        self.net = net
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        # transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2,1,0)) # if using RGB instead of BGR
        self.transformer = transformer

        with open(class_labels_file) as f:
            labels_df = pd.DataFrame([
                {
                    'synset_id': l.strip().split(' ')[0],
                    'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
                }
                for l in f.readlines()
            ])
        self.labels = labels_df.sort('synset_id')['name'].values

    def classify_image(self, image):
        try:
            net = self.net
            net.blobs['data'].data[...] = self.transformer.preprocess('data', image)

            starttime = time.time()
            # scores = self.net.predict([image], oversample=True).flatten()
            out = net.forward()
            proba = out['prob'][0]
            scores = net.blobs['fc8'].data[0]
            endtime = time.time()

            indices = (-proba).argsort()[:3]
            predictions = self.labels[indices]

            # In addition to the prediction text, we will also produce
            # the length for the progress bar visualization.
            meta_proba = [
                (p, '%.5f' % proba[i])
                for i, p in zip(indices, predictions)
            ]

            score = [
                (p, '%.5f' % scores[i])
                for i, p in zip(indices, predictions)
            ]
            logging.info('result: %s', str(meta_proba))

            return (True, score, meta_proba, '%.3f' % (endtime - starttime))

        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')


def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)
    parser.add_option(
        '-g', '--gpu',
        help="use gpu mode",
        action='store_true', default=False)

    opts, args = parser.parse_args()
    ImagenetClassifier.default_args.update({'gpu_mode': opts.gpu})

    # Initialize classifier + warm start by forward for allocation
    app.clf = ImagenetClassifier(**ImagenetClassifier.default_args)
    app.clf.net.forward()

    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    start_from_terminal(app)
