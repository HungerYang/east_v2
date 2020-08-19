import zipfile
import glob
import os
gts_to_zip = False #True

det_to_zip = True #False #True

if gts_to_zip:
    res_path = '/home/xhyang/datasets/ocr_test_dataset/'
    dirlist = glob.glob(os.path.join(res_path, '*data'))
    for dirname in dirlist:
        outname = os.path.join('./', 'gt_'+os.path.basename(dirname)+'.zip')

        with zipfile.ZipFile(outname, mode='w', allowZip64=True) as zipf:
            txtlist = glob.glob(os.path.join(dirname, '*.txt'))
            for txtname in txtlist:
                zipf.write(txtname, os.path.basename(txtname))

if det_to_zip:
    #res_path = '../result/res101k/'
    res_path = '../result/tmp6/'
    dirlist = glob.glob( os.path.join(res_path, '*data'))
    
    for dirname in dirlist:
        outname = os.path.join('./dets/', os.path.basename(dirname)+'.zip')
        with zipfile.ZipFile(outname, mode='w', allowZip64=True) as zipf:
            txtlist = glob.glob(os.path.join(dirname, '*.txt'))
            for txtname in txtlist:
                zipf.write(txtname, os.path.basename(txtname))

param = {
    'g':'', #'./gts/gt_icpr_test_data.zip', 
    's':'', #'./dets/icpr_test_data.zip', 
    'o':'', #'./result/icpr_test_data/',
    }

import rrc_evaluation_funcs
from script import default_evaluation_params, validate_data, evaluate_method

det_path = './dets/'
gts_path = './gts/'
res_path = './result/'
detlist = glob.glob(os.path.join(det_path, '*.zip'))
for dname in detlist:
    gtname = os.path.join(gts_path, 'gt_'+os.path.basename(dname))
    resp = os.path.join(res_path, os.path.basename(dname))
    if not os.path.exists(resp):
        os.mkdir(resp)
    resname = os.path.join(resp, os.path.basename(dname))
    
    param['g'] = gtname
    param['s'] = dname
    param['o'] = resname
    
    print('Evaluating: ', os.path.basename(dname))
    rrc_evaluation_funcs.main_evaluation(param, default_evaluation_params, validate_data, evaluate_method)
    print('\n') 

