import numpy as np
import datetime
import time
from collections import defaultdict
# from . import mask as maskUtils
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import scipy.io as sio

# font = {'family' : 'Tahoma',        
#         'size'   : 22}  
# matplotlib.rc('font', **font)

font = {'size'   : 22}  
matplotlib.rc('font', **font)

import os, pdb

class COCOeval:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt   = cocoGt              # ground truth COCO API
        self.cocoDt   = cocoDt              # detections COCO API
        self.params   = {}                  # evaluation parameters
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = Params(iouType=iouType) # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())


    def _prepare(self, id_setup):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        p = self.params
        if p.useCats:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            # gt['ignore'] = 1 if (gt['height'] < self.params.HtRng[id_setup][0] or gt['height'] > self.params.HtRng[id_setup][1]) or \
            #    ( gt['vis_ratio'] < self.params.VisRng[id_setup][0] or gt['vis_ratio'] > self.params.VisRng[id_setup][1]) else gt['ignore']
            gbox = gt['bbox']
            gt['ignore'] = 1 if (gt['height'] < self.params.HtRng[id_setup][0] or gt['height'] > self.params.HtRng[id_setup][1] or \
               gt['occlusion'] not in self.params.OccRng[id_setup] or \
               gbox[0] < self.params.bndRng[0] or gbox[1] < self.params.bndRng[1] or \
               gbox[0]+gbox[2] > self.params.bndRng[2] or gbox[1]+gbox[3] > self.params.bndRng[3])  else gt['ignore']
        
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results

    def evaluate(self, id_setup):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params=p

        self._prepare(id_setup)
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        computeIoU = self.computeIoU

        self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                        for imgId in p.imgIds
                        for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        HtRng = self.params.HtRng[id_setup]
        # VisRng = self.params.VisRng[id_setup]
        OccRng = self.params.OccRng[id_setup]
        # self.evalImgs = [evaluateImg(imgId, catId, HtRng, VisRng, maxDet)
        #          for catId in catIds
        #          for imgId in p.imgIds
        #      ]
        self.evalImgs = [evaluateImg(imgId, catId, HtRng, OccRng, maxDet)
                 for catId in catIds
                 for imgId in p.imgIds
             ]
        
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))


    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]


        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')


        # compute iou between each dt and gt region
        iscrowd = [int(o['ignore']) for o in gt]
        ious = self.iou(d,g,iscrowd)
        return ious

    def iou( self, dts, gts, pyiscrowd ):
        dts = np.asarray(dts)
        gts = np.asarray(gts)
        pyiscrowd = np.asarray(pyiscrowd)
        ious = np.zeros((len(dts), len(gts)))
        for j, gt in enumerate(gts):
            gx1 = gt[0]
            gy1 = gt[1]
            gx2 = gt[0] + gt[2]
            gy2 = gt[1] + gt[3]
            garea = gt[2] * gt[3]
            for i, dt in enumerate(dts):
                dx1 = dt[0]
                dy1 = dt[1]
                dx2 = dt[0] + dt[2]
                dy2 = dt[1] + dt[3]
                darea = dt[2] * dt[3]

                unionw = min(dx2,gx2)-max(dx1,gx1)
                if unionw <= 0:
                    continue
                unionh = min(dy2,gy2)-max(dy1,gy1)
                if unionh <= 0:
                    continue
                t = unionw * unionh
                if pyiscrowd[j]:
                    unionarea = darea
                else:
                    unionarea = darea + garea - t

                ious[i, j] = float(t)/unionarea
        return ious



    def evaluateImg(self, imgId, catId, hRng, oRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        try:
            p = self.params
            if p.useCats:
                gt = self._gts[imgId,catId]
                dt = self._dts[imgId,catId]
            else:
                gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
                dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
            if len(gt) == 0 and len(dt) ==0:
                return None

            for g in gt:
                if g['ignore']:
                    g['_ignore'] = 1
                else:
                    g['_ignore'] = 0
            # sort dt highest score first, sort gt ignore last
            gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
            gt = [gt[i] for i in gtind]
            dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
            dt = [dt[i] for i in dtind[0:maxDet]]
            # exclude dt out of height range
            dt = [d for d in dt if d['height'] >= hRng[0] / self.params.expFilter and d['height'] < hRng[1] * self.params.expFilter]
            dtind = np.array([int(d['id'] - dt[0]['id']) for d in dt])


            if len(dt) == 0:
                return None

            # load computed ious        
            ious = self.ious[imgId, catId][dtind, :] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]
            # ious = self.ious[imgId, catId][dtind, :] if self.ious[imgId, catId].shape[1] > 0 else self.ious[imgId, catId]
            ious = ious[:,gtind]

            T = len(p.iouThrs)
            G = len(gt)
            D = len(dt)
            gtm  = np.zeros((T,G))
            dtm  = np.zeros((T,D))
            gtIg = np.array([g['_ignore'] for g in gt])
            dtIg = np.zeros((T,D))
            if not len(ious)==0:
                for tind, t in enumerate(p.iouThrs):
                    for dind, d in enumerate(dt):
                        # information about best match so far (m=-1 -> unmatched)
                        iou = min([t,1-1e-10])
                        bstOa = iou
                        bstg = -2
                        bstm = -2
                        for gind, g in enumerate(gt):
                            m = gtm[tind,gind]
                            # if this gt already matched, and not a crowd, continue
                            if m>0:
                                continue
                            # if dt matched to reg gt, and on ignore gt, stop
                            if bstm!=-2 and gtIg[gind] == 1:
                                break
                            # continue to next gt unless better match made
                            if ious[dind,gind] < bstOa:
                                continue
                            # if match successful and best so far, store appropriately
                            bstOa=ious[dind,gind]
                            bstg = gind
                            if gtIg[gind] == 0:
                                bstm = 1
                            else:
                                bstm = -1

                        # if match made store id of match for both dt and gt
                        if bstg ==-2:
                            continue
                        dtIg[tind,dind] = gtIg[bstg]
                        dtm[tind,dind]  = gt[bstg]['id']
                        if bstm == 1:
                            gtm[tind,bstg]     = d['id']
        except Exception as ex:
            import traceback, sys

            ex_type, ex_value, ex_traceback = sys.exc_info()            

            # Extract unformatter stack traces as tuples
            trace_back = traceback.extract_tb(ex_traceback)

            # Format stacktrace
            stack_trace = list()

            for trace in trace_back:
                stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))

            sys.stderr.write("[Error] Exception type : %s \n" % ex_type.__name__)
            sys.stderr.write("[Error] Exception message : %s \n" %ex_value)
            for trace in stack_trace:
                sys.stderr.write("[Error] (Stack trace) %s\n" % trace)

            
        


        # store results for given image and category
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'hRng':         hRng,
                'oRng':         oRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }

    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)
        R           = len(p.fppiThrs)
        K           = len(p.catIds) if p.useCats else 1
        M           = len(p.maxDets)
        ys   = -np.ones((T,R,K,M)) # -1 for the precision of absent categories

        xx_graph = []
        yy_graph = []

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = [1] #_pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]

        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds) # n

        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0*I0
            for m, maxDet in enumerate(m_list):
                E = [self.evalImgs[Nk + i] for i in i_list]  # len(self.evalImgs): 2252(GT), len(i_list): 2252 (num of test images)
                # len(E): 2252
                E = [e for e in E if not e is None] # len(E): 1127
                if len(E) == 0:
                    continue

                # single image can have multiple dtboxes therefore multiple gtScores

                dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E]) # len(dtScores) : 3046 same with E[i]['dtScores']

                # different sorting method generates slightly different results.
                # mergesort is used to be consistent as Matlab implementation.

                inds = np.argsort(-dtScores, kind='mergesort') # sorted in desc order

                dtm = np.concatenate([e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:, inds] # len(dtm[0]) : 3046 / organized in dtScore order
                dtIg = np.concatenate([e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:, inds] # len(dtIg[0]): 3046
                gtIg = np.concatenate([e['gtIgnore'] for e in E]) # len(gtIg): 3658 
             
                npig = np.count_nonzero(gtIg == 0)
                if npig == 0:
                    continue
                tps = np.logical_and(dtm, np.logical_not(dtIg))
                fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg)) # detected boxes which didn't match with gt boxes and is not intended to ignore
                
                inds = np.where(dtIg==0)[1]
                tps = tps[:,inds]
                fps = fps[:,inds] # len(fps[0]): 1886 is dtIg is 0 suppress

                tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float64)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float64) # calculate accumulated sum in each row

                print('There are {}tps out of {} gt boxes'.format(tp_sum[0][-1], npig))
                quit()

            
                for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)): # fp_sum => a1 + a2 + ... + an
                    tp = np.array(tp) #len(tp): 1886
                    fppi = np.array(fp)/I0 # how many images to consider
                    nd = len(tp)
                    recall = tp / npig # npig: number of gt boxes (not ignored)
                    q = np.zeros((R,))

                    
                    
                    xx_graph.append( fppi )
                    yy_graph.append( 1-recall )

                    # numpy is slow without cython optimization for accessing elements
                    # use python array gets significant speed improvement
                    recall = recall.tolist()
                    q = q.tolist()

                    for i in range(nd - 1, 0, -1):
                        if recall[i] < recall[i - 1]:
                            recall[i - 1] = recall[i]

                    inds = np.searchsorted(fppi, p.fppiThrs, side='right') - 1 # number of elements in inds: R (9)
                    try:
                        for ri, pi in enumerate(inds):
                            q[ri] = recall[pi]
                    except:
                        pass
                    ys[t,:,k,m] = np.array(q)
        
        self.eval = {
            'params': p,
            'counts': [T, R, K, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'TP':   ys,
            'xx': xx_graph,
            'yy': yy_graph
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))


    def draw_figure(self, ax, filename='result.jpg'):
        ### Draw

        mrs = 1 - self.eval['TP']
        mean_s = np.log(mrs[mrs<2])
        mean_s = np.mean(mean_s)
        mean_s = float( np.exp(mean_s) * 100 )

        xx = self.eval['xx']
        yy = self.eval['yy']
        
        # plt.clf()
        # fig, ax = plt.subplots(figsize=(12,9))

        ax.cla()

        
        ax.plot( xx[0], yy[0], linewidth=3, label='{:.2f}%, {:s}'.format(mean_s, os.path.basename(filename)) )
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend()

        yt = [1,5] + list(range(10,60,10)) + [64, 80]
        yticklabels=[ '.{:02d}'.format(num) for num in yt ]

        yt += [100]
        yt = [ yy/100.0 for yy in yt ]
        yticklabels += [1]
        
        ax.set_yticks(yt)
        ax.set_yticklabels(yticklabels)
        plt.grid(which='major', axis='both')
        plt.ylim(0.01, 1)
        plt.xlim(2e-4, 50)
        plt.ylabel('miss rate')
        plt.xlabel('false positives per image')
        plt.savefig(filename)


    def summarize(self,id_setup, res_file=None):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize(iouThr=None, maxDets=100 ):
            OCC_TO_TEXT = ['none', 'partial_occ', 'heavy_occ']

            p = self.params
            iStr = ' {:<18} {} @ {:<18} [ IoU={:<9} | height={:>6s} | visibility={:>6s} ] = {:0.2f}%'
            titleStr = 'Average Miss Rate'
            typeStr = '(MR)'
            setupStr = p.SetupLbl[id_setup]
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)
            heightStr = '[{:0.0f}:{:0.0f}]'.format(p.HtRng[id_setup][0], p.HtRng[id_setup][1])
            # occlStr = '[{:0.2f}:{:0.2f}]'.format(p.VisRng[id_setup][0], p.VisRng[id_setup][1])
            occlStr = '[' + '+'.join([ '{:s}'.format(OCC_TO_TEXT[occ]) for occ in p.OccRng[id_setup] ]) + ']'

            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

            # dimension of precision: [TxRxKxAxM]
            s = self.eval['TP']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            mrs = 1-s[:,:,:,mind]

            if len(mrs[mrs<2])==0:
                mean_s = -1
            else:
                mean_s = np.log(mrs[mrs<2])
                mean_s = np.mean(mean_s)
                mean_s = np.exp(mean_s)
            print(iStr.format(titleStr, typeStr,setupStr, iouStr, heightStr, occlStr, mean_s*100))

            if res_file:
                res_file.write(iStr.format(titleStr, typeStr,setupStr, iouStr, heightStr, occlStr, mean_s*100))
                res_file.write('\n')
            return mean_s

        if not self.eval:
            raise Exception('Please run accumulate() first')
        
        return _summarize(iouThr=.5,maxDets=1000)

    def __str__(self):
        self.summarize()

class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value

        self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01).astype(np.int8) + 1, endpoint=True)
        self.fppiThrs = np.array([0.0100,    0.0178,    0.0316,    0.0562,    0.1000,    0.1778,    0.3162,    0.5623,    1.0000])
        self.maxDets = [1000]
        self.expFilter = 1.25
        self.useCats = 1

        self.iouThrs = np.array([0.5])  # np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)

        self.HtRng = [[55, 1e5 ** 2], [50,75], [50, 1e5 ** 2], [20, 1e5 ** 2]]
        # self.VisRng = [[0.65, 1e5 ** 2], [0.65, 1e5 ** 2], [0.2,0.65], [0.2, 1e5 ** 2]]
        self.OccRng = [[0, 1], [0, 1], [2], [0, 1, 2]]
        self.SetupLbl = ['Reasonable', 'Reasonable_small','Reasonable_occ=heavy', 'All']


        self.bndRng = [5, 5, 635, 507];     # discard bbs outside this pixel range


    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None

        
# import numpy as np
# import datetime
# import time
# from collections import defaultdict
# # from . import mask as maskUtils
# import copy
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# import scipy.io as sio

# # font = {'family' : 'Tahoma',        
# #         'size'   : 22}  
# # matplotlib.rc('font', **font)

# font = {'size'   : 22}  
# matplotlib.rc('font', **font)

# import os, pdb

# class COCOeval:
#     # Interface for evaluating detection on the Microsoft COCO dataset.
#     #
#     # The usage for CocoEval is as follows:
#     #  cocoGt=..., cocoDt=...       # load dataset and results
#     #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
#     #  E.params.recThrs = ...;      # set parameters as desired
#     #  E.evaluate();                # run per image evaluation
#     #  E.accumulate();              # accumulate per image results
#     #  E.summarize();               # display summary metrics of results
#     # For example usage see evalDemo.m and http://mscoco.org/.
#     #
#     # The evaluation parameters are as follows (defaults in brackets):
#     #  imgIds     - [all] N img ids to use for evaluation
#     #  catIds     - [all] K cat ids to use for evaluation
#     #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
#     #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
#     #  areaRng    - [...] A=4 object area ranges for evaluation
#     #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
#     #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
#     #  iouType replaced the now DEPRECATED useSegm parameter.
#     #  useCats    - [1] if true use category labels for evaluation
#     # Note: if useCats=0 category labels are ignored as in proposal scoring.
#     # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
#     #
#     # evaluate(): evaluates detections on every image and every category and
#     # concats the results into the "evalImgs" with fields:
#     #  dtIds      - [1xD] id for each of the D detections (dt)
#     #  gtIds      - [1xG] id for each of the G ground truths (gt)
#     #  dtMatches  - [TxD] matching gt id at each IoU or 0
#     #  gtMatches  - [TxG] matching dt id at each IoU or 0
#     #  dtScores   - [1xD] confidence of each dt
#     #  gtIgnore   - [1xG] ignore flag for each gt
#     #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
#     #
#     # accumulate(): accumulates the per-image, per-category evaluation
#     # results in "evalImgs" into the dictionary "eval" with fields:
#     #  params     - parameters used for evaluation
#     #  date       - date evaluation was performed
#     #  counts     - [T,R,K,A,M] parameter dimensions (see above)
#     #  precision  - [TxRxKxAxM] precision for every evaluation setting
#     #  recall     - [TxKxAxM] max recall for every evaluation setting
#     # Note: precision and recall==-1 for settings with no gt objects.
#     #
#     # See also coco, mask, pycocoDemo, pycocoEvalDemo
#     #
#     # Microsoft COCO Toolbox.      version 2.0
#     # Data, paper, and tutorials available at:  http://mscoco.org/
#     # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
#     # Licensed under the Simplified BSD License [see coco/license.txt]
#     def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
#         '''
#         Initialize CocoEval using coco APIs for gt and dt
#         :param cocoGt: coco object with ground truth annotations
#         :param cocoDt: coco object with detection results
#         :return: None
#         '''
#         if not iouType:
#             print('iouType not specified. use default iouType segm')
#         self.cocoGt   = cocoGt              # ground truth COCO API
#         self.cocoDt   = cocoDt              # detections COCO API
#         self.params   = {}                  # evaluation parameters
#         self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
#         self.eval     = {}                  # accumulated evaluation results
#         self._gts = defaultdict(list)       # gt for evaluation
#         self._dts = defaultdict(list)       # dt for evaluation
#         self.params = Params(iouType=iouType) # parameters
#         self._paramsEval = {}               # parameters for evaluation
#         self.stats = []                     # result summarization
#         self.ious = {}                      # ious between all gts and dts
#         if not cocoGt is None:
#             self.params.imgIds = sorted(cocoGt.getImgIds())
#             self.params.catIds = sorted(cocoGt.getCatIds())


#     def _prepare(self, id_setup):
#         '''
#         Prepare ._gts and ._dts for evaluation based on params
#         :return: None
#         '''
#         p = self.params
#         if p.useCats:
#             gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
#             dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
#         else:
#             gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
#             dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

#         # set ignore flag
#         for gt in gts:
#             gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
#             # gt['ignore'] = 1 if (gt['height'] < self.params.HtRng[id_setup][0] or gt['height'] > self.params.HtRng[id_setup][1]) or \
#             #    ( gt['vis_ratio'] < self.params.VisRng[id_setup][0] or gt['vis_ratio'] > self.params.VisRng[id_setup][1]) else gt['ignore']
#             gbox = gt['bbox']
#             gt['ignore'] = 1 if (gt['height'] < self.params.HtRng[id_setup][0] or gt['height'] > self.params.HtRng[id_setup][1] or \
#                gt['occlusion'] not in self.params.OccRng[id_setup] or \
#                gbox[0] < self.params.bndRng[0] or gbox[1] < self.params.bndRng[1] or \
#                gbox[0]+gbox[2] > self.params.bndRng[2] or gbox[1]+gbox[3] > self.params.bndRng[3])  else gt['ignore']

#         self._gts = defaultdict(list)       # gt for evaluation
#         self._dts = defaultdict(list)       # dt for evaluation
#         for gt in gts:
#             self._gts[gt['image_id'], gt['category_id']].append(gt)
#         for dt in dts:
#             self._dts[dt['image_id'], dt['category_id']].append(dt)
#         self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
#         self.eval     = {}                  # accumulated evaluation results

#     def evaluate(self, id_setup):
#         '''
#         Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
#         :return: None
#         '''
#         tic = time.time()
#         print('Running per image evaluation...')
#         p = self.params
#         # add backward compatibility if useSegm is specified in params
#         if not p.useSegm is None:
#             p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
#             print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
#         print('Evaluate annotation type *{}*'.format(p.iouType))
#         p.imgIds = list(np.unique(p.imgIds))
#         if p.useCats:
#             p.catIds = list(np.unique(p.catIds))
#         p.maxDets = sorted(p.maxDets)
#         self.params=p

#         self._prepare(id_setup)
#         # loop through images, area range, max detection number
#         catIds = p.catIds if p.useCats else [-1]

#         computeIoU = self.computeIoU

#         self.ious = {(imgId, catId): computeIoU(imgId, catId) \
#                         for imgId in p.imgIds
#                         for catId in catIds}

#         evaluateImg = self.evaluateImg
#         maxDet = p.maxDets[-1]
#         HtRng = self.params.HtRng[id_setup]
#         # VisRng = self.params.VisRng[id_setup]
#         OccRng = self.params.OccRng[id_setup]
#         # self.evalImgs = [evaluateImg(imgId, catId, HtRng, VisRng, maxDet)
#         #          for catId in catIds
#         #          for imgId in p.imgIds
#         #      ]
#         self.evalImgs = [evaluateImg(imgId, catId, HtRng, OccRng, maxDet)
#                  for catId in catIds
#                  for imgId in p.imgIds
#              ]
        
#         self._paramsEval = copy.deepcopy(self.params)
#         toc = time.time()
#         print('DONE (t={:0.2f}s).'.format(toc-tic))


#     def computeIoU(self, imgId, catId):
#         p = self.params
#         if p.useCats:
#             gt = self._gts[imgId,catId]
#             dt = self._dts[imgId,catId]
#         else:
#             gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
#             dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
#         if len(gt) == 0 and len(dt) ==0:
#             return []
#         inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
#         dt = [dt[i] for i in inds]
#         if len(dt) > p.maxDets[-1]:
#             dt=dt[0:p.maxDets[-1]]


#         if p.iouType == 'segm':
#             g = [g['segmentation'] for g in gt]
#             d = [d['segmentation'] for d in dt]
#         elif p.iouType == 'bbox':
#             g = [g['bbox'] for g in gt]
#             d = [d['bbox'] for d in dt]
#         else:
#             raise Exception('unknown iouType for iou computation')


#         # compute iou between each dt and gt region
#         iscrowd = [int(o['ignore']) for o in gt]
#         ious = self.iou(d,g,iscrowd)
#         return ious

#     def iou( self, dts, gts, pyiscrowd ):
#         dts = np.asarray(dts)
#         gts = np.asarray(gts)
#         pyiscrowd = np.asarray(pyiscrowd)
#         ious = np.zeros((len(dts), len(gts)))
#         for j, gt in enumerate(gts):
#             gx1 = gt[0]
#             gy1 = gt[1]
#             gx2 = gt[0] + gt[2]
#             gy2 = gt[1] + gt[3]
#             garea = gt[2] * gt[3]
#             for i, dt in enumerate(dts):
#                 dx1 = dt[0]
#                 dy1 = dt[1]
#                 dx2 = dt[0] + dt[2]
#                 dy2 = dt[1] + dt[3]
#                 darea = dt[2] * dt[3]

#                 unionw = min(dx2,gx2)-max(dx1,gx1)
#                 if unionw <= 0:
#                     continue
#                 unionh = min(dy2,gy2)-max(dy1,gy1)
#                 if unionh <= 0:
#                     continue
#                 t = unionw * unionh
#                 if pyiscrowd[j]:
#                     unionarea = darea
#                 else:
#                     unionarea = darea + garea - t

#                 ious[i, j] = float(t)/unionarea
#         return ious



#     def evaluateImg(self, imgId, catId, hRng, oRng, maxDet):
#         '''
#         perform evaluation for single category and image
#         :return: dict (single image results)
#         '''
#         try:
#             p = self.params
#             if p.useCats:
#                 gt = self._gts[imgId,catId]
#                 dt = self._dts[imgId,catId]
#             else:
#                 gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
#                 dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
#             if len(gt) == 0 and len(dt) ==0:
#                 return None

#             for g in gt:
#                 if g['ignore']:
#                     g['_ignore'] = 1
#                 else:
#                     g['_ignore'] = 0
#             # sort dt highest score first, sort gt ignore last
#             gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
#             gt = [gt[i] for i in gtind]
#             dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
#             dt = [dt[i] for i in dtind[0:maxDet]]
#             # exclude dt out of height range
#             dt = [d for d in dt if d['height'] >= hRng[0] / self.params.expFilter and d['height'] < hRng[1] * self.params.expFilter]
#             dtind = np.array([int(d['id'] - dt[0]['id']) for d in dt])

#          

#             if len(dt) == 0:
#                 return None

#             # load computed ious        
#             ious = self.ious[imgId, catId][dtind, :] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]
#             # ious = self.ious[imgId, catId][dtind, :] if self.ious[imgId, catId].shape[1] > 0 else self.ious[imgId, catId]
#             ious = ious[:,gtind]


#             T = len(p.iouThrs)
#             G = len(gt)
#             D = len(dt)
#             gtm  = np.zeros((T,G))
#             dtm  = np.zeros((T,D))
#             gtIg = np.array([g['_ignore'] for g in gt])
#             dtIg = np.zeros((T,D))
#             if not len(ious)==0:
#                 for tind, t in enumerate(p.iouThrs):
#                     for dind, d in enumerate(dt):
#                         # information about best match so far (m=-1 -> unmatched)
#                         iou = min([t,1-1e-10])
#                         bstOa = iou
#                         bstg = -2
#                         bstm = -2
#                         for gind, g in enumerate(gt):
#                             m = gtm[tind,gind]
#                             # if this gt already matched, and not a crowd, continue
#                             if m>0:
#                                 continue
#                             # if dt matched to reg gt, and on ignore gt, stop
#                             if bstm!=-2 and gtIg[gind] == 1:
#                                 break
#                             # continue to next gt unless better match made
#                             if ious[dind,gind] < bstOa:
#                                 continue
#                             # if match successful and best so far, store appropriately
#                             bstOa=ious[dind,gind]
#                             bstg = gind
#                             if gtIg[gind] == 0:
#                                 bstm = 1
#                             else:
#                                 bstm = -1

#                         # if match made store id of match for both dt and gt
#                         if bstg ==-2:
#                             continue
#                         dtIg[tind,dind] = gtIg[bstg]
#                         dtm[tind,dind]  = gt[bstg]['id']
#                         if bstm == 1:
#                             gtm[tind,bstg]     = d['id']
#         except Exception as ex:
#             import traceback, sys

#             ex_type, ex_value, ex_traceback = sys.exc_info()            

#             # Extract unformatter stack traces as tuples
#             trace_back = traceback.extract_tb(ex_traceback)

#             # Format stacktrace
#             stack_trace = list()

#             for trace in trace_back:
#                 stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))

#             sys.stderr.write("[Error] Exception type : %s \n" % ex_type.__name__)
#             sys.stderr.write("[Error] Exception message : %s \n" %ex_value)
#             for trace in stack_trace:
#                 sys.stderr.write("[Error] (Stack trace) %s\n" % trace)



#         # store results for given image and category
#         return {
#                 'image_id':     imgId,
#                 'category_id':  catId,
#                 'hRng':         hRng,
#                 'oRng':         oRng,
#                 'maxDet':       maxDet,
#                 'dtIds':        [d['id'] for d in dt],
#                 'gtIds':        [g['id'] for g in gt],
#                 'dtMatches':    dtm,
#                 'gtMatches':    gtm,
#                 'dtScores':     [d['score'] for d in dt],
#                 'gtIgnore':     gtIg,
#                 'dtIgnore':     dtIg,
#             }

#     def accumulate(self, p = None):
#         '''
#         Accumulate per image evaluation results and store the result in self.eval
#         :param p: input params for evaluation
#         :return: None
#         '''
#         print('Accumulating evaluation results...')
#         tic = time.time()
#         if not self.evalImgs:
#             print('Please run evaluate() first')
#         # allows input customized parameters
#         if p is None:
#             p = self.params
#         p.catIds = p.catIds if p.useCats == 1 else [-1]
#         T           = len(p.iouThrs)
#         R           = len(p.fppiThrs)
#         K           = len(p.catIds) if p.useCats else 1
#         M           = len(p.maxDets)
#         ys   = -np.ones((T,R,K,M)) # -1 for the precision of absent categories

#         xx_graph = []
#         yy_graph = []

#         # create dictionary for future indexing
#         _pe = self._paramsEval
#         catIds = [1] #_pe.catIds if _pe.useCats else [-1]
#         setK = set(catIds)
#         setM = set(_pe.maxDets)
#         setI = set(_pe.imgIds)
#         # get inds to evaluate
#         k_list = [n for n, k in enumerate(p.catIds)  if k in setK]

#         m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
#         i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
#         I0 = len(_pe.imgIds)

#         # retrieve E at each category, area range, and max number of detections
#         for k, k0 in enumerate(k_list):
#             Nk = k0*I0
#             for m, maxDet in enumerate(m_list):
#                 E = [self.evalImgs[Nk + i] for i in i_list]
#                 E = [e for e in E if not e is None]
#                 if len(E) == 0:
#                     continue

#                 dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

#                 # different sorting method generates slightly different results.
#                 # mergesort is used to be consistent as Matlab implementation.

#                 inds = np.argsort(-dtScores, kind='mergesort')

#                 dtm = np.concatenate([e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:, inds]
#                 dtIg = np.concatenate([e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:, inds]
#                 gtIg = np.concatenate([e['gtIgnore'] for e in E])
#                 npig = np.count_nonzero(gtIg == 0)
#                 if npig == 0:
#                     continue
#                 tps = np.logical_and(dtm, np.logical_not(dtIg))
#                 fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))
#                 inds = np.where(dtIg==0)[1]
#                 tps = tps[:,inds]
#                 fps = fps[:,inds]

#                 tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
#                 fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)

            
#                 for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
#                     tp = np.array(tp)
#                     fppi = np.array(fp)/I0
#                     nd = len(tp)
#                     recall = tp / npig
#                     q = np.zeros((R,))


#                     xx_graph.append( fppi )
#                     yy_graph.append( 1-recall )

#                     # numpy is slow without cython optimization for accessing elements
#                     # use python array gets significant speed improvement
#                     recall = recall.tolist()
#                     q = q.tolist()

#                     for i in range(nd - 1, 0, -1):
#                         if recall[i] < recall[i - 1]:
#                             recall[i - 1] = recall[i]

#                     inds = np.searchsorted(fppi, p.fppiThrs, side='right') - 1
#                     try:
#                         for ri, pi in enumerate(inds):
#                             q[ri] = recall[pi]
#                     except:
#                         pass
#                     ys[t,:,k,m] = np.array(q)
        
#         self.eval = {
#             'params': p,
#             'counts': [T, R, K, M],
#             'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#             'TP':   ys,
#             'xx': xx_graph,
#             'yy': yy_graph
#         }
#         toc = time.time()
#         print('DONE (t={:0.2f}s).'.format( toc-tic))


#     def draw_figure(self, ax, filename='result.jpg'):
#         ### Draw
#   
#         mrs = 1 - self.eval['TP']
#         mean_s = np.log(mrs[mrs<2])
#         mean_s = np.mean(mean_s)
#         mean_s = float( np.exp(mean_s) * 100 )

#         xx = self.eval['xx']
#         yy = self.eval['yy']
        
#         # plt.clf()
#         # fig, ax = plt.subplots(figsize=(12,9))

#         ax.cla()

        
#         ax.plot( xx[0], yy[0], linewidth=3, label='{:.2f}%, {:s}'.format(mean_s, os.path.basename(filename)) )
#         ax.set_yscale('log')
#         ax.set_xscale('log')
#         ax.legend()

#         yt = [1,5] + list(range(10,60,10)) + [64, 80]
#         yticklabels=[ '.{:02d}'.format(num) for num in yt ]

#         yt += [100]
#         yt = [ yy/100.0 for yy in yt ]
#         yticklabels += [1]
        
#         ax.set_yticks(yt)
#         ax.set_yticklabels(yticklabels)
#         plt.grid(which='major', axis='both')
#         plt.ylim(0.01, 1)
#         plt.xlim(2e-4, 50)
#         plt.ylabel('miss rate')
#         plt.xlabel('false positives per image')
#         plt.savefig(filename)


#     def summarize(self,id_setup, res_file=None):
#         '''
#         Compute and display summary metrics for evaluation results.
#         Note this functin can *only* be applied on the default parameter setting
#         '''
#         def _summarize(iouThr=None, maxDets=100 ):
#             OCC_TO_TEXT = ['none', 'partial_occ', 'heavy_occ']

#             p = self.params
#             iStr = ' {:<18} {} @ {:<18} [ IoU={:<9} | height={:>6s} | visibility={:>6s} ] = {:0.2f}%'
#             titleStr = 'Average Miss Rate'
#             typeStr = '(MR)'
#             setupStr = p.SetupLbl[id_setup]
#             iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
#                 if iouThr is None else '{:0.2f}'.format(iouThr)
#             heightStr = '[{:0.0f}:{:0.0f}]'.format(p.HtRng[id_setup][0], p.HtRng[id_setup][1])
#             # occlStr = '[{:0.2f}:{:0.2f}]'.format(p.VisRng[id_setup][0], p.VisRng[id_setup][1])
#             occlStr = '[' + '+'.join([ '{:s}'.format(OCC_TO_TEXT[occ]) for occ in p.OccRng[id_setup] ]) + ']'

#             mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

#             # dimension of precision: [TxRxKxAxM]
#             s = self.eval['TP']
#             # IoU
#             if iouThr is not None:
#                 t = np.where(iouThr == p.iouThrs)[0]
#                 s = s[t]
#             mrs = 1-s[:,:,:,mind]

#             if len(mrs[mrs<2])==0:
#                 mean_s = -1
#             else:
#                 mean_s = np.log(mrs[mrs<2])
#                 mean_s = np.mean(mean_s)
#                 mean_s = np.exp(mean_s)
#             print(iStr.format(titleStr, typeStr,setupStr, iouStr, heightStr, occlStr, mean_s*100))

#             if res_file:
#                 res_file.write(iStr.format(titleStr, typeStr,setupStr, iouStr, heightStr, occlStr, mean_s*100))
#                 res_file.write('\n')
#             return mean_s

#         if not self.eval:
#             raise Exception('Please run accumulate() first')
        
#         return _summarize(iouThr=.5,maxDets=1000)

#     def __str__(self):
#         self.summarize()

# class Params:
#     '''
#     Params for coco evaluation api
#     '''
#     def setDetParams(self):
#         self.imgIds = []
#         self.catIds = []
#         # np.arange causes trouble.  the data point on arange is slightly larger than the true value

#         self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
#         self.fppiThrs = np.array([0.0100,    0.0178,    0.0316,    0.0562,    0.1000,    0.1778,    0.3162,    0.5623,    1.0000])
#         self.maxDets = [1000]
#         self.expFilter = 1.25
#         self.useCats = 1

#         self.iouThrs = np.array([0.5])  # np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)

#         self.HtRng = [[55, 1e5 ** 2], [50,75], [50, 1e5 ** 2], [20, 1e5 ** 2]]
#         # self.VisRng = [[0.65, 1e5 ** 2], [0.65, 1e5 ** 2], [0.2,0.65], [0.2, 1e5 ** 2]]
#         self.OccRng = [[0, 1], [0, 1], [2], [0, 1, 2]]
#         self.SetupLbl = ['Reasonable', 'Reasonable_small','Reasonable_occ=heavy', 'All']


#         self.bndRng = [5, 5, 635, 507];     # discard bbs outside this pixel range


#     def __init__(self, iouType='segm'):
#         if iouType == 'segm' or iouType == 'bbox':
#             self.setDetParams()
#         else:
#             raise Exception('iouType not supported')
#         self.iouType = iouType
#         # useSegm is deprecated
#         self.useSegm = None