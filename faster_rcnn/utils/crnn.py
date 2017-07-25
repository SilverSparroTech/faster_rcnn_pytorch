import torch
from torch.autograd import Variable
import torch.nn as nn
from faster_rcnn import utility
from faster_rcnn import network
from warpctc_pytorch import CTCLoss


# class BidirectionalLSTM(nn.Module):
#
#     def __init__(self, nIn, nHidden, nOut):
#         super(BidirectionalLSTM, self).__init__()
#
#         self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
#         self.embedding = nn.Linear(nHidden * 2, nOut)
#
#     def forward(self, input):
#         recurrent, _ = self.rnn(input)
#         T, b, h = recurrent.size()
#         t_rec = recurrent.view(T * b, h)
#
#         output = self.embedding(t_rec)  # [T * b, nOut]
#         output = output.view(T, b, -1)
#
#         return output


"""
class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, ngpu, n_rnn=2, leakyRelu=False):
        print("CRNN")
        super(CRNN, self).__init__()
        self.inputs = np.zeros((128,1,32,128))
        self.loss = None
        self.ngpu = ngpu
        self.image = torch.cuda.FloatTensor(128, 1, 32, 128)
        self.image = Variable(self.image,requires_grad=True)
        #print(self.image.size())

        # self.image=self.image.cuda()
        #print(self.image.size())
        self.frcnn = FasterRCNN(21,False)
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 3,2]
        ps = [1, 1, 1, 1, 1, 1, 1,0]
        ss = [2, 1, 1, 1, 1, 1, 1,1]
        nm = [64, 128, 256, 256, 512, 512, 512, 512]

        cnn = nn.Sequential()
        self.criterion = CTCLoss()
        self.criterion = self.criterion.cuda()
        self.prevcost = torch.cuda.FloatTensor(1)
        self.prevcost[0] = 0.0
        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 2),
                                                            (2, 1),
                                                            (1, 0)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 2),
                                                            (2, 1),
                                                            (1, 0)))  # 512x2x16
        convRelu(6, True)  # 512x1x16
        convRelu(7)
        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh, ngpu),
            BidirectionalLSTM(nh, nh, nclass, ngpu)
        )

    def forward(self,image_from_train,shape,im_data, im_info,gt_ocr, gt_boxes, gt_ishard, dontcare_areas):
        # conv features
        # pretrained_model = 'data/pretrained_model/VGG_imagenet.npy'
        # network.weights_normal_init(self.frcnn, dev=0.01)
        #
        # network.load_pretrained_npy(self.frcnn, pretrained_model)
        self.inputs,cpu_texts= self.frcnn.forward(image_from_train,shape,im_data, im_info,gt_ocr, gt_boxes, gt_ishard, dontcare_areas)
        #print(self.inputs.shape)
        # print cpu_texts
        # os.exit()

        #check
        cpu_texts=tuple(cpu_texts.reshape(1, -1)[0])

        #assert False
        self.image = network.np_to_variable(self.inputs)

        #
        # utility.loadData(self.image, self.inputs)
        print (self.image,"yoyo")
        alphabet='0123456789abcdefghijklmnopqrstuvwxyz:-#\'&!"$%&()*+-.:;<=>? ,/'
        # cpu_texts = self.frcnn.ocr

        print (cpu_texts),"cpppppppppu texxxxxxxxxxxts"
        converter = utility.strLabelConverter(alphabet)
        t, l = converter.encode(cpu_texts)
        # print '~~~~~~~',cpu_texts,t,l,self.image,'-------------'
        # conv = utility.data_parallel(self.cnn, self.image, self.ngpu)
        conv = self.cnn(self.image)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features

        output = self.rnn(conv)
        # print output.requires_grad
        #print(output.size(),"output size")
        #features, rois = self.frcnn.rpn(im_data, im_info, gt_boxes, gt_ishard, dontcare_areas)

        if self.training:

            text = torch.IntTensor(self.image.size(0) * 5)
            text=Variable(text)
            length = torch.IntTensor(self.image.size(0))
            length = Variable(length)
            utility.loadData(text, t)
            utility.loadData(length, l)
            #print(output,"tttttttttttttectxt")
            preds_size = Variable(torch.IntTensor([output.size(0)] * self.image.size(0)))
            # print text,length, preds_size,output

            cost = self.criterion(output, text, preds_size, length) / self.image.size(0)

            # print cost.requires_grad,"cost"
            # if np.isnan(np.sum(cost.data.cpu().numpy())):
            #     print('NaNaNaNaNaNa')
            #     cost=Variable(torch.from_numpy(self.prevcost.data.cpu().numpy()))

            cost=cost.cuda()
            # cost.zero_grad()

            # self.prevcost=cost
            _, preds = output.max(2)
            preds = preds.squeeze(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            # print sim_preds.requires_grad
            for pred, target in zip(sim_preds, cpu_texts):
                print(sim_preds)
            # loss=torch.cuda.FloatTensor((1))
            # loss=Variable(loss,requires_grad=True)
            #
            # loss[0]=cost
            print (cost,self.frcnn.loss,self.frcnn.rpn.loss,"all three losses")
            return cost+(self.frcnn.loss+self.frcnn.rpn.loss)

        #return output





    # def loss_out(self,):
    #     return self.loss
    """
class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut, ngpu):
        super(BidirectionalLSTM, self).__init__()
        self.ngpu = ngpu

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = utility.data_parallel(
            self.rnn, input, self.ngpu)  # [T, b, h * 2]

        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = utility.data_parallel(
            self.embedding, t_rec, self.ngpu)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, ngpu, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        self.ngpu = ngpu
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()
        self.criterion = CTCLoss()
        self.criterion = self.criterion.cuda()
        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 2),
                                                            (2, 1),
                                                            (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 2),
                                                            (2, 1),
                                                            (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh,ngpu),
            BidirectionalLSTM(nh, nh, nclass,ngpu)
        )

    def forward(self, input ,cpu_texts):
        # conv features
        image = network.np_to_variable(input)
        # conv = self.cnn(image)
        conv = utility.data_parallel(self.cnn, image, self.ngpu)

        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = utility.data_parallel(self.rnn, conv, self.ngpu)

        # return output
        # cpu_texts = tuple(cpu_texts.reshape(1, -1)[0])

        cpu_texts=tuple(cpu_texts)
        # assert False


        #
        # utility.loadData(self.image, self.inputs)
        # print (image, "yoyo")
        alphabet = '0123456789abcdefghijklmnopqrstuvwxyz:-#\&\'!"$%&()*+-.:;<=>? ,/'
        # cpu_texts = self.frcnn.ocr

        print (cpu_texts), "cpppppppppu texxxxxxxxxxxts"
        converter = utility.strLabelConverter(alphabet)
        t, l = converter.encode(cpu_texts)

        text = torch.IntTensor(image.size(0) * 5)
        text=Variable(text)
        length = torch.IntTensor(image.size(0))
        length = Variable(length)
        utility.loadData(text, t)
        utility.loadData(length, l)
        #print(output,"tttttttttttttectxt")
        preds_size = Variable(torch.IntTensor([output.size(0)] * image.size(0)))
        # print text,length, preds_size,output

        cost = self.criterion(output, text, preds_size, length) / image.size(0)
        cost = cost.cuda()
        # cost.zero_grad()

        # self.prevcost=cost
        _, preds = output.max(2)
        preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        # print sim_preds.requires_grad
        for pred, target in zip(sim_preds, cpu_texts):
            print(sim_preds)
        return cost

