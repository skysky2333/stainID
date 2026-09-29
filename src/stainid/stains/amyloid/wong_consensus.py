"""Model classes from the Wong et al. (2022) amyloid consensus CNNs (Keiser lab), kept verbatim so the published weights unpickle."""
import math

import torch
import torch.nn as nn


class Net(nn.Module):
    """
    The CNN architecture
    """
    def __init__(self, fc_nodes=512, num_classes=3, dropout=0.5):
        super(Net, self).__init__()
        self.drop = 0.2
        self.features = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1),
                                      nn.BatchNorm2d(16),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      
                                      nn.Conv2d(16, 32, 3, padding=1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      
                                      nn.Conv2d(32, 48, 3, padding=1),
                                      nn.BatchNorm2d(48),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      
                                      nn.Conv2d(48, 64, 3, padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      
                                      nn.Conv2d(64, 80, 3, padding=1),
                                      nn.BatchNorm2d(80),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      
                                      nn.Conv2d(80, 96, 3, padding=1),
                                      nn.BatchNorm2d(96),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),)
        
        self.classifier = nn.Sequential(nn.Linear(96 * 4 * 4, num_classes))
        self.train_loss_curve = []
        self.dev_loss_curve = []
        self.train_auprc = []
        self.dev_auprc = []

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class CustomizedLinearFunction(torch.autograd.Function):
    """
    Autograd function which masks it's weights by 'mask'. 
    reference: https://github.com/uchida-takumi/CustomizedLinear
    """
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias, mask is an optional argument
    def forward(ctx, input, weight, bias=None, mask=None):
        if mask is not None:
            # change weight to 0 where mask == 0
            weight = weight * mask
        output = input.mm(weight.t()) ##mm is matrix multiplication
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        ctx.save_for_backward(input, weight, bias, mask)
        return output
    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_mask = None
        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require them is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
            if mask is not None:
                # change grad_weight to 0 where mask == 0
                grad_weight = grad_weight * mask
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
        return grad_input, grad_weight, grad_bias, grad_mask

class EnsembleNet(nn.Module):
    """
    Ensemble net that weights individual constituent models using sparse feed forward connections
    Each consituent model's final layer (each of size 3) is concatenated together into a single layer
    Mask is a 2D matrix that specifies which connections from this concatenated layer of size(# of models x 3) to keep (1), and which connections to delete (0) when wiring to the final 3 neuron layer of the ensemble net
    Each EnsembleNet will have all 7 annotator models as attributes no matter what, BUT we'll selectively choose which models to actually ensemble in the forward definition 
    This way we have the option to forward novice nets if we want to (but in this particular study, we do not)
    Random nets will be models 8 - 12, and instantiated if passed as parameters, else won't be attributes
    """
    def __init__(self, mask, model1, model2, model3, model4, model5, model6, model7,model8=None,model9=None,model10=None,model11=None,model12=None,bias=True,amateur_param=False):
        super(EnsembleNet, self).__init__()
        self.ensemble = True
        self.equally_weighted = False
        self.amateur_param = amateur_param
        self.input_features = mask.shape[0]
        self.output_features = mask.shape[1]
        if isinstance(mask, torch.Tensor):
            self.mask = mask.type(torch.float).t()
        else:
            self.mask = torch.tensor(mask, dtype=torch.float).t()
        self.mask = nn.Parameter(self.mask, requires_grad=False)
        self.weight = nn.Parameter(torch.Tensor(self.output_features, self.input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_features))
        else:
            # You should always register all possible parameters, but the optional ones can be None if you want.
            self.register_parameter('bias', None)
        self.reset_parameters()
        # mask weight
        self.weight.data = self.weight.data * self.mask
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4
        self.model5 = model5
        self.model6 = model6
        self.model7 = model7
        if model8 != None: ##using one random constituent net 
            self.model8 = model8
        else:
            self.model8 = None
        if model9 != None: ##using multiple constituent nets
            self.model9 = model9
            self.model10 = model10
            self.model11 = model11
            self.model12 = model12
        else:
            self.model9, self.model10, self.model11, self.model12 = None, None, None, None
        self.train_loss_curve = []
        self.dev_loss_curve = []
        self.train_auprc = []
        self.dev_auprc = []

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        x3 = self.model3(x)
        x4 = self.model4(x)
        x5 = self.model5(x)
        x6 = self.model6(x)
        x7 = self.model7(x)
        ##if random constituent nets are being used they will be forwarded
        if self.model8 != None: 
            x8 = self.model8(x)
        if self.model9 != None:
            x9 = self.model9(x)
            x10 = self.model10(x)
            x11 = self.model11(x)
            x12 = self.model12(x)

        ##if we're using (forwarding) the two amateur raters in our ensemble 
        if self.amateur_param:
            ##plus one random constituent net
            if self.model8 != None and self.model9 == None:
                x = torch.cat((x1,x2,x3,x4,x5,x6,x7,x8), dim=1) 
            ##plus multiple random nets
            elif self.model9 != None:
                x = torch.cat((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12), dim=1) 
            ##no random nets
            else:
                x = torch.cat((x1,x2,x3,x4,x5,x6,x7), dim=1)
        ##if we're NOT using (forwarding) the two amateur raters in our ensemble 
        else:
            if self.model8 != None and self.model9 == None:
                x = torch.cat((x2,x4,x5,x6,x7,x8), dim=1)
            elif self.model9 != None:
                x = torch.cat((x2,x4,x5,x6,x7,x8,x9,x10,x11,x12), dim=1)
            else:
                x = torch.cat((x2,x4,x5,x6,x7), dim=1)
        return CustomizedLinearFunction.apply(x, self.weight, self.bias, self.mask)
        
    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )
   
class EquallyWeightedEnsembleNet(nn.Module):
    """
    Ensemble that simply forwards each constituent net with equal weighting
    """
    def __init__(self, model1, model2, model3, model4, model5, model6, model7,model8=None,model9=None,model10=None,model11=None,amateur_param=True):
        super(EquallyWeightedEnsembleNet, self).__init__()
        self.ensemble = True
        self.equally_weighted = True
        self.amateur_param = amateur_param
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4
        self.model5 = model5
        self.model6 = model6
        self.model7 = model7
        if model8 != None:
            self.model8 = model8
        else:
            self.model8 = None
        if model9 != None:
            self.model9 = model9
            self.model10 = model10
            self.model11 = model11
            self.model12 = model12
        else:
            self.model9, self.model10, self.model11, self.model12 = None, None, None, None
        self.train_loss_curve = []
        self.dev_loss_curve = []
        self.train_auprc = []
        self.dev_auprc = []

    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        x3 = self.model3(x)
        x4 = self.model4(x)
        x5 = self.model5(x)
        x6 = self.model6(x)
        x7 = self.model7(x)
        if self.model8 != None:
            x8 = self.model8(x)
        if self.model9 != None:
            x9 = self.model9(x)
            x10 = self.model10(x)
            x11 = self.model11(x)
            x12 = self.model12(x)
        if self.amateur_param:
            if self.model8 != None and self.model9 == None:
                x = torch.stack((x1,x2,x3,x4,x5,x6,x7,x8)) 
            elif self.model9 != None:
                x = torch.stack((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12)) 
            else:
                x = torch.stack((x1,x2,x3,x4,x5,x6,x7)) 
        else:
            if self.model8 != None and self.model9 == None:
                x = torch.stack((x2,x4,x5,x6,x7,x8))
            elif self.model9 != None:
                x = torch.stack((x2,x4,x5,x6,x7,x8,x9,x10,x11,x12))
            else:
                x = torch.stack((x2,x4,x5,x6,x7)) 
        avg = torch.mean(x, dim=0) 
        return avg 
