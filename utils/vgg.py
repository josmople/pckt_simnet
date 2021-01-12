from torch.nn import Module as _Module

import torchvision.models.vgg as models

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

NORMALIZE_MEAN_255 = [v * 255 for v in NORMALIZE_MEAN]
NORMALIZE_STD_255 = [v * 255 for v in NORMALIZE_STD]


def normalize(tensor, mean=NORMALIZE_MEAN, std=NORMALIZE_STD, inplace=False):
    """
    Normalizes image tensor using VGG mean and std.
    Assumes image has [0,1] values.
    Accepts CxHxW or BxCxHxW images.
    """
    from torch import as_tensor

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = as_tensor(mean, dtype=dtype, device=tensor.device)
    std = as_tensor(std, dtype=dtype, device=tensor.device)

    if tensor.dim() == 4:
        tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    elif tensor.dim() == 3:
        tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
    else:
        raise NotImplementedError("Only accepts 3D or 4D tensor")

    return tensor


def normalize_255(tensor, mean=NORMALIZE_MEAN_255, std=NORMALIZE_STD_255, inplace=False):
    """
    Normalizes image tensor using VGG mean and std.
    Assumes image has [0,255] values.
    Accepts CxHxW or BxCxHxW images.
    """
    return normalize(tensor, mean, std, inplace)


def denormalize(tensor, mean=NORMALIZE_MEAN, std=NORMALIZE_STD, inplace=False):
    """
    Denormalizes image tensor using VGG mean and std.
    Assumes image has [0,1] values.
    Accepts CxHxW or BxCxHxW images.
    """
    from torch import as_tensor

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = as_tensor(mean, dtype=dtype, device=tensor.device)
    std = as_tensor(std, dtype=dtype, device=tensor.device)

    if tensor.dim() == 4:
        tensor.mul_(std[None, :, None, None]).add_(mean[None, :, None, None])
    elif tensor.dim() == 3:
        tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
    else:
        raise NotImplementedError("Only accepts 3D or 4D tensor")

    return tensor


def denormalize_255(tensor, mean=NORMALIZE_MEAN_255, std=NORMALIZE_STD_255, inplace=False):
    """
    Denormalizes image tensor using VGG mean and std.
    Assumes image has [0,255] values.
    Accepts CxHxW or BxCxHxW images.
    """
    return denormalize(tensor, mean, std, inplace)


class VGGExtractor(_Module):

    """
    Allows extraction of features from specific layers of the VGG network.
    """

    @staticmethod
    def layername_index_mapping(features):
        """
        Generates a mapping of {layername: index} based on the VGG layers (i.e. features)
        """
        from torch.nn import Conv2d, ReLU, BatchNorm2d, MaxPool2d

        mapping = {}

        n, m = 1, 0
        i = 0
        for i, layer in enumerate(features):
            if isinstance(layer, Conv2d):
                m += 1
                mapping["conv{}_{}".format(n, m)] = i
            elif isinstance(layer, ReLU):
                mapping["relu{}_{}".format(n, m)] = i
            elif isinstance(layer, BatchNorm2d):
                mapping["batch{}_{}".format(n, m)] = i
            elif isinstance(layer, MaxPool2d):
                mapping["pool{}".format(n)] = i
                n += 1
                m = 0
        mapping["last"] = i
        return mapping

    @staticmethod
    def fetches_to_idxs(fetches, mapping):
        """
        Turns a List[Union[str, int]] to List[int]
        int values in fetches refers to layer index
        str values in fetches refers to layer name (see self.layernames or mapping)
        Special indexes
        -1 -> corresponds to input post-normalization, if no normalization has similar value with '-2'
        -2 -> corresponds to raw input
        """
        idxs = []
        for idx in fetches:
            if isinstance(idx, int):
                idxs.append(idx)
            elif isinstance(idx, str):
                try:
                    idx = mapping[idx]
                except:
                    raise ValueError("Layer `{}` not found".format(idx))
                idxs.append(idx)
            else:
                raise ValueError("Expected `fetches` to be list[int], list[str]")
        return idxs

    def __init__(self, vgg_nlayer=19, vgg_bn=False, requires_grad=False, inplace_relu=False, max_layer=None, normalize_fn=None, pretrained=False, **kwds):
        super().__init__()

        assert isinstance(vgg_nlayer, (int, str))
        assert isinstance(vgg_bn, bool)
        assert isinstance(requires_grad, bool)
        assert isinstance(inplace_relu, bool)
        assert callable(normalize_fn) or normalize_fn is None

        model_name = f"vgg{vgg_nlayer}{'_bn' if vgg_bn else ''}"
        self.model = getattr(models, model_name)(pretrained=pretrained, **kwds).features
        self.mapping = self.__class__.layername_index_mapping(self.model)
        self.normalize = normalize_fn

        max_idx = (self.mapping[str(max_layer)] + 1) if isinstance(max_layer, str) else max_layer
        self.model = self.model[:max_idx]

        for param in self.model.parameters():
            param.requires_grad_(requires_grad)

        from torch.nn import ReLU
        for layer in self.model:
            if isinstance(layer, ReLU):
                layer.inplace = inplace_relu

    @property
    def layernames(self):
        return list(self.mapping.keys())

    @property
    def layernames_valid(self):
        return self.layernames[:len(self.model)]

    @property
    def fetchlist(self):
        out = [-2, *range(len(self.model)), *self.layernames_valid]
        if self.normalize is not None:
            out.insert(0, -1)
        return out

    def forward(self, x, fetches=None):
        if fetches is None:
            if self.normalize is not None:
                x = self.normalize(x)
            return self.model(x)
        if isinstance(fetches, (int, str)):
            fetches = [fetches]
            fetches = self.__class__.fetches_to_idxs(fetches, self.mapping)
            return self._forward(x, fetches)[0]
        if isinstance(fetches, (tuple, list)):
            fetches = self.__class__.fetches_to_idxs(fetches, self.mapping)
            return self._forward(x, fetches)
        raise ValueError("Expected `fetches` to be int, str, list[int], list[str]")

    def _forward(self, x, idxs):
        assert isinstance(idxs, (list, tuple))
        assert all([isinstance(idx, int) for idx in idxs])
        assert all([idx >= -2 for idx in idxs])

        outputs = {}

        y = x
        if -2 in idxs:
            outputs[-2] = y

        if self.normalize is not None:
            y = self.normalize(y)
        if -1 in idxs:
            outputs[-1] = y

        last_idx = max(idxs)

        for idx, layer in enumerate(self.model):
            y = layer(y)

            if idx in idxs:
                outputs[idx] = y

            if idx >= last_idx:
                break

        return [outputs[idx] for idx in idxs]


del _Module

if __name__ == "__main__":
    from torch import no_grad, randn

    with no_grad():
        # Will call torchvision.models.vgg.vgg16_bn(**kwds)
        instance = VGGExtractor(vgg_nlayer=16, vgg_bn=True)

        print("# Backbone Network")
        print(instance.model)
        print("_________________________________")

        print("# Layer aliasing")
        for name, idx in instance.mapping.items():
            print(f"{idx:2} {name}")
        print("_________________________________")

        # Dummy input
        print("# Dummy input")
        x = randn(10, 3, 64, 64)
        print(x.size())
        print("_________________________________")

        print("# Full output of VGG")
        print(instance(x).size())
        print("_________________________________")

        print("# Output of VGG at Layer 5 / relu1_2")
        out1, out2 = instance(x, [5, "relu1_2"])
        print("Layer 5 and Layer relu1_2 are the same layers: ", (out1 == out2).all().item())
        print("_________________________________")

        print("# Outputs follows the fetch order")
        fetches = [3, 29, "batch5_2", 3, "relu1_2", 19]
        outs = instance(x, fetches)
        for layer, out in zip(fetches, outs):
            print(layer, " ->", out.size())
        print("_________________________________")

        print("# Special fetch indeces")
        instance = VGGExtractor(vgg_nlayer=13, vgg_bn=False, normalize_fn=normalize)
        fetches = [-2, -1, "last"]
        same_as_x, after_normalize, last = instance(x, fetches)
        print("Index -1 is the same as the input: ", (same_as_x == x).all().item())
        print("Index -2 is the output after normalize : ", (after_normalize == normalize(x)).all().item())
        print("Fetch 'last' is final output : ", (last == instance(x)).all().item())
        print("_________________________________")

        print("# Index -2 and -1 are same without normalization function")
        fetches = [-2, -1]
        instance = VGGExtractor(vgg_nlayer=13, vgg_bn=False, normalize_fn=None)
        same_as_x, after_normalize = instance(x, fetches)
        print("Index -1 and Index -2 are same: ", (same_as_x == after_normalize).all().item())
        print("_________________________________")
