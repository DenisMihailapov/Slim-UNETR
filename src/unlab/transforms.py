from typing import Any
from monai.data import MetaTensor, get_track_meta
from monai.utils import convert_to_tensor

from monai.transforms import Flip, Rotate
from monai.transforms import RandomizableTransform
from monai.transforms.traits import InvertibleTrait
from monai.transforms.spatial import array


class RandFlip(RandomizableTransform, InvertibleTrait):
    backend = Flip.backend

    def __init__(self, prob: float = 0.1, spatial_axis=1) -> None:
        RandomizableTransform.__init__(self, prob)
        self._transform = Flip(spatial_axis=spatial_axis, lazy=False)

    def __call__(self, img: MetaTensor, randomize: bool = True) -> MetaTensor:
        if randomize:
            self.randomize(None)

        if self._do_transform:
            img = self._transform(img.as_tensor())
            img = convert_to_tensor(img, track_meta=get_track_meta())

        return img

    def inverse(self, img: MetaTensor) -> MetaTensor:
        if not self._do_transform:
            return img
        img = self._transform(img.as_tensor())
        img = convert_to_tensor(img, track_meta=get_track_meta())
        return img


class RandRotate(RandomizableTransform, InvertibleTrait):
    backend = Rotate.backend

    def __init__(
            self,
            prob=0.1,
            range_x=0.,
            range_y=0.,
            range_z=0.,
            mode="nearest",
            padding_mode="zeros"
    ) -> None:
        RandomizableTransform.__init__(self, prob)
        self.range_x = tuple(sorted([-range_x, range_x]))
        self.range_y = tuple(sorted([-range_y, range_y]))
        self.range_z = tuple(sorted([-range_z, range_z]))

        self.x, self.y, self.z = 0., 0., 0.
        self._transform = Rotate(
            angle=(self.x, self.y, self.z),
            mode=mode,
            padding_mode=padding_mode,
            keep_size=True,
            align_corners=True,
            lazy=False
        )

    def randomize(self, data: Any | None = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self.x = self.R.uniform(low=self.range_x[0], high=self.range_x[1])
        self.y = self.R.uniform(low=self.range_y[0], high=self.range_y[1])
        self.z = self.R.uniform(low=self.range_z[0], high=self.range_z[1])
        self._transform.angle = (self.x, self.y, self.z)

    def __call__(self, img: MetaTensor, randomize: bool = True) -> MetaTensor:
        if randomize:
            self.randomize(None)

        if self._do_transform:
            img = self._transform(img.as_tensor())
            img = convert_to_tensor(img, track_meta=get_track_meta())

        return img

    def inverse(self, img: MetaTensor) -> MetaTensor:
        if not self._do_transform:
            return img
        self._transform.angle = (-self.x, -self.y, -self.z)
        img = self._transform(img.as_tensor())
        self._transform.angle = (self.x, self.y, self.z)
        img = convert_to_tensor(img, track_meta=get_track_meta())
        return img


class Transforms:

    def __init__(
            self,
            rot_prob: float = 0.8,
            rot_range_x: float = 0.0,
            rot_range_y: float = 0.0,
            rot_range_z: float = 0.1 * 3.141592,
            flip_prob: float = 0.3,
            final_image_size=128
    ):
        self._transforms = [
            RandRotate(  # RandAffine
                prob=rot_prob,
                range_x=rot_range_x,
                range_y=rot_range_y,
                range_z=rot_range_z,
                mode="nearest",
                padding_mode="zeros"
            ),
            RandFlip(
                prob=flip_prob,
                spatial_axis=0
            )
        ]

        self.final_image_size = final_image_size
        self._dim_z_resize = None

    def _center_crop_fc(self, image):
        # image  B, C=1, H, W, D
        image = image[:, 0, ...]
        center = image.shape[2] // 2
        window = self.final_image_size // 2
        crop_range = [
            center - window,
            center + window,
        ]
        self._dim_z_resize = array.Resize(
            spatial_size=(image.shape[2], image.shape[2], self.final_image_size)
        )
        image = self._dim_z_resize(image)
        return image[
               :,
               crop_range[0]:crop_range[1],
               crop_range[0]:crop_range[1],
               :
               ][:, None, ...]

    def __call__(self, img):
        img = img[:, 0, ...]
        for i in range(len(self._transforms)):
            img = self._transforms[i](img)
        img = img[:, None, ...]
        img = self._center_crop_fc(img)
        return img

    def inverse(self, img):
        img = img[:, 0, ...]
        for i in range(len(self._transforms))[::-1]:
            img = self._transforms[i].inverse(img)
        img = img[:, None, ...]
        return img

    def inverse_predict(self, predict):
        orig_shape = predict.shape
        rand_rot_lab = self.inverse(predict.reshape(
            *(orig_shape[0] * orig_shape[1], orig_shape[2], orig_shape[2], orig_shape[2]))).reshape(*orig_shape)

        return rand_rot_lab